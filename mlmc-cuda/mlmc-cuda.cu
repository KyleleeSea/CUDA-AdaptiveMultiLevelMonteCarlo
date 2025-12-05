#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <math.h>
#include <unistd.h> 
#include <iostream>
#include <random>
#include <chrono>
#include <iomanip> 
#include <functional>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <curand_kernel.h>

#include "mlmc-cuda.h"

////////////////////////////////////////////////////////////////////////////////////////
// Constants, Definitions, Etc
///////////////////////////////////////////////////////////////////////////////////////
#define MAX_LEVELS 21  
#define THREAD_DIM 1024
#define MAX_BLOCK_X 256
#define MAX_NUM_THREADS (THREAD_DIM)*(MAX_BLOCK_X) // 2^(9*2) = 262,144
#define RAND_SEED 67
#define SAMPLE_LEVEL_CUTOFF 8 // Found experimentally

////////////////////////////////////////////////////////////////////////////////////////
// All cuda kernels (GPU code) here
///////////////////////////////////////////////////////////////////////////////////////

// Precondition: curandState* is malloc'd to size MAX_NUM_THREADS 
// Initializes MAX_NUM_THREADS independent curand states
__global__ void setup_curand(curandState* state) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // Initialize the state for the current thread
    curand_init(RAND_SEED, tid, 0, &state[tid]); 
}

__device__ __forceinline__
double milstein_factor(double r, double sig, double h, double dW)
{
    float dw2 = dW * dW;
    return 1.0f + r*h + sig*dW + 0.5f*sig*sig*(dw2 - h);
}

__device__ __forceinline__
void reduce_two(double* fine_factors, double* coarse_factors)
{
    int tid = threadIdx.x;

    //Tree reduction: multiply in place down to index 0, assmes blockDim.x is a power of 2
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            fine_factors[tid]   *= fine_factors[tid + offset];
            coarse_factors[tid] *= coarse_factors[tid + offset];
        }
        __syncthreads();
    }
    //fine_factors[0] and coarse_factors[0] hold the end result
}


/*
Kernel to compute samples at level 0, which is a special case where no
sample coupling is necessary and each sample is only one "timestep"

Input:
- sums_ptr: malloc'd to size of MAX_NUM_THREADS, store partial sums here
- squared_sums_ptr: malloc'd to size MAX_NUM_THREADS, store partial squared sums
- samples_per_thread: number of samples each thread should take
- total_samples: total number of samples that need to be taken
- T: time to maturity
- sqrtT: square root of T
- independent_term: K + r*K*T
- modifier_a: sig*K
- modifier_b: 0.5f*sig*sig*K
- exp_term: exp(-r*T)
- K: strike price
- curand_state: array of independent curand states,
   malloc'd and initialized to MAX_NUM_THREADS 

Output:
- sums_ptr[tid] = partial sum of samples for this thread
- squared_sums_ptr[tid] = partial squared sum of samples for this thread
*/
__global__ void mlmc_l0(double* sums_ptr, double* squared_sums_ptr, unsigned long samples_per_thread, 
    unsigned long total_samples, double T, double sqrtT, double independent_term, double modifier_a, double modifier_b, 
    double exp_term, double K, curandState* curand_state) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        // early exit if already got enough samples
        unsigned long curr = (unsigned long)tid * samples_per_thread;
        if (curr >= total_samples) return;
        unsigned long rem = total_samples-curr;
        unsigned long num_samples = (rem < samples_per_thread) ? rem : samples_per_thread;

        double local_sum = 0;
        double local_squared_sum = 0;

        // It is more efficient to generate 2 normals at a time, so we process in
        // batches of 2
        for (size_t i = 0; i < num_samples;) {
            float2 normals = curand_normal2(&curand_state[tid]);
            float dWf_arr[2];
            dWf_arr[0] = sqrtT * normals.x;
            dWf_arr[1] = sqrtT * normals.y;

            for (size_t j = 0; j < 2 && i<num_samples; j++) {
                float dWf = dWf_arr[j];
                // Xf  = K + r*K*T + sig*K*dWf + 0.5f*sig*sig*K*(dWf*dWf-T);
                float Xf = independent_term + modifier_a*dWf + modifier_b*(dWf*dWf-T);
                
                float dP = exp_term*fmaxf(0.0f,Xf-K);
                local_sum += dP;
                local_squared_sum += dP * dP;
                i++;
            }
        }

        // Write to global device memory
        sums_ptr[tid] += local_sum;
        squared_sums_ptr[tid] += local_squared_sum;
}

/*
Kernel to compute samples at level > 0 where we take samples in parallel

Input:
- sums_ptr: malloc'd to size of MAX_NUM_THREADS, store partial sums here
- squared_sums_ptr: malloc'd to size MAX_NUM_THREADS, store partial squared sums
- samples_per_thread: number of samples each thread should take
- total_samples: total number of samples that need to be taken
- l: the current level
- T: time to maturity
- r: risk free interest rate
- sig: asset volatility
- K: strike price
- curand_state: array of independent curand states,
   malloc'd and initialized to MAX_NUM_THREADS 

Output:
- sums_ptr[tid] = partial sum of samples for this thread
- squared_sums_ptr[tid] = partial squared sum of samples for this thread
*/
__global__ void mlmc_parallel_sample_level(double* sums_ptr, double* squared_sums_ptr, unsigned long samples_per_thread, 
    unsigned long total_samples, int l, double T, double r, double sig, double K, curandState* curand_state) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    unsigned long curr = (unsigned long)tid * samples_per_thread;
    if (curr >= total_samples) return; // early exit if already got enough samples
    unsigned long rem = total_samples-curr;
    unsigned long num_samples = (rem < samples_per_thread) ? rem : samples_per_thread;

    int   nf, nc;
    float hf, hc;

    nf = 1<<l;
    nc = nf/2;

    hf = T / ((float) nf);
    hc = T / ((float) nc);

    double local_sum = 0;
    double local_squared_sum = 0;

    for (size_t i = 0; i < num_samples; i++) {
        float Xf, Xc, dWc, Pf, Pc, dP;
        float2 normals;
        float dWf[2];

        Xf = K;
        Xc = Xf;

        for (int n=0; n<nc; n++) {
            normals = curand_normal2(&curand_state[tid]);
            dWf[0] = sqrt(hf) * normals.x;
            dWf[1] = sqrt(hf) * normals.y;
            
            for (int m=0; m<2; m++) {
                Xf  = Xf + r*Xf*hf + sig*Xf*dWf[m]
                        + 0.5f*sig*sig*Xf*(dWf[m]*dWf[m]-hf);
            }

            dWc = dWf[0] + dWf[1];

            Xc  = Xc + r*Xc*hc + sig*Xc*dWc + 0.5f*sig*sig*Xc*(dWc*dWc-hc);
        }

        Pf  = fmaxf(0.0f,Xf-K);
        Pc  = fmaxf(0.0f,Xc-K);

        dP  = exp(-r*T)*(Pf-Pc);
        local_sum += dP;
        local_squared_sum += dP * dP;
    }

    // Write to global device memory
    sums_ptr[tid] += local_sum;
    squared_sums_ptr[tid] += local_squared_sum;
}

/*
Kernel to compute samples at level > 0 where we take timesteps within a sample
in parallel

Input:
- sums_ptr: malloc'd to size of MAX_NUM_THREADS, store partial sums here
- squared_sums_ptr: malloc'd to size MAX_NUM_THREADS, store partial squared sums
- samples_per_block: number of samples each thread block should take
- total_samples: total number of samples that need to be taken
- l: the current level
- T: time to maturity
- r: risk free interest rate
- sig: asset volatility
- K: strike price
- curand_state: array of independent curand states,
   malloc'd and initialized to MAX_NUM_THREADS 

Output:
- sums_ptr[tid] = partial sum of samples for this thread
- squared_sums_ptr[tid] = partial squared sum of samples for this thread
*/
__global__ void mlmc_parallel_timestep_level(double* sums_ptr, double* squared_sums_ptr, unsigned long samples_per_block, 
    unsigned long total_samples, int l, double T, double r, double sig, double K, curandState* curand_state){
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    unsigned long curr = (unsigned long)blockIdx.x * samples_per_block;
    if (curr >= total_samples) return; // early exit if already got enough samples
    unsigned long rem = total_samples-curr;
    unsigned long num_samples = (rem < samples_per_block) ? rem : samples_per_block;
        
    int   nf, nc;
    float hf, hc;

    nf = 1<<l;
    nc = nf/2;

    hf = T / ((float) nf);
    hc = T / ((float) nc);

    double local_sum = 0;
    double local_squared_sum = 0;

    // Use shared block memory to improve runtime
    __shared__ double coarse_factors[THREAD_DIM];
    __shared__ double fine_factors[THREAD_DIM];

    for (size_t i = 0; i < num_samples; i++) {

        float Xf, Xc, dWc, Pf, Pc, dP;
        float2 normals;
        float dWf[2];

        Xf = K;
        Xc = Xf;
 
        for (int base = 0; base < nc; base += blockDim.x) {

            int sid = base + threadIdx.x;
           
            if (sid < nc) {
                normals = curand_normal2(&curand_state[tid]);
                dWf[0] = sqrt(hf) * normals.x;
                dWf[1] = sqrt(hf) * normals.y;
                dWc = dWf[0] + dWf[1];
                fine_factors[threadIdx.x] = milstein_factor(r, sig, hf, dWf[0]) * milstein_factor(r, sig, hf, dWf[1]);
                coarse_factors[threadIdx.x] =  milstein_factor(r, sig, hc, dWc);
            } else {
                fine_factors[threadIdx.x] = 1.0;
                coarse_factors[threadIdx.x] = 1.0;
            }

            __syncthreads();

            reduce_two(fine_factors, coarse_factors);
            if (threadIdx.x == 0) {
                Xf *= fine_factors[0];
                Xc *= coarse_factors[0];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            Pf  = fmaxf(0.0f,Xf-K);
            Pc  = fmaxf(0.0f,Xc-K);

            dP  = exp(-r*T)*(Pf-Pc);
            local_sum += dP;
            local_squared_sum += dP * dP;
        }
    }

    if (threadIdx.x == 0) {
        sums_ptr[blockIdx.x] += local_sum;
        squared_sums_ptr[blockIdx.x] += local_squared_sum;
    }
}


////////////////////////////////////////////////////////////////////////////////////////
// All CPU code here 
///////////////////////////////////////////////////////////////////////////////////////


float mlmc(int Lmin, int Lmax, int N0, float eps, bool diag, double T, double r, double sigma, double K) {

  /*
  suml: accumulator over all samples taken for the given mlmc level
  suml[0][l] = total # steps taken over all samples for the given level
  suml[1][l] = sum over |Y_l - Y_{l-1}|
  suml[2][l] = sum over |Y_l - Y_{l-1}|^2

  runtimes[l]: array of the total time spent at level l
  runtime_by_part[1]: total time spent in Thrust array initialization
  runtime_by_part[2]: total time spent in kernel runtime
  runtime_by_part[3]: total time spent in Thrust reduction

  ml[l] = mean |Y_l - Y_{l-1}| at level l 
  Vl[l] = variance at level l 
  Cl[l] = # steps to take per sample at level l
  alpha, beta, gamma are hyperparameters
  dNl[l] = number of additional samples to take next iteration
  L = current number of levels
  converged = 1 if converged and 0 if not
  */
  double suml[3][MAX_LEVELS], runtimes[MAX_LEVELS], runtime_by_part[3];
  float  ml[MAX_LEVELS], Vl[MAX_LEVELS], Cl[MAX_LEVELS], alpha, beta, gamma, sum, theta;
  int    L, converged;
  unsigned long dNl[MAX_LEVELS];

  //
  // check input parameters
  //
  if (Lmin<2) {
    fprintf(stderr,"error: needs Lmin >= 2 \n");
    exit(1);
  }
  if (Lmax<Lmin) {
    fprintf(stderr,"error: needs Lmax >= Lmin \n");
    exit(1);
  }

  if (Lmax > MAX_LEVELS) {
    fprintf(stderr,"error: Lmax exceeds max_levels \n");
    exit(1);
  }

  if (N0<=0 || eps<=0.0f) {
    fprintf(stderr,"error: needs N>0, eps>0 \n");
    exit(1);
  }

  //
  // initialisation
  //
  alpha = 1.0f; // Based on Milstein discretization
  beta  = 1.0f; // Based on Milstein discretization
  gamma = 2.0f; // Based on Milstein discretization
  theta = 0.25f;// MSE split between bias^2 and variance (given by Giles)

  L = Lmin; // we start with Lmin levels
  converged = 0;

  for(int l=0; l<=Lmax; l++) { // initialize all the variables.
    Cl[l]   = powf(2.0f,(float)l*gamma);
    for(int n=0; n<3; n++) suml[n][l] = 0.0; // zero out the running sums for each possible level
    runtimes[l] = 0;
    runtime_by_part[0] = 0;
    runtime_by_part[1] = 0;
    runtime_by_part[2] = 0;
   }

  for(int l=0; l<=Lmin; l++) {dNl[l] = N0;} // Set initial number of samples to take

    // Setup curand
    auto curand_setup_start = std::chrono::steady_clock::now();

    curandState* curand_states;
    cudaMalloc((void**)&curand_states, (MAX_NUM_THREADS) * sizeof(curandState));

    dim3 pixelBlockDim(THREAD_DIM);
    dim3 pixelGridDim(MAX_BLOCK_X);
    setup_curand<<<pixelGridDim, pixelBlockDim>>>(curand_states);
    cudaDeviceSynchronize();

    auto curand_setup_end = std::chrono::steady_clock::now();
    double curand_setup_time = std::chrono::duration_cast<std::chrono::duration<double>>(curand_setup_end - curand_setup_start).count();

  //
  // main loop
  //
  while (!converged) {
    //
    // update sample sums
    //
    
    for (int l=0; l<=L; l++) {
      if (dNl[l]>0) { // skip levels where we don't need more samples
            auto start_timer = std::chrono::steady_clock::now();

            // Initialize thrust device vectors
            auto init_start = std::chrono::steady_clock::now();
            unsigned long vector_size = std::min(static_cast<unsigned long>(MAX_NUM_THREADS), dNl[l]);
            vector_size = std::max(static_cast<unsigned long>(MAX_BLOCK_X), vector_size);
            thrust::device_vector<double> l0_sums(vector_size, 0);
            thrust::device_vector<double> l0_squared_sums(vector_size, 0);

            // Populate sums and squared sums
            double* l0_sums_array = thrust::raw_pointer_cast( l0_sums.data() );
            double* l0_squared_sums_array = thrust::raw_pointer_cast( l0_squared_sums.data() );

            unsigned long samples_per_thread = (dNl[l] + (MAX_NUM_THREADS) - 1) / (MAX_NUM_THREADS);
            unsigned long blocks_per_thread = (dNl[l] + (MAX_BLOCK_X) - 1) / (MAX_BLOCK_X);

            auto init_end = std::chrono::steady_clock::now();
            runtime_by_part[0] += std::chrono::duration_cast<std::chrono::duration<double>>(init_end - init_start).count();

            auto kernel_start = std::chrono::steady_clock::now();
            if (l == 0) {
              double sqrt_T = sqrt(T);
              double indep_term = K + r*K*T;
              double modifier_a = sigma*K;
              double modifier_b = 0.5f*sigma*sigma*K;
              double exp_term = exp(-r*T);
              mlmc_l0<<<pixelGridDim, pixelBlockDim>>>(l0_sums_array, l0_squared_sums_array,
                  samples_per_thread, dNl[l], T, sqrt_T, indep_term, modifier_a, modifier_b,
                  exp_term, K, curand_states);
              cudaDeviceSynchronize();
            } else if (l < SAMPLE_LEVEL_CUTOFF) {
              mlmc_parallel_sample_level<<<pixelGridDim, pixelBlockDim>>>(l0_sums_array, l0_squared_sums_array, 
              samples_per_thread, dNl[l], l, T, r, sigma, K, curand_states);
              cudaDeviceSynchronize();
            } else {
              mlmc_parallel_timestep_level<<<pixelGridDim, pixelBlockDim>>>(l0_sums_array, l0_squared_sums_array, 
              blocks_per_thread, dNl[l], l, T, r, sigma, K, curand_states);
              cudaDeviceSynchronize();
            }

            auto kernel_end = std::chrono::steady_clock::now();
            runtime_by_part[1] += std::chrono::duration_cast<std::chrono::duration<double>>(kernel_end - kernel_start).count();

            // Sum reduce all the samples
            auto reduce_start = std::chrono::steady_clock::now();
            double total_sum = thrust::reduce(l0_sums.begin(), l0_sums.end());

            double total_squared_sum = thrust::reduce(l0_squared_sums.begin(), l0_squared_sums.end());
            auto reduce_end = std::chrono::steady_clock::now();
            runtime_by_part[2] += std::chrono::duration_cast<std::chrono::duration<double>>(reduce_end - reduce_start).count();
            auto end_timer = std::chrono::steady_clock::now();
            runtimes[l] += std::chrono::duration_cast<std::chrono::duration<double>>(end_timer - start_timer).count();

            suml[0][l] += (float) dNl[l]; // Total # samples taken at level l so far
            suml[1][l] += total_sum; // |Y_l - Y_{l-1}|
            suml[2][l] += total_squared_sum; //  |Y_l - Y_{l-1}|^2
      }
    }

    //
    // compute absolute average, variance and cost,
    // correct for possible under-sampling,
    // and set optimal number of new samples
    //
    sum = 0.0f;

    for (int l=0; l<=L; l++) { 
      ml[l] = fabs(suml[1][l]/suml[0][l]); // mean difference between Y_l and Y_{l-1} at level l
      Vl[l] = fmaxf(suml[2][l]/suml[0][l] - ml[l]*ml[l], 0.0f); // Variance E[Y_l^2] - E[Y_l]^2

      sum += sqrtf(Vl[l]*Cl[l]);
    }

    for (int l=0; l<=L; l++) {
      dNl[l] = ceilf( fmaxf( 0.0f, // formula to calculate how many more samples to take at each level
                       sqrtf(Vl[l]/Cl[l])*sum/((1.0f-theta)*eps*eps)
                     - suml[0][l] ) );
    }

    //
    // if (almost) converged, estimate remaining error and decide 
    // whether a new level is required
    //

    sum = 0.0; // Checks if more samples need to be added
    for (int l=0; l<=L; l++) {
      sum += fmaxf(0.0f, (float)dNl[l]-0.01f*suml[0][l]);
    } // we subtract 0.01f*suml[0][l] to fuzzy approx so it doesn't have to be exactly 0

    if (sum==0) {
      if (diag) printf(" achieved variance target \n");

      converged = 1;
      float rem = ml[L] / (powf(2.0f,alpha)-1.0f); 
    
      if (rem > sqrtf(theta)*eps) { 
        if (L==Lmax)
            printf("*** failed to achieve weak convergence *** \n");
        else {
          converged = 0;
          L++;
          Vl[L] = Vl[L-1]/powf(2.0f,beta);
          Cl[L] = Cl[L-1]*powf(2.0f,gamma);

          if (diag) printf(" L = %d \n",L);

          sum = 0.0f;
          for (int l=0; l<=L; l++) sum += sqrtf(Vl[l]*Cl[l]);
          for (int l=0; l<=L; l++) // calculate number of samples to add per level 
            dNl[l] = ceilf( fmaxf( 0.0f, 
                            sqrtf(Vl[l]/Cl[l])*sum/((1.0f-theta)*eps*eps)
                          - suml[0][l] ) );
        }
      }
    }

    if (diag) {
        for (int l = 0; l <= L; ++l) {
            if (l == L) {
                std::cout << "level " << l << " Cl=" << Cl[l] << " dNl=" << dNl[l] << std::endl;
            } else {
                double mean = suml[1][l] / suml[0][l];
                double var  = suml[2][l] / suml[0][l] - mean*mean;
                std::cout << "level " << l << ": N=" << suml[0][l] << " mean=" << mean << " var=" << var << " Cl=" << Cl[l] << " dNl=" << dNl[l] << std::endl;
            }
        }
    }
  }

  //
  // finally, evaluate multilevel estimator and set outputs
  //
  float P = 0.0f;
  for (int l=0; l<=L; l++) {
    P += suml[1][l]/suml[0][l]; // sum of differences
  }

  cudaFree(curand_states);

  if (diag) {
    std::cout << "FINAL DIAGNOSTICS" << std::endl;
    std::cout << "Total levels: " << L << std::endl;
    std::cout << "Samples per level: " << std::endl;
    for (int l = 0; l <= L; l++) {
        std::cout << "level: " << l << " total samples: " << suml[0][l] << " timesteps per sample: " << Cl[l] << std::endl;
    }
  

    for (int l = 0; l <= L; l++) {
      std::cout << "level: " << l << " runtime " << runtimes[l] << std::endl;
    }

    std::cout << "runtime diagnostics: " << "init: " << runtime_by_part[0] << " kernel: " << runtime_by_part[1] << " reduce: " << runtime_by_part[2] << std::endl;
    std::cout << "curand setup time: " << curand_setup_time << std::endl;
  }

    std::cout << "Final payoff: " << P << std::endl;
    return P;
}
