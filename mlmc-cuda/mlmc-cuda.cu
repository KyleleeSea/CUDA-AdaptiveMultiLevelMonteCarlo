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
#define MAX_BLOCK_X 64
#define MAX_NUM_THREADS (THREAD_DIM)*(MAX_BLOCK_X) // 2^(9*2) = 262,144
#define RAND_SEED 67

////////////////////////////////////////////////////////////////////////////////////////
// All cuda kernels (GPU code) here
///////////////////////////////////////////////////////////////////////////////////////
__global__ void setup_curand(curandState* state) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // Initialize the state for the current thread
    curand_init(RAND_SEED, tid, 0, &state[tid]); 
}

/*
In: 
- sums_ptr, GPU device array
- squared_sums_ptr, GPU device array
- samples_per_thread, the number of samples each thread must take
- total_samples, the total number of samples all threads will take combined
- T, time to maturity
- sqrtT
- independent_term, K + r*K*T;
- modifier_a, sigma*K;
- modifier_b, 0.5f*sigma*sigma*K;
- K, strike price
- curand_state, device GPU array of independent curand states

Takes up to samples_per_thread samples and stores the sum of their payoffs
in sums_ptr[tid] and the squared sum squared_sums_ptr[tid] 
*/
__global__ void mlmc_l0(double* sums_ptr, double* squared_sums_ptr, unsigned long samples_per_thread, 
    unsigned long total_samples, double T, double sqrtT, double independent_term, double modifier_a, double modifier_b, double K,
    curandState* curand_state) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        if (tid > 0 && (tid-1)*samples_per_thread >= total_samples) { // early exit if already got enough samples
            return;
        }

        double local_sum = 0;
        double local_squared_sum = 0;


        // We know (tid-1)*samples_per_thread < total_samples
        size_t num_samples = fminf(samples_per_thread, total_samples-((tid-1)*samples_per_thread));
        for (size_t i = 0; i < num_samples; i++) {
            float dWf = sqrtT * curand_normal(&curand_state[tid]);
            float Xf = independent_term + modifier_a*dWf + modifier_b*(dWf*dWf-T);
            
            float dP = fmaxf(0.0f,Xf-K);
            local_sum += dP;
            local_squared_sum += dP * dP;
        }

        // Write to global device memory
        sums_ptr[tid] += local_sum;
        squared_sums_ptr[tid] += local_squared_sum;
}


////////////////////////////////////////////////////////////////////////////////////////
// All CPU code here 
///////////////////////////////////////////////////////////////////////////////////////

// RANDOMIZATION GENERATORS
// https://people.maths.ox.ac.uk/~gilesm/mlmc/c++/mlmc_rng.cpp
std::default_random_engine rng;
std::normal_distribution<float> normal(0.0f,1.0f);

auto next_normal = std::bind(std::ref(normal), std::ref(rng));

void rng_initialisation() {
    rng.seed(1234);
    normal.reset();
}

void mlmc_l(int l, unsigned long N, double *sums, double T, double r, double sig, double K) {
  int   nf, nc;
  float hf, hc;

  nf = 1<<l;
  nc = nf/2;

  hf = T / ((float) nf);
  hc = T / ((float) nc);

  for (int k=0; k<3; k++) sums[k] = 0.0;  // zero out sums
  
  for (unsigned long np = 0; np<N; np++) {
    float Xf, Xc, dWc, Pf, Pc, dP;
    float dWf[2];

    Xf = K;
    Xc = Xf;

    if (l==0) {
      dWf[0] = sqrt(hf) * next_normal();

      Xf  = Xf + r*Xf*hf + sig*Xf*dWf[0]
               + 0.5f*sig*sig*Xf*(dWf[0]*dWf[0]-hf);
    }

    else {
      for (int n=0; n<nc; n++) {
        dWf[0] = sqrt(hf) * next_normal();
        dWf[1] = sqrt(hf) * next_normal();

        for (int m=0; m<2; m++) {
          Xf  = Xf + r*Xf*hf + sig*Xf*dWf[m]
                   + 0.5f*sig*sig*Xf*(dWf[m]*dWf[m]-hf);
        }

        dWc = dWf[0] + dWf[1];

        Xc  = Xc + r*Xc*hc + sig*Xc*dWc + 0.5f*sig*sig*Xc*(dWc*dWc-hc);
      }
    }

    Pf  = fmaxf(0.0f,Xf-K);
    Pc  = fmaxf(0.0f,Xc-K);

    dP  = exp(-r*T)*(Pf-Pc);
    Pf  = exp(-r*T)*Pf;

    if (l==0) dP = Pf;

    sums[0] += nf;     // add number of timesteps as cost
    sums[1] += dP;
    sums[2] += dP*dP;
  }
}

float mlmc(int Lmin, int Lmax, int N0, float eps, bool diag, double T, double r, double sigma, double K) {
    /*
    sums: holds output of of mlmc level. sums[0] = total # steps at that mlmc level,
    sums[1] = |Y_l - Y_{l-1}|, sums[2] = |Y_l - Y_{l-1}|^2

    suml: accumulator over all samples taken for the given mlmc level
    suml[0] = total # steps taken over all samples for the given level
    suml[1] = sum over |Y_l - Y_{l-1}|
    suml[2] = sum over |Y_l - Y_{l-1}|^2

    ml[l] = mean |Y_l - Y_{l-1}| at level l 
    Vl[l] = variance at level l 
    Cl[l] = # steps to take per sample at level l
    dNl[l] = number of additional samples to take
    L = current number of levels
    */
  double sums[3], suml[3][MAX_LEVELS], runtimes[MAX_LEVELS], l0_runtimes[2];
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
    Cl[l]   = powf(2.0f,(float)l*gamma); // Cost per sample at level l = # time steps per sample at level l = 2^l (gamma=1)
    for(int n=0; n<3; n++) suml[n][l] = 0.0; // Zero out the running sums for each possible level
    runtimes[l] = 0;
    l0_runtimes[0] = 0;
    l0_runtimes[1] = 0;
}

  for(int l=0; l<=Lmin; l++) {dNl[l] = N0;} // Set initial number of samples to take

    // Setup curand
    auto curand_setup_start = std::chrono::steady_clock::now();

    curandState* curand_states;
    cudaMalloc((void**)&curand_states, (MAX_NUM_THREADS) * sizeof(curandState));

    dim3 pixelBlockDim(THREAD_DIM); // 1024 threads
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
        for(int n=0; n<3; n++) sums[n] = 0.0; // zero out results array

        // Parallelize l == 0 case
        if (l == 0) {
            auto start_timer = std::chrono::steady_clock::now();
            // Initialize thrust device vectors
            unsigned long vector_size = std::min(static_cast<unsigned long>(MAX_NUM_THREADS), dNl[l]);
            thrust::device_vector<double> l0_sums(vector_size, 0);
            thrust::device_vector<double> l0_squared_sums(vector_size, 0);

            // Populate sums and squared sums
            double* l0_sums_array = thrust::raw_pointer_cast( l0_sums.data() );
            double* l0_squared_sums_array = thrust::raw_pointer_cast( l0_squared_sums.data() );

            unsigned long samples_per_thread = (dNl[l] + (MAX_NUM_THREADS) - 1) / (MAX_NUM_THREADS);

            auto kernel_start = std::chrono::steady_clock::now();
            double sqrt_T = sqrt(T);
            double indep_term = K + r*K*T;
            double modifier_a = sigma*K;
            double modifier_b = 0.5f*sigma*sigma*K;
            mlmc_l0<<<pixelGridDim, pixelBlockDim>>>(l0_sums_array, l0_squared_sums_array,
                samples_per_thread, dNl[l], T, sqrt_T, indep_term, modifier_a, modifier_b,
                K, curand_states);
            cudaDeviceSynchronize();
            auto kernel_end = std::chrono::steady_clock::now();
            l0_runtimes[0] += std::chrono::duration_cast<std::chrono::duration<double>>(kernel_end - kernel_start).count();

            // Sum reduce all the samples
            auto reduce_start = std::chrono::steady_clock::now();
            double total_sum = thrust::reduce(l0_sums.begin(), l0_sums.end());

            double total_squared_sum = thrust::reduce(l0_squared_sums.begin(), l0_squared_sums.end());
            auto reduce_end = std::chrono::steady_clock::now();
            l0_runtimes[1] += std::chrono::duration_cast<std::chrono::duration<double>>(reduce_end - reduce_start).count();
            auto end_timer = std::chrono::steady_clock::now();
            runtimes[l] += std::chrono::duration_cast<std::chrono::duration<double>>(end_timer - start_timer).count();

            suml[0][l] += (float) dNl[l];
            suml[1][l] += total_sum; // |Y_l - Y_{l-1}|, accumulating sum of diffs 
            suml[2][l] += total_squared_sum; //  |Y_l - Y_{l-1}|^2, accumulating sum of diffs squared

        } else { // Otherwise keep it sequential for now
            auto start_timer = std::chrono::steady_clock::now();
            mlmc_l(l,dNl[l],sums, T, r, sigma, K); // Takes dNl[l] additional samples at level l, puts results in sums
            auto end_timer = std::chrono::steady_clock::now();
            runtimes[l] += std::chrono::duration_cast<std::chrono::duration<double>>(end_timer - start_timer).count();
            suml[0][l] += (float) dNl[l]; // += # additional samples taken, accumulating # of samples
            suml[1][l] += sums[1]; // |Y_l - Y_{l-1}|, accumulating sum of diffs 
            suml[2][l] += sums[2]; //  |Y_l - Y_{l-1}|^2, accumulating sum of diffs squared
        }
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

      sum += sqrtf(Vl[l]*Cl[l]); // Keep sum of square roots of variance * cost
    }

    for (int l=0; l<=L; l++) {
      dNl[l] = ceilf( fmaxf( 0.0f, // Formula to calculate how many more samples to take at each level
                       sqrtf(Vl[l]/Cl[l])*sum/((1.0f-theta)*eps*eps)
                     - suml[0][l] ) ); // This gets optimal number of samples then subtracts out suml[0][l] so we only get extra samples
    }

    //
    // if (almost) converged, estimate remaining error and decide 
    // whether a new level is required
    //

    sum = 0.0; // checks if more samples need to be added
      for (int l=0; l<=L; l++)
        sum += fmaxf(0.0f, (float)dNl[l]-0.01f*suml[0][l]); // check if we need to add more samples at this level
        // We subtract 0.01f*suml[0][l] to fuzzy approx so it doesn't have to be exactly 0

    if (sum==0) { // 0 if no levels need more samples added
      if (diag) printf(" achieved variance target \n");

      converged = 1;
      float rem = ml[L] / (powf(2.0f,alpha)-1.0f); 
    
      if (rem > sqrtf(theta)*eps) {
        if (L==Lmax)
            printf("*** failed to achieve weak convergence *** \n");
        else {
          converged = 0;
          L++; // add a level
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

    std::cout << "level 0 runtime diagnostics: " << "kernel: " << l0_runtimes[0] << " reduce: " << l0_runtimes[1] << std::endl;
    std::cout << "curand setup time: " << curand_setup_time << std::endl;
    std::cout << "Final payoff: " << P << std::endl;
  }

  return P;
}
