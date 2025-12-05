#include "mlmc-cuda.h"

////////////////////////////////////////////////////////////////////////////////////////
// CUDA kernels
///////////////////////////////////////////////////////////////////////////////////////

// Kernel to initialize the currand state for state[tid]
__global__ void setup_curand(curandState* state) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(RAND_SEED, tid, 0, &state[tid]); 
}

// Calculates the milstein factor given r, sig, h and dW
__device__ __forceinline__
double milstein_factor(double r, double sig, double h, double dW)
{
    float dw2 = dW * dW;
    return 1.0f + r*h + sig*dW + 0.5f*sig*sig*(dw2 - h);
}

// Reduces fine_factors and coarse_factors together
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
// CPU code 
///////////////////////////////////////////////////////////////////////////////////////
/*
Gets dNl samples at level l and places the results in sums

Input:
sums: sums[0] = |Y_l - Y_{l-1}|, sums[1] = |Y_l - Y_{l-1}|^2
dNl: number of samples to take
l: current level
T: time to maturity of asset
r: risk free interest rate
sigma: volatility of asset
K: strike price
curand_state: array of independent curand states,
   malloc'd and initialized to MAX_NUM_THREADS 
runtime_by_part: 
  runtime_by_part[0]: total time spent in Thrust array initialization
  runtime_by_part[1]: total time spent in kernel runtime
  runtime_by_part[2]: total time spent in Thrust reduction
runtime_by_level: runtime_by_level[l] = total runtime spent at level l

Output:
Sets sums[0] = |Y_l - Y_{l-1}|, sums[1] = |Y_l - Y_{l-1}|^2 and updates
runtime diagnostics
*/
void mlmc_cuda(double *sums, unsigned long dNl, int l, double T, double r, double sigma, double K, curandState* curand_state, double* runtime_by_part, double* runtime_by_level) {
    // Initialize thrust device vectors
    auto start_timer = std::chrono::steady_clock::now();
    auto init_start = std::chrono::steady_clock::now();
    unsigned long vector_size = std::min(static_cast<unsigned long>(MAX_NUM_THREADS), dNl);
    thrust::device_vector<double> l0_sums(vector_size, 0);
    thrust::device_vector<double> l0_squared_sums(vector_size, 0);

    // Populate sums and squared sums
    double* l0_sums_array = thrust::raw_pointer_cast( l0_sums.data() );
    double* l0_squared_sums_array = thrust::raw_pointer_cast( l0_squared_sums.data() );

    unsigned long samples_per_thread = (dNl + (MAX_NUM_THREADS) - 1) / (MAX_NUM_THREADS);

    dim3 pixelBlockDim(THREAD_DIM);
    dim3 pixelGridDim(MAX_BLOCK_X);

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
    
    auto reduce_start = std::chrono::steady_clock::now();
    double total_sum = thrust::reduce(l0_sums.begin(), l0_sums.end());
    double total_squared_sum = thrust::reduce(l0_squared_sums.begin(), l0_squared_sums.end());

    auto reduce_end = std::chrono::steady_clock::now();
    runtime_by_part[2] += std::chrono::duration_cast<std::chrono::duration<double>>(reduce_end - reduce_start).count();

    sums[0] += total_sum; // |Y_l - Y_{l-1}|
    sums[1] += total_squared_sum; //  |Y_l - Y_{l-1}|^2

    auto end_timer = std::chrono::steady_clock::now();
    runtime_by_level[l] += std::chrono::duration_cast<std::chrono::duration<double>>(end_timer - start_timer).count();
}

// Populates curand_state
void setup_curand_host(curandState* curand_state) {
    dim3 pixelBlockDim(THREAD_DIM);
    dim3 pixelGridDim(MAX_BLOCK_X);
    setup_curand<<<pixelGridDim, pixelBlockDim>>>(curand_state);
    cudaDeviceSynchronize();
}
