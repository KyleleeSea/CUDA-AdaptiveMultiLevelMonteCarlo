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

////////////////////////////////////////////////////////////////////////////////////////
// Constants
///////////////////////////////////////////////////////////////////////////////////////
#define MAX_LEVELS 21  
#define THREAD_DIM 1024
#define MAX_BLOCK_X 64
#define MAX_NUM_THREADS (THREAD_DIM)*(MAX_BLOCK_X) // 2^(9*2) = 262,144
#define RAND_SEED 67
#define SAMPLE_LEVEL_CUTOFF 8 // Found experimentally

////////////////////////////////////////////////////////////////////////////////////////
// GPU function wrappers
///////////////////////////////////////////////////////////////////////////////////////
void mlmc_cuda(double *sums, unsigned long dNl, int l, double T, double r, double sigma, double K, curandState* curand_state, double* runtime_by_part, double* runtime_by_level);

void setup_curand_host(curandState* curand_state);