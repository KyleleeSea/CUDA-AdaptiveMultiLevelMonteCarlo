#include "mpi-driver.h"
#include "mlmc-cuda.h"

/*
Runs adaptive multi level monte carlo for European option payoff estimation for the
asset respresented by T, r, sigma, K

Input:
Lmin: # levels to start at
Lmax: maximum number of levels, fails to converge if exceeded
N0: number of samples to start with
eps: convergence epsilon value, runtime increases as eps decreases
diag: diagnostics printed if true
T: asset time to maturity
r: risk free interest rate
sigma: asset volatility
K: asset strike price

Output:
Estimated option payoff
*/
float mlmc(int Lmin, int Lmax, int N0, float eps, bool diag, double T, double r, double sigma, double K) {
  /*
  sums[0] = local sum of |Y_l - Y_{l-1}| for this iteration
  sums[1] = local sum of |Y_l - Y_{l-1}|^2 for this iteration

  diff_sum[l] = total sum of |Y_l - Y_{l-1}| across iterations 
  squared_diff_sum[l] = total sum of |Y_l - Y_{l-1}|^2 across iterations 

  total_samples[l] = total number of samples taken at level l across iterations

  runtimes[l]: array of the total time spent at level l
  runtime_by_component[0]: total time spent in Thrust array initialization
  runtime_by_component[1]: total time spent in kernel runtime
  runtime_by_component[2]: total time spent in Thrust reduction

  ml[l] = mean |Y_l - Y_{l-1}| at level l 
  Vl[l] = variance at level l 
  Cl[l] = # steps to take per sample at level l
  alpha, beta, gamma are hyperparameters
  dNl[l] = number of additional samples to take next iteration
  L = current number of levels
  converged = 1 if converged and 0 if not

  iter_data: data communciated by MPI to synchronize iterations
    iter_data[0] to iter_data[MAX_LEVELS-1] is dNl[l]
    iter_data[MAX_LEVELS] = L
    iter_data[MAX_LEVELS+1] = converged
  */
  double sums[2], diff_sum[MAX_LEVELS], squared_diff_sum[MAX_LEVELS], total_samples[MAX_LEVELS], 
    runtime_by_level[MAX_LEVELS], runtime_by_component[3];
  float  ml[MAX_LEVELS], Vl[MAX_LEVELS], Cl[MAX_LEVELS], alpha, beta, gamma, sum, theta;
  int    L, converged;
  unsigned long iter_data[MAX_LEVELS+2]; 

  // Setup MPI
  int pid, nproc;
  // Get process rank
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  // Get total number of processes
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  // Set cuda device
  int nDevices;
  cudaGetDeviceCount(&nDevices);

  int device_id = pid % nDevices;
  cudaSetDevice(device_id);

  //
  // check input parameters
  //
  if (Lmin<2) {
    if (pid == ROOT) {
        fprintf(stderr,"error: needs Lmin >= 2 \n");
    }
    exit(1);
  }
  if (Lmax<Lmin) {
    if (pid == ROOT) {
        fprintf(stderr,"error: needs Lmax >= Lmin \n");
    }
    exit(1);
  }

  if (Lmax > MAX_LEVELS) {
    if (pid == ROOT) {
        fprintf(stderr,"error: Lmax exceeds max_levels \n");
    }
    exit(1);
  }

  if (N0<=0 || eps<=0.0f) {
    if (pid == ROOT) {
        fprintf(stderr,"error: needs N>0, eps>0 \n");
    }
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

  for(int l=0; l<=Lmax; l++) { // Initialize all the variables
    Cl[l]   = powf(2.0f,(float)l*gamma); 
    diff_sum[l] = 0;
    squared_diff_sum[l] = 0;
    total_samples[l] = 0;
    runtime_by_level[l] = 0;
  }

  for (int i = 0; i < 3; i++) {
    runtime_by_component[i] = 0;
  }

  for(int l=0; l<=Lmin; l++) {iter_data[l] = N0;} // Set initial number of samples to take

  // Setup curand
  auto curand_setup_start = std::chrono::steady_clock::now();

  curandState* curand_state;
  cudaMalloc((void**)&curand_state, (MAX_NUM_THREADS) * sizeof(curandState));
  setup_curand_host(curand_state);

  auto curand_setup_end = std::chrono::steady_clock::now();
  double curand_setup_time = std::chrono::duration_cast<std::chrono::duration<double>>(curand_setup_end - curand_setup_start).count();

  //
  // main loop
  //

  while (!converged) {
    // Invariant: send_buf[0] to send_buf[MAX_LEVELS-1] is normal sums
    // send_buf[MAX_LEVELS] to MAX_LEVELS[2*MAX_LEVELS-1] is squared sums
    double send_buf[2*MAX_LEVELS]; 
    for (int l = 0; l <= L; l++) {
        send_buf[l] = 0;
        send_buf[MAX_LEVELS+l] = 0;
    }

    //
    // update sample sums
    //

    // Split samples across nodes
    for (int l = 0; l <= L; l++) { // partition samples across nodes
        total_samples[l] += iter_data[l];

        unsigned long floor = iter_data[l] / nproc; // take floor
        if (pid == ROOT) {
            iter_data[l] = floor+(iter_data[l]-floor*nproc); // give remainder to root
        } else {
            iter_data[l] = floor;
        }
    }
    
    for (int l=0; l<=L; l++) {
      if (iter_data[l]>0) { // skip levels where we don't need more samples

        for (int i = 0; i < 2; i++) {sums[i] = 0;}

        mlmc_cuda(sums, iter_data[l], l, T, r, sigma, K, curand_state, runtime_by_component, runtime_by_level);

        send_buf[l] += sums[0]; // Sums
        send_buf[MAX_LEVELS+l] += sums[1]; // Squared sums 
      }
    }

    // Reduce the local sums
    double recv_buf[2*MAX_LEVELS]; // 0 to MAX_LEVELS-1 is diff sum. MAX_LEVELS to 2*MAX_LEVELS-1 is squared sum.

    MPI_Reduce (&send_buf, &recv_buf, 2*MAX_LEVELS, MPI_DOUBLE, MPI_SUM, ROOT, MPI_COMM_WORLD);
    for (size_t l = 0; l <= L; l++) {
        diff_sum[l] += recv_buf[l];
        squared_diff_sum[l] += recv_buf[MAX_LEVELS+l];
    }

    if (pid == ROOT) {
    //
    // compute absolute average, variance and cost,
    // correct for possible under-sampling,
    // and set optimal number of new samples
    //
    sum = 0.0f;

    for (int l=0; l<=L; l++) { 
      ml[l] = fabs(diff_sum[l]/total_samples[l]); // mean difference between Y_l and Y_{l-1} at level l
      Vl[l] = fmaxf(squared_diff_sum[l]/total_samples[l] - ml[l]*ml[l], 0.0f); // Variance E[Y_l^2] - E[Y_l]^2

      sum += sqrtf(Vl[l]*Cl[l]); 
    }

    for (int l=0; l<=L; l++) {
      iter_data[l] = ceilf( fmaxf( 0.0f, // Calculate how many more samples to take at each level
                       sqrtf(Vl[l]/Cl[l])*sum/((1.0f-theta)*eps*eps)
                     - total_samples[l] ) );
    }

    //
    // if (almost) converged, estimate remaining error and decide 
    // whether a new level is required
    //
    sum = 0.0; // checks if more samples need to be added
      for (int l=0; l<=L; l++) {
        sum += fmaxf(0.0f, (float)iter_data[l]-0.01f*total_samples[l]); 
      }
        

    if (sum==0) { // 0 if no levels need more samples added
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
          for (int l=0; l<=L; l++) // Calculate number of samples to add per level 
            iter_data[l] = ceilf( fmaxf( 0.0f, 
                            sqrtf(Vl[l]/Cl[l])*sum/((1.0f-theta)*eps*eps)
                          - total_samples[l] ) );
        }
      }
    }

    if (pid == ROOT && diag) {
        for (int l = 0; l <= L; ++l) {
            if (l == L) {
                std::cout << "level " << l << " Cl=" << Cl[l] << " dNl=" << iter_data[l] << std::endl;
            } else {
                double mean = diff_sum[l] / total_samples[l];
                double var  = squared_diff_sum[l] / total_samples[l] - mean*mean;
                std::cout << "level " << l << ": N=" << total_samples[l] << " mean=" << mean << " var=" << var << " Cl=" << Cl[l] << " dNl=" << iter_data[l] << std::endl;
            }
        }
    }
    }

    // Also broadcast L and converged in the same message
    iter_data[MAX_LEVELS] = L;
    iter_data[MAX_LEVELS+1] = converged;

    // Broadcast next iteration data
    MPI_Bcast (&iter_data, MAX_LEVELS+2, MPI_UNSIGNED_LONG, ROOT, MPI_COMM_WORLD);

    L = iter_data[MAX_LEVELS];
    converged = iter_data[MAX_LEVELS+1];
    
  }
  

  //
  // finally, evaluate multilevel estimator and set outputs
  // only ROOT has the correct answer at the end
  //
  float P = 0.0f;
  for (int l=0; l<=L; l++) {
    P += diff_sum[l]/total_samples[l]; // sum of differences
  }

  cudaFree(curand_state);

  if (diag && pid == ROOT) {
    std::cout << "FINAL DIAGNOSTICS" << std::endl;
    std::cout << "Total levels: " << L << std::endl;
    std::cout << "Samples per level: " << std::endl;
    for (int l = 0; l <= L; l++) {
        std::cout << "level: " << l << " total samples: " << total_samples[l] << " timesteps per sample: " << Cl[l] << std::endl;
    }

    for (int l = 0; l <= L; l++) {
        std::cout << "level: " << l << " runtime " << runtime_by_level[l] << std::endl;
    }

    std::cout << "runtme diagnostics: " << "init: " << runtime_by_component[0] << " kernel: " << runtime_by_component[1] << " reduce: " << runtime_by_component[2] << std::endl;
    std::cout << "curand setup time: " << curand_setup_time << std::endl;

    std::cout << "Final payoff: " << P << std::endl;
  }

    return P;
}
