/*
   multilevel Monte Carlo control routine

   Lmin  = minimum level of refinement       >= 2
   Lmax  = maximum level of refinement       >= Lmin
   N0    = initial number of samples         > 0
   eps   = desired accuracy (rms error)      > 0 
 
   mlmc_l(l,N,sums)
        l       = level
        N       = number of paths
        sums[0] = sum(cost)
        sums[1] = sum(Y)
        sums[2] = sum(Y^2)
        where Y are iid samples with expected value:
        E[P_0]           on level 0
        E[P_l - P_{l-1}] on level l>0

   P   = value
   Nl  = number of samples at each level
   Cl  = average cost of samples at each level
   Starter code: https://people.maths.ox.ac.uk/gilesm/mlmc/c++/mlmc.cpp
*/

#include <math.h>
#include <unistd.h> 
#include <iostream>
#include <random>
#include <chrono>
#include <iomanip> 
#include <functional>
#include "mlmc-sequential.h"

#define MAX_LEVELS 21

// RANDOMIZATION GENERATORS
// Credit: https://people.maths.ox.ac.uk/~gilesm/mlmc/c++/mlmc_rng.cpp
std::default_random_engine rng;
std::normal_distribution<float> normal(0.0f,1.0f);

auto next_normal = std::bind(std::ref(normal), std::ref(rng));

void rng_initialisation() {
    rng.seed(1234);
    normal.reset();
}

/*
In:
- l, level number. We take 2^l steps at level l
- N, number of samples to take
- sums, where we place outputs 
- T, amount of time in years for option to mature
- r, risk free interest rate
- sigma, underlying asset volatility 0 < sigma < 1
- K, strike price

Out:
- sums[0] = number of timesteps taken per sample
- sums[1] = |Y_l - Y_{l-1}|
- sums[2] = |Y_l - Y_{l-1}|^2

Calculates for a European option
*/
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

    sums[0] += nf;     // Add number of timesteps as cost
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
  double sums[3], suml[3][MAX_LEVELS];
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

  for(int l=0; l<=Lmax; l++) { // Initialize all the variables.
    Cl[l]   = powf(2.0f,(float)l*gamma); // Cost per sample at level l = # time steps per sample at level l = 2^l (gamma=1)
    for(int n=0; n<3; n++) suml[n][l] = 0.0; // zero out the running sums for each possible level
  }

  for(int l=0; l<=Lmin; l++) dNl[l] = N0; // Set initial number of samples to take

  //
  // main loop
  //
  while (!converged) {
    //
    // update sample sums
    //
    
    for (int l=0; l<=L; l++) {
      if (dNl[l]>0) { // Skip levels where we don't need more samples
        for(int n=0; n<3; n++) sums[n] = 0.0; // Zero out results array
        mlmc_l(l,dNl[l],sums, T, r, sigma, K); // Takes dNl[l] additional samples at level l, puts results in sums
        suml[0][l] += (float) dNl[l]; // Accumulating total # of samples
        suml[1][l] += sums[1]; // |Y_l - Y_{l-1}|, accumulating sum of diffs 
        suml[2][l] += sums[2]; //  |Y_l - Y_{l-1}|^2, accumulating sum of diffs squared
      }
    }

    // compute absolute average, variance and cost, set optimal number of new samples
    sum = 0.0f;

    for (int l=0; l<=L; l++) { 
      ml[l] = fabs(suml[1][l]/suml[0][l]); // Mean difference between Y_l and Y_{l-1} at level l
      Vl[l] = fmaxf(suml[2][l]/suml[0][l] - ml[l]*ml[l], 0.0f); // Variance E[Y_l^2] - E[Y_l]^2

      sum += sqrtf(Vl[l]*Cl[l]);
    }

    for (int l=0; l<=L; l++) {
      dNl[l] = ceilf( fmaxf( 0.0f, // formula to calculate how many more samples to take at each level
                       sqrtf(Vl[l]/Cl[l])*sum/((1.0f-theta)*eps*eps)
                     - suml[0][l] ) ); // This gets optimal number of samples then subtracts out suml[0][l] so we only get extra samples
    }

    //
    // if (almost) converged, estimate remaining error and decide 
    // whether a new level is required
    //

    sum = 0.0; // checks if more samples need to be added
      for (int l=0; l<=L; l++)
        sum += fmaxf(0.0f, (float)dNl[l]-0.01f*suml[0][l]); // Check if we need to add more samples at this level
        // We subtract 0.01f*suml[0][l] to fuzzy check so it doesn't have to be exactly 0

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
          for (int l=0; l<=L; l++) // Calculate number of samples to add per level 
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
    P += suml[1][l]/suml[0][l]; // Sum of differences
  }

  if (diag) {
    std::cout << "FINAL DIAGNOSTICS" << std::endl;
    std::cout << "Total levels: " << L << std::endl;
    std::cout << "Samples per level: " << std::endl;
    for (int l = 0; l <= L; l++) {
        std::cout << "level: " << l << " total samples: " << suml[0][l] << " timesteps per sample: " << Cl[l] << std::endl;
    }
  }

    std::cout << "Final payoff: " << P << std::endl;
    return P;
}
