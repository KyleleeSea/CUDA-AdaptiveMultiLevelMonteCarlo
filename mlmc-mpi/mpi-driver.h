#include <math.h>
#include <unistd.h> 
#include <iostream>
#include <random>
#include <chrono>
#include <iomanip> 
#include <functional>
#include <mpi.h>

#define ROOT 0

float mlmc(int Lmin, int Lmax, int N0, float eps, bool diag,
           double T, double r, double sigma, double K);

