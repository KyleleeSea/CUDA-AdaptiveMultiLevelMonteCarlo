// mlmc-sequential.h

void rng_initialisation();

float mlmc(int Lmin, int Lmax, int N0, float eps, bool diag,
           double T, double r, double sigma, double K);