#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include "mpi-driver.h"

struct TestCase {
    std::string name;
    double T;       // time to maturity (years)
    double r;       // risk-free rate
    double sigma;   // volatility
    double K;       // strike price
};

int main(int argc, char *argv[]) {
    int pid;
    int nproc;
    // Initialize MPI
    MPI_Init(&argc, &argv);
    // Get process rank
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    // Get total number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    // Define test cases
    std::vector<TestCase> tests = {
        { "AAPL (1-year European Call)", 1.0, 0.02, 0.25, 280.0 },
        { "Tesla (6-month Call, high vol)", 0.5, 0.02, 0.30, 430.0 },
        { "SPY (Index ETF, stable)", 2.0, 0.02, 0.10, 680.0 }
    };

    // Different epsilon values to test convergence
    std::vector<float> eps_values = { 0.005f, 0.0025, 0.001f };

    const int Lmin = 2;
    const int Lmax = 20;
    const int N0 = 100000;
    const bool diag = false;

    std::cout << std::fixed << std::setprecision(6);

    if (pid == ROOT) {
        std::cout << "Running Multilevel Monte Carlo Test Suite\n";
        std::cout << "------------------------------------------\n";
    }

    for (const auto &test : tests) {
        if (pid == ROOT) {
            std::cout << "\n===== " << test.name << " =====\n";
            std::cout << "Parameters: T=" << test.T << ", r=" << test.r
                    << ", sigma=" << test.sigma << ", K=" << test.K << std::endl;
        }

        for (float eps : eps_values) {
            if (pid == ROOT) {
                std::cout << "\n--- epsilon = " << eps << " ---" << std::endl;
            }

            auto start = std::chrono::steady_clock::now();

            float payoff = mlmc(Lmin, Lmax, N0, eps, diag,
                                test.T, test.r, test.sigma, test.K);

            auto end = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

            if (pid == ROOT) {
                std::cout << "Estimated payoff: " << payoff << std::endl;
                std::cout << "Runtime: " << elapsed << " seconds\n";
            }
        }
    }

    std::cout << "\nAll tests completed.\n";
    // Cleanup
    MPI_Finalize();
    return 0;
}
