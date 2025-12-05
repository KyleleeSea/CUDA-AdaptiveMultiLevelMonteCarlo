#include <iostream>
#include <chrono>
#include <limits>
#include <string>
#include <iomanip>

#include "mlmc-cuda.h"

// Struct for preset assets
struct AssetPreset {
    std::string name;
    double T;       // maturity
    double sigma;   // volatility
    double K;       // strike price
};

int main() {

    // Preset asset list
    AssetPreset presets[] = {
        {"AAPL",  1.0, 0.25, 280.0},
        {"TSLA",  0.5, 0.30, 430.0},
        {"SPY",   2.0, 0.10, 680.0},
        {"NVDA",  1.0, 0.40, 900.0},
        {"AMZN",  1.0, 0.28, 150.0}
    };
    const int NUM_PRESETS = sizeof(presets) / sizeof(presets[0]);

    // Fixed MLMC parameters
    float eps  = 0.05f;
    bool diag  = true;
    int Lmin   = 2;
    int Lmax   = 20;
    int N0     = 100000;

    std::cout << "===============================\n";
    std::cout << " Multi-Level Monte Carlo (MLMC)\n";
    std::cout << " European Option Pricing – CUDA\n";
    std::cout << "===============================\n\n";

    std::cout << "Choose input mode:\n";
    std::cout << "  1. Select preconfigured asset\n";
    std::cout << "  2. Enter T, sigma, K manually\n";
    std::cout << "Enter choice (1 or 2): ";

    int choice;
    std::cin >> choice;

    double T, sigma, K;

    if (choice == 1) {
        std::cout << "\nAvailable underlying assets:\n";
        for (int i = 0; i < NUM_PRESETS; i++) {
            std::cout << "  " << (i + 1) << ". " << presets[i].name
                      << "  (T=" << presets[i].T
                      << ", σ=" << presets[i].sigma
                      << ", K=" << presets[i].K << ")\n";
        }

        int pick = 0;
        while (true) {
            std::cout << "\nSelect asset (1-" << NUM_PRESETS << "): ";
            std::cin >> pick;
            if (pick >= 1 && pick <= NUM_PRESETS) break;
            std::cout << "Invalid selection. Try again.\n";
        }

        T     = presets[pick - 1].T;
        sigma = presets[pick - 1].sigma;
        K     = presets[pick - 1].K;

        std::cout << "\nUsing preset: " << presets[pick - 1].name << "\n";

    } else if (choice == 2) {
        // Manual input mode
        while (true) {
            std::cout << "Enter maturity T (0.1 to 3.0): ";
            std::cin >> T;
            if (T >= 0.1 && T <= 3.0) break;
            std::cout << "Invalid T. Try again.\n";
        }

        while (true) {
            std::cout << "Enter volatility sigma (0 < sigma < 0.5): ";
            std::cin >> sigma;
            if (sigma > 0.0 && sigma < 0.5) break;
            std::cout << "Invalid sigma. Try again.\n";
        }

        while (true) {
            std::cout << "Enter strike price K (0 < K <= 1000): ";
            std::cin >> K;
            if (K > 0.0 && K <= 1000.0) break;
            std::cout << "Invalid K. Try again.\n";
        }

    } else {
        std::cout << "Invalid choice. Exiting.\n";
        return 1;
    }

    double r = 0.02;  // risk-free rate (can be modified as needed)

    std::cout << "\nRunning MLMC...\n";
    auto start = std::chrono::steady_clock::now();

    float payoff = mlmc(Lmin, Lmax, N0, eps, diag, T, r, sigma, K);

    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    
    std::cout << "\n===============================\n";
    std::cout << "     MLMC ESTIMATED PRICE\n";
    std::cout << "===============================\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Estimated Option Payoff = " << payoff << "\n\n";

    std::cout << "\n===============================\n";
    std::cout << "          MLMC RUNTIME\n";
    std::cout << "===============================\n";
    std::cout << "Runtime = " << elapsed << " seconds\n";


    return 0;
}
