# CUDA Adaptive Multilevel Monte Carlo for European Option Pricing

This repository contains **sequential**, **CUDA**, and **MPI** implementations of an **Adaptive Multilevel Monte Carlo (MLMC)** algorithm for pricing European options. It includes GPU-accelerated kernels, benchmarking tools, and a full set of test suites.

🔗 **Project Website:**  
https://kyleleesea.github.io/CUDA-AdaptiveMultiLevelMonteCarlo/

---

## Repository Structure

The project provides three independent implementations of the MLMC algorithm:

### **1. Sequential Implementation**
Located in: /mlmc-sequential

### **2. CUDA GPU Implementation**
Located in: /mlmc-cuda

### **3. MPI + CUDA Distributed Implementation**
Located in: /mlmc-mpi


Each implementation includes its own `Makefile`, source files, and executables.

---

## ⚙️ Building

For **all implementations**, navigate into the corresponding directory and run:

```bash
make
```

After building, run one of the available executables:
### Test Suite
Runs predefined test cases:
```bash
./test-suite
```

### Command-Line Interface (CLI)
```bash
./mlmc-cli
```

## References 
We utilized Mike Giles' sequential MLMC C++ implementation found here:
https://people.maths.ox.ac.uk/~gilesm/mlmc/