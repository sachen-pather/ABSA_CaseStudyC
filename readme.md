# CUDA Asian Options Monte Carlo Pricer

A high-performance CUDA implementation for pricing Asian options using Monte Carlo simulation with antithetic variance reduction. Optimized for NVIDIA RTX 3050 and similar GPUs.

## Features

- **Antithetic Variance Reduction**: Implements proper antithetic sampling to reduce Monte Carlo variance
- **Memory-Efficient Design**: Adaptive kernel selection based on time step requirements
- **GPU Acceleration**: Optimized for NVIDIA RTX 3050 with configurable thread blocks
- **Convergence Analysis**: Built-in testing framework with multiple simulation sizes
- **Performance Monitoring**: Real-time progress tracking and performance metrics
- **CSV Export**: Convergence data export for analysis and visualization

## Requirements

### Hardware

- NVIDIA GPU with Compute Capability 3.0+ (RTX 3050 recommended)
- Minimum 4GB GPU memory

### Software

- CUDA Toolkit 11.0+
- C++ compiler with C++11 support
- Linux/Windows with NVIDIA drivers

## Installation

1. **Install CUDA Toolkit**

   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install nvidia-cuda-toolkit

   # Or download from NVIDIA developer website
   ```

2. **Clone/Download the Code**

   ```bash
   # Save the code as asian_options.cu
   ```

3. **Compile**

   ```bash
   nvcc -O3 -arch=sm_86 asian_options.cu -o asian_options -lcurand
   ```

   **Note**: Adjust `-arch=sm_86` based on your GPU:

   - RTX 3050/3060/3070/3080/3090: `sm_86`
   - GTX 1060/1070/1080: `sm_61`
   - RTX 2060/2070/2080: `sm_75`

## Usage

### Basic Execution

```bash
./asian_options
```

### Default Parameters

The program uses these financial parameters:

- **Spot Price (S0)**: $100.00
- **Strike Price (K)**: $105.00
- **Risk-free Rate (r)**: 5% (0.05)
- **Volatility (σ)**: 30% (0.30)
- **Time to Maturity (T)**: 2.0 years
- **Time Steps**: 730 (daily observations)

### Output Example

```
=== FIXED CUDA Asian Option Pricer ===
Parameters: S0=100.00, K=105.00, r=0.050, σ=0.300, T=2.0
Time steps: 730, dt=0.002740

🎯 Test 1: 1000000 simulations (1M)
Monte Carlo Price: $8.423156
Completed simulations: 1000000
Performance: 15.24 million sims/sec
```

## Code Structure

### Main Components

1. **OptionParams Structure**

   - Contains all option parameters and precomputed values
   - Drift and diffusion coefficients for efficiency

2. **CUDA Kernels**

   - `asian_monte_carlo_kernel_fixed`: Memory-optimized for ≤730 steps
   - `asian_monte_carlo_kernel_memory_efficient`: For larger step counts

3. **CUDAAsianPricerFixed Class**

   - GPU memory management
   - Batch processing for large simulations
   - Performance monitoring

4. **Analytical Formulas**
   - Case study approximation for validation
   - Error analysis framework

### Key Features

#### Antithetic Variance Reduction

```cpp
// Regular path
S *= expf(drift + diffusion * z);

// Antithetic path (same random number, opposite sign)
S *= expf(drift + diffusion * (-z));
```

#### Adaptive Memory Management

- Stores random numbers for steps ≤730
- Regenerates for larger step counts to save memory

#### Batched Processing

- Processes large simulations in configurable batches
- Prevents GPU memory overflow
- Maintains accuracy across batches

## Performance Optimization

### GPU Configuration

```cpp
#define THREADS_PER_BLOCK 256    // Optimal for RTX 3050
#define MAX_BLOCKS 2048          // Maximum concurrent blocks
#define BATCH_SIZE 1000000       // Simulations per batch
```

### Memory Layout

- **Coalesced Memory Access**: Optimized thread-to-memory mapping
- **Register Usage**: Minimized for high occupancy
- **Shared Memory**: Efficient random state management

### Expected Performance

- **RTX 3050**: ~15-20 million simulations/second
- **RTX 3080**: ~40-60 million simulations/second
- **Memory Usage**: ~200MB for 1B simulations

## Convergence Analysis

The program tests multiple simulation sizes to demonstrate Monte Carlo convergence:

| Simulations | Expected Error | Runtime   |
| ----------- | -------------- | --------- |
| 1K - 100K   | ~10-1%         | <1 sec    |
| 1M - 10M    | ~0.1-0.03%     | 1-5 sec   |
| 100M - 1B   | ~0.01-0.003%   | 30-60 sec |

### CSV Export

Results are automatically saved to `asian_convergence_data.csv`:

```csv
Simulations,MC_Price,Analytical_Price,Error_Percent,Theoretical_StdError
1000000,8.423156,8.425000,0.022,0.1000
```

## Mathematical Background

### Asian Option Payoff

```
Payoff = max(Average_Price - Strike, 0)
Average_Price = (1/n) * Σ(S_i) for i=1 to n
```

### Monte Carlo Estimation

```
Option_Price = e^(-rT) * E[Payoff]
             ≈ e^(-rT) * (1/N) * Σ(Payoff_i)
```

### Antithetic Variance Reduction

For each simulation path using random numbers Z_i:

1. **Regular Path**: S*i = S*{i-1} _ exp(drift + diffusion _ Z_i)
2. **Antithetic Path**: S*i = S*{i-1} _ exp(drift + diffusion _ (-Z_i))
3. **Final Estimate**: (Payoff_regular + Payoff_antithetic) / 2

## Troubleshooting

### Common Issues

1. **Compilation Errors**

   ```bash
   # Check CUDA installation
   nvcc --version

   # Verify GPU compute capability
   nvidia-smi
   ```

2. **Runtime Errors**

   ```bash
   # Check GPU memory
   nvidia-smi

   # Reduce BATCH_SIZE if out of memory
   #define BATCH_SIZE 500000
   ```

3. **Performance Issues**
   - Ensure NVIDIA drivers are up to date
   - Check GPU temperature and throttling
   - Verify no other GPU-intensive processes

### Error Messages

- **"CUDA Error"**: Check GPU memory and driver installation
- **"Out of memory"**: Reduce BATCH_SIZE or MAX_BLOCKS
- **"Invalid device"**: Verify CUDA-compatible GPU is present

## Customization

### Modifying Parameters

Edit the main function to change option parameters:

```cpp
float S0 = 100.0f;    // Spot price
float K = 105.0f;     // Strike price
float r = 0.05f;      // Risk-free rate
float sigma = 0.3f;   // Volatility
float T = 2.0f;       // Time to maturity
int nsteps = 730;     // Number of time steps
```

### Performance Tuning

Adjust GPU configuration for your hardware:

```cpp
#define THREADS_PER_BLOCK 512    // Try 128, 256, 512, 1024
#define MAX_BLOCKS 1024          // Adjust based on GPU memory
#define BATCH_SIZE 2000000       // Larger for more GPU memory
```

## Academic Use

This implementation is designed for educational purposes and demonstrates:

- Monte Carlo simulation in finance
- CUDA parallel programming
- Variance reduction techniques
- Convergence analysis
- Performance optimization

### Assignment Integration

The code includes analytical formula comparison to show the accuracy improvements of Monte Carlo simulation over simplified approximations.

## License

This code is provided for educational and research purposes. Please cite appropriately if used in academic work.

## Support

For issues or questions:

1. Check GPU compatibility and CUDA installation
2. Verify compilation flags match your GPU architecture
3. Review error messages for memory or driver issues
4. Test with smaller simulation sizes first

## References

- Hull, J. C. (2018). _Options, Futures, and Other Derivatives_
- Glasserman, P. (2004). _Monte Carlo Methods in Financial Engineering_
- NVIDIA CUDA Programming Guide
- Antithetic Variates in Monte Carlo Simulation
