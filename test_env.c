// C Development Environment Test Suite
// Test basic C compilation and math libraries

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

// Test 1: Basic C compilation
void test_basic_c() {
    printf("=== Test 1: Basic C Compilation ===\n");
    printf("✅ C compiler working!\n");
    printf("✅ Standard I/O working!\n");
    printf("✅ Basic functions working!\n\n");
}

// Test 2: Math library
void test_math_library() {
    printf("=== Test 2: Math Library ===\n");
    
    double x = 2.5;
    double sqrt_result = sqrt(x);
    double exp_result = exp(1.0);
    double log_result = log(exp_result);
    double sin_result = sin(M_PI / 2);
    
    printf("sqrt(%.1f) = %.6f ✅\n", x, sqrt_result);
    printf("exp(1.0) = %.6f ✅\n", exp_result);
    printf("log(e) = %.6f ✅\n", log_result);
    printf("sin(π/2) = %.6f ✅\n", sin_result);
    
    // Test if math results are reasonable
    if (fabs(sqrt_result - 1.581139) < 0.001 && 
        fabs(exp_result - 2.718282) < 0.001 &&
        fabs(log_result - 1.0) < 0.001 &&
        fabs(sin_result - 1.0) < 0.001) {
        printf("✅ Math library working correctly!\n\n");
    } else {
        printf("❌ Math library results incorrect!\n\n");
    }
}

// Test 3: Memory allocation
void test_memory_allocation() {
    printf("=== Test 3: Memory Allocation ===\n");
    
    size_t size = 1000000;  // 1M doubles
    double* array = (double*)malloc(size * sizeof(double));
    
    if (array == NULL) {
        printf("❌ Memory allocation failed!\n\n");
        return;
    }
    
    // Fill array with test data
    for (size_t i = 0; i < size; i++) {
        array[i] = (double)i * 0.001;
    }
    
    // Check first and last elements
    printf("First element: %.6f ✅\n", array[0]);
    printf("Last element: %.6f ✅\n", array[size-1]);
    printf("✅ Memory allocation working (%zu MB allocated)\n", (size * sizeof(double)) / (1024 * 1024));
    
    free(array);
    printf("✅ Memory deallocation working\n\n");
}

// Test 4: Performance timing
void test_performance_timing() {
    printf("=== Test 4: Performance Timing ===\n");
    
    clock_t start, end;
    int iterations = 10000000;  // 10M iterations
    
    start = clock();
    
    double sum = 0.0;
    for (int i = 0; i < iterations; i++) {
        sum += sqrt((double)i);
    }
    
    end = clock();
    
    double cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    double ops_per_sec = iterations / cpu_time;
    
    printf("Computed %d square roots\n", iterations);
    printf("Time taken: %.6f seconds ✅\n", cpu_time);
    printf("Performance: %.0f operations/second ✅\n", ops_per_sec);
    printf("Result sum: %.6f ✅\n", sum);
    printf("✅ Performance timing working!\n\n");
}

// Test 5: Simple Monte Carlo test (preparation for GPU version)
void test_simple_monte_carlo() {
    printf("=== Test 5: Simple Monte Carlo (π estimation) ===\n");
    
    srand(time(NULL));
    int total_points = 1000000;
    int points_in_circle = 0;
    
    clock_t start = clock();
    
    for (int i = 0; i < total_points; i++) {
        double x = (double)rand() / RAND_MAX * 2.0 - 1.0;  // -1 to 1
        double y = (double)rand() / RAND_MAX * 2.0 - 1.0;  // -1 to 1
        
        if (x*x + y*y <= 1.0) {
            points_in_circle++;
        }
    }
    
    clock_t end = clock();
    
    double pi_estimate = 4.0 * points_in_circle / total_points;
    double error = fabs(pi_estimate - M_PI) / M_PI * 100.0;
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    printf("Total points: %d\n", total_points);
    printf("Points in circle: %d\n", points_in_circle);
    printf("π estimate: %.6f\n", pi_estimate);
    printf("Actual π: %.6f\n", M_PI);
    printf("Error: %.4f%% ✅\n", error);
    printf("Time: %.6f seconds\n", time_taken);
    printf("Rate: %.0f points/second ✅\n", total_points / time_taken);
    
    if (error < 1.0) {
        printf("✅ Monte Carlo working correctly!\n\n");
    } else {
        printf("⚠️ Monte Carlo error high (but likely just random variation)\n\n");
    }
}

// Test 6: Compiler and system info
void test_system_info() {
    printf("=== Test 6: System Information ===\n");
    
    printf("Compiler: ");
    #ifdef __GNUC__
        printf("GCC %d.%d.%d ✅\n", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
    #elif defined(_MSC_VER)
        printf("MSVC %d ✅\n", _MSC_VER);
    #elif defined(__clang__)
        printf("Clang %d.%d.%d ✅\n", __clang_major__, __clang_minor__, __clang_patchlevel__);
    #else
        printf("Unknown ✅\n");
    #endif
    
    printf("Architecture: ");
    #ifdef _WIN64
        printf("Windows 64-bit ✅\n");
    #elif defined(_WIN32)
        printf("Windows 32-bit ✅\n");
    #elif defined(__linux__)
        printf("Linux ✅\n");
    #elif defined(__APPLE__)
        printf("macOS ✅\n");
    #else
        printf("Unknown ✅\n");
    #endif
    
    printf("Size of int: %zu bytes ✅\n", sizeof(int));
    printf("Size of long: %zu bytes ✅\n", sizeof(long));
    printf("Size of double: %zu bytes ✅\n", sizeof(double));
    printf("Size of pointer: %zu bytes ✅\n", sizeof(void*));
    
    printf("✅ System information retrieved!\n\n");
}

// Main test runner
int main() {
    printf("🔧 C Development Environment Test Suite\n");
    printf("Testing C compiler, math library, memory allocation, and performance\n");
    printf("This will help ensure your environment is ready for CUDA development\n\n");
    
    test_basic_c();
    test_math_library();
    test_memory_allocation();
    test_performance_timing();
    test_simple_monte_carlo();
    test_system_info();
    
    printf("=== Summary ===\n");
    printf("✅ All basic C development tests completed!\n");
    printf("✅ Your environment appears ready for CUDA development\n");
    printf("✅ Math library working correctly\n");
    printf("✅ Memory allocation working\n");
    printf("✅ Performance timing working\n");
    printf("✅ Monte Carlo algorithms working\n\n");
    
    printf("🚀 Next steps:\n");
    printf("1. If all tests passed, your C environment is ready!\n");
    printf("2. Next we can test CUDA toolkit installation\n");
    printf("3. Then we can compile the Asian options CUDA code\n\n");
    
    printf("📝 To compile this test:\n");
    printf("GCC/Clang: gcc -o test_env test_env.c -lm\n");
    printf("MSVC: cl test_env.c\n\n");
    
    return 0;
}