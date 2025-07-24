#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

// Structure to hold matrix
typedef struct {
    double** data;
    int size;
} Matrix;

// Function prototypes
Matrix* createMatrix(int size);
void freeMatrix(Matrix* m);
void printMatrix(Matrix* m);
void fillRandomMatrix(Matrix* m);
void copyMatrix(Matrix* src, Matrix* dest, int row, int col, int size);
void addMatrix(Matrix* a, Matrix* b, Matrix* result, int size);
void subtractMatrix(Matrix* a, Matrix* b, Matrix* result, int size);
Matrix* naiveMultiply(Matrix* a, Matrix* b);
Matrix* strassenMultiply(Matrix* a, Matrix* b);
Matrix* strassenHelper(Matrix* a, Matrix* b, int size);
int nextPowerOf2(int n);
void runBenchmark();
void saveResultsToFile(int* sizes, double* naive_times, double* strassen_times, double* naive_std, double* strassen_std, int count);

// Create a matrix of given size
Matrix* createMatrix(int size) {
    Matrix* m = (Matrix*)malloc(sizeof(Matrix));
    m->size = size;
    m->data = (double**)malloc(size * sizeof(double*));
    for (int i = 0; i < size; i++) {
        m->data[i] = (double*)calloc(size, sizeof(double));
    }
    return m;
}

// Free matrix memory
void freeMatrix(Matrix* m) {
    if (m) {
        for (int i = 0; i < m->size; i++) {
            free(m->data[i]);
        }
        free(m->data);
        free(m);
    }
}

// Print matrix (for small matrices)
void printMatrix(Matrix* m) {
    for (int i = 0; i < m->size; i++) {
        for (int j = 0; j < m->size; j++) {
            printf("%8.2f ", m->data[i][j]);
        }
        printf("\n");
    }
}

// Fill matrix with random values
void fillRandomMatrix(Matrix* m) {
    for (int i = 0; i < m->size; i++) {
        for (int j = 0; j < m->size; j++) {
            m->data[i][j] = (double)(rand() % 100) / 10.0;
        }
    }
}

// Copy submatrix from source to destination
void copyMatrix(Matrix* src, Matrix* dest, int row, int col, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            dest->data[i][j] = src->data[row + i][col + j];
        }
    }
}

// Add two matrices
void addMatrix(Matrix* a, Matrix* b, Matrix* result, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            result->data[i][j] = a->data[i][j] + b->data[i][j];
        }
    }
}

// Subtract two matrices
void subtractMatrix(Matrix* a, Matrix* b, Matrix* result, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            result->data[i][j] = a->data[i][j] - b->data[i][j];
        }
    }
}

// Naive matrix multiplication O(n^3)
Matrix* naiveMultiply(Matrix* a, Matrix* b) {
    Matrix* result = createMatrix(a->size);
    for (int i = 0; i < a->size; i++) {
        for (int j = 0; j < a->size; j++) {
            result->data[i][j] = 0;
            for (int k = 0; k < a->size; k++) {
                result->data[i][j] += a->data[i][k] * b->data[k][j];
            }
        }
    }
    return result;
}

// Find next power of 2 greater than or equal to n
int nextPowerOf2(int n) {
    int p = 1;
    while (p < n) {
        p *= 2;
    }
    return p;
}

// Strassen multiplication with zero-padding
Matrix* strassenMultiply(Matrix* a, Matrix* b) {
    int original_size = a->size;
    int padded_size = nextPowerOf2(original_size);
    
    // Create padded matrices
    Matrix* a_padded = createMatrix(padded_size);
    Matrix* b_padded = createMatrix(padded_size);
    
    // Copy original matrices to padded matrices
    for (int i = 0; i < original_size; i++) {
        for (int j = 0; j < original_size; j++) {
            a_padded->data[i][j] = a->data[i][j];
            b_padded->data[i][j] = b->data[i][j];
        }
    }
    
    // Perform Strassen multiplication on padded matrices
    Matrix* result_padded = strassenHelper(a_padded, b_padded, padded_size);
    
    // Extract result back to original size
    Matrix* result = createMatrix(original_size);
    for (int i = 0; i < original_size; i++) {
        for (int j = 0; j < original_size; j++) {
            result->data[i][j] = result_padded->data[i][j];
        }
    }
    
    // Clean up
    freeMatrix(a_padded);
    freeMatrix(b_padded);
    freeMatrix(result_padded);
    
    return result;
}

// Strassen helper function (recursive)
Matrix* strassenHelper(Matrix* a, Matrix* b, int size) {
    // Base case: use naive multiplication for small matrices
    if (size <= 32) {
        return naiveMultiply(a, b);
    }
    
    int half = size / 2;
    
    // Create submatrices
    Matrix* a11 = createMatrix(half);
    Matrix* a12 = createMatrix(half);
    Matrix* a21 = createMatrix(half);
    Matrix* a22 = createMatrix(half);
    Matrix* b11 = createMatrix(half);
    Matrix* b12 = createMatrix(half);
    Matrix* b21 = createMatrix(half);
    Matrix* b22 = createMatrix(half);
    
    // Fill submatrices
    copyMatrix(a, a11, 0, 0, half);
    copyMatrix(a, a12, 0, half, half);
    copyMatrix(a, a21, half, 0, half);
    copyMatrix(a, a22, half, half, half);
    copyMatrix(b, b11, 0, 0, half);
    copyMatrix(b, b12, 0, half, half);
    copyMatrix(b, b21, half, 0, half);
    copyMatrix(b, b22, half, half, half);
    
    // Create temporary matrices for Strassen computations
    Matrix* m1 = createMatrix(half);
    Matrix* m2 = createMatrix(half);
    Matrix* m3 = createMatrix(half);
    Matrix* m4 = createMatrix(half);
    Matrix* m5 = createMatrix(half);
    Matrix* m6 = createMatrix(half);
    Matrix* m7 = createMatrix(half);
    Matrix* temp1 = createMatrix(half);
    Matrix* temp2 = createMatrix(half);
    
    // M1 = (A11 + A22) * (B11 + B22)
    addMatrix(a11, a22, temp1, half);
    addMatrix(b11, b22, temp2, half);
    Matrix* m1_result = strassenHelper(temp1, temp2, half);
    for (int i = 0; i < half; i++) {
        for (int j = 0; j < half; j++) {
            m1->data[i][j] = m1_result->data[i][j];
        }
    }
    freeMatrix(m1_result);
    
    // M2 = (A21 + A22) * B11
    addMatrix(a21, a22, temp1, half);
    Matrix* m2_result = strassenHelper(temp1, b11, half);
    for (int i = 0; i < half; i++) {
        for (int j = 0; j < half; j++) {
            m2->data[i][j] = m2_result->data[i][j];
        }
    }
    freeMatrix(m2_result);
    
    // M3 = A11 * (B12 - B22)
    subtractMatrix(b12, b22, temp2, half);
    Matrix* m3_result = strassenHelper(a11, temp2, half);
    for (int i = 0; i < half; i++) {
        for (int j = 0; j < half; j++) {
            m3->data[i][j] = m3_result->data[i][j];
        }
    }
    freeMatrix(m3_result);
    
    // M4 = A22 * (B21 - B11)
    subtractMatrix(b21, b11, temp2, half);
    Matrix* m4_result = strassenHelper(a22, temp2, half);
    for (int i = 0; i < half; i++) {
        for (int j = 0; j < half; j++) {
            m4->data[i][j] = m4_result->data[i][j];
        }
    }
    freeMatrix(m4_result);
    
    // M5 = (A11 + A12) * B22
    addMatrix(a11, a12, temp1, half);
    Matrix* m5_result = strassenHelper(temp1, b22, half);
    for (int i = 0; i < half; i++) {
        for (int j = 0; j < half; j++) {
            m5->data[i][j] = m5_result->data[i][j];
        }
    }
    freeMatrix(m5_result);
    
    // M6 = (A21 - A11) * (B11 + B12)
    subtractMatrix(a21, a11, temp1, half);
    addMatrix(b11, b12, temp2, half);
    Matrix* m6_result = strassenHelper(temp1, temp2, half);
    for (int i = 0; i < half; i++) {
        for (int j = 0; j < half; j++) {
            m6->data[i][j] = m6_result->data[i][j];
        }
    }
    freeMatrix(m6_result);
    
    // M7 = (A12 - A22) * (B21 + B22)
    subtractMatrix(a12, a22, temp1, half);
    addMatrix(b21, b22, temp2, half);
    Matrix* m7_result = strassenHelper(temp1, temp2, half);
    for (int i = 0; i < half; i++) {
        for (int j = 0; j < half; j++) {
            m7->data[i][j] = m7_result->data[i][j];
        }
    }
    freeMatrix(m7_result);
    
    // Calculate result submatrices
    Matrix* c11 = createMatrix(half);
    Matrix* c12 = createMatrix(half);
    Matrix* c21 = createMatrix(half);
    Matrix* c22 = createMatrix(half);
    
    // C11 = M1 + M4 - M5 + M7
    addMatrix(m1, m4, temp1, half);
    subtractMatrix(temp1, m5, temp2, half);
    addMatrix(temp2, m7, c11, half);
    
    // C12 = M3 + M5
    addMatrix(m3, m5, c12, half);
    
    // C21 = M2 + M4
    addMatrix(m2, m4, c21, half);
    
    // C22 = M1 - M2 + M3 + M6
    subtractMatrix(m1, m2, temp1, half);
    addMatrix(temp1, m3, temp2, half);
    addMatrix(temp2, m6, c22, half);
    
    // Combine result submatrices
    Matrix* result = createMatrix(size);
    for (int i = 0; i < half; i++) {
        for (int j = 0; j < half; j++) {
            result->data[i][j] = c11->data[i][j];
            result->data[i][j + half] = c12->data[i][j];
            result->data[i + half][j] = c21->data[i][j];
            result->data[i + half][j + half] = c22->data[i][j];
        }
    }
    
    // Clean up
    freeMatrix(a11); freeMatrix(a12); freeMatrix(a21); freeMatrix(a22);
    freeMatrix(b11); freeMatrix(b12); freeMatrix(b21); freeMatrix(b22);
    freeMatrix(m1); freeMatrix(m2); freeMatrix(m3); freeMatrix(m4);
    freeMatrix(m5); freeMatrix(m6); freeMatrix(m7);
    freeMatrix(temp1); freeMatrix(temp2);
    freeMatrix(c11); freeMatrix(c12); freeMatrix(c21); freeMatrix(c22);
    
    return result;
}

// Save benchmark results to file for plotting
void saveResultsToFile(int* sizes, double* naive_times, double* strassen_times, double* naive_std, double* strassen_std, int count) {
    FILE* file = fopen("benchmark_results.txt", "w");
    if (file) {
        fprintf(file, "Size,Naive_Time,Naive_Std,Strassen_Time,Strassen_Std,Speedup\n");
        for (int i = 0; i < count; i++) {
            double speedup = naive_times[i] / strassen_times[i];
            fprintf(file, "%d,%.6f,%.6f,%.6f,%.6f,%.2f\n", 
                    sizes[i], naive_times[i], naive_std[i], strassen_times[i], strassen_std[i], speedup);
        }
        fclose(file);
        printf("Results saved to benchmark_results.txt\n");
    }
}

// Run comprehensive benchmark
void runBenchmark() {
    printf("=== Strassen vs Naive Matrix Multiplication Benchmark ===\n");
    printf("Running 10 tests per matrix size for statistical reliability\n\n");

    int test_sizes[] = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    int num_runs = 10;
    
    double naive_times[num_tests];
    double strassen_times[num_tests];
    double naive_std[num_tests];
    double strassen_std[num_tests];
    
    printf("%-8s %-15s %-18s %-10s\n", "Size", "Naive(s)±σ", "Strassen(s)±σ", "Speedup");
    printf("-----------------------------------------------\n");
    
    for (int i = 0; i < num_tests; i++) {
        int size = test_sizes[i];
        printf("Testing %dx%d matrices... ", size, size);
        fflush(stdout);
        
        double naive_run_times[num_runs];
        double strassen_run_times[num_runs];
        
        // Run multiple tests for this size
        for (int run = 0; run < num_runs; run++) {
            // Create test matrices
            Matrix* a = createMatrix(size);
            Matrix* b = createMatrix(size);
            fillRandomMatrix(a);
            fillRandomMatrix(b);
            
            // Benchmark naive multiplication
            clock_t start = clock();
            Matrix* naive_result = naiveMultiply(a, b);
            clock_t end = clock();
            naive_run_times[run] = ((double)(end - start)) / CLOCKS_PER_SEC;
            
            // Benchmark Strassen multiplication
            start = clock();
            Matrix* strassen_result = strassenMultiply(a, b);
            end = clock();
            strassen_run_times[run] = ((double)(end - start)) / CLOCKS_PER_SEC;
            
            // Clean up
            freeMatrix(a);
            freeMatrix(b);
            freeMatrix(naive_result);
            freeMatrix(strassen_result);
        }
        
        // Calculate statistics
        double naive_sum = 0, strassen_sum = 0;
        for (int run = 0; run < num_runs; run++) {
            naive_sum += naive_run_times[run];
            strassen_sum += strassen_run_times[run];
        }
        naive_times[i] = naive_sum / num_runs;
        strassen_times[i] = strassen_sum / num_runs;
        
        // Calculate standard deviation
        double naive_var = 0, strassen_var = 0;
        for (int run = 0; run < num_runs; run++) {
            naive_var += (naive_run_times[run] - naive_times[i]) * (naive_run_times[run] - naive_times[i]);
            strassen_var += (strassen_run_times[run] - strassen_times[i]) * (strassen_run_times[run] - strassen_times[i]);
        }
        naive_std[i] = sqrt(naive_var / num_runs);
        strassen_std[i] = sqrt(strassen_var / num_runs);
        
        // Calculate speedup
        double speedup = naive_times[i] / strassen_times[i];
        
        printf("Done\n");
        // printf("%-8d %-15s %-18s %-10.2fx", 
        //        size, 
        //        "", // Will be filled in next printf
        //        "", // Will be filled in next printf
        //        speedup);
        
        // Print times with standard deviation on separate line for better formatting
        printf("%-8d %.3f±%.3f   %.3f±%.3f      %-.2fx", 
               size, naive_times[i], naive_std[i], strassen_times[i], strassen_std[i], speedup);
        
        printf("\n");
    }
    
    // Save results for visualization
    saveResultsToFile(test_sizes, naive_times, strassen_times, naive_std, strassen_std, num_tests);
    
    // Generate simple ASCII visualization
    printf("\n=== Performance Visualization (Log Scale) ===\n");
    printf("Execution Time Comparison:\n\n");
    
    // Find min and max times for log scaling
    double min_time = naive_times[0];
    double max_time = naive_times[0];
    for (int i = 0; i < num_tests; i++) {
        if (naive_times[i] > 0 && naive_times[i] < min_time) min_time = naive_times[i];
        if (strassen_times[i] > 0 && strassen_times[i] < min_time) min_time = strassen_times[i];
        if (naive_times[i] > max_time) max_time = naive_times[i];
        if (strassen_times[i] > max_time) max_time = strassen_times[i];
    }
    
    // Use log scale for better visualization
    double log_min = log2(min_time);
    double log_max = log2(max_time);
    double log_range = log_max - log_min;
    
    for (int i = 0; i < num_tests; i++) {
        printf("%4dx%-4d: ", test_sizes[i], test_sizes[i]);
        
        // Calculate log-scaled bar lengths
        int naive_bar = 0, strassen_bar = 0;
        if (naive_times[i] > 0) {
            naive_bar = (int)((log2(naive_times[i]) - log_min) * 40 / log_range);
        }
        if (strassen_times[i] > 0) {
            strassen_bar = (int)((log2(strassen_times[i]) - log_min) * 40 / log_range);
        }
        
        // Ensure bars are at least 1 character if time > 0
        if (naive_times[i] > 0 && naive_bar == 0) naive_bar = 1;
        if (strassen_times[i] > 0 && strassen_bar == 0) strassen_bar = 1;
        
        printf("Naive    [");
        for (int j = 0; j < naive_bar; j++) printf("#");
        for (int j = naive_bar; j < 40; j++) printf(" ");
        printf("] %.4fs", naive_times[i]);
        
        // Show speedup indicator
        double speedup = naive_times[i] / strassen_times[i];
        if (speedup > 1.0) {
            printf(" ↗ %.2fx faster", speedup);
        } else {
            printf(" ↘ %.2fx slower", 1.0/speedup);
        }
        printf("\n");
        
        printf("           Strassen [");
        for (int j = 0; j < strassen_bar; j++) printf("=");
        for (int j = strassen_bar; j < 40; j++) printf(" ");
        printf("] %.4fs\n\n", strassen_times[i]);
    }
    
    // Add scale reference
    printf("Scale: Log2 time from %.1e to %.1e seconds\n", min_time, max_time);
    printf("Legend: # = Naive O(n³)    = = Strassen O(n^%.2f)\n", log2(7));
}

// Demonstration function
void demonstrateCorrectness() {
    printf("=== Correctness Demonstration ===\n");
    
    // Test with a small non-power-of-2 matrix
    int size = 5;
    Matrix* a = createMatrix(size);
    Matrix* b = createMatrix(size);
    
    // Fill with simple values for easy verification
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            a->data[i][j] = i + j + 1;
            b->data[i][j] = (i == j) ? 1 : 0; // Identity-like matrix
        }
    }
    
    printf("Matrix A (%dx%d):\n", size, size);
    printMatrix(a);
    printf("\nMatrix B (%dx%d):\n", size, size);
    printMatrix(b);
    
    Matrix* naive_result = naiveMultiply(a, b);
    Matrix* strassen_result = strassenMultiply(a, b);
    
    printf("\nNaive Result:\n");
    printMatrix(naive_result);
    printf("\nStrassen Result:\n");
    printMatrix(strassen_result);
    
    // Check if results match
    int match = 1;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (fabs(naive_result->data[i][j] - strassen_result->data[i][j]) > 1e-10) {
                match = 0;
                break;
            }
        }
        if (!match) break;
    }
    
    printf("\nResults match: %s\n\n", match ? "YES" : "NO");
    
    freeMatrix(a);
    freeMatrix(b);
    freeMatrix(naive_result);
    freeMatrix(strassen_result);
}

int main() {
    srand(time(NULL));
    
    printf("Strassen Matrix Multiplication with Zero-Padding\n");
    printf("================================================\n\n");
    
    // Demonstrate correctness with small matrices
    // demonstrateCorrectness();
    
    // Run comprehensive benchmark
    runBenchmark();
    
    // printf("\nNote: For visualization, you can plot the data from 'benchmark_results.txt'\n");
    // printf("using tools like gnuplot, Python matplotlib, or Excel.\n");
    
    return 0;
}
