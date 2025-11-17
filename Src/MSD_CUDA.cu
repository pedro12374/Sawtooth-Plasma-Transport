#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <cuda_runtime.h>
#include <math.h>

// Define the error-checking macro
#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// Grid and Block dimensions
#define GRID_DIM_X 512
#define GRID_DIM_Y 512
#define BLOCK_SIZE 16

// Integration parameters
const double dt = 0.01;
const double FINAL_TIME = 1000.0;
const int NUM_TIME_STEPS = FINAL_TIME / dt + 1;
const int NUM_PARTICLES = GRID_DIM_X * GRID_DIM_Y;

// Physical parameters (consistent with Lyapunov code)
const double kx1 = 6.0;
const double ky1 = 3.0;

const double kx2 = -3.5;
const double ky2 = -1.5;

const double kx3 = -2.5;
const double ky3 = -1.5;

const double w1 = 0.476;
const double w2 = 0.476;
const double w3 = 0.476;

const double A1 = 1.0;

const double X_MIN = -M_PI + 0.001;
const double X_MAX = M_PI - 0.001;
const double Y_MIN = -2 * M_PI + 0.001;
const double Y_MAX = 2 * M_PI - 0.001;


// Helper functions for writing to files
template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6) {
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}

void writeMatrixToFile(const std::string& filename, const double* data, int numRows, int numCols) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return;
    }
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            outFile << data[i * numCols + j] << " ";
        }
        outFile << std::endl;
    }
    outFile.close();
    std::cout << "Matrix written to file: " << filename << std::endl;
}

void writeVectorToFile(const std::string& filename, const double* data, int numElements) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return;
    }
    for (int i = 0; i < numElements; ++i) {
        outFile << data[i] << std::endl;
    }
    outFile.close();
    std::cout << "Vector written to file: " << filename << std::endl;
}

// ========================================================================================
// IMPROVEMENT: Using the exact same vector field function 'f' as in the Lyapunov code
// ========================================================================================
__device__ void f(double x, double y, double t, double& dxdt, double& dydt, double A2, double tau, double beta) {
    // Amplitude modulation functions
    const double A3 = A2 / 3.0;

    // Sawtooth wave for A1's amplitude
    double A1_amplitude = (t / tau) - floor(t / tau);

    // Square pulse for A2 and A3's amplitude
    double A2_amplitude = 0.0;
    if (fabs(beta) < 1e-9) {
        A2_amplitude = 1.0; // If beta is near zero, the pulse is always on
    } else {
        // This logic is now identical to the Lyapunov code
        if (fmod(t, tau) <= (tau / beta)) {
            A2_amplitude = 1.0;
        }
    }

    // Velocities
    const double v = fabs(w2 / ky2 - w1 / ky1);
    const double v3 = fabs(w3 / ky3 - w1 / ky1);

    // dxdt component
    // Note: sin(a + PI/2) = cos(a)
    dxdt = A1 * A1_amplitude * ky1 * sin(kx1 * x) * sin(ky1 * y) +
           A2 * A2_amplitude * ky2 * cos(kx2 * x) * sin(ky2 * (y - v * t)) +
           A3 * A2_amplitude * ky3 * sin(kx3 * x) * sin(ky3 * (y - v3 * t));

    // dydt component
    // Note: cos(a + PI/2) = -sin(a)
    dydt = A1 * A1_amplitude * kx1 * cos(kx1 * x) * cos(ky1 * y) -
           A2 * A2_amplitude * kx2 * sin(kx2 * x) * cos(ky2 * (y - v * t)) +
           A3 * A2_amplitude * kx3 * cos(kx3 * x) * cos(ky3 * (y - v3 * t));
}

// Standard RK4 integrator for particle trajectories
__device__ void rungeKutta(double& x, double& y, double t, double dt, double A2, double tau, double beta) {
    double k1x, k1y, k2x, k2y, k3x, k3y, k4x, k4y;
    f(x, y, t, k1x, k1y, A2, tau, beta);
    f(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, t + 0.5 * dt, k2x, k2y, A2, tau, beta);
    f(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, t + 0.5 * dt, k3x, k3y, A2, tau, beta);
    f(x + dt * k3x, y + dt * k3y, t + dt, k4x, k4y, A2, tau, beta);
    x += (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
    y += (dt / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y);
}

// Kernel for calculating MSD and Final Displacement
__global__ void solve_system_kernel(double* Displ, double* Displ_x, double* Displ_y, double* MSD, double A2, double tau, double beta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int flat_idx = idy * gridDim.x * blockDim.x + idx;

    if (idx < GRID_DIM_X && idy < GRID_DIM_Y) {
        // Initialize position based on grid
        const double xi = X_MIN + (X_MAX - X_MIN) * idx / (double)(GRID_DIM_X - 1);
        const double yi = Y_MIN + (Y_MAX - Y_MIN) * idy / (double)(GRID_DIM_Y - 1);
        double x = xi;
        double y = yi;

        // Main time evolution loop
        for (int i = 0; i < NUM_TIME_STEPS; ++i) {
            double t = i * dt;
            rungeKutta(x, y, t, dt, A2, tau, beta);
            
            // IMPROVEMENT: Added comment clarifying why PBC is not applied here.
            // For a true Mean Square Displacement, we need the "unwrapped" trajectory.
            // Applying periodic boundary conditions would incorrectly cap the displacement at the system size.
            
            // MSD Calculation using atomicAdd to safely sum contributions from all threads.
            double sq_disp = (x - xi) * (x - xi) + (y - yi) * (y - yi);
            atomicAdd(&MSD[i], sq_disp);
        }

        // Final Displacement Calculation
        Displ[flat_idx] = sqrt((x - xi) * (x - xi) + (y - yi) * (y - yi));
        Displ_x[flat_idx] = x - xi;
        Displ_y[flat_idx] = y - yi;
    }
}

// Main solver function
void solver(double A2, double tau, double beta) {
    double *Displ, *Displ_x, *Displ_y, *MSD;
    double *dev_Displ, *dev_Displ_x, *dev_Displ_y, *dev_MSD;

    // Allocate host memory
    Displ = new double[NUM_PARTICLES];
    Displ_x = new double[NUM_PARTICLES];
    Displ_y = new double[NUM_PARTICLES];
    MSD = new double[NUM_TIME_STEPS];

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&dev_Displ, NUM_PARTICLES * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dev_Displ_x, NUM_PARTICLES * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dev_Displ_y, NUM_PARTICLES * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dev_MSD, NUM_TIME_STEPS * sizeof(double)));

    // Initialize MSD sums to zero
    CUDA_CHECK(cudaMemset(dev_MSD, 0, NUM_TIME_STEPS * sizeof(double))); 

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(GRID_DIM_X / BLOCK_SIZE, GRID_DIM_Y / BLOCK_SIZE);

    std::cout << "Calculating for A2=" << A2 << ", tau=" << tau << ", beta=" << beta << "..." << std::endl;
    solve_system_kernel<<<gridSize, blockSize>>>(dev_Displ, dev_Displ_x, dev_Displ_y, dev_MSD, A2, tau, beta);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "Kernel finished." << std::endl;

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(Displ, dev_Displ, NUM_PARTICLES * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Displ_x, dev_Displ_x, NUM_PARTICLES * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Displ_y, dev_Displ_y, NUM_PARTICLES * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(MSD, dev_MSD, NUM_TIME_STEPS * sizeof(double), cudaMemcpyDeviceToHost));

    // Normalize MSD on the host
    for (int i = 0; i < NUM_TIME_STEPS; ++i) {
        MSD[i] /= NUM_PARTICLES;
    }

    // Save data to files
    std::string base_name = "A2_" + to_string_with_precision(A2, 4) + "_tau_" + to_string_with_precision(tau, 4) + "_beta_" + to_string_with_precision(beta, 4);
    writeVectorToFile("../MSD/" + base_name + ".csv", MSD, NUM_TIME_STEPS);
    writeMatrixToFile("../Displ/" + base_name + ".csv", Displ, GRID_DIM_Y, GRID_DIM_X);
    writeMatrixToFile("../Displ_x/" + base_name + ".csv", Displ_x, GRID_DIM_Y, GRID_DIM_X);
    writeMatrixToFile("../Displ_y/" + base_name + ".csv", Displ_y, GRID_DIM_Y, GRID_DIM_X);
    
    // Free memory
    delete[] Displ; delete[] Displ_x; delete[] Displ_y; delete[] MSD;
    CUDA_CHECK(cudaFree(dev_Displ)); CUDA_CHECK(cudaFree(dev_Displ_x)); CUDA_CHECK(cudaFree(dev_Displ_y)); CUDA_CHECK(cudaFree(dev_MSD));
}

int main(int argc, char* argv[]) {
    // Parameter loops for your study
    double Vtau[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    double Vbeta[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double vA2[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};

    for (double A2 : vA2) {
        for (double tau : Vtau) {
             // IMPROVEMENT: Added safety check for tau to prevent division by zero, for robustness.
            if (fabs(tau) < 1e-9) {
                std::cout << "Skipping computations for tau near zero." << std::endl;
                continue;
            }
            for (double beta : Vbeta) {
                solver(A2, tau, beta);
            }
        }
    }
    
    std::cout << "Finished all computations." << std::endl;
    return 0;
}
