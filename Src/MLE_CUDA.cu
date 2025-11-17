#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cuda_runtime.h>
#include <math.h>

// Grid and Block dimensions
#define GRID_SIZE_X 1024
#define GRID_SIZE_Y 1024
#define BLOCK_SIZE 16

// Integration parameters
const double dt = 0.01;
const double FINAL_TIME = 10000.0;
const int RENORM_STEPS = 10; // Renormalize the tangent vector every 10 steps

// System Parameters
const double kx1 = 6.0;
const double ky1 = 3.0;
const double kx2 = -3.5;
const double ky2 = -1.5;
const double kx3 = -2.5;
const double ky3 = -1.5;

// Wave frequencies
const double w1 = 0.476;
const double w2 = 0.476;
const double w3 = 0.476;

const double A1 = 1.0;
// const double v = fabs(w2 / ky2 - w1 / ky1); // v is now calculated inside device functions

// Domain for initial conditions
const double X_MIN = -M_PI + 0.001;
const double X_MAX = M_PI - 0.001;
const double Y_MIN = -2 * M_PI + 0.001;
const double Y_MAX = 2 * M_PI - 0.001;

// Helper function for filename formatting
template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6) {
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}

// Helper function to write results to a file
void writeMatrixToFile(const std::string& filename, const double* matrix, int numRows, int numCols) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return;
    }
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            outFile << matrix[i * numCols + j] << " ";
        }
        outFile << std::endl;
    }
    outFile.close();
    std::cout << "Matrix written to file: " << filename << std::endl;
}

// ========================================================================================
// MODIFIED: The vector field function 'f' updated with your new equations
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
        // Your condition simplified: fmod(t, tau) for t>0 is always positive.
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

// ========================================================================================
// MODIFIED: Jacobian of the new vector field 'f'
// ========================================================================================
__device__ void jacobian(double x, double y, double t, double A2, double tau, double beta, double& J11, double& J12, double& J21, double& J22) {
    // Recreate the same amplitude modulation as in the function 'f'
    const double A3 = A2 / 3.0;
    double A1_amplitude = (t / tau) - floor(t / tau);
    double A2_amplitude = 0.0;
    if (fabs(beta) < 1e-9) {
        A2_amplitude = 1.0;
    } else {
        if (fmod(t, tau) <= (tau / beta)) {
            A2_amplitude = 1.0;
        }
    }

    // Velocities
    const double v = fabs(w2 / ky2 - w1 / ky1);
    const double v3 = fabs(w3 / ky3 - w1 / ky1);
    
    // J11 = d(dxdt)/dx
    J11 = A1 * A1_amplitude * ky1 * kx1 * cos(kx1 * x) * sin(ky1 * y) -
          A2 * A2_amplitude * ky2 * kx2 * sin(kx2 * x) * sin(ky2 * (y - v * t)) +
          A3 * A2_amplitude * ky3 * kx3 * cos(kx3 * x) * sin(ky3 * (y - v3 * t));

    // J12 = d(dxdt)/dy
    J12 = A1 * A1_amplitude * ky1 * ky1 * sin(kx1 * x) * cos(ky1 * y) +
          A2 * A2_amplitude * ky2 * ky2 * cos(kx2 * x) * cos(ky2 * (y - v * t)) +
          A3 * A2_amplitude * ky3 * ky3 * sin(kx3 * x) * cos(ky3 * (y - v3 * t));

    // J21 = d(dydt)/dx
    J21 = -A1 * A1_amplitude * kx1 * kx1 * sin(kx1 * x) * cos(ky1 * y) -
          A2 * A2_amplitude * kx2 * kx2 * cos(kx2 * x) * cos(ky2 * (y - v * t)) -
          A3 * A2_amplitude * kx3 * kx3 * sin(kx3 * x) * cos(ky3 * (y - v3 * t));

    // J22 = d(dydt)/dy
    J22 = -A1 * A1_amplitude * kx1 * ky1 * cos(kx1 * x) * sin(ky1 * y) +
           A2 * A2_amplitude * kx2 * ky2 * sin(kx2 * x) * sin(ky2 * (y - v * t)) -
           A3 * A2_amplitude * kx3 * ky3 * cos(kx3 * x) * sin(ky3 * (y - v3 * t));
}

// RK4 integrator for the system and its tangent vector evolution
__device__ void rungeKutta_mle(double& x, double& y, double& dx, double& dy, double t, double dt, double A2, double tau, double beta) {
    double k1x, k1y, k1dx, k1dy, k2x, k2y, k2dx, k2dy;
    double k3x, k3y, k3dx, k3dy, k4x, k4y, k4dx, k4dy;
    double J11, J12, J21, J22;

    // K1
    f(x, y, t, k1x, k1y, A2, tau, beta);
    jacobian(x, y, t, A2, tau, beta, J11, J12, J21, J22);
    k1dx = J11 * dx + J12 * dy; k1dy = J21 * dx + J22 * dy;

    // K2
    f(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, t + 0.5 * dt, k2x, k2y, A2, tau, beta);
    jacobian(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, t + 0.5 * dt, A2, tau, beta, J11, J12, J21, J22);
    k2dx = J11 * (dx + 0.5 * dt * k1dx) + J12 * (dy + 0.5 * dt * k1dy);
    k2dy = J21 * (dx + 0.5 * dt * k1dx) + J22 * (dy + 0.5 * dt * k1dy);

    // K3
    f(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, t + 0.5 * dt, k3x, k3y, A2, tau, beta);
    jacobian(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, t + 0.5 * dt, A2, tau, beta, J11, J12, J21, J22);
    k3dx = J11 * (dx + 0.5 * dt * k2dx) + J12 * (dy + 0.5 * dt * k2dy);
    k3dy = J21 * (dx + 0.5 * dt * k2dx) + J22 * (dy + 0.5 * dt * k2dy);

    // K4
    f(x + dt * k3x, y + dt * k3y, t + dt, k4x, k4y, A2, tau, beta);
    jacobian(x + dt * k3x, y + dt * k3y, t + dt, A2, tau, beta, J11, J12, J21, J22);
    k4dx = J11 * (dx + dt * k3dx) + J12 * (dy + dt * k3dy);
    k4dy = J21 * (dx + dt * k3dx) + J22 * (dy + dt * k3dy);

    // Update position and tangent vector
    x += (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
    y += (dt / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y);
    dx += (dt / 6.0) * (k1dx + 2.0 * k2dx + 2.0 * k3dx + k4dx);
    dy += (dt / 6.0) * (k1dy + 2.0 * k2dy + 2.0 * k3dy + k4dy);
}

// The main CUDA kernel that computes the MLE for each initial condition.
__global__ void calculate_mle_kernel(double* mle_results, double A2, double tau, double beta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < GRID_SIZE_X && idy < GRID_SIZE_Y) {
        double x = X_MIN + (X_MAX - X_MIN) * idx / (double)(GRID_SIZE_X - 1);
        double y = Y_MIN + (Y_MAX - Y_MIN) * idy / (double)(GRID_SIZE_Y - 1);
        
        double dx = 1.0, dy = 0.0, lyap_sum = 0.0;

        int num_steps = FINAL_TIME / dt;
        for (int i = 0; i < num_steps; ++i) {
            rungeKutta_mle(x, y, dx, dy, i * dt, dt, A2, tau, beta);
            
            // Periodic boundary conditions for y from [0, 2*PI]
            y = fmod(y, 2.0 * M_PI);
            if (y < 0.0) y += 2.0 * M_PI;
            
            // Renormalization step
            if ((i + 1) % RENORM_STEPS == 0) {
                double d = sqrt(dx * dx + dy * dy);
                if (d > 1.0e-15) { // Avoid log(0)
                    lyap_sum += log(d);
                    dx /= d;
                    dy /= d;
                }
            }
        }
        
        mle_results[idy * GRID_SIZE_X + idx] = lyap_sum / FINAL_TIME;
    }
}

// Solver function to manage memory and kernel launch for a single parameter set.
void solver(double A2, double tau, double beta) {
    const int num_elements = GRID_SIZE_X * GRID_SIZE_Y;
    double* mle_values = new double[num_elements];
    double* dev_mle_values;

    cudaMalloc((void**)&dev_mle_values, num_elements * sizeof(double));

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((GRID_SIZE_X + BLOCK_SIZE - 1) / BLOCK_SIZE, (GRID_SIZE_Y + BLOCK_SIZE - 1) / BLOCK_SIZE);

    std::cout << "Calculating MLE for A2=" << A2 << ", tau=" << tau << ", beta=" << beta << "..." << std::endl;
    calculate_mle_kernel<<<gridSize, blockSize>>>(dev_mle_values, A2, tau, beta);
    cudaDeviceSynchronize();

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
    }

    cudaMemcpy(mle_values, dev_mle_values, num_elements * sizeof(double), cudaMemcpyDeviceToHost);
    
    // Filename includes all parameters
    std::string filename = "../NLE/A2_" + to_string_with_precision(A2, 4) +
                           "_tau_" + to_string_with_precision(tau, 4) +
                           "_beta_" + to_string_with_precision(beta, 4) + ".csv";
    writeMatrixToFile(filename, mle_values, GRID_SIZE_Y, GRID_SIZE_X);

    delete[] mle_values;
    cudaFree(dev_mle_values);
}

// ========================================================================================
// MODIFIED: Main function with more robust parameter loops
// ========================================================================================
int main(int argc, char* argv[]) {
    
    double Vtau[] = { 0.4,0.6};
    double Vbeta[] = { 5.0};
    double vA2[] = {0.1};

    for (int i = 0; i < sizeof(Vtau)/sizeof(Vtau[0]); i++) {
        // MODIFIED: Added a check to prevent division by zero when tau is zero.
        if (fabs(Vtau[i]) < 1e-9) {
            std::cout << "Skipping computations for tau = 0 to avoid division by zero." << std::endl;
            continue;
        }
        for (int j = 0; j < sizeof(Vbeta)/sizeof(Vbeta[0]); j++) {
            for (int k = 0; k < sizeof(vA2)/sizeof(vA2[0]); k++) {
                solver(vA2[k], Vtau[i], Vbeta[j]);
            }
        }
    }
    
    std::cout << "Finished all computations." << std::endl;

    return 0;
}
