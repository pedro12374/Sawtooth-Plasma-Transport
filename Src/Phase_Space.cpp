#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

using namespace std;
const double kx1 = 6;
const double ky1 = 3;

const double kx2 = -3.5;
const double ky2 = -1.5;

const double kx3 = -2.5;
const double ky3 = -1.5;


const double w1 = 0.476;
const double w2 = 0.476;
const double w3 = 0.476;

const double A1 = 1.0;

double x_min = 0.01, x_max = M_PI-0.01;
double y_min = 0, y_max = 2*M_PI-0.01;





template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6)
{
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}



void f(double x, double y, double t, double& dxdt, double& dydt, double A2, double tau, double beta) {
    
    // --- Sawtooth Wave Modulation using the floor() function ---
    const double T =  tau; // Period
    const double A3 = A2/3.0;
    // This calculates a sawtooth amplitude that ramps from 0 to 1
    double A1_amplitude = (t / T) - floor(t / T);
    double A2_amplitude = 0;
    if(fmod(t,tau)>= -1.0*(tau/beta) && fmod(t,tau)<=(tau/beta)) {
        A2_amplitude = 1.0; // Full amplitude at multiples of beta
    } else {
        A2_amplitude = 0.0; // Zero amplitude otherwise
    }
    // --- The rest of the function ---
    const double v = fabs(w2 / ky2 - w1 / ky1);
    const double v3 = fabs(w3 / ky3 - w1 / ky1);
    dxdt =  A1*A1_amplitude * ky1 * sin(kx1 * x) * sin(ky1 * y) + A2 * A2_amplitude * ky2 * sin(kx2 * x+M_PI/2.0) * sin(ky2 * (y - v * t)) + A3 * A2_amplitude * ky3 * sin(kx3 * x) * sin(ky3 * (y - v3 * t));
    dydt =  A1*A1_amplitude * kx1 * cos(kx1 * x) * cos(ky1 * y) + A2 * A2_amplitude * kx2 * cos(kx2 * x+M_PI/2.0) * cos(ky2 * (y - v * t)) + A3 * A2_amplitude * kx3 * cos(kx3 * x) * cos(ky3 * (y - v3 * t));
}

void rungeKutta(double x, double y, double t, double dt, double& nx, double& ny,double A2,double tau = 0.5, double beta = 0.0) {
    double k1x, k1y, k2x, k2y, k3x, k3y, k4x, k4y;
    
    // Step 1
    f(x, y, t, k1x, k1y, A2, tau, beta);
    // Step 2
    f(x + 0.5f * dt * k1x, y + 0.5f * dt * k1y, t + 0.5f * dt, k2x, k2y, A2, tau, beta);
    // Step 3
    f(x + 0.5f * dt * k2x, y + 0.5f * dt * k2y, t + 0.5f * dt, k3x, k3y, A2, tau, beta);
    // Step 4
    f(x + dt * k3x, y + dt * k3y, t + dt, k4x, k4y, A2, tau, beta);

    // Compute next point using weighted sum
    nx = x + (dt / 6.0f) * (k1x + 2.0f * k2x + 2.0f * k3x + k4x);
    ny = y + (dt / 6.0f) * (k1y + 2.0f * k2y + 2.0f * k3y + k4y);
}




int solver(double A2, double tau = 0.5, double beta = 0.0) {



    int grid_size = 15; // Number of points along each axis

    double dx = (x_max - x_min) / (grid_size - 1);
    double dy = (y_max - y_min) / (grid_size - 1);
    double dt = 0.01;
    
    std::string strobo_file = "../Strobo/A2_" + to_string_with_precision(A2,4) + "_" + to_string_with_precision(tau,4) + "_" + to_string_with_precision(beta,4) + ".csv";

    ofstream MyFile(strobo_file);

	double const v = fabs(w2/ky2 - w1/ky1    );
	double T = 2*M_PI/beta;

    for (int i = 0; i < grid_size; ++i) {
        for (int j = 0; j < grid_size; ++j) {



            double x = x_min + i * dx;
            double y = y_min + j * dy;

            double nx, ny;

            double t=0;
            while (t <= 1000*T) {

                rungeKutta(x, y, t, dt, nx, ny,A2,tau,beta);

                if(ny>2*M_PI){
                    y = ny-2*M_PI;
                }else if(ny<0){
                    y = ny+2*M_PI;
                }else{
                    y = ny;
                }
                
                x = nx;
                //y = ny;
                t += dt;
                
                
                if(fmod(t,T) >= -0.01   && fmod(t,T) <= 0.01  ){
                
                        MyFile<<t<<"\t"<< x << " \t " << y << endl;    
                }
                
                
            }
            MyFile << endl; // Add spacing between runs
        }
    }

    return 0;
}


int main(int argc,char* argv[]) {
    
    std::ifstream param(argv[1]);

    double A2_i,A2_f,A2_step;
    
    param >> A2_i>> A2_f >>A2_step;
    
   
    double Vtau[] = {0.1};
    double Vbeta[] = {1.0};
    double vA2[] = {0.1};

    for (double A2 : vA2) {
        for (double tau : Vtau) {
            for (double beta : Vbeta) {
                solver(A2, tau, beta);
            }
        }
    }



    return 0;
}
