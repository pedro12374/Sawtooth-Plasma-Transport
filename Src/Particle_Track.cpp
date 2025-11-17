#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <sstream>
#include <omp.h>

using namespace std;
const double kx1 = 1;
const double ky1 = 1;

const double kx2 = 1;
const double ky2 = 1;

const double w1 = 1.1;
const double w2 = 0.1;

const double A1 = 1.0;

double x_min = 0, x_max = M_PI-0.01;
double y_min = 0, y_max = 2*M_PI-0.01;



double const v = fabs(w2/ky2 - w1/ky1    );
double const T = 4*M_PI/v;


template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6)
{
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}


void f(double x, double y, double t, double& dxdt, double& dydt,double A2,double tau = 0.5, double beta = 0.0) {

    double sinetau = sin(tau*t+beta)*sin(tau*t+beta);
    dxdt = A1*sinetau*ky1*sinf(kx1*x)*sinf(ky1*y) + A2*sinetau*ky2*sinf(kx2*x)*sinf(ky2*(y-v*t));
    dydt = A1*sinetau*kx1*cosf(kx1*x)*cosf(ky1*y) + A2*sinetau*kx2*cosf(kx2*x)*cosf(ky2*(y-v*t));
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


    int grid_size = 10; // Number of points along each axis

    double dx = (x_max - x_min) / (grid_size - 1);
    double dy = (y_max - y_min) / (grid_size - 1);
    double dt = 0.01;

    std::string strobo_file = "../Particle/A2_" + to_string_with_precision(A2,4) + "_" + to_string_with_precision(tau,4) + "_" + to_string_with_precision(beta,4) + ".csv";

    ofstream MyFile(strobo_file);


            double x = 1.5;
            double y = 3.0;

            double nx, ny;

            double t=0;
            while (t <= 1000) {

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
                
                
        
                
             //if(fmod(t,T) >= -0.01   && fmod(t,T) <= 0.01  ){
                
                        MyFile<<t<<"\t"<< x << " \t " << y << endl;    
              //  }
                
                
            }
            MyFile << endl; // Add spacing between runs

    return 0;
}

int main(int argc,char* argv[]) {
    
    std::ifstream param(argv[1]);

    double A2_i,A2_f,A2_step;
    
    param >> A2_i>> A2_f >>A2_step;
    
    double Vtau[5] = {0.0,M_PI_4,M_PI_2,3*M_PI_4,M_PI};
    double Vbeta[2] = {0.0,M_PI_2};
    

     std::cout << "Starting parallel computation on " << omp_get_max_threads() << " threads." << std::endl;
    double start_time = omp_get_wtime();
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 2; j++) {
            if(i==0 && j==0){
                // Skip the case where tau = 0 and beta = 0
                continue;
            }
            
            for(double A2 = A2_i;A2<=A2_f;A2+=A2_step){
                solver(A2,Vtau[i],Vbeta[j]);
            }
        }
    }
    double end_time = omp_get_wtime();
    std::cout << "Finished all computations." << std::endl;
    std::cout << "Total time taken: " << end_time - start_time << " seconds." << std::endl;
 
    return 0;
}
