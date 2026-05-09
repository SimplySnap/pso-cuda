#include "evals.cuh"
/*Levy Function - 3 dim*/
__global__ void levy_fn(float fitness, const float x[], int n_dim) {
    /*
    Takes as input point (x[]) and fitness, writes to fitness for this given point,
    n_dim is number of dims
    Code from wiseodd on github
    ASSUME n_dim > 2
    */
    //error check
    if (n_dim < 3) {
        std::throw_error("Not enough dimensions!");
        return;
    }
    float res = 0;
    float y1 = 1 + (x[0] - 1) / 4;
    float yn = 1 + (x[n_dim - 1] - 1) / 4;

    res += powf(sinf(pi * y1), 2);

    for (int i = 0; i < n_dim - 1; i++){
        float y = 1 + (x[i] - 1) / 4;
        float yp = 1 + (x[i + 1] - 1) / 4;

        res += powf(y - 1, 2) * (1 + 10 * powf(sinf(pi * yp), 2)) 
                + powf(yn - 1, 2);
    }
    fitness = res;
}

/*Rastrigin Function. NOT parallelized*/
__global__ void rastrigin_fn(float fitness, const float x[], int n_dim) {
    /* 
    Rastrigin Function
    Takes as input point (x[]) fitness to write to, n_dim number of dims
    */
    float res = 10 * n_dim;
    for (int i = 0; i < n_dim; ++i) {
        res += ((x[i]*x[i])-10*cosf(2*pi*x[i]));
    }
    fitness = res;
}

/*Schaffer F2 Function*/
__global__ void schaffer_f2_fn(float fitness, const float x[], int n_dim) {
    /*
    Schaffer F2 function - 2 dim MUST BE
    takes as input (x[]), fitness to write to, n_dim number of dims
    */
    if (n_dims != 2){
        std::throw_error("Not 2 dimensions!");
        return;
    }
    float res = 0.5;
    float frac = (sinf(x[0]*x[0]-x[1]*x[1])**2 - 0.5)/(1+0.001*(x[0]*x[0]+x[1]*x[1]))**2;
    fitness += frac;
}

