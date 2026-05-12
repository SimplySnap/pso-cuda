#include "evals.cuh"
/*Levy Function - 3 dim*/
__device__ float levy_fn(const float* x, int n_dim) {
    /*
    Takes as input point (x[]) and fitness, writes to fitness for this given point,
    n_dim is number of dims
    Code from wiseodd on github
    ASSUME n_dim > 2
    */
    //error check
    if (n_dim < 3) {
        printf("fn only defined for over 2 dimensions");
        return INFINITY; //not defined-> never wins fitness;
    }
    //y1, yn outside loop
    float y1 = 1.0f + (x[0] - 1.0f) / 4.0f;
    float yn = 1.0f + (x[n_dim - 1] - 1.0f) / 4.0f;

    float res = powf(sinf(M_PI_F * y1), 2.0f);

    for (int i = 0; i < n_dim - 1; i++) {
        float y  = 1.0f + (x[i]     - 1.0f) / 4.0f;
        float yp = 1.0f + (x[i + 1] - 1.0f) / 4.0f;
        res += powf(y - 1.0f, 2.0f) * (1.0f + 10.0f * powf(sinf(M_PI_F * yp), 2.0f));
    }

    // Terminal term added ONCE, outside the loop
    res += powf(yn - 1.0f, 2.0f) * (1.0f + powf(sinf(2.0f * M_PI_F * yn), 2.0f));

    return res;
}

/*Rastrigin Function. NOT parallelized*/
__device__ float rastrigin_fn(const float* x, int n_dim) {
    /* 
    Rastrigin Function
    Takes as input point (x[]) fitness to write to, n_dim number of dims
    */
    float res = 10 * n_dim;
    for (int i = 0; i < n_dim; ++i) {
        res += ((x[i]*x[i])-10*cosf(2*M_PI_F*x[i]));
    }
    return res;
}

/*Schaffer F2 Function*/
__device__ float schaffer_f2_fn(const float* x, int n_dim) {
    /*
    Schaffer F2 function - 2 dim MUST BE
    takes as input (x[]), fitness to write to, n_dim number of dims
    */
    if (n_dim != 2){
        printf("fn only defined for 2 dimensions");
        return INFINITY; //again, not defined NEVER wins fitness
    }
    float res = 0.5;
    float frac = (powf(sinf(x[0]*x[0] - x[1]*x[1]), 2.0f) - 0.5)/(1+0.001*(x[0]*x[0]+x[1]*x[1]));
    float frac_sq = frac*frac;
    //add to res and return
    res += frac_sq;
    return res;

}

__device__ EvaluatorFn d_levy_ptr = levy_fn;
__device__ EvaluatorFn d_rastrigin_ptr = rastrigin_fn;
__device__ EvaluatorFn d_schaffer_ptr = schaffer_f2_fn;
