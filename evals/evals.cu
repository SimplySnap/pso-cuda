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

/* =============================================================================
 * TSP via random-keys (Smallest-Position-Value) encoding.
 * ===========================================================================*/

// Instance storage in constant memory: coordinates [x0,y0,x1,y1,...] and the
// active city count. __constant__ is broadcast-friendly — every thread reads
// the same coords, so the per-warp access pattern is ideal here.
__constant__ float c_tsp_coords[2 * MAX_TSP_CITIES];
__constant__ int   c_tsp_n;

static int h_tsp_n = 0; // host-side mirror of the uploaded city count

/* TSP tour length under the random-keys decode.
 *
 * The particle position x[] holds one continuous key per city. The tour is the
 * permutation that sorts the keys ascending (SPV decode); fitness is the length
 * of that closed tour (returns to the start). n_dim must equal the uploaded
 * city count.
 *
 * Decode is an in-register selection sort over a local index buffer — O(n^2),
 * which is the intended "harder" cost for stress-testing the swarm. n is capped
 * at MAX_TSP_CITIES so the perm[] buffer fits in per-thread local memory.
 */
__device__ float tsp_fn(const float* x, int n_dim) {
    int n = c_tsp_n;
    if (n <= 0) {
        printf("tsp_fn: no instance uploaded\n");
        return INFINITY;
    }
    if (n_dim != n) {
        printf("tsp_fn: D (%d) must equal city count (%d)\n", n_dim, n);
        return INFINITY; // mis-sized -> never wins fitness
    }

    // argsort x ascending into perm via selection sort (stable enough; ties are
    // resolved by index, which is fine — equal keys map to an arbitrary but
    // deterministic order).
    int perm[MAX_TSP_CITIES];
    for (int i = 0; i < n; ++i) perm[i] = i;
    for (int i = 0; i < n - 1; ++i) {
        int min_j = i;
        for (int j = i + 1; j < n; ++j) {
            if (x[perm[j]] < x[perm[min_j]]) min_j = j;
        }
        int tmp     = perm[i];
        perm[i]     = perm[min_j];
        perm[min_j] = tmp;
    }

    // Closed-tour Euclidean length: city perm[i] -> perm[i+1], wrapping to start.
    float total = 0.0f;
    for (int i = 0; i < n; ++i) {
        int a = perm[i];
        int b = perm[(i + 1) % n];
        float dx = c_tsp_coords[2 * a]     - c_tsp_coords[2 * b];
        float dy = c_tsp_coords[2 * a + 1] - c_tsp_coords[2 * b + 1];
        total += sqrtf(dx * dx + dy * dy);
    }
    return total;
}

__device__ EvaluatorFn d_levy_ptr = levy_fn;
__device__ EvaluatorFn d_rastrigin_ptr = rastrigin_fn;
__device__ EvaluatorFn d_schaffer_ptr = schaffer_f2_fn;
__device__ EvaluatorFn d_tsp_ptr = tsp_fn;
EvaluatorFn h_fn = nullptr; //initialized in main.cu by cudaMemcpyFromSymbol

// Host: push a TSP instance into constant memory. Defined here so it can name
// the __constant__ symbols directly (they aren't visible across .cu files).
cudaError_t tsp_upload_instance(const float* xy, int n_cities) {
    if (n_cities <= 0 || n_cities > MAX_TSP_CITIES) {
        fprintf(stderr,
            "tsp_upload_instance: n_cities=%d out of range (1..%d)\n",
            n_cities, MAX_TSP_CITIES);
        return cudaErrorInvalidValue;
    }
    cudaError_t err = cudaMemcpyToSymbol(
        c_tsp_coords, xy, sizeof(float) * 2 * n_cities);
    if (err != cudaSuccess) return err;
    err = cudaMemcpyToSymbol(c_tsp_n, &n_cities, sizeof(int));
    if (err != cudaSuccess) return err;
    h_tsp_n = n_cities;
    return cudaSuccess;
}

int tsp_num_cities() { return h_tsp_n; }
