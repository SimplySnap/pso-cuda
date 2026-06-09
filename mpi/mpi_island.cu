#include "mpi_island.h"
#include "../pso/cuda_check.cuh"
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>

/* =============================================================================
 * Device gather/scatter for migration sync.
 * -----------------------------------------------------------------------------
 * pbest_pos is SoA [D * N] (component d of particle i at d*N + i), so a single
 * particle's D coordinates are strided across D columns. The old sync pulled
 * each whole column D->H to pick out m migrants — O(D) copies of N floats each.
 * These kernels do the strided gather/scatter on-device so only the packed
 * m*D migrants cross PCIe. Packed layout mirrors the host buffers: [dim*m + mi].
 * ===========================================================================*/

static inline int grid_for(int n, int block) { return (n + block - 1) / block; }

// Gather m migrants' coords from pbest_pos into packed out[dim*m + mi].
__global__ void gather_migrants_kernel(const float* __restrict__ pbest_pos,
                                       const int* __restrict__ idx,
                                       float* __restrict__ out,
                                       int N, int D, int m) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= m * D) return;
    int dim = t / m;
    int mi  = t % m;
    out[t] = pbest_pos[dim * N + idx[mi]];
}

// Scatter packed in[dim*m + mi] back into pbest_pos at the given slot indices.
__global__ void scatter_migrants_kernel(float* __restrict__ pbest_pos,
                                        const int* __restrict__ slot,
                                        const float* __restrict__ in,
                                        int N, int D, int m) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= m * D) return;
    int dim = t / m;
    int mi  = t % m;
    pbest_pos[dim * N + slot[mi]] = in[t];
}

// Scatter m fitnesses into pbest_fit at the given slot indices.
__global__ void scatter_fit_kernel(float* __restrict__ pbest_fit,
                                   const int* __restrict__ slot,
                                   const float* __restrict__ in, int m) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= m) return;
    pbest_fit[slot[t]] = in[t];
}

// Gather the gbest particle's (at *gbest_idx) D coords into contiguous out[D].
__global__ void gather_gbest_kernel(const float* __restrict__ pbest_pos,
                                    const int* __restrict__ gbest_idx,
                                    float* __restrict__ out, int N, int D) {
    int dim = blockIdx.x * blockDim.x + threadIdx.x;
    if (dim >= D) return;
    out[dim] = pbest_pos[dim * N + *gbest_idx];
}

// Scatter a contiguous gbest position[D] into pbest_pos slot 0.
__global__ void scatter_gbest_kernel(float* __restrict__ pbest_pos,
                                     const float* __restrict__ in, int N, int D) {
    int dim = blockIdx.x * blockDim.x + threadIdx.x;
    if (dim >= D) return;
    pbest_pos[dim * N] = in[dim];
}

void island_sync_data_alloc(IslandSyncData* data, MPI_Comm comm,
                             int n_migrate, int D) {
    /**
     * @brief Populates an IslandSyncData with rank info and allocates host buffers.
     *
     * @param data      Output struct to populate.
     * @param comm      MPI communicator.
     * @param n_migrate Particles to exchange per sync.
     * @param D         Problem dimensionality.
     *
     * @returns void. Exits on malloc failure.
     *
     * @Structure
     *   - MPI_Comm_rank/size
     *   - malloc send/recv pos [n_migrate*D] and fit [n_migrate] buffers
     *   - malloc h_gbest_pos [D]
     */
    data->comm      = comm;
    data->n_migrate = n_migrate;
    data->D         = D;
    MPI_Comm_rank(comm, &data->rank);
    MPI_Comm_size(comm, &data->n_ranks);

    data->h_send_pos  = (float*)std::malloc(sizeof(float) * n_migrate * D);
    data->h_send_fit  = (float*)std::malloc(sizeof(float) * n_migrate);
    data->h_recv_pos  = (float*)std::malloc(sizeof(float) * n_migrate * D);
    data->h_recv_fit  = (float*)std::malloc(sizeof(float) * n_migrate);
    data->h_gbest_pos = (float*)std::malloc(sizeof(float) * D);

    if (!data->h_send_pos || !data->h_send_fit ||
        !data->h_recv_pos || !data->h_recv_fit || !data->h_gbest_pos) {
        std::fprintf(stderr, "island_sync_data_alloc: malloc failed\n");
        std::exit(EXIT_FAILURE);
    }

    //device scratch for the gather/scatter kernels — sized by n_migrate and D
    CUDA_CHECK(cudaMalloc(&data->d_send_pos, sizeof(float) * n_migrate * D));
    CUDA_CHECK(cudaMalloc(&data->d_recv_pos, sizeof(float) * n_migrate * D));
    CUDA_CHECK(cudaMalloc(&data->d_recv_fit, sizeof(float) * n_migrate));
    CUDA_CHECK(cudaMalloc(&data->d_idx,      sizeof(int)   * n_migrate));
    CUDA_CHECK(cudaMalloc(&data->d_gbest_pos, sizeof(float) * D));
}

void island_sync_data_free(IslandSyncData* data) {
    /**
     * @brief Frees all host buffers in an IslandSyncData.
     *
     * @param data Struct to clean up.
     *
     * @returns void.
     *
     * @Structure
     *   - free each buffer, null the pointer
     */
    std::free(data->h_send_pos);  data->h_send_pos  = nullptr;
    std::free(data->h_send_fit);  data->h_send_fit  = nullptr;
    std::free(data->h_recv_pos);  data->h_recv_pos  = nullptr;
    std::free(data->h_recv_fit);  data->h_recv_fit  = nullptr;
    std::free(data->h_gbest_pos); data->h_gbest_pos = nullptr;

    cudaFree(data->d_send_pos);  data->d_send_pos  = nullptr;
    cudaFree(data->d_recv_pos);  data->d_recv_pos  = nullptr;
    cudaFree(data->d_recv_fit);  data->d_recv_fit  = nullptr;
    cudaFree(data->d_idx);       data->d_idx       = nullptr;
    cudaFree(data->d_gbest_pos); data->d_gbest_pos = nullptr;
}

static void island_gbest_exchange(IslandState* state, void* user_data,
                                  bool force) {
    /**
     * @brief Finds the globally best fitness across all islands and conditionally
     *        adopts that island's gbest position on each rank.
     *
     * @param state     Device state snapshot for this island.
     * @param user_data IslandSyncData* cast from void*.
     * @param force     If true, always overwrite local gbest (fc behaviour).
     *                  If false, only overwrite when the global best strictly
     *                  improves on this island's current gbest (ring behaviour).
     *
     * @returns void.
     *
     * @Structure
     *   - cudaMemcpy d_gbest_val -> local_val
     *   - MPI_Allreduce MPI_MINLOC -> best_pair {val, winning_rank}
     *   - winning rank gathers gbest pos D->H
     *   - MPI_Bcast position vector from winning_rank
     *   - always: write best_pair.val -> d_gbest_val (correct reporting on all ranks)
     *   - gated on (force || best_pair.val < local_val):
     *       scatter broadcast position into pbest_pos slot 0 and reset d_gbest_idx=0
     */
    IslandSyncData* d = (IslandSyncData*)user_data;

    //pull local gbest scalar to host
    float local_val = 0.0f;
    CUDA_CHECK(cudaMemcpy(&local_val, state->d_gbest_val,
        sizeof(float), cudaMemcpyDeviceToHost));

    //pack as {float val, int rank} for MPI_MINLOC
    struct { float val; int rank; } local_pair = { local_val, d->rank };
    struct { float val; int rank; } best_pair  = { 0.0f, 0 };
    MPI_Allreduce(&local_pair, &best_pair, 1, MPI_FLOAT_INT,
                  MPI_MINLOC, d->comm);

    //winning rank gathers its gbest position on-device
    if (d->rank == best_pair.rank) {
        gather_gbest_kernel<<<grid_for(state->D, 256), 256>>>(
            state->d_pbest_pos, state->d_gbest_idx, d->d_gbest_pos,
            state->N, state->D);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(d->h_gbest_pos, d->d_gbest_pos,
            sizeof(float) * state->D, cudaMemcpyDeviceToHost));
    }

    //broadcast winning position to all ranks (needed even if not adopted,
    //so the bcast is always unconditional and all ranks stay in sync)
    MPI_Bcast(d->h_gbest_pos, state->D, MPI_FLOAT, best_pair.rank, d->comm);

    //always update d_gbest_val so final_gbest reporting is accurate on all ranks
    CUDA_CHECK(cudaMemcpy(state->d_gbest_val, &best_pair.val,
        sizeof(float), cudaMemcpyHostToDevice));

    //only inject the foreign position attractor when force=true (fc) or when
    //the global best is strictly better than this island's current best (ring)
    if (force || best_pair.val < local_val) {
        CUDA_CHECK(cudaMemcpy(d->d_gbest_pos, d->h_gbest_pos,
            sizeof(float) * state->D, cudaMemcpyHostToDevice));
        scatter_gbest_kernel<<<grid_for(state->D, 256), 256>>>(
            state->d_pbest_pos, d->d_gbest_pos, state->N, state->D);
        CUDA_CHECK(cudaGetLastError());
        int zero = 0;
        CUDA_CHECK(cudaMemcpy(state->d_gbest_idx, &zero,
            sizeof(int), cudaMemcpyHostToDevice));
    }
}

//helper: find indices of the n_migrate particles with lowest pbest_fit
static std::vector<int> top_indices(const float* h_fit, int N, int n_migrate) {
    /*
    Returns indices of the n_migrate smallest values in h_fit[N].
    Simple partial sort — N is at most a few thousand so this is fine.
    */
    std::vector<int> idx(N);
    for (int i = 0; i < N; ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + n_migrate, idx.end(),
        [&](int a, int b){ return h_fit[a] < h_fit[b]; });
    idx.resize(n_migrate);
    return idx;
}

void island_migrate_ring(IslandState* state, void* user_data) {
    /**
     * @brief Ring migration: send top n_migrate particles to right neighbor,
     *        receive n_migrate particles from left neighbor, then exchange gbest.
     *
     * @param state     Device state snapshot.
     * @param user_data IslandSyncData*.
     *
     * @returns void.
     *
     * @Structure
     *   - copy full pbest_fit D->H to find top indices
     *   - pack h_send_pos (SoA rows for top particles) and h_send_fit
     *   - MPI_Sendrecv to/from ring neighbors
     *   - inject received particles into last n_migrate slots of d_pbest_pos/fit
     *   - island_gbest_exchange()
     */
    IslandSyncData* d = (IslandSyncData*)user_data;
    int N = state->N;
    int D = state->D;
    int m = d->n_migrate;

    //pull pbest_fit to host to rank particles
    std::vector<float> h_fit(N);
    CUDA_CHECK(cudaMemcpy(h_fit.data(), state->d_pbest_fit,
        sizeof(float) * N, cudaMemcpyDeviceToHost));

    std::vector<int> top = top_indices(h_fit.data(), N, m);

    //pack send buffers on-device: copy top indices H->D, gather into d_send_pos
    //[dim*m + mi], then one m*D D->H copy into the host MPI buffer.
    CUDA_CHECK(cudaMemcpy(d->d_idx, top.data(),
        sizeof(int) * m, cudaMemcpyHostToDevice));
    gather_migrants_kernel<<<grid_for(m * D, 256), 256>>>(
        state->d_pbest_pos, d->d_idx, d->d_send_pos, N, D, m);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(d->h_send_pos, d->d_send_pos,
        sizeof(float) * m * D, cudaMemcpyDeviceToHost));
    for (int mi = 0; mi < m; ++mi)
        d->h_send_fit[mi] = h_fit[top[mi]];

    int right = (d->rank + 1)           % d->n_ranks;
    int left  = (d->rank - 1 + d->n_ranks) % d->n_ranks;

    MPI_Sendrecv(
        d->h_send_pos, m * D, MPI_FLOAT, right, 0,
        d->h_recv_pos, m * D, MPI_FLOAT, left,  0,
        d->comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(
        d->h_send_fit, m, MPI_FLOAT, right, 1,
        d->h_recv_fit, m, MPI_FLOAT, left,  1,
        d->comm, MPI_STATUS_IGNORE);

    //inject received particles into the m weakest slots (partial_sort descending)
    std::vector<int> idx(N);
    for (int i = 0; i < N; ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + m, idx.end(),
        [&](int a, int b){ return h_fit[a] > h_fit[b]; }); //descending
    std::vector<int> worst_idx(idx.begin(), idx.begin() + m);

    //push recv buffers + worst slots to device, scatter positions and fitness.
    CUDA_CHECK(cudaMemcpy(d->d_idx, worst_idx.data(),
        sizeof(int) * m, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d->d_recv_pos, d->h_recv_pos,
        sizeof(float) * m * D, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d->d_recv_fit, d->h_recv_fit,
        sizeof(float) * m, cudaMemcpyHostToDevice));
    scatter_migrants_kernel<<<grid_for(m * D, 256), 256>>>(
        state->d_pbest_pos, d->d_idx, d->d_recv_pos, N, D, m);
    CUDA_CHECK(cudaGetLastError());
    scatter_fit_kernel<<<grid_for(m, 256), 256>>>(
        state->d_pbest_fit, d->d_idx, d->d_recv_fit, m);
    CUDA_CHECK(cudaGetLastError());

    island_gbest_exchange(state, user_data,false);
}

void island_migrate_fc(IslandState* state, void* user_data) {
    /**
     * @brief Fully-connected migration: share top n_migrate particles with ALL
     *        ranks via MPI_Allgather, inject globally best n_migrate not owned
     *        by this rank into worst slots, then exchange gbest.
     *
     * @param state     Device state snapshot.
     * @param user_data IslandSyncData*.
     *
     * @returns void.
     *
     * @Structure
     *   - pull pbest_fit D->H, find top m indices
     *   - pack h_send_pos / h_send_fit
     *   - MPI_Allgather into gathered pos [n_ranks * m * D] and fit [n_ranks * m]
     *   - pick best m particles globally that don't belong to this rank
     *   - find worst m indices on this island
     *   - inject selected particles into worst slots of d_pbest_pos / d_pbest_fit
     *   - island_gbest_exchange()
     */
    IslandSyncData* d = (IslandSyncData*)user_data;
    int N       = state->N;
    int D       = state->D;
    int m       = d->n_migrate;
    int nranks  = d->n_ranks;

    //pull pbest_fit to host
    std::vector<float> h_fit(N);
    CUDA_CHECK(cudaMemcpy(h_fit.data(), state->d_pbest_fit,
        sizeof(float) * N, cudaMemcpyDeviceToHost));

    std::vector<int> top = top_indices(h_fit.data(), N, m);

    //pack send buffers on-device (see ring): gather top migrants into d_send_pos,
    //then a single m*D D->H copy into the host MPI buffer.
    CUDA_CHECK(cudaMemcpy(d->d_idx, top.data(),
        sizeof(int) * m, cudaMemcpyHostToDevice));
    gather_migrants_kernel<<<grid_for(m * D, 256), 256>>>(
        state->d_pbest_pos, d->d_idx, d->d_send_pos, N, D, m);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(d->h_send_pos, d->d_send_pos,
        sizeof(float) * m * D, cudaMemcpyDeviceToHost));
    for (int mi = 0; mi < m; ++mi)
        d->h_send_fit[mi] = h_fit[top[mi]];

    //gather all islands' top particles — gathered_pos[rank * m * D + dim * m + mi]
    std::vector<float> gathered_pos(nranks * m * D);
    std::vector<float> gathered_fit(nranks * m);
    MPI_Allgather(d->h_send_pos, m * D, MPI_FLOAT,
                  gathered_pos.data(), m * D, MPI_FLOAT, d->comm);
    MPI_Allgather(d->h_send_fit, m, MPI_FLOAT,
                  gathered_fit.data(), m, MPI_FLOAT, d->comm);

    //collect all foreign particles sorted by fitness ascending
    //skip own rank's block
    struct Candidate { float fit; int rank; int mi; };
    std::vector<Candidate> cands;
    cands.reserve((nranks - 1) * m);
    for (int r = 0; r < nranks; ++r) {
        if (r == d->rank) continue;
        for (int mi = 0; mi < m; ++mi)
            cands.push_back({ gathered_fit[r * m + mi], r, mi });
    }
    std::sort(cands.begin(), cands.end(),
        [](const Candidate& a, const Candidate& b){ return a.fit < b.fit; });

    //take the best m foreign candidates
    int n_inject = std::min(m, (int)cands.size());

    //find worst m slots on this island
    std::vector<int> idx(N);
    for (int i = 0; i < N; ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + n_inject, idx.end(),
        [&](int a, int b){ return h_fit[a] > h_fit[b]; }); //descending = worst first
    std::vector<int> worst_idx(idx.begin(), idx.begin() + n_inject);

    //compact the chosen foreign particles into a contiguous packed buffer
    //[dim*n_inject + i] (host side — gathered_pos is already host memory), plus
    //their fitnesses and target slots, then push to device and scatter.
    for (int dim = 0; dim < D; ++dim)
        for (int i = 0; i < n_inject; ++i) {
            int r  = cands[i].rank;
            int mi = cands[i].mi;
            d->h_recv_pos[dim * n_inject + i] =
                gathered_pos[r * m * D + dim * m + mi];
        }
    for (int i = 0; i < n_inject; ++i)
        d->h_recv_fit[i] = cands[i].fit;

    CUDA_CHECK(cudaMemcpy(d->d_idx, worst_idx.data(),
        sizeof(int) * n_inject, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d->d_recv_pos, d->h_recv_pos,
        sizeof(float) * n_inject * D, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d->d_recv_fit, d->h_recv_fit,
        sizeof(float) * n_inject, cudaMemcpyHostToDevice));
    scatter_migrants_kernel<<<grid_for(n_inject * D, 256), 256>>>(
        state->d_pbest_pos, d->d_idx, d->d_recv_pos, N, D, n_inject);
    CUDA_CHECK(cudaGetLastError());
    scatter_fit_kernel<<<grid_for(n_inject, 256), 256>>>(
        state->d_pbest_fit, d->d_idx, d->d_recv_fit, n_inject);
    CUDA_CHECK(cudaGetLastError());

    island_gbest_exchange(state, user_data,true);
}