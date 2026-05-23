#include "mpi_island.h"
#include "../pso/cuda_check.cuh"
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>

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
}

void island_gbest_exchange(IslandState* state, void* user_data) {
    /**
     * @brief Finds the globally best fitness across all islands and broadcasts
     *        that island's gbest position to every rank.
     *
     * @param state     Device state snapshot for this island.
     * @param user_data IslandSyncData* cast from void*.
     *
     * @returns void.
     *
     * @Structure
     *   - copy d_gbest_val D->H
     *   - MPI_Allreduce with MPI_FLOAT_INT / MPI_MINLOC -> {val, winning_rank}
     *   - winning rank gathers gbest_pos from pbest_pos[D*N + gbest_idx] on device
     *   - MPI_Bcast position vector from winning_rank
     *   - all ranks write new gbest into d_gbest_val, d_gbest_idx=-1 (position
     *     is now injected directly — idx is stale across ranks so we invalidate it)
     * 
     * Known design choice: overwrites particle 0's pbest. Fine for large swarms
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

    //winning rank copies its gbest position from device to h_gbest_pos
    if (d->rank == best_pair.rank) {
        int gbest_idx = -1;
        CUDA_CHECK(cudaMemcpy(&gbest_idx, state->d_gbest_idx,
            sizeof(int), cudaMemcpyDeviceToHost));

        //gather D position values from pbest_pos[d * N + gbest_idx]
        //use a small temp device buffer to do a single coalesced copy
        std::vector<float> h_pbest_col(state->N);
        for (int dim = 0; dim < state->D; ++dim) {
            CUDA_CHECK(cudaMemcpy(h_pbest_col.data(),
                state->d_pbest_pos + dim * state->N,
                sizeof(float) * state->N, cudaMemcpyDeviceToHost));
            d->h_gbest_pos[dim] = h_pbest_col[gbest_idx];
        }
    }

    //broadcast winning position to all ranks
    MPI_Bcast(d->h_gbest_pos, state->D, MPI_FLOAT, best_pair.rank, d->comm);

    //write new gbest val to device on all ranks
    CUDA_CHECK(cudaMemcpy(state->d_gbest_val, &best_pair.val,
        sizeof(float), cudaMemcpyHostToDevice));

    //gbest_idx is now meaningless across ranks — kernel_update reads
    //pbest_pos[dim*N + *d_gbest_idx], so we inject the broadcast position
    //into slot 0 of pbest_pos and point d_gbest_idx there
    for (int dim = 0; dim < state->D; ++dim) {
        CUDA_CHECK(cudaMemcpy(
            state->d_pbest_pos + dim * state->N,
            &d->h_gbest_pos[dim],
            sizeof(float), cudaMemcpyHostToDevice));
    }
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(state->d_gbest_idx, &zero,
        sizeof(int), cudaMemcpyHostToDevice));
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

    //pack send buffers — SoA: h_send_pos[dim * m + mi]
    std::vector<float> h_pbest_row(N);
    for (int dim = 0; dim < D; ++dim) {
        CUDA_CHECK(cudaMemcpy(h_pbest_row.data(),
            state->d_pbest_pos + dim * N,
            sizeof(float) * N, cudaMemcpyDeviceToHost));
        for (int mi = 0; mi < m; ++mi)
            d->h_send_pos[dim * m + mi] = h_pbest_row[top[mi]];
    }
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

    //inject received particles into last m slots (overwrite weakest)
    std::vector<int> worst = top_indices(h_fit.data(), N, m); //reuse — these are best; invert
    //actually replace the worst: partial_sort descending
    std::vector<int> idx(N);
    for (int i = 0; i < N; ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + m, idx.end(),
        [&](int a, int b){ return h_fit[a] > h_fit[b]; }); //descending
    std::vector<int> worst_idx(idx.begin(), idx.begin() + m);

    for (int dim = 0; dim < D; ++dim) {
        CUDA_CHECK(cudaMemcpy(h_pbest_row.data(),
            state->d_pbest_pos + dim * N,
            sizeof(float) * N, cudaMemcpyDeviceToHost));
        for (int mi = 0; mi < m; ++mi)
            h_pbest_row[worst_idx[mi]] = d->h_recv_pos[dim * m + mi];
        CUDA_CHECK(cudaMemcpy(
            state->d_pbest_pos + dim * N,
            h_pbest_row.data(),
            