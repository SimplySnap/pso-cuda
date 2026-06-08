//mpi/mpi_island.h
#pragma once
#include <mpi.h>
#include "../pso/pso.h"

// IslandState lives in pso/pso.h — re-used here without redefinition.

/**
 * @brief Per-island MPI context, passed as on_sync_data through PSOConfig.
 *
 * @param comm          MPI communicator (usually MPI_COMM_WORLD).
 * @param rank          This process's MPI rank.
 * @param n_ranks       Total number of MPI ranks / islands.
 * @param n_migrate     Number of top particles to migrate per sync.
 * @param h_send_pos    Host buffer for outgoing migrant positions [n_migrate * D].
 * @param h_send_fit    Host buffer for outgoing migrant fitnesses [n_migrate].
 * @param h_recv_pos    Host buffer for incoming migrant positions [n_migrate * D].
 * @param h_recv_fit    Host buffer for incoming migrant fitnesses [n_migrate].
 * @param h_gbest_pos   Host buffer for gbest position broadcast [D].
 * @param D             Problem dimensionality (sizes the device scratch buffers).
 * @param d_send_pos    Device scratch for packed outgoing migrants [n_migrate * D].
 * @param d_recv_pos    Device scratch for packed incoming migrants [n_migrate * D].
 * @param d_recv_fit    Device scratch for incoming migrant fitnesses [n_migrate].
 * @param d_idx         Device scratch for migrant slot indices [n_migrate].
 * @param d_gbest_pos   Device scratch for the gathered gbest position [D].
 *
 * @Structure
 *   Allocated once in the MPI main before pso_run() via island_sync_data_alloc().
 *   Freed after pso_run() returns via island_sync_data_free().
 *   Host and device buffers are reused across every sync to avoid per-sync malloc.
 *   The device scratch lets the per-sync gather/scatter run as kernels and move
 *   only the n_migrate*D migrants (and D gbest values) across PCIe, instead of
 *   the old per-dimension host-staged copies that dragged whole N-float columns.
 */
typedef struct {
    MPI_Comm comm;
    int      rank;
    int      n_ranks;
    int      n_migrate;
    float*   h_send_pos;  //[n_migrate * D]
    float*   h_send_fit;  //[n_migrate]
    float*   h_recv_pos;  //[n_migrate * D]
    float*   h_recv_fit;  //[n_migrate]
    float*   h_gbest_pos; //[D]
    int      D;           //problem dimensionality
    //device scratch — packed [dim*n_migrate + mi] to mirror the host layout
    float*   d_send_pos;  //[n_migrate * D]
    float*   d_recv_pos;  //[n_migrate * D]
    float*   d_recv_fit;  //[n_migrate]
    int*     d_idx;       //[n_migrate] — top (gather) or worst (scatter) slots
    float*   d_gbest_pos; //[D]
} IslandSyncData;

/**
 * @brief Allocates host buffers inside an IslandSyncData struct.
 *
 * @param data      Pointer to an IslandSyncData to populate.
 * @param comm      MPI communicator to bind to this island.
 * @param n_migrate Number of top particles to exchange per sync.
 * @param D         Number of problem dimensions.
 *
 * @returns void. Calls exit(EXIT_FAILURE) on allocation failure.
 *
 * @Structure
 *   - MPI_Comm_rank / MPI_Comm_size to fill rank and n_ranks
 *   - malloc four host buffers sized by n_migrate and D
 */
void island_sync_data_alloc(IslandSyncData* data, MPI_Comm comm,
                             int n_migrate, int D);

/**
 * @brief Frees host buffers allocated by island_sync_data_alloc.
 *
 * @param data Pointer to the IslandSyncData to clean up.
 *
 * @returns void.
 */
void island_sync_data_free(IslandSyncData* data);

/**
 * @brief SyncCallback: exchanges gbest across all islands (fully connected).
 *        Uses MPI_MINLOC to find the globally best rank, then MPI_Bcast
 *        to distribute that rank's gbest position into every island's
 *        d_gbest_val, d_gbest_idx, and d_pbest_pos.
 *
 * @param state      Current island's device state snapshot.
 * @param user_data  Pointer to IslandSyncData cast from void*.
 *
 * @returns void.
 *
 * @Structure
 *   - cudaMemcpy d_gbest_val -> host scalar
 *   - MPI_Allreduce MPI_FLOAT_INT / MPI_MINLOC -> winning {val, rank}
 *   - winning rank cudaMemcpy gbest_pos D->H into h_gbest_pos
 *   - MPI_Bcast h_gbest_pos from winning rank
 *   - all ranks cudaMemcpy h_gbest_pos H->D into pbest_pos[D * N + gbest_idx]
 *     and update d_gbest_val / d_gbest_idx
 */
void island_gbest_exchange(IslandState* state, void* user_data);

/**
 * @brief SyncCallback: ring-topology particle migration + gbest exchange.
 *        Each rank sends its top n_migrate particles (by pbest_fit) to its
 *        right neighbor and receives n_migrate particles from its left neighbor.
 *        Follows migration with a call to island_gbest_exchange.
 *
 * @param state      Current island's device state snapshot.
 * @param user_data  Pointer to IslandSyncData cast from void*.
 *
 * @returns void.
 *
 * @Structure
 *   - find top n_migrate indices by sorting/scanning pbest_fit on host
 *   - cudaMemcpy those particle rows from d_pbest_pos H->H into h_send_pos
 *   - MPI_Sendrecv h_send_pos -> right neighbor, h_recv_pos <- left neighbor
 *   - cudaMemcpy h_recv_pos into the last n_migrate slots of d_pbest_pos
 *   - overwrite their d_pbest_fit entries with h_recv_fit values
 *   - call island_gbest_exchange(state, user_data)
 */
void island_migrate_ring(IslandState* state, void* user_data);

/**
 * @brief SyncCallback: fully-connected migration + gbest exchange.
 *        Each rank shares its top n_migrate particles with ALL other ranks
 *        via MPI_Allgather, then each island injects the globally best
 *        n_migrate particles it doesn't already own.
 *        Follows migration with a call to island_gbest_exchange.
 *
 * @param state      Current island's device state snapshot.
 * @param user_data  Pointer to IslandSyncData cast from void*.
 *
 * @returns void.
 *
 * @Structure
 *   - cudaMemcpy top n_migrate particles D->H into h_send_pos / h_send_fit
 *   - MPI_Allgather h_send_pos/fit from all ranks into a gathered buffer
 *   - select globally best n_migrate particles not owned by this rank
 *   - cudaMemcpy selected particles H->D into last n_migrate slots of d_pbest_pos
 *   - call island_gbest_exchange(state, user_data)
 */
void island_migrate_fc(IslandState* state, void* user_data);