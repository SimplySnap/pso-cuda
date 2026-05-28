# pso-cuda MPI scaling — M4 summary

## Strong scaling (total N = 4096, D = 30, iters = 500)

Single-GPU baseline (pso_cuda, N=4096): **total_ms = 14.91**

| topology   |   n_islands |    N |   eval_ms |   reduce_ms |   update_ms |   sync_ms |   total_ms |   speedup_vs_mpi1 |   efficiency_vs_mpi1 |   speedup_vs_single |   final_gbest |
|:-----------|------------:|-----:|----------:|------------:|------------:|----------:|-----------:|------------------:|---------------------:|--------------------:|--------------:|
| ring       |           1 | 4096 |     7.308 |       3.571 |       4.13  |    73.751 |     88.761 |             1     |                1     |               0.168 |        16.914 |
| fc         |           1 | 4096 |     7.3   |       3.522 |       4.129 |    69.468 |     84.418 |             1     |                1     |               0.177 |        16.914 |
| ring       |           2 | 2048 |     7.47  |       2.696 |       3.825 |   106.328 |    120.319 |             0.738 |                0.369 |               0.124 |        16.914 |
| fc         |           2 | 2048 |     7.812 |       2.692 |       3.849 |    90.263 |    104.616 |             0.807 |                0.403 |               0.143 |        16.914 |
| ring       |           4 | 1024 |     7.349 |       2.827 |       3.686 |   113.122 |    126.985 |             0.699 |                0.175 |               0.117 |        13.929 |
| fc         |           4 | 1024 |     7.532 |       2.79  |       3.678 |    97.495 |    111.494 |             0.757 |                0.189 |               0.134 |        13.929 |

## Weak scaling (N = 1024 per rank, D = 30, iters = 500)

Single-GPU baseline (pso_cuda, N=1024): **total_ms = 13.27**

| topology   |   n_islands |    N |   eval_ms |   reduce_ms |   update_ms |   sync_ms |   total_ms |   speedup_vs_mpi1 |   efficiency_vs_mpi1 |   speedup_vs_single |   final_gbest |
|:-----------|------------:|-----:|----------:|------------:|------------:|----------:|-----------:|------------------:|---------------------:|--------------------:|--------------:|
| ring       |           1 | 1024 |     6.863 |       2.815 |       3.637 |    60.37  |     73.685 |             1     |                1     |               0.18  |        42.783 |
| fc         |           1 | 1024 |     7.234 |       2.808 |       3.627 |    59.072 |     72.742 |             1     |                1     |               0.182 |        46.763 |
| ring       |           2 | 1024 |     7.547 |       2.815 |       3.674 |    82.073 |     96.109 |             0.767 |                0.383 |               0.138 |        21.889 |
| fc         |           2 | 1024 |     7.294 |       2.799 |       3.701 |    81.458 |     95.253 |             0.764 |                0.382 |               0.139 |        21.889 |
| ring       |           4 | 1024 |     7.456 |       2.823 |       3.693 |    88.859 |    102.832 |             0.717 |                0.179 |               0.129 |        13.929 |
| fc         |           4 | 1024 |     7.354 |       2.941 |       3.687 |   100.08  |    114.062 |             0.638 |                0.159 |               0.116 |        13.929 |

## Notes

- `speedup_vs_mpi1` uses `pso_{topo} -np 1` as the baseline (MPI-native efficiency).
- `speedup_vs_single` uses the single-GPU `pso_cuda` binary as the baseline (honest absolute speedup).
- `sync_ms` is host wall time inside the on_sync callback, measured via `std::chrono::steady_clock`.
- `total_ms = eval_ms + reduce_ms + update_ms + sync_ms`.
