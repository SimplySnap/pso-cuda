# bench/ — measurement & analysis layout

Generated artifacts and scripts for the M4 + Phase E + Phase G work on branch `oliver-m4`.

**Read [`M4_REPORT.md`](M4_REPORT.md) first** — it's the comprehensive writeup, organized by the M4 rubric. This README is just a per-file index for navigation.

## Reproducing everything from scratch

```bash
module load course/cme213/nvhpc/24.1
make clean && make all && make mpi

# in order, submit one at a time:
sbatch bench/smoke.sh           # 1-node smoke test, ~2 min
sbatch bench/correctness.sh     # correctness sweep across rank counts, ~3 min
sbatch bench/scaling.sh         # M4 strong + weak scaling at N_total=4096, ~5 min
sbatch bench/sweeps.sh          # Phase E: N-sweep, sync-Pareto, large-N strong/weak, ~7 min
sbatch bench/sweep_NxRxD.sh     # Phase G: D × ranks × N matrix at sync=25, m=N/100, ~14 min
sbatch bench/nsys.sh            # Nsight Systems trace, ~3 min

python3 bench/mpi_analyze.py    # regenerate all MPI figures + table_mpi.md
python3 bench/analyze.py        # M3 single-GPU figures (carryover, optional)
```

## Scripts

| Script | What it produces | Resources |
|---|---|---|
| `smoke.sh` | One MPI run per topology to verify sync hook fires | 1 node, 2 ranks |
| `correctness.sh` | `correctness.csv` — single + ring/fc × {1,2,4} for rastrigin/levy | 4 nodes, 4 ranks |
| `scaling.sh` | `scaling_{strong,weak,baseline}.csv` — M4 N_total=4096 study | 4 nodes, 4 ranks |
| `sweeps.sh` | Phase E: `sweep_N.csv`, `sweep_sync.csv`, `sweep_{strong,weak}_largeN.csv` | 4 nodes, 4 ranks |
| `sweep_NxRxD.sh` | Phase G: `sweep_NxRxD.csv` + baseline + levy. Uses sync=25, m=N/100 | 4 nodes, 16 ranks (1/GPU) |
| `nsys.sh` | `trace_ring_rank_*.nsys-rep` + `nsys_summary.txt` | 2 nodes, 2 ranks |
| `analyze.py` | M3 single-GPU figures (legacy) | login node, ~5 sec |
| `mpi_analyze.py` | All MPI figures + `table_mpi.md` | login node, ~15 sec |

## Data files

### M4 base (`m=5` fixed, `sync=10`)

| CSV | What |
|---|---|
| `correctness.csv` | gbest + total_ms per (impl, topology, ranks, evaluator) |
| `scaling_strong.csv` | N_total=4096 split across ranks (ring + fc, np=1,2,4) |
| `scaling_weak.csv` | per-rank N=1024 fixed (ring + fc, np=1,2,4) |
| `scaling_baseline.csv` | pso_cuda single-GPU at the relevant N values |
| `results.csv` | M3 single-GPU sweep (carryover) |

### Phase E (still `m=5`, varying sync and N)

| CSV | What |
|---|---|
| `sweep_N.csv` | N ∈ {1024, 4096, 16384, 65536} × ring/fc at np=4, sync=10 |
| `sweep_N_baseline.csv` | pso_cuda at same N values |
| `sweep_sync.csv` | sync_interval Pareto, 5 seeds × 6 sync × 2 topo = 60 rows |
| `sweep_strong_largeN.csv` | N_total=16384 strong scaling |
| `sweep_weak_largeN.csv` | per-rank N=16384 weak scaling |
| `sweep_largeN_baseline.csv` | pso_cuda at N=16384 |

### Phase G (`m = max(5, N/100)`, sync=25, D varies)

| CSV | What |
|---|---|
| `sweep_NxRxD.csv` | 36-cell matrix: D × ranks × topology × N for rastrigin |
| `sweep_NxRxD_baseline.csv` | pso_cuda single-GPU at the matching (D, N) cells |
| `sweep_NxRxD_levy.csv` | 3 levy sanity rows, one per D ∈ {30, 100, 300} |

**Schema note:** All MPI CSVs written via `--csv_path` are *headerless*. The column order is:
```
topology,evaluator,n_islands,N,D,iters,seed,eval_ms,reduce_ms,update_ms,sync_ms,total_ms,final_gbest
```
This is documented as `MPI_CSV_COLS` at the top of `mpi_analyze.py`.

`sweep_sync.csv` has a custom schema (extra `sync_interval,seed` columns) because the bash sweep parses stdout — see the script header.

## Figures

| File | Generated from | Shows |
|---|---|---|
| `fig_mpi_scaling.png` | scaling_{strong,weak,baseline}.csv | M4 strong+weak speedup/efficiency vs rank count |
| `fig_mpi_breakdown.png` | scaling_{strong,weak}.csv | stacked-bar: eval/reduce/update/sync ms per (rank, topology) |
| `fig_sweep_N.png` | sweep_N.csv | Phase E N-sweep: sync_ms, compute_ms, sync/compute ratio vs N |
| `fig_sweep_sync.png` | sweep_sync.csv | Phase E sync Pareto: sync_ms vs gbest with ±std bands |
| `fig_sweep_largeN_scaling.png` | sweep_{strong,weak}_largeN.csv | rank scaling at N_total=16384 |
| `fig_sweep_NxRxD.png` | sweep_NxRxD.csv | Phase G: 3 panels (D=30/100/300), total_ms vs N with 60s ceiling |

## Profiling traces

| File | What |
|---|---|
| `trace_ring_rank_0.nsys-rep` | Per-rank Nsight Systems profile (rank 0 of np=2 ring run) |
| `trace_ring_rank_1.nsys-rep` | Same, rank 1 |
| `nsys_summary.txt` | `nsys stats` text output — cuda_api_sum + cuda_gpu_kern_sum |

## Slurm logs

`*.err` files retained for traceability (mostly empty — runs succeeded). `*.out` files are gitignored (per `.gitignore`).

## Key findings (one-line each — full discussion in `M4_REPORT.md`)

1. **The sync hook in `pso_run()` was the M4 blocker** — restored in commit `68eebee`; without it MPI binaries ran as independent islands with no migration.
2. **`--sync 25` is empirically Pareto-optimal**, not the `--sync 10` we used in early sweeps.
3. **Phase E's "MPI wins at large N" prediction was an artifact** of fixed `m=5`. Under proportional `m = N/100`, sync/compute ratio stays near 1.0 and MPI doesn't beat single-GPU.
4. **fc topology breaks at np=16** — Allgather's O(p²) cost makes sync 10× compute. Ring stays viable.
5. **Algorithmic differentiation requires high D.** At D=30, multi-island doesn't help. At D=100+, migration gives ~20-34% better gbest.
6. **Host-staging is the bottleneck** — confirmed by Nsight: 28.4 ms cudaMemcpy vs 3.8 ms GPU kernel time per 200 iters.

## Where to build next

See §8 of [`M4_REPORT.md`](M4_REPORT.md) for the prioritized task list toward the Jun 8 final. Top three:
1. Pack-once gather kernel (replace D-per-dim cudaMemcpys with one contiguous transfer)
2. CUDA-aware MPI (pass device pointers directly to MPI primitives)
3. Async-stream overlap (run sync on stream 1 while next-iter compute runs on stream 0)
