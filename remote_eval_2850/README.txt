WSP2850 evaluation

From the repository root, using the existing simulator Python:
python scripts/run_wsp2850_eval.py

Defaults: 16 environments, cuda:0, seeds 42/43/44, original and WSP.
Total: 96 first episodes. CUDA availability must be verified on the host.
CPU fallback is explicit: --device cpu
A short pilot can use --seeds 42.

Results are saved under evaluation_results/wsp2850/ab2850_*.
Each run contains run.json, per-seed logs/results, and summary.json on success.
This directory is not ignored by Git. Commit and push it using GitHub Desktop
or git add evaluation_results/wsp2850 followed by commit and push.
No automatic Git push occurs. Running logs are snapshots, not completed results.
The outputs/ directory contains only the extracted execution bundle/cache.

Original checkpoints and existing result folders are preserved.
Student-A benchmark does not prove the original checkpoint training variant.
