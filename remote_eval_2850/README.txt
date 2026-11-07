MODEL 2850 ORIGINAL / WSP EVALUATION

Copy this entire remote_eval_2850 folder into:
C:\Users\COM\Documents\RLForHexapod

Run exactly once in remote PowerShell:
& 'C:\Users\COM\Documents\RLForHexapod\.venv\Scripts\python.exe' 'C:\Users\COM\Documents\RLForHexapod\remote_eval_2850\run_eval.py'

Default: CPU, newton_gl, 128 environments, seeds 42/43/44.
Six simulator processes run sequentially, original then WSP per seed.
First episode per environment only, phase 0 starts at x uniform [-0.1, 0.1].
Original height-map observation noise remains enabled. No training or exports.
Model and pinned code hashes are verified; installed robot/stair assets must match.
The original training variant cannot be proven from checkpoint metadata.

Results are written beneath ab2850_<UTC>_<id> inside this folder.
run.json tracks progress; each subprocess has a .log and a result .json.
summary.json compares top-rate, falls, settled success, and all-episode capped top time.
Early falls and noncompletion retain the full horizon in the capped score.
The summary refuses to compare different initial physical states.
No candidate is automatically promoted or deployed.

Optional overrides: --num-envs 256 --seeds 42 43 44 --device cuda:0
Use a new invocation for a fresh run; existing results are preserved.
