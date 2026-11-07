# Pinned Hexapod student compatibility task

This package supplies `WSP-Hexapod-Stair-Student-A` when an installed older
RLforHexapod checkout has only the teacher tasks. It contains task source from
RLForHexapod commit `ed53031ea774e167982bc9563a28435daefc619b`.

Place this directory beside the playback wrapper, then register explicitly:

```python
from hexapod_student_compat import SOURCE_COMMIT, register

task_id = register()
```

Importing the package does not register a task. Registration creates a distinct
task ID and does not modify the installed project or its original task IDs.
The wrapper should prefer the installed `Hexapod-Stair-Student-A` when available.

The upstream task implementation is retained, including scene, actions, rewards,
student noise, and reset behavior. Its only source edit relocates the task's MDP
import into this private package. Robot and stair assets still come from the
installed `RLforHexapod.assets.ROBOT_USDA` and `OBSTACLES_DIR` locations.
The installed assets and Isaac Lab runtime must match the training environment.
Before creating the environment, the wrapper should call
`inspect_asset_provenance(ROBOT_USDA, OBSTACLES_DIR)`, save its JSON-ready report,
and reject `matches_pinned == False`. `asset_manifest.json` pins the nine robot
USD package files and `stairs_160mm.usda` using their upstream Git LFS SHA-256
hashes and sizes.
Missing files, unresolved LFS pointers, and changed bytes are reported separately.
This checker does not download or replace assets. A matching model architecture
alone does not establish that an older installed robot asset matches training.

Expected model contract: student 271 observations, teacher 91 observations,
18 actions, and a one-layer GRU with 256 hidden units. The configured student
MLP uses hidden dimensions 256 and 128, ELU, and observation normalization.
Student height-map noise remains enabled with standard deviation 0.01.
Reset buffer sampling is disabled, but the two upstream initial x ranges still
select phase 0 or phase 1. This package does not silently change the benchmark
to start only at phase 0.

`provenance.json` records original URLs, original and vendored SHA-256 hashes,
Git blob hashes, and the exact import replacement. `LICENSE` retains the
upstream BSD-3-Clause notice. To verify bytes, source transformations, Python
syntax, and private imports without importing Isaac Lab:

```sh
python hexapod_student_compat/verify_snapshot.py
```

This static verification does not establish simulation compatibility or policy
performance. The playback wrapper must verify runtime observation and action
dimensions, runner type, and checkpoint tensor shapes before rollout.
