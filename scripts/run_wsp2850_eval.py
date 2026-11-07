"""Extract the versioned evaluation bundle and run with this Python environment."""
import hashlib
from pathlib import Path
import subprocess
import sys
import zipfile


def main():
    repo = Path(__file__).resolve().parents[1]
    archive = repo / 'checkpoints/remote_eval_2850.zip'
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()[:16]
    destination = repo / 'outputs' / ('wsp2850_' + digest)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for name in bundle.namelist():
            target = (destination / name).resolve()
            if destination.resolve() not in target.parents:
                raise ValueError('Archive path escapes destination')
        bundle.extractall(destination)
    launcher = destination / 'remote_eval_2850/run_eval.py'
    raise SystemExit(subprocess.call([sys.executable, str(launcher),
                                     '--python', sys.executable, *sys.argv[1:]], cwd=repo))


if __name__ == '__main__':
    main()
