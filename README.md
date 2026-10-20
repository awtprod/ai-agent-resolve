<img src="assets/icon.svg" width="64" height="64" alt="">

# AI-Agent-Resolve

Created in [T3 Code](https://t3.codes).

## Continuous integration

GitHub Actions runs lint, formatting, and the full non-live test suite on Linux with Python
3.11 and 3.12. It installs ffmpeg so media tests exercise their real tool path. Windows with
Python 3.12 runs the job-store portability tests and the full non-live node suite; those tests
do not use ffmpeg.

The exact Linux equivalent, starting from a clean checkout on Debian or Ubuntu, is:

```bash
sudo apt-get update && sudo apt-get install --yes ffmpeg
for version in 3.11 3.12; do
  "python${version}" -m venv ".venv-ci-${version}"
  ".venv-ci-${version}/bin/python" -m pip install -e '.[dev]'
  AAR_LIVE=0 ".venv-ci-${version}/bin/ruff" check .
  AAR_LIVE=0 ".venv-ci-${version}/bin/ruff" format --check .
  AAR_LIVE=0 ".venv-ci-${version}/bin/pytest" -q \
    --junitxml="junit-linux-${version}.xml"
done
```

The exact Windows PowerShell equivalent for the node job is:

```powershell
py -3.12 -m venv .venv-ci
.venv-ci\Scripts\python -m pip install -e '.[dev]'
$env:AAR_LIVE = "0"
.venv-ci\Scripts\pytest -q tests/node/test_job_store_portability.py `
  --junitxml=junit-windows-portability.xml
.venv-ci\Scripts\pytest -q tests/node -m "not live" `
  --junitxml=junit-windows-node.xml
```
