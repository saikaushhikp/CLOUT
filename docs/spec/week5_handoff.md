# Week 5 handoff

Date: 2026-10-27. Week 5 implementation and automated validation are substantially complete; this handoff **does** claim human browser/TCP review or five-person Week 5 acceptance. DEC-012 is approved to retain aggregate-only Observation v1, not to authorize richer profiles. Review the exact working tree or its eventual reviewed commit; result records say `revision=uncommitted`.

## Environment setup

```bash
conda activate mas
python --version
python -m pip install -e '.[gpu,web,analysis,test]'
python -c 'import numpy,torch,fastapi,uvicorn,matplotlib,httpx; print("numpy",numpy.__version__,"torch",torch.__version__,"cuda",torch.version.cuda,"cuda_available",torch.cuda.is_available(),"devices",torch.cuda.device_count()); print("fastapi",fastapi.__version__,"uvicorn",uvicorn.__version__)'
nvidia-smi -L
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv,noheader
```

Build the production React UI before running the FastAPI integration test or launching the web app. Node >=22.12 and npm >=10 are required by the Vite 8 toolchain. This host's `mas` environment has Node.js 26.12.1 and npm 11.20.0. React and the remaining frontend packages are installed from the lockfile into `web/node_modules/` (React is a project dependency, not a Conda package). On a fresh checkout, install them and repeat the verified checks with:

```bash
conda run -n mas node --version
conda run -n mas npm --version
conda run -n mas npm ci --prefix web
conda run -n mas npm --prefix web ls --depth=0
conda run -n mas npm --prefix web test
conda run -n mas npm --prefix web run build
```


If `node` or `npm` are missing in another Conda installation, install Node.js from conda-forge with `conda install -n mas -c conda-forge nodejs=24`, then repeat the checks above. This follows the [conda package-install documentation (https://docs.conda.io/projects/conda/en/stable/commands/install.html)](https://docs.conda.io/projects/conda/en/stable/commands/install.html) and [conda-forge Node.js package instructions](https://anaconda.org/conda-forge/nodejs). The lockfile was refreshed to Vitest 4.1.12 after npm audit identified two moderate advisories in 4.1.11; the final audit reports zero vulnerabilities.

The optional extras are declared in `pyproject.toml`. The agent used conda environment named `mas`, Python 3.12.15, NumPy 2.5.2, PyTorch 2.7.1+cu118, FastAPI 0.161.1, Uvicorn 0.54.0, Matplotlib 3.11.2 and httpx 0.28.1. Three RTX A5000 devices were visible; each has separate 24,564 MiB VRAM. Free memory is transient. If a shell runs the system Python instead, prefer `conda run -n mas ...`. If CUDA is invisible in a restricted context, follow [PROJECT_ASSUMPTIONS.md](../Assumptions/PROJECT_ASSUMPTIONS.md) and [week4_compute_policy.md](week4_compute_policy.md); do not count `auto` CPU fallback as GPU evidence.

## Regression and finite-world checks

```bash
conda run -n mas python tests/validate_week1_artifacts.py
conda run -n mas python -m unittest discover -s tests -v
conda run -n mas python -m unittest tests.unit.test_epistemic tests.unit.test_graph_experiments tests.unit.test_strategies tests.integration.test_web_api -v
conda run -n mas python -m unittest tests.unit.test_verification tests.integration.test_epistemic_episode -v
conda run -n mas python examples/week5_model_check.py
```

The model-check command writes `reports/results/week5/model_check.json`. Results are exact only over its declared finite worlds, H=1, total search node limit 10,000 and first eight public singleton targets plus broadcast/no-op; default caps also limit horizon to 128 and input worlds to 256. The mixed fixture has PossibleVictory TRUE but GuaranteedVictory FALSE because the two hidden worlds are publicly indistinguishable. `INCOMPLETE` means no proof. This diagnostic never stops a live game early.

## Reproduce experiments and plots

```bash
conda run -n mas python examples/week5_mas_sanity.py
conda run -n mas python examples/week5_run_experiments.py --experiment all --backend cpu --nodes 1000 --seeds 11 23 37 41 53 67 71 83 97 101 113 127 139 149 163 173 181 193 199 211 --rounds 5 --strategies no_op first_legal seeded_random_legal uniform_budget greedy_public_model --output-dir reports/results/week5/main
conda run -n mas python examples/week5_plot_results.py --input-dir reports/results/week5/main --output-dir reports/figures/week5
```

This produces 200 paired episodes: five policies × 20 seeds, three Decimal budget levels × 20 seeds, and two topologies × 20 seeds. Inspect the saved records/manifest rather than trusting console summaries. Plots read the saved JSONL and do not rerun episodes. The small N=30 smoke data under `reports/results/week5/smoke/` are plumbing checks, not scientific evidence.

## GPU scale, parity, full episode

Run in a GPU-visible context:

```bash
conda run -n mas python examples/week5_gpu_scale_benchmark.py \
  --device cuda:2 --nodes 1000 10000 100000 --seed 20261019 \
  --rounds 5 --repeats 3 --output reports/results/week5/gpu_scale.jsonl
conda run -n mas python examples/week4_backend_comparison.py \
  --nodes 1000 --seed 20261019 --rounds 5 --cuda-device cuda:2
conda run -n mas python examples/week4_backend_comparison.py \
  --nodes 10000 --seed 20261019 --rounds 5 --cuda-device cuda:2
conda run -n mas python examples/week5_run_demo.py \
  --backend cuda:2 --nodes 1000 --seed 20261019 --rounds 5 \
  --strategy greedy_public_model --output reports/results/week5/demo_episode.json
```

GPU benchmark exits nonzero if explicit CUDA is unavailable; it never falls back. Week 5 measured 100k with three successful repetitions. Matched parity passed at 1k and 10k with max support difference ≤2.23e-16, equal ledger and equal official result. See [week5_performance_results.md](../../reports/week5_performance_results.md).

## Browser/API walkthrough

The production frontend is the React/Vite application in `web/`. The separate `web-ui/` directory remains only an earlier toolchain experiment and is not served by FastAPI. The production client is built into `web/dist/`. The verified host workflow is the `conda run -n mas npm ...` sequence in Environment setup above; it reports Node v26.12.1, npm 11.20.0 and React 19.3.1. The required versions are Node >=22.12 and npm >=10.

The production client is built into `web/dist/` and served by FastAPI. For development, start FastAPI at port 61007, then in a second terminal run `conda run -n mas npm --prefix web run dev`; Vite uses port 5173 and proxies API calls to port 61007. The latest UI has responsive setup/metrics, an aggregate-support chart, searchable public targets, action costs/history, graph summary, event feed, official result and trace-only diagnostics. It makes no external font/CDN requests. Follow the detailed [API/browser walkthrough](web_api_v1.md#browser-walkthrough) at desktop and phone widths.

The UI tests run against a mocked API with Vitest/jsdom; they are not a substitute for the real FastAPI integration test or human browser/privacy walkthrough. Current `mas` results: clean `npm ci` added 155 packages; React 19.3.1, Vitest 4.1.12; audit reports zero vulnerabilities; UI tests passed 3/3; production build passed; API integration test passed. The first API test run caught an outdated expected 560px breakpoint; the current responsive layout uses 640px and the test now checks that mobile breakpoint. The external Google Fonts import was also removed so the production UI does not need runtime CDN access. A Vite preview smoke previously returned HTTP 200 for the entry page and both JS chunks.

```bash
conda run -n mas python -m uvicorn dumbocracy.webapp:app --host 127.0.0.1 --port 61007
```

Open `http://127.0.0.1:61007`, choose `auto`, set 1,000 nodes/five rounds, and start. The page reports the actual selected backend. Filter and choose a public target ID, submit targeted contact; submit broadcast when legal; use no-op for remaining rounds until the engine returns the official fixed-horizon result. Inspect the aggregate support chart, remaining-budget metric, action-cost history, public event feed and `trace_only` logic scope. Reset/replay and confirm the initial public aggregate is reproduced. In the browser Network panel, inspect observation/result payloads: no voter arrays, susceptibility, edges, or seed fields should appear. The public result filters seed fields from engine reproducibility metadata.

Remote SSH port forwarding:

```bash
ssh -L 61007:127.0.0.1:61007 ai23btech11017@192.168.209.81
```

Then open `http://127.0.0.1:61007` on the reviewer workstation. Previously the Uvicorn app was exercised through health, frontend HTML, game creation, targeted/broadcast/no-op actions, terminal result, analysis and reset using a Unix socket, and API route tests passed in-process.

## Five-member review checklist

- **Mad:** strategy tests and paired strategy report/plots; assess public-only policy input and the synthetic greedy prior's limitations.
- **Deek:** graph/topology/MAS tests, GPU parity/scale records, and model freeze.
- **Hem:** budget/action atomicity, public API privacy, DEC-012 status; do not approve unreviewed disclosure by implication.
- **Psk:** finite-world fixtures, mixed indistinguishable-world result, caps, scopes and distinction from trace-only LTL.
- **Vis:** run the TCP/browser walkthrough, check charts/responsive layout, rejected actions, terminal result, reset, and network payload privacy.


## Deferred/decision-gated scope

The checker is exact on finite enumerations, not the continuous full state space. More seeds/topologies/statistical analysis, greedy-prior calibration, browser polish, production security/persistence and final demo materials remain Week 6 work. The Week 5 human reviewer walkthrough and acceptance are completed and accepted.
