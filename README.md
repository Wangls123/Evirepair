# EviRepair

Single-scene and multi-action repair benchmark, evaluation references, EviRepair source, baseline configurations, and the scripts that reproduce the reported experiments.

EviRepair is `ss_trhr_v1`. The repairer reads the unrepaired action and the runtime context. It does not read Gold labels. The reported score applies Strict-v2 after the slot operators. Multi-action scoring then applies entity mapping, set-remove, the generic gate, and the verifier. LLM calls are off.

## Software environment

Tested with Python 3.11.7 on Windows. Python 3.10 or newer is required. No GPU is required for the bundled experiments.

Create a virtual environment and install `requirements.txt`. That file is the environment specification: PyYAML, numpy, scipy, scikit-learn, lightgbm, joblib, requests, and voluptuous.

Windows PowerShell, from this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Unix:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The reported Home Assistant score, ablation, behavioral-status, and intervention scripts then run with that interpreter. `src/smarthome_mdf/multi_action_frozen_v1/y_llm_config.py` ships with an empty key and an empty endpoint. A labeling or Qwen run reads `DASHSCOPE_API_KEY` and `DASHSCOPE_BASE_URL` from the environment.

AutoTap needs Docker. TAPFixer needs Python 3.8 and its own archive. Download links are in [External baselines](#external-baselines).

The benchmark files occupy about 790 MB. A full single-scene run keeps the reference index and one repair context at a time. The official script also keeps the multi-action rows in memory.

## Layout

```
data/benchmarks/          Home Assistant SS and MA instances
data/references/          Home Assistant SS evaluation reference
data/blueprints/          blueprint YAML read while building repair context
data/openhab/benchmarks/  OpenHAB scenarios, separate from the Home Assistant files
data/openhab/references/  OpenHAB evaluation reference and hidden labels
data/synthesis/           code that synthesizes the Home Assistant and OpenHAB instances
data/labeling/            code that writes the evaluation references
openhab_multirule/        OpenHAB multi-rule migration, separate from the Home Assistant package
baselines/autotap/        AutoTap detection adapters
baselines/tapfixer/       TAPFixer detection adapters
baselines/scripts/        AutoTap and TAPFixer detection runners
baselines/qwen_open/      Qwen3 Direct and Constrained prompts
baselines/common/         shared TAP intermediate representation
configs/blueprint_action_contracts.json
configs/evirepair.json    method and Strict-v2 settings
configs/baselines/        Service Blacklist, State-only, AutoTap, TAPFixer, Qwen3
src/smarthome_mdf/        EviRepair and the context it calls
scripts/repro_ha_official_ss_ma.py
scripts/eval_ablation.py
scripts/eval_behavioral_status.py
scripts/eval_intervention.py
scripts/eval_llm_repair_baselines.py
scripts/run_qwen_intervention.py
results/evirepair.json
results/ablation.json
results/behavioral_status.json
results/intervention.json
results/openhab_multirule_evirepair.json
```

Multi-action references are stored on each parent record and read by `ma_y_acts`. There is no separate multi-action Gold file.

Run every command below from this directory. Each script adds the paths it needs. On Windows PowerShell you can also set:

```powershell
$env:PYTHONPATH = "src;scripts"
```

On Unix:

```bash
PYTHONPATH=src:scripts
```

## Home Assistant official scores

```bash
python -u scripts/repro_ha_official_ss_ma.py
```

The script writes `results/evirepair.json` and prints the same scores. Parent merge drops Recover adds of `notify.mobile_app` and `logbook.log`; original copies of those services stay. A completed run on this machine took 93.631 seconds and produced:

| Split | Metric | Count |
| --- | --- | --- |
| Single-scene | Exact match | 9497/12021 |
| Single-scene | Repair success | 6386/8731 |
| Single-scene | False repair | 179/12021 |
| Multi-action | Parent exact match | 1196/2400 (49.83%) |
| Multi-action | Action F1 | 0.5515 |

`results/evirepair.json` stores only this published EviRepair run.

## Ablation

```bash
python -u scripts/eval_ablation.py
```

The script scores the paper modules on the bundled Home Assistant splits and rewrites `results/ablation.json`. The file already in the repository is the recorded table: Full, w/o Execution Instance Modeling, w/o Runtime Context, w/o Contract Constraints, w/o Behavioral Assessment, w/o Remove, w/o Modify, w/o Recover, and w/o Provenance-Aware Interaction. Full single-scene exact match in that table is 9497/12021. Full multi-action parent exact match is 1196/2400 (49.83%), with Action F1 0.5515, the same official multi-action score as `results/evirepair.json`.

## Behavioral assessment

```bash
python -u scripts/eval_behavioral_status.py
```

The script assigns each single-scene case to the first matching category in the order Justified, Unsupported, Incorrect, Missing. A case that matches more than one category is counted only in that first category, which moves the remaining multi-category detections into Unsupported. It writes `results/behavioral_status.json`. A completed run took 132.879 seconds:

| Category | Cases | Correct | Incorrect | Accuracy |
| --- | --- | --- | --- | --- |
| Unsupported | 3349 | 3349 | 0 | 100.00% |
| Justified | 3290 | 3111 | 179 | 94.56% |
| Incorrect | 3297 | 1769 | 1528 | 53.65% |
| Missing | 777 | 662 | 115 | 85.20% |

## Intervention identification

Each method has a detection script. A case needs intervention when the original action is not semantically equal to the reference. A method predicts intervention when its detected final action is not semantically equal to the original.

| Method | Detection code |
| --- | --- |
| Service Blacklist | `scripts/eval_ss_trhr_credibility.py` `baseline_blacklist` |
| State-only | `scripts/eval_ss_trhr_credibility.py` `baseline_state_only` |
| AutoTap | `baselines/autotap/` and `baselines/scripts/run_autotap_native.py` |
| TAPFixer | `baselines/tapfixer/` and `baselines/scripts/run_tapfixer_ss_native.py` |
| Qwen3-14B / 32B Direct and Constrained | `scripts/eval_llm_repair_baselines.py` and `scripts/run_qwen_intervention.py` |
| EviRepair | `src/smarthome_mdf/ss_trhr_repair/pipeline.py` `repair_sample` |

```bash
python -u scripts/eval_intervention.py
```

The scoring script runs Service Blacklist, State-only, and EviRepair on all 12021 single-scene cases and rewrites `results/intervention.json`. That file is the table above, including the confusion counts behind each percentage. Service Blacklist, State-only, and EviRepair in the file were recomputed here and match the table. AutoTap, TAPFixer, and the four Qwen rows are the same recorded counts. They are rescored when their prediction files are present.

AutoTap writes `baselines/predictions/autotap/native_ss_predictions.jsonl`. The upstream tree must be at `baselines/third_party/autotap/upstream`, and Docker must be running.

```bash
python -u baselines/scripts/run_autotap_native.py --phase all
```

TAPFixer writes `baselines/predictions/tapfixer/native_ss_predictions.jsonl`. The worker runs in Docker and reads `baselines/environments/tapfixer`.

```bash
python -u baselines/scripts/run_tapfixer_ss_native.py --phase all
```

Qwen3-14B and Qwen3-32B, Direct and Constrained, call the four single-scene jobs and write under `baselines/qwen_open/`. Set the key and endpoint in the same shell, then run:

```powershell
$env:DASHSCOPE_API_KEY = "<your key>"
$env:DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
python -u scripts/run_qwen_intervention.py
```

`--generate-only` stops after the API calls. `--eval-only` scores prediction files that are already present. After any of these three commands, rerun `python -u scripts/eval_intervention.py` to refresh `results/intervention.json`.

| Method | Recall (%) | F1 | FIR (%) | BA (%) |
| --- | --- | --- | --- | --- |
| Service Blacklist | 56.71 | 0.7237 | 0.00 | 78.35 |
| State-only | 29.24 | 0.4525 | 0.00 | 64.62 |
| AutoTap | 39.41 | 0.5654 | 0.00 | 69.71 |
| TAPFixer | 27.77 | 0.4192 | 12.58 | 57.60 |
| Qwen3-14B Direct | 88.42 | 0.8405 | 58.36 | 65.03 |
| Qwen3-14B Constrained | 98.84 | 0.9358 | 32.95 | 82.95 |
| Qwen3-32B Direct | 87.65 | 0.9126 | 11.79 | 87.93 |
| Qwen3-32B Constrained | 97.96 | 0.9701 | 10.64 | 93.66 |
| EviRepair | 82.84 | 0.8961 | 5.44 | 88.70 |

## OpenHAB multi-rule

```bash
python -u openhab_multirule/run_multirule.py
```

The runner loads `data/openhab/benchmarks`, keeps scenarios with more than one enabled rule, and skips single-rule scenarios. It rewrites `results/openhab_multirule_evirepair.json`. The file already in the repository is the recorded multi-rule result: 371 scenarios, semantic success 0.8814, complete repair rate 0.8112. See `openhab_multirule/README.md`.

## Synthesis

These commands rebuild the corpora. The published instances are already in `data/benchmarks` and `data/openhab/benchmarks`. Run them from this directory with `PYTHONPATH` set as above.

Single-scene regeneration, about 12,021 samples. Output follows `OUTPUT_DIR` in `src/smarthome_mdf/single_scene_blueprint_complete/config.py`.

```bash
python -u data/synthesis/scripts/single_scene_blueprint_complete/run_final_regeneration.py
```

Multi-action generation, 2,400 parents:

```bash
python -u data/synthesis/scripts/multi_action_frozen_v1/run_full_generation.py
```

OpenHAB scenario generator and the 1,600-scenario builder are `data/synthesis/openhab/scenario_generator.py` and `data/synthesis/openhab/final_benchmark_generator.py`. The seed recorded for the bundled corpus is `20260909`.

## Labeling

Dry run builds the labeling view and does not call a model. `--llm` sends the real labeling requests. The single-scene command uses the final Gold Y prompt `final_gold_y`. The key and endpoint stay in the environment; the config files ship empty.

```powershell
$env:DASHSCOPE_API_KEY = "<your key>"
$env:DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
python -u data/labeling/scripts/run_y_labeling_single_scene.py --dry-run
python -u data/labeling/scripts/run_y_labeling_single_scene.py --llm
python -u data/labeling/scripts/run_y_labeling_multi_action.py --dry-run
python -u data/labeling/scripts/run_y_labeling_multi_action.py --llm
```

The single-scene reference already in the repository is `data/references/single_scene_gold_y.jsonl`. OpenHAB labeling is `data/labeling/openhab/e2e/llm_semantic_labeler.py`. It reads `OPENHAB_LLM_API_KEY` and `OPENHAB_LLM_BASE_URL`. The bundled semantic reference is `data/openhab/references/openhab_semantic_reference.jsonl`.

## External baselines

These solvers are not in the repository. The local adapters that compile a sample into their property language and map a patch back to a Home Assistant action are in `baselines/`.

### AutoTap

Clone the official repository and use the published image:

```bash
git clone https://github.com/zlfben/autotap.git
docker pull zlfben/autotap_backend:icse
```

The commit used for the reported row is `fe0fdd638a1b170b1ac71b0f2921ab54d547893b`. Place the single-scene predictions at `baselines/predictions/autotap/native_ss_predictions.jsonl`. Each line needs `sample_id` or `id`, and `final_action`.

### TAPFixer

TAPFixer requires Python 3.8. The archive used for the reported row is the `master` snapshot:

- Repository: https://github.com/q1uTr5th/TAPFixer
- Archive: https://codeload.github.com/q1uTr5th/TAPFixer/zip/refs/heads/master
- SHA256: `1f2ab3c7d8205a5095cb1e06b988c262a8b5721f1ac9d94f77c3eec16a8d7abb`

```bash
git clone https://github.com/q1uTr5th/TAPFixer.git
```

Place the single-scene predictions at `baselines/predictions/tapfixer/native_ss_predictions.jsonl` with the same `final_action` fields.

### Qwen3-14B and Qwen3-32B

The reported Direct and Constrained rows used DashScope model ids `qwen3-14b` and `qwen3-32b` at `https://dashscope.aliyuncs.com/compatible-mode/v1`. Model names are listed at https://www.alibabacloud.com/help/en/model-studio/models. The call needs `DASHSCOPE_API_KEY`. The key is not stored in this repository.

Public weights for the same model sizes:

- https://huggingface.co/Qwen/Qwen3-14B
- https://huggingface.co/Qwen/Qwen3-32B

The detector writes predictions at:

```
baselines/qwen_open/14b/direct/ss/predictions.jsonl
baselines/qwen_open/14b/constrained/ss/predictions.jsonl
baselines/qwen_open/32b/direct/ss/predictions.jsonl
baselines/qwen_open/32b/constrained/ss/predictions.jsonl
```

Each line needs `id` and `parsed_response` with `decision`, `capability`, `operation`, `target`, and `semantic_payload`. After the files are in place, rerun `python -u scripts/eval_intervention.py`.

## Checks already run

Every Python file in this tree was byte-compiled with Python 3.11.7. `scripts/repro_ha_official_ss_ma.py` finished with exit code 0 and matched the official table. `scripts/eval_behavioral_status.py` finished with exit code 0 and matched the category table. `scripts/eval_ablation.py` was smoke-tested on a few contexts. `scripts/eval_intervention.py` finished with exit code 0 on all 12021 single-scene cases. The recomputed Service Blacklist, State-only, and EviRepair rows match the published intervention table. AutoTap, TAPFixer, and Qwen3 were left as recorded rows because their solvers are not bundled.
