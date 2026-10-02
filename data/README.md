# Data

`benchmarks/single_scene.jsonl` is the synthesized single-scene benchmark: 12,021 instances.

`benchmarks/multi_action.jsonl` is the synthesized multi-action benchmark: 2,400 parent instances. Component samples are referenced by `single_scene_sample_id`.

`references/single_scene_gold_y.jsonl` is the single-scene evaluation reference. It is not an input to EviRepair.

Multi-action evaluation references are stored on each parent record and read by `ma_y_acts`. There is no separate multi-action Gold file.

`blueprints/raw` and `blueprints/vision` hold the 22 blueprint YAML files read while the repair context is built. They are inputs to context construction, not evaluation references.

`openhab/` is a separate OpenHAB corpus: 1,600 scenarios under `openhab/benchmarks`, with evaluation references under `openhab/references`. It is not mixed with the Home Assistant files above.

`synthesis/` holds the code that builds these instances. `single_scene_blueprint_complete/` synthesizes the Home Assistant single-scene samples. `multi_action_frozen_v1/`, `multi_action_frozen_v2/`, and `formal_b0/` synthesize the multi-action samples and the formal B0 blocks. `openhab/` synthesizes the OpenHAB scenarios. The runner scripts are under `synthesis/scripts/`.

`labeling/` holds the code that writes the evaluation references. `multi_action_frozen_v1/` and `labeling/` label the Home Assistant Gold Y. `openhab/` labels the OpenHAB semantic reference. The runner scripts are under `labeling/scripts/`.

LLM endpoint addresses and API keys are not stored in these files. `Y_LLM_API_KEY` and `Y_LLM_BASE_URL` are empty. A labeling run reads the key and the endpoint from the environment.
