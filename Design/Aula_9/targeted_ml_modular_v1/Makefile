PROJECT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PYTHON ?= python
SPEC ?= $(PROJECT_DIR)/specs/activity.yaml
OUTPUT_ROOT ?= $(PROJECT_DIR)/build

.PHONY: validate-spec build build-modelled build-ml build-report export-serving score-modelled score-frame score-raw streamlit-app check-compatibility test docker-build dbt-parse dbt-docs clean-local trim-delivery-view

validate-spec:
	$(PYTHON) -m targeted_ml validate-spec --analysis-spec $(SPEC)

build:
	$(PYTHON) -m targeted_ml build --analysis-spec $(SPEC) --output-root $(OUTPUT_ROOT)

build-modelled:
	$(PYTHON) -m targeted_ml build-modelled --analysis-spec $(SPEC) --output-root $(OUTPUT_ROOT)

build-ml:
	$(PYTHON) -m targeted_ml build-ml --analysis-spec $(SPEC) --output-root $(OUTPUT_ROOT)

build-report:
	$(PYTHON) -m targeted_ml build-report --analysis-spec $(SPEC) --output-root $(OUTPUT_ROOT)

export-serving:
	$(PYTHON) -m targeted_ml export-serving --analysis-spec $(SPEC) --output-root $(OUTPUT_ROOT)

score-modelled:
	$(PYTHON) -m targeted_ml score-modelled --analysis-spec $(SPEC) --output-root $(OUTPUT_ROOT)

score-frame:
	$(PYTHON) -m targeted_ml score-frame --analysis-spec $(SPEC) --output-root $(OUTPUT_ROOT) --scoring-frame $(SCORING_FRAME)

score-raw:
	$(PYTHON) -m targeted_ml score-raw --analysis-spec $(SPEC) --dataset-root $(DATASET_ROOT) --output-root $(OUTPUT_ROOT)

streamlit-app:
	$(PYTHON) -m streamlit run $(PROJECT_DIR)/targeted_ml/apps/streamlit_app.py

check-compatibility:
	$(PYTHON) -m targeted_ml check-compatibility --output-root $(OUTPUT_ROOT)

test:
	$(PYTHON) -m pytest -q

docker-build:
	docker build -t targeted-ml-modular-v1 .

dbt-parse:
	TARGETED_ML_BUILD_DUCKDB=$(OUTPUT_ROOT)/duckdb/build.duckdb \
	TARGETED_ML_MODELLED_DUCKDB=$(OUTPUT_ROOT)/modelled/duckdb/base_modelada_v2.duckdb \
	dbt parse --project-dir $(PROJECT_DIR)/dbt_lineage --profiles-dir $(PROJECT_DIR)/dbt_lineage

dbt-docs:
	TARGETED_ML_BUILD_DUCKDB=$(OUTPUT_ROOT)/duckdb/build.duckdb \
	TARGETED_ML_MODELLED_DUCKDB=$(OUTPUT_ROOT)/modelled/duckdb/base_modelada_v2.duckdb \
	dbt docs generate --project-dir $(PROJECT_DIR)/dbt_lineage --profiles-dir $(PROJECT_DIR)/dbt_lineage

clean-local:
	find $(PROJECT_DIR) -type d -name '__pycache__' -prune -exec rm -rf {} +
	rm -rf \
		$(PROJECT_DIR)/.pytest_cache \
		$(PROJECT_DIR)/.ruff_cache \
		$(OUTPUT_ROOT)/logs \
		$(OUTPUT_ROOT)/staging \
		$(OUTPUT_ROOT)/local_runtime_duckdb \
		$(OUTPUT_ROOT)/local_runtime_staging \
		$(OUTPUT_ROOT)/app_specs \
		$(OUTPUT_ROOT)/app_uploads \
		$(PROJECT_DIR)/dbt_lineage/logs \
		$(PROJECT_DIR)/dbt_lineage/target/compiled
	rm -f \
		$(PROJECT_DIR)/dbt_lineage/target/graph.gpickle \
		$(PROJECT_DIR)/dbt_lineage/target/graph_summary.json \
		$(PROJECT_DIR)/dbt_lineage/target/partial_parse.msgpack \
		$(PROJECT_DIR)/dbt_lineage/target/perf_info.json \
		$(PROJECT_DIR)/dbt_lineage/target/run_results.json

trim-delivery-view:
	printf '%s\n' \
		"import json" \
		"import shutil" \
		"from pathlib import Path" \
		"project_dir = Path('$(PROJECT_DIR)')" \
		"output_root = Path('$(OUTPUT_ROOT)')" \
		"runs_dir = output_root / 'inference_runs'" \
		"keep_runs = {'example_modelled_inference', 'example_raw_inference'}" \
		"if runs_dir.exists():" \
		"    for path in runs_dir.iterdir():" \
		"        if path.is_dir() and path.name not in keep_runs:" \
		"            shutil.rmtree(path)" \
		"latest_json = output_root / 'serving' / 'latest.json'" \
		"exports_dir = output_root / 'serving' / 'exports'" \
		"keep_export = None" \
		"if latest_json.exists():" \
		"    payload = json.loads(latest_json.read_text(encoding='utf-8'))" \
		"    latest_export_dir = payload.get('latest_export_dir')" \
		"    if latest_export_dir:" \
		"        keep_export = Path(latest_export_dir).name" \
		"if exports_dir.exists() and keep_export:" \
		"    for path in exports_dir.iterdir():" \
		"        if path.is_dir() and path.name != keep_export:" \
		"            shutil.rmtree(path)" \
		"compiled_dir = project_dir / 'dbt_lineage' / 'target' / 'compiled'" \
		"if compiled_dir.exists():" \
		"    shutil.rmtree(compiled_dir)" \
		"for rel in ['dbt_lineage/target/graph.gpickle', 'dbt_lineage/target/graph_summary.json', 'dbt_lineage/target/partial_parse.msgpack', 'dbt_lineage/target/perf_info.json', 'dbt_lineage/target/run_results.json']:" \
		"    target = project_dir / rel" \
		"    if target.exists():" \
		"        target.unlink()" \
	| $(PYTHON) -
