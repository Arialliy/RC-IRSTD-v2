PYTHON ?= python3
DETECTOR_CONFIG ?= configs/detector.yaml
CALIBRATOR_CONFIG ?= configs/calibrator.yaml
PIPELINE_CONFIG ?= configs/pipeline.yaml

.PHONY: install test compile audit-data smoke train calibrator pipeline outer-lodo clean-smoke

install:
	$(PYTHON) -m pip install -e ".[dev]"

compile:
	$(PYTHON) -m compileall -q rc_irstd certification data_ext evaluation losses model risk_curve scripts utils *.py

test: compile
	$(PYTHON) -m pytest -q

audit-data:
	$(PYTHON) scripts/audit_dataset_splits.py \
		--dataset-dir datasets/NUAA-SIRST \
		--dataset-dir datasets/NUDT-SIRST \
		--dataset-dir datasets/IRSTD-1K \
		--output artifacts/dataset_split_audit.json

smoke:
	bash scripts/smoke_full_pipeline.sh

train:
	$(PYTHON) train_detector.py --config $(DETECTOR_CONFIG)

calibrator:
	$(PYTHON) train_calibrator.py --config $(CALIBRATOR_CONFIG)

pipeline:
	$(PYTHON) run_pipeline.py --config $(PIPELINE_CONFIG)

outer-lodo:
	PYTHON_BIN=$(PYTHON) bash scripts/run_outer_lodo_3gpu.sh

clean-smoke:
	rm -rf artifacts/smoke_data artifacts/smoke_pipeline
