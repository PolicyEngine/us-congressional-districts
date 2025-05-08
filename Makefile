.PHONY: data

targets:
	python src/us_congressional_districts/pull_age_targets.py
	python src/us_congressional_districts/pull_district_geometries.py
	python src/us_congressional_districts/pull_state_geometries.py

data:
	python src/us_congressional_districts/calibrate.py

install-uv:
	uv pip install -e .
	uv pip install policyengine-us

documentation:
	jb build docs

test:
	echo "No tests yet"