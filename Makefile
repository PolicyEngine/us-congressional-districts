targets:
	python src/us_congressional_districts/pull_age_targets.py
	python src/us_congressional_districts/pull_district_geometries.py
	python src/us_congressional_districts/pull_state_geometries.py

data:
	python src/us_congressional_districts/calibrate.py