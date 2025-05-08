targets:
	python src/us_congressional_districts/pull-age-targets.py
	python src/us_congressional_districts/pull-district-geometries.py
	python src/us_congressional_districts/pull-state-geometries.py

data:
	python src/us_congressional_districts/calibrate.py