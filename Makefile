setup-env:
	poetry shell

extract-graph: setup-env
	python src/pipelines/extract_graphs.py

extract-features: setup-env
	echo Features

extract: extract-graph-all extract-features
