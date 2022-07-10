setup-env:
	poetry shell

extract-graph: setup-env
	python3 src/pipelines/extract_graphs.py

extract-features: setup-env
	python3 src/pipelines/extract_embeddings.py

extract: extract-graph extract-features
