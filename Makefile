setup-env:
	poetry activate

extract-graph:
	python ex

extract-features:
	echo Features

extract: extract-graph-all extract-features
