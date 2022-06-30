# CodeGraphClassification

This is the official repository of the paper: **_CodeGraphClassification: Classification of Source Code Repositories_**

## Public Data

The data used to train the models is available at the following link:

Check the [Model Training](#Model Training) section to see how to use.

**NOTE:** The compressed data is XGB, when uncompressed, it will used around XXGB of your disk.

## Pretrained Models

The pretrained models are available to download:

Check the [Inference](#Inference) section to see how to use.

___

## Reproducibility

### Setup

1. Install Poetry using the following guide: [https://python-poetry.org/docs/](https://python-poetry.org/docs/);
2. Setup the environment by running: `make setup-env`;
3. Set up the graph extraction tool [ARCAN](www.arcan.tech).

### Prerequisites

The first thing required to run the repository is to have a list of repositories in the format `user/repo`, in our case,
this file is located in the `data` folder in a CSV format. The file name is `project_list.csv`

### Data preparation

The first step in to extract the graph of each project in the `project_list.csv`. After setting up ARCAN, we use the
following command to run the extraction for the projects in out list:

```commandline
make extract-graph
```

After extracting the graphs, we need to extract the features

```commandline
make extract-features
```

### Model Training

### Inference

___

## How to Cite

