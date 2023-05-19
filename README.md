<<<<<<< HEAD

# CodeGraphClassification

This is the official repository of the paper: **_CodeGraphClassification: Classification of Source Code Repositories_**

## Data

The data for this work is available at the following link:

**NOTE:** The compressed data is 25GB, when uncompressed, it will use around 198GB of your disk.

## Reproducibility

1. Install Poetry using the following guide: [https://python-poetry.org/docs/](https://python-poetry.org/docs/);
2. Setup the environment using poetry.
3. Set up the graph extraction tool [ARCAN](www.arcan.tech).
4. Run the pipelines in the [./src/pipelines](./src/pipelines) file. To set the configs use
   hydra ([https://hydra.cc/docs/intro/](https://hydra.cc/docs/intro/)).

## Project structure

```
.
├── data
│  ├── interim   # intermediate files
│  │  └── arcanOutput  # Arcan output files
│  ├── models # Word embeddings models
│  ├── processed # Final processed files
│  │  ├── annotations # File level annotation for all configs
│  │  │  ├── ensemble
│  │  │  │  └── best
│  │  │  │      ├── cascade
│  │  │  │      ├── exp_voting
│  │  │  │      ├── jsd
│  │  │  │      ├── max
│  │  │  │      └── voting
│  │  │  ├── keyword
│  │  │  │  ├── identifiers
│  │  │  │  │  └── yake
│  │  │  │  └── name
│  │  │  │      └── yake
│  │  │  └── similarity
│  │  │      └── name
│  │  │          ├── BERT
│  │  │          ├── fastText
│  │  │          └── w2v-so
│  │  ├── content  # Preprocessed content of the source code file for quick analysis
│  │  │  ├── identifiers
│  │  │  ├── methods
│  │  │  └── name
│  │  ├── keywords # Extracted keywords for the labels
│  │  │  ├── identifiers
│  │  │  │  └── yake
│  │  │  │      └── similarity
│  │  │  ├── name
│  │  │  │  └── yake
│  │  │  │      └── similarity
│  │  ├── labelled_graph  # Labelled arcan graphs 
│  │  │  ├── ensemble
│  │  │  │  └── best
│  │  │  │      └── voting
│  │  │  │          └── none
│  │  │  │              └── none
│  │  │  └── keyword
│  │  │      ├── identifiers
│  │  │      │  └── yake
│  │  │      └── name
│  │  │          └── yake
│  │  ├── manual_eval  # Human eval files
│  │  ├── package_labels # Package-level labels
│  │  │  ├── ensemble
│  │  │  │  └── best
│  │  │  │      ├── cascade
│  │  │  │      ├── exp_voting
│  │  │  │      ├── jsd
│  │  │  │      ├── max
│  │  │  │      └── voting
│  │  │  ├── keyword
│  │  │  │  ├── identifiers
│  │  │  │  │  └── yake
│  │  │  │  └── name
│  │  │  │      └── yake
│  │  │  └── similarity
│  │  │      └── name
│  │  │          ├── BERT
│  │  │          ├── fastText
│  │  │          └── w2v-so
│  │  └── project_labels # Project-level labels
│  │      ├── ensemble
│  │      │  └── best
│  │      │      ├── cascade
│  │      │      ├── exp_voting
│  │      │      ├── jsd
│  │      │      ├── max
│  │      │      └── voting
│  │      ├── keyword
│  │      │  ├── identifiers
│  │      │  │  └── yake
│  │      │  └── name
│  │      │      └── yake
│  │      └── similarity
│  │          └── name
│  │              ├── BERT
│  │              ├── fastText
│  │              └── w2v-so
│  └── raw
├── docker
│  └── Dockerfile
├── logs # log folder
│  └── arcan
├── notebooks  # Misc notbooks
│  ├── analysis
│  └── test
│      └── resources
├── output  # Outputs for some analysis
│  └── stats
├── reports  # Plots folder
│  ├── plots
│  │  ├── annotations
│  │  └── statistics
│  └── raw
├── scripts
│  ├── bash
│  ├── python
│  └── R  # Scripts for generating the plots
├── src  # Project source code files
│  ├── analysis
│  ├── annotation
│  ├── conf  # Configuration files for Hydra
│  │  ├── content
│  │  ├── embedding
│  │  ├── ensemble
│  │  ├── extraction
│  │  ├── filtering
│  │  ├── keyword
│  │  ├── local
│  │  ├── node_annotation
│  │  └── transformation
│  ├── data
│  ├── ensemble_annotations
│  ├── feature
│  └── pipelines  # Pipelines used for the paper
├── tools  # Auxiliary tools
│  └── arcan  # Arcan folder (.jar - Request to Arcan Devs)
│      └── lib
├── languages.so
├── Makefile
├── pyproject.toml   # Poetry config file to set up the python env
└── README.md   # THis file
```