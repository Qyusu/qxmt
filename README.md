# QXMT: Quantum Experiment Management Tool
![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)
[![unit tests](https://github.com/kenya-sk/qxmt/actions/workflows/unit_tests.yaml/badge.svg)](https://github.com/kenya-sk/qxmt/actions/workflows/unit_tests.yaml)
[![Docs](https://img.shields.io/badge/docs-online-brightgreen)](https://qyusu.github.io/qxmt/)
[![PyPI version](https://img.shields.io/pypi/v/qxmt.svg)](https://pypi.org/project/qxmt/)
[![Downloads](https://static.pepy.tech/badge/qxmt)](https://pepy.tech/project/qxmt)


QXMT is an open-source tool designed for managing experiments in quantum machine learning. Its primary focus is on minimizing the cost of managing experiments and ensuring reproducibility. To reduce costs, QXMT aims to minimize the amount of implementation code needed for experiment management, allowing developers and researchers to focus on implementing new experiments. For reproducibility, QXMT manages experimental configurations via configuration files, enabling not only the original developers but also collaborators to easily reproduce the same results without significant additional effort.

QXMT includes a variety of datasets, machine learning models, and visualization methods to support experiment management. By using these components together, users can avoid the need to develop entire workflows from scratch while also ensuring that experiments can be evaluated under consistent conditions. These default features will continue to be expanded in future updates.

### Limitation
QXMT is a newly released tool, and its features are currently limited. The quantum libraries and devices that have been tested are listed below. For future development plans, please refer to the roadmap. Even if your environment is not listed, you can still manage experiments by implementing according to the interfaces provided by QXMT. For details on how to implement these interfaces, please refer to the documentation.


| Quantum Library              | Simulator | Real Machine (IBMQ, Amazon Braket) |
|---------------------|-----------|--------------|
|<p align="center">[pennylane](https://github.com/PennyLaneAI/pennylane)</p>|<p align="center">✅</p>|<p align="center">❌</p>|
|<p align="center">[Qulacs](https://github.com/qulacs/qulacs)</p>           |<p align="center">❌</p>|<p align="center">❌</p>|
|<p align="center">[Qiskit](https://github.com/Qiskit/qiskit)</p>           |<p align="center">❌</p>|<p align="center">❌</p>|
|<p align="center">[Cirq](https://github.com/quantumlib/Cirq)</p>           |<p align="center">❌</p>|<p align="center">❌</p>|


## Installation
QXMT is tested and supported on 64-bit systems with:
- Python 3.10, 3.11

You can install QXMT with Python's pip package manager:
```bash
pip install qxmt
```

When installing QXMT, you have the option to enable the LLM functionality. By default, it is not installed. By enabling the LLM feature, you can automatically generate experiment summaries based on code differences. If needed, please install it using the following command:
```bash
pip install qxmt[llm]
```

## Tool Overview
QXMT manages experiments in the following folder structure.
```bash
<your_project>
├── data
├── configs
│   ├── config_1.yaml
│   ├──   ⋮
│   └── config_n.yaml
└── experiments
    ├── <your_experiment_1>
    │   ├── experiment.json
    │   ├── run_1
    │   │   ├── config.yaml
    │   │   ├── shots.h5
    │   │   └── model.pkl
    │   ├── run_2
    │   ├──   ⋮
    │   └── run_n
    │   ⋮
    └── <your_experiment_n>

```

### Keywords
- **data**: Contains the raw data used in the experiments.
- **configs**: Holds YAML files that define the configurations for each experiment run.
- **experiments**: Contains the results of the experiments.
    - A dedicated folder is automatically created for each experiment, based on the name provided when the experiment is initialized.
    - Each experiment folder includes an experiment.json file and subfolders that manage the individual runs of the experiment.


## Getting Started
### 1. Start new experiment
```python
import qxmt

# initialize experiment setting
experiment = qxmt.Experiment(
    name="operation_check",  # set your experiment name
    desc="operation check of experiment package",  # set your experiment description
    auto_gen_mode=False,  # if True, each experimental description is automatically generated by LLM
).init()

# run experiment. each experiment defined in config file or instance.
# see documentation for details on instance mode
# run from config
artifact, result = experiment.run(config_source="configs/template-openml.yaml")

# get instance of each experiment artifact
dataset = artifact.dataset
model = artifact.model

# output result
# result table convert to pandas dataframe
experiment.runs_to_dataframe()

# visualization (Below are some of the features. See documentation for details.)
model.plot_train_test_kernel_matrix(dataset.X_train, dataset.X_test, n_jobs=5)
```

### 2. Load existing experiment
```python
# load existing experiment from json file
experiment = qxmt.Experiment().load_experiment(
    "experiments/operation_check/experiment.json")

# reproduction target run artifact
reproduction_model = experiment.reproduce(run_id=1)

# run new experiment
artifact, result = experiment.run(config_source="configs/template-openml.yaml")

# output result
experiment.runs_to_dataframe()
```


## Contributing
We happily welcome contributions to QXMT. For details on how to contribute, please refer to our [Contribution Guide](./CONTRIBUTING.md).
