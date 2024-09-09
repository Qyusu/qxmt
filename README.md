# QXMT: Quantum Experiment Management Tool
QXMT is an open-source experiment management tool for quantum machine learning. The development focus is on low-cost experiment management and ensuring reproducibility. For low-cost management, the goal is to minimize the implementation code required for managing experiments, allowing developers and researchers to focus solely on new experimental implementations. To ensure reproducibility, QXMT manages experimental information as configuration file, enabling not only the original developer but also collaborators to reproduce the same experimental results without investing significant time.

QXMT provides several datasets, machine learning models, and visualization methods to facilitate experiment management. By combining these, users can not only avoid developing the entire workflow themselves but also ensure that many people can evaluate their experiments based on the same standards. These default features will be continuously expanded in future development.

### Limitation
QXMT is newly released, and the available features are still limited. The quantum libraries and devices that have been tested are listed below. For future development plans, please refer to the roadmap. Even in environments not listed below, you can manage experiments by implementing according to the interfaces provided by QXMT. For details on how to implement, please refer to the documentation.


| Quantum Library              | Simulator | Real Machine |
|---------------------|-----------|--------------|
|<p align="center">[pennylane](https://github.com/PennyLaneAI/pennylane)</p>|<p align="center">✅</p>|<p align="center">❌</p>|
|<p align="center">[Qulacs](https://github.com/qulacs/qulacs)</p>           |<p align="center">❌</p>|<p align="center">❌</p>|
|<p align="center">[Qiskit](https://github.com/Qiskit/qiskit)</p>           |<p align="center">❌</p>|<p align="center">❌</p>|
|<p align="center">[Cirq](https://github.com/quantumlib/Cirq)</p>           |<p align="center">❌</p>|<p align="center">❌</p>|


## Installation
<!--
QXMT is tested and supported on 64-bit systems with:
- Python 3.11
- macOS 14.6.1 or later
-->

You can install QXMT with Python's pip package manager:
```bash
pip install qxmt
```

When installing QXMT, you have the option to enable the LLM functionality. By default, it is not installed. By enabling the LLM feature, you can automatically generate experiment summaries based on code differences. If needed, please install it using the following command:
```bash
pip install qxmt[llm]
```


## Getting Started

## Documentation

## Contributing
