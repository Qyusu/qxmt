.. qxmt documentation master file, created by
   sphinx-quickstart on Wed Jul 31 14:05:39 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to qxmt's documentation!
================================

QXMT is an open-source experiment management tool for quantum machine learning. The development focus is on low-cost experiment management and ensuring reproducibility. For low-cost management, the goal is to minimize the implementation code required for managing experiments, allowing developers and researchers to focus solely on new experimental implementations. To ensure reproducibility, QXMT manages experimental information as configuration file, enabling not only the original developer but also collaborators to reproduce the same experimental results without investing significant time.

QXMT provides several datasets, machine learning models, and visualization methods to facilitate experiment management. By combining these, users can not only avoid developing the entire workflow themselves but also ensure that many people can evaluate their experiments based on the same standards. These default features will be continuously expanded in future development.


Contents
==================
.. toctree::
   :maxdepth: 1

   qxmt
   tutorials/tutorial_top.md

Limitation
==================
QXMT is newly released, and the available features are still limited. The quantum libraries and devices that have been tested are listed below. For future development plans, please refer to the roadmap. Even in environments not listed below, you can manage experiments by implementing according to the interfaces provided by QXMT. For details on how to implement, please refer to the documentation.

+-------------------------------------------------------------------+------------+--------------+
| Quantum Library                                                   | Simulator  | Real Machine |
+-------------------------------------------------------------------+------------+--------------+
| `pennylane <https://github.com/PennyLaneAI/pennylane>`_           |    ✅      |     ❌       |
+-------------------------------------------------------------------+------------+--------------+
| `Qulacs <https://github.com/qulacs/qulacs>`_                      |    ❌      |     ❌       |
+-------------------------------------------------------------------+------------+--------------+
| `Qiskit <https://github.com/Qiskit/qiskit>`_                      |    ❌      |     ❌       |
+-------------------------------------------------------------------+------------+--------------+
| `Cirq <https://github.com/quantumlib/Cirq>`_                      |    ❌      |     ❌       |
+-------------------------------------------------------------------+------------+--------------+


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
