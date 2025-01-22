.. qxmt documentation master file, created by
   sphinx-quickstart on Wed Jul 31 14:05:39 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to QXMT documentation!
================================

QXMT is an open-source tool designed for managing experiments in quantum machine learning. Its primary focus is on minimizing the cost of managing experiments and ensuring reproducibility. To reduce costs, QXMT aims to minimize the amount of implementation code needed for experiment management, allowing developers and researchers to focus on implementing new experiments. For reproducibility, QXMT manages experimental configurations via configuration files, enabling not only the original developers but also collaborators to easily reproduce the same results without significant additional effort.

QXMT includes a variety of datasets, machine learning models, and visualization methods to support experiment management. By using these components together, users can avoid the need to develop entire workflows from scratch while also ensuring that experiments can be evaluated under consistent conditions. These default features will continue to be expanded in future updates.

Installation
==================
To install QXMT package, you can use `pip`, the Python package installer. Run the following command in your terminal:

.. code-block:: bash

   pip install qxmt

Make sure you have Python 3.10 or more recent version installed. If you need LLM mode or more detailed instructions, please refer to `the official PyPI page <https://pypi.org/project/qxmt/>`__.


Limitation
==================
QXMT is a newly released tool, and its features are currently limited. We are currently developing with the assumption of usage in `PennyLane <https://github.com/PennyLaneAI/pennylane>`__. PennyLane allows the use of simulators from other quantum libraries as plugins. Please refer to the `documentation <https://qyusu.github.io/qxmt/tutorials/en/tool_reference.html#specifying-the-simulator>`__ for details on how to use them. Additionally, as of now, support for real quantum hardware is limited to `IBMQ <https://quantum.ibm.com/>`__ and `Amazon Braket <https://aws.amazon.com/braket>`__.

Even if your environment is not listed, you can still manage experiments by implementing according to the interfaces provided by QXMT. For details on how to implement these interfaces, please refer to the documentation.

Contents
==================
.. toctree::
   :maxdepth: 1

   qxmt
   tutorials/en/tutorial_top.md

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
