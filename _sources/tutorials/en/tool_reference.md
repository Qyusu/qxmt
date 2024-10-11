# Tool Reference


## Load Datasets with OpenML
In QXMT, the [OpenML](https://www.openml.org/) API can be utilized via a config file to simplify dataset preparation. This section explains the process from searching for the desired dataset to setting it up in the config file for use.

Let's assume you want to use the "Fashion-MNIST" dataset for an experiment. First, search for the relevant dataset on [OpenML's dataset search page](https://www.openml.org/search?type=data&sort=runs&status=active) and navigate to its detail page. On the detail page, you will be able to see information about the dataset, as shown below. From this information, take note of the "ID" and add it to the config (in this case, the ID is "40996").

<img src="../../_static/images/tutorials/tools/openml_detail.png" alt="Dataset Detail Page" title="Dataset Detail Page" width="50%">

Based on the information from the search, configure the Run settings in the config as shown below. (Only the necessary parts are extracted for clarity)

``` yaml
dataset:
  type: "openml"
  openml:
    name: "Fashion-MNIST"
    id: 40996
    return_format: "numpy"
    save_path: "data/openml/Fashion-MNIST/dataset.npz"
```

- **type**: Specifies the dataset type to be used in QXMT. Set this to `openml` when using a dataset from OpenML.
- **openml.name**: The name of the dataset on OpenML.
- **openml.id**: The ID of the dataset on OpenML.
- **openml.return_format**: Specifies the format of the dataset. It can be either pandas or numpy.
- **openml.save_path**: Specifies the path to save the downloaded dataset. If set to `null`, the dataset will not be saved.

Both `openml.name` and `openml.id` can be used individually. If only `openml.name` is specified, the dataset will be searched internally via the API. Since `openml.id` uniquely identifies the dataset, it is recommended to use this value. If both `openml.name` and `openml.id` are set, the value of `openml.id` will take precedence.

## Using Projected Kernel
In kernel-based machine learning models, such as QSVM, there are various algorithms available for kernel computation. This section explains how to configure the settings when using the `Projected Kernel` [1].

A simple Projected Kernel is expressed by the following equation, where the scale parameter `γ` and the method of projecting quantum states into classical states can be specified for distance computation.

<img src="../../_static/images/tutorials/tools/projected_kernel_definition.png" alt="Definition of Projected Kernel" title="Definition of Projected Kernel" width="50%">

*Source: Equation (9) from “Power of data in quantum machine learning” [1]*

The kernel-related settings are managed collectively under the `kernel` section in the config file. Here, you can specify the type of kernel to use and its parameters. To use the Projected Kernel, configure the settings in the config file as follows.

``` yaml
kernel:
  module_name: "qxmt.kernels.pennylane"
  implement_name: "ProjectedKernel"
  params:
    gamma: 1.0
    projection: "xyz_sum"
```

- **module_name**: Specifies the name of the module where the kernel method is implemented. In this case, use the one provided by QXMT as indicated above.
- **implement_name**: Specifies the class name that implements the kernel method. In this case, use the one provided by QXMT as indicated above.
- **params.gamma**: The scale parameter for kernel computation.
- **params.projection**: The method for projecting quantum states into classical states (available options are "x", "y", "z", "xyz", and "xyz_sum").


[1] Hsin-Yuan Huang, Michael Broughton, Masoud Mohseni, Ryan Babbush, Sergio Boixo, Hartmut Neven, and Jarrod R McClean, “Power of data in quantum machine learning,” [Nature Communications 12, 1–9 (2021)](https://www.nature.com/articles/s41467-021-22539-9).

---

### Version Information
| Environment | Version |
|----------|----------|
| document | 2024/10/10 |
| QXMT| v0.2.3 |
