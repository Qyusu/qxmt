# MNISTデータセットに対して独自のカスタム関数を利用した実践的なケース

今回のチュートリアルでは、より実践的な例としてMNISTを用いたカスタム機能について紹介します。QXMTの利用が初めての場合は、「[デフォルトのデータセットとモデルのみを利用したシンプルなケース](./default_simple.md)」で全体像を掴むことから始めることをおすすめします。


## 1. データセットの準備
まず、実験管理を始めるにあたり利用するデータセットをダウンロードします。今回はMNISTを利用します。
主要なデータセットについては、`scikit-learn`の[fetch_openml](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html)メソッドを活用すると容易にダウンロードすることができます。

今回のチュートリアルでは実験に利用するデータセットを自身でダウンロードしていますが、QXMTではconfigファイル経由で主要なデータセットを[OpenML](https://www.openml.org/)からダウンロードする機能もあります。詳細はOpenMLを活用したチュートリアルを参照して下さい。

``` python
import numpy as np
from sklearn import datasets

# Load MNIST784 dataset
digits_dataset = datasets.fetch_openml("mnist_784")
print(f"Dataset Shape: {digits_dataset.data.shape}")
print(f"Unique Label Number: {len(np.unique(digits_dataset.target))}")

# output
# Dataset Shape: (70000, 784)
# Unique Label Number: 10

# convert to numpy array
X = digits_dataset.data.to_numpy()
y = digits_dataset.target.to_numpy()

# save dataset on local environment
np.save("../data/mnist_784/dataset_X.npy", X)
np.save("../data/mnist_784/dataset_y.npy", y)
```

## 2. カスタム機能の実装
QXMTでは、大きく以下の5つをカスタム機能として利用者が独自に定義することができます。

- **Dataset**: 読み込んだデータセットのフィルタリングや外れ値除去等の前処理 (`raw_preprocess_logic`)と離散化や次元圧縮等の変形処理 (`transform_logic`)を定義することができます。
- **Feature Map**: 独自の特徴マップを関数や量子回路として定義することができます
- **Kernel**: 独自のカーネル関数を定義することができます
- **Model**: 独自の量子機械学習モデルを定義することができます
- **Evaluation**: 独自の評価指標を定義し、各実験のログとして管理することができます

このチュートリアルでは、利用頻度の高い`Dataset`、`Feature Map`、`Evaluation`の3つを独自実装しQXMTで実験管理する方法を紹介します。その他の項目についても同様に実装・呼び出し・管理することができるため、ぜひ各自で挑戦してみてください。

### 2.1 データセットの処理プロセスを独自定義
データセットについては、前処理ロジックと変形ロジックの2つを独自に定義することができます。

まずは、前処理のロジックから実装していきます。
ここでは全10クラスあるMNISTのデータセットから特定のラベルのみに絞り込む処理を実装しています。また、量子カーネルの計算量は大きく、サンプル数が大きなデータセットでは試行錯誤が困難なため、データのサンプル数を絞り込む機能も追加しています。

``` python
# File: your_project/custom/raw_preprocess_logic.py

import numpy as np


def sampling_by_each_class(
    X: np.ndarray, y: np.ndarray, n_samples: int, labels: list[int]
) -> tuple[np.ndarray, np.ndarray]:

    y = np.array([int(label) for label in y])
    indices = np.where(np.isin(y, labels))[0]
    X, y = X[indices][:n_samples], y[indices][:n_samples]

    return X, y
```

次にデータの変形ロジックを実装していきます。
ここではPCAを用いて入力データの次元圧縮を行っています。MNISTは、一つの画像が784次元で構成されているため必要なqubit数が膨大となります。そのため、利用している環境で計算可能なサイズまで引数`n_components`で圧縮します。圧縮する次元数は、config経由で設定できるように引数として設定しておきます。

``` python
# File: your_project/custom/transform_logic.py

from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def dimension_reduction_by_pca(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val) if X_val is not None else None
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=n_components)
    pca.fit(X_train_scaled)
    X_train_pca = pca.transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled) if X_val_scaled is not None else None
    X_test_pca = pca.transform(X_test_scaled)

    return X_train_pca, y_train, X_val_pca, y_val, X_test_pca, y_test
```

### 2.2 特徴マップを独自定義
QXMTでは、Rotation Gateで構築されたFeature MapやZZFeatureMapのような基本的なものはデフォルトで用意されており、configで指定することでそのまま使用することができます。しかし、実務や研究ではより複雑なFeature Mapを利用する機会が多いと思います。そこで、独自のFeature Mapを実装し`config`経由で呼び出すことができます。この方法を用いることで、Feature Mapの設計などの実験も行いやすくなります。

このチュートリアルでは、ZZFeatureMapを適用した後、XXFeatureMapを適用するPennylaneの量子回路を特徴マップとして実装しました。独自のFeature Mapを定義する際には、抽象クラスである`qxmt.feature_maps.BaseFeatureMap`を継承して実装するようにしてください。そうすることで、QXMT内で互換性を保ったまま様々な機能を利用することができます。都度クラスを作成するのが手間な場合は、KernelクラスにFeatureMapを実装した関数だけを渡す方法もありますので、API Referenceを参照してください。

``` python
# File: your_project/custom/feature_map.py

import numpy as np
import pennylane as qml

from qxmt.feature_maps import BaseFeatureMap


class CustomFeatureMap(BaseFeatureMap):
    def __init__(self, n_qubits: int, reps: int) -> None:
        super().__init__("pennylane", n_qubits)
        self.reps: int = reps

    def feature_map(self, x: np.ndarray) -> None:
        for _ in range(self.reps):
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
                qml.RZ(x[i], wires=i)
            for i in range(0, self.n_qubits - 1):
                qml.IsingZZ(2 * (np.pi - x[i]) * (np.pi - x[i + 1]), wires=[i, i + 1])

        for _ in range(self.reps):
            for i in range(self.n_qubits):
                qml.RX(x[i], wires=i)
            for i in range(0, self.n_qubits - 1):
                qml.IsingXX(2 * (np.pi - x[i]) * (np.pi - x[i + 1]), wires=[i, i + 1])
```

### 2.3 評価指標を独自定義
この章の最後に独自の評価指標を定義します。デフォルトの指標としてAccuracy, Precision, Recall, F-Measureが用意されていますが、それぞれの目的に応じて追加の評価指標が必要になることも多々あると思います。その際には、ここで紹介する方法でデフォルトの指標と同様に実験管理を行うことができます。

今回のチュートリアルでは、追加の評価指標として特異度を定義します。独自の評価指標を定義する際には、`qxmt.evaluation.BaseMetric`を継承したクラスとして定義し、`evaluate`メソッドに評価値を計算するロジックを実装します。

``` python
# File: your_project/custom/evaluation.py

from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix

from qxmt.evaluation.metrics import BaseMetric


class CustomMetric(BaseMetric):
    def __init__(self) -> None:
        super().__init__("specificity")

    @staticmethod
    def evaluate(actual: np.ndarray, predicted: np.ndarray, **kwargs: Any) -> float:
        tn, fp, fn, tp = confusion_matrix(actual, predicted).ravel()
        return tn / (tn + fp)
```

ここまででデータセット、特徴マップ、評価指標に関わる独自ロジックの実装が完了しました。今回実装したロジックはいくつか引数を取ることができます。それらはRunの`config`経由で設定することができ、異なる条件での実験も容易に実行できるようになっています。詳細については第3章で紹介します。

## 3. Runのconfig設定
ここでは実装した独自のメソッドを利用するために`config`で設定が必要な項目について紹介します。
追加で設定が必要な箇所には`[SETUP]`タグを付与しています。主な設定方法は、独自ロジックを実装したモジュールのパスと関数またはクラス名を指定します。それぞれで利用するパラメータについても、`params`配下に記載することで実行時に引数に渡すことができます。

```yaml
# File: your_project/configs/custom.yaml

description: "Configuration file for the custom MNIST case"

global_settings:
  random_seed: &global_seed 42
  task_type: "classification"

dataset:
  file: # [SETUP] full path or relative path from the root of the project
    data_path: "data/mnist_784/dataset_X.npy"
    label_path: "data/mnist_784/dataset_y.npy"
    label_name: null
  random_seed: *global_seed
  split:
    train_ratio: 0.8
    validation_ratio: 0.0
    test_ratio: 0.2
    shuffle: true
  features: null
  raw_preprocess_logic: # [SETUP] your logic path and parameter
    module_name: "your_project.custom.raw_preprocess_logic"
    implement_name: "sampling_by_each_class"
    params:
        n_samples: 100
        labels: [0, 1]
  transform_logic: # [SETUP] your logic path and parameter
    module_name: "your_project.custom.transform_logic"
    implement_name: "dimension_reduction_by_pca"
    params:
        n_components: 2

device:
  platform: "pennylane"
  name: "default.qubit"
  n_qubits: 2
  shots: null

feature_map: # [SETUP] your logic path and parameter
  module_name: "your_project.custom.feature_map"
  implement_name: "CustomFeatureMap"
  params:
    reps: 2

kernel:
  module_name: "qxmt.kernels.pennylane"
  implement_name: "FidelityKernel"
  params: {}

model:
  name: "qsvc"
  file_name: "model.pkl"
  params:
    C: 1.0
    gamma: 0.05

evaluation: # [SETUP] your logic path
  default_metrics:
  - "accuracy"
  - "precision"
  - "recall"
  - "f1_score"
  custom_metrics:
  - {"module_name": "your_project.custom.evaluation", "implement_name": "CustomMetric"}
```

## 4. 実験の実行と評価
では、最後にQXMTの実験管理インスタンスを生成し、上記で定義した`config`ファイルを渡してRunを実行します。

もし、実行時に`ModuleNotFound`等のエラーが出る場合はPythonの実行pathを確認し、再実行して下さい。
例えば、カスタムモジュールを`/root/hoge/your_project/custom/raw_preprocess_logic.py`に実装している場合は`sys.path.append("/root/hoge")`コマンドでpathを追加することができます。

``` python
import qxmt

# initialize experiment for custom tutorial
exp = qxmt.Experiment(
    name="custom_tutorial",
    desc="A custome experiment for MNIST dataset",
    auto_gen_mode=False,
).init()

# execute run of custom method
config_path = "../configs/custom.yaml"
artifact, result = exp.run(config_source=config_path)

# check evaluation result
metrics_df = exp.runs_to_dataframe()
metrics_df.head()
# output
#       run_id  accuracy  precision  recall  f1_score
#	run_id	accuracy	precision	recall	f1_score	specificity
# 0	     1	    0.6	         0.66	  0.66	    0.66	       0.5
```

独自定義した評価指標も含めて、実験の結果を可視化します。

``` python
from qxmt.visualization import plot_metrics_side_by_side

# get run result as dataframe
df = exp.runs_to_dataframe()

# add your custom metrics on metrics list
plot_metrics_side_by_side(
  df=df,
  metrics=["accuracy", "recall", "precision", "f1_score", "specificity"],
  run_ids=[1],
  save_path=exp.experiment_dirc / "side_by_side.png"
  )
```
<img src="../../_static/images/tutorials/custom/side_by_side.png" alt="評価指標の比較" title="評価指標の比較">

---

### バージョン情報
| Environment | Version |
|----------|----------|
| document | 2024/10/22 |
| QXMT| v0.3.1 |
