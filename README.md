# odenet

ODENetを用いた機械学習システムのテスト

## Description

常微分方程式を用いたニューラルネットワークモデルであるODENetを用いた機械学習、特に教師あり学習を行う。ネットワークの学習および、データの予測、学習中の学習曲線の表示などを行う。

## Requirement

* Python 3.6.5
* python-dotenv 0.15.0
* tqdm 4.56.0
* numpy 1.19.5
* pandas 1.1.5
* seaborn 0.11.1
* scikit-learn 0.24.1
* dash 1.19.0

## Installation

```bash
git clone https://github.com/aaa-yt/odenet.git
cd odenet
pip install -r requirement.txt
```

## Data

* config/parameter.conf : ハイパーパラメータの設定ファイル。

```conf
[MODEL]
input_dimension = 1
output_dimension = 1
maximum_time = 1.0
weights_division = 100
function_type = sigmoid
initializer_type = zero
initializer_parameter = 1.0
solver_type = euler

[TRAINER]
loss_type = MSE
optimizer_type = SGD
learning_rate = 0.01
momentum = 0.9
decay = 0.99
decay2 = 0.999
regularizer_type = None
regularizer_rate = 0.0
epoch = 1000
batch_size = 1
is_accuracy = 0
save_step = 1
```

* data/processed/data.json : 教師データのファイル。

```json
{
    "Train": {
        "Input": (n_train, input_dimension)次元の配列,
        "Output": (n_train, output_dimension)次元の配列
    },
    "Validation": {
        "Input": (n_validation, input_dimension)次元の配列,
        "Output": (n_validation, output_dimension)次元の配列
    },
    "Test": {
        "Input": (n_test, input_dimension)次元の配列,
        "Output": (n_test, output_dimension)次元の配列
    }
}
```

* model/model.json : modelのパラメータのファイル。
* logs/main.log : logファイル。

## Usage

### Training

* ODENetの学習を行う。
* config/parameter.confおよびdata/processed/data.jsonを用意する。
* 以下のコマンドを実行する。

```bash
python src/run.py train
```

### Predict

* 学習済みODENetを用いてデータの予測を行う。
* config/parameter.conf、data/processed/data.jsonおよびmodel/model.jsonを用意する。
* 以下のコマンドを実行する。

```bash
python src/run.py predict
```

### Visualize

* 学習中の学習曲線の表示を行う。
* 以下のコマンドを実行する。

```bash
python src/run.py visualize
```

* http://127.0.0.1:8050/ にアクセスし、学習曲線および、パラメータを表示する。
* http://127.0.0.1:8050/data にアクセスし、教師データの学習の様子を表示する。

## Note

* Windows10にて動作確認済み。
* visualizeの http://127.0.0.1:8050/data はデータの種類によっては表示されない場合があります。

## Author

* Yuto
* mail to: ay.futsal.univ@gmail.com
* Twitter: https://twitter.com/aaa_ytooooo
