# event-flow-baselines

イベントカメラの光フロー推定で、**「過去のイベントをどこに持たせるか(入力表現か、ネットワークの状態か)」**を比べるためのベースライン実験一式です。Hebbian機構を入れる前の参照表を作ることが目的です。

- 土台: [tudelft/event_flow](https://github.com/tudelft/event_flow)(Hagenaars et al., NeurIPS 2021、MITライセンス)。履歴ごと取り込み、取り込み時点を `upstream-base` ブランチとして残しています。`git diff origin/upstream-base` でこのリポジトリでの変更が全部見えます。元のREADMEは [docs/README_event_flow.md](docs/README_event_flow.md) です。
- 損失: contrast maximization(Gallego et al., CVPR 2018)に基づく自己教師あり損失。元コードのまま変更していません。
- 対象モデルは次の3つの出典に限定しています: Hagenaars 2021、Gallego 2018、EV-LayerSegNet([論文](https://arxiv.org/abs/2506.06596) / [コード](https://github.com/Fulmen67/event_segmentation))。

## 現在の状態

| 段階 | 内容 | 状態 |
|---|---|---|
| 0 | コードが今の環境(PyTorch 2系)で動くか | 合成データのスモークテストは通過(CPU)。**Izar上では未実行** |
| 1 | 対象6モデルの配布済み重みで論文Table 1を再現 | スクリプト準備済み。**データ未取得・未実行** |
| 2 | 対象6モデルを再学習して再現 | config 6本を生成済み。**未実行** |
| 追加 | 記憶なしの4条件、追加シード、入力窓の長さ、AffineObjects(論文にない。再現の後) | configのみ。**未実行** |

実データでの学習・評価は一切行っていません。数値はまだ1つもありません。

## 再現の対象(6モデル)

論文の14モデルのうち、**過去の持たせ方だけが違うANNの6モデル**に絞っています。Hebbian機構と直接比べる相手になるものです。

| 記憶の持ち方 | EV-FlowNet系 | FireNet系(軽量) | コード上のクラス |
|---|---|---|---|
| ConvGRU(論文の主モデル) | EV-FlowNet | FireNet | `RecEVFlowNet` / `FireNet` |
| 素のRNN | RNN-EV-FlowNet | RNN-FireNet | `RNNRecEVFlowNet` / `RNNFireNet` |
| リーク(状態を持つニューロン) | Leaky-EV-FlowNet | Leaky-FireNet | `LeakyRecEVFlowNet` / `LeakyFireNet` |

- SNNの8モデル(LIF、ALIF、PLIF、XLIF)は対象外です。配布済みの重みは手元に来るので、評価だけなら `MODELS` にクラス名を足せば加えられます。再学習の設定は用意していません。
- さらに減らすなら、EV-FlowNet系の3つだけでも記憶の比較は成立します。

## 論文(Hagenaars et al. 2021)との対応

学習と評価の手順は論文に合わせています。

| 項目 | 論文 | このリポジトリ |
|---|---|---|
| 学習データ | UZH-FPV(128×128の切り出し140本) | 同じ(配布データをそのまま使用) |
| 検証セット | なし(記載なし) | なし |
| 保存する重み | 記載なし。公開コードは学習損失が最良のエポックを保存 | 公開コードのまま |
| 評価 | MVSECの4系列(`outdoor_day1`、`indoor_flying1`〜`3`)でAEEと外れ値率、dt=1とdt=4 | 同じ(`mvsec_dt1`、`mvsec_dt4`) |
| GTなしの評価 | ECDとHQFでFWL、RSAT | 同じ(`ecd`、`hqf`) |
| 学習回数 | モデルごとに1つの結果。シード数の記載なし | 再現は1シード(seed 0) |

論文にないもの(再現の後に使う追加。下の「追加の実験」):

- 記憶なしの4条件(`evflownet-cnt`、`evflownet-voxel5`、`fireflownet-cnt`、`fireflownet-voxel5`)と追加シード
- 入力窓の長さのスイープ、AffineObjects

## 実行手順(Izar)

すべてリポジトリのルートで実行します。

### 0. セットアップとスモークテスト

```bash
git clone https://github.com/umetyu/event-flow-baselines.git && cd event-flow-baselines
bash cluster/setup_izar.sh
cat logs/smoke-<JOB_ID>.out      # 最後が "All smoke tests passed." なら合格
```

- `umetyu/event_segmentation` の `benchmarks/flow_comparison` と同じvenv(`$HOME/venvs/event-flow-torch251-cu121`、PyTorch 2.5.1+cu121)を再利用し、不足パッケージだけ追加します。別の場所にしたい場合は `EVENT_FLOW_VENV` を設定してください。
- スモークテストは合成データで全モデルの「学習 → 保存 → 評価」を通します。**動くことの確認であり、精度とは無関係です。**

### データの入手

| データ | 置き場所 | 入手元 |
|---|---|---|
| 論文のデータ一式(UZH-FPV学習用、MVSEC、ECD、HQF) | `$EVFLOW_DATA`(既定 `datasets/data/`)の下に `training/` `MVSEC/` `ECD/` `HQF/` | [docs/README_event_flow.md](docs/README_event_flow.md) の "Download datasets"(19.4 GB、展開後94 GB) |
| 論文の学習済みモデル | `mlruns/` | 同 "Download models" |
| AffineObjects | `$AFFINE_DATA`(既定 `$HOME/projects/event_segmentation/dataset`)の下に `train/h5/` `test/recordings_h5/` | 既にIzarにあるものを使用 |

- 配布元はOneDriveの共有リンクです。このリポジトリを準備した環境からは到達できず、**リンクが現在も有効かは未確認**です。ブラウザで落としてからIzarへ転送する必要があるかもしれません。
- 展開に約115 GBの空きが必要です。ホームの容量上限に注意してください。置き場所を変える場合は `export EVFLOW_DATA=/path/to/data` としてからジョブを投入します。

### 1. 論文の再現(学習なし)

```bash
sbatch --export=ALL,SUITES="mvsec_dt1 mvsec_dt4",MODELS=core,REFERENCE=1 cluster/eval.sbatch
cat results/summary.md
```

`mlruns/` にある配布済みの重みのうち対象の6モデルをMVSECで評価し、[experiments/reference/hagenaars2021_table1.yml](experiments/reference/hagenaars2021_table1.yml) の論文値との差を表にします。論文は小数2桁なので、差が0.01 px程度に収まれば再現できています。ここが合わないうちは先へ進まないでください。

- 参照値はPDFから機械的に書き起こしたものです。引用する前に論文と照合してください。
- 配布モデルは古いMLflow・PyTorchで保存されています。メタデータが読めない場合でも重みを探して読むようにし、**重みが見つからないときは評価を中止します**(元コードは黙って未学習の重みで評価を続けます)。この経路は合成データで模擬して確認しましたが、実物では未確認です。

### 2. 再学習して再現(6モデル)

```bash
# まず1本だけ流して1エポックの時間を確認
bash cluster/submit_train.sh configs/exp/s2_gru-evflownet-cnt_w1000_seed0.yml
# 残りの5本
bash cluster/submit_train.sh configs/exp/s2_{rnn,leaky}-evflownet-cnt_w1000_seed0.yml \
                             configs/exp/s2_{gru,rnn,leaky}-firenet-cnt_w1000_seed0.yml
# 学習後、Table 1と比較
sbatch --export=ALL,SUITES="mvsec_dt1 mvsec_dt4",RUNS="step2",MODELS=core,REFERENCE=1 cluster/eval.sbatch
```

- 条件は論文と同じです(UZH-FPVの学習データ全部、1,000イベントごとに順伝播、10,000イベントごとに損失、128×128、バッチ8、100エポック、Adam 2e-4)。`rnn-evflownet` だけ元コードのコメントに従い学習率5e-5です。
- 1エポックの所要時間は未計測です。最初のジョブのログ(`Epoch 0000 done | ... s`)の秒数×100が1モデルの所要時間です。3日の制限に収まらないようなら相談してください。**途中から再開する機能はありません**(`--prev_runid` で重みを引き継いだ新規学習は可能)。
- 見積もり(未実測): 1モデル約10時間(6〜20時間)、6モデルで約2 GPU日。FireNet系はもっと短いはずです。
- 再学習の結果は論文値と小数2桁まで一致するとは限りません。論文は1回の学習結果だけを報告しており、初期値やシードが分からないためです。

## 追加の実験(論文にない。再現の後)

### 記憶なしの条件と追加シード

```bash
bash cluster/submit_train.sh configs/exp/s2_{evflownet,fireflownet}-{cnt,voxel5}_w1000_seed0.yml
```

| 入力 | EV-FlowNet系 | FireNet系 | 役割 |
|---|---|---|---|
| カウント | `evflownet-cnt` | `fireflownet-cnt` | 下限(入力にも状態にも過去がない) |
| voxel 5ビン | `evflownet-voxel5` | `fireflownet-voxel5` | 過去を手設計で入力に入れる |

残りのシードは `configs/exp/s2_*_seed1.yml` と `seed2` です。

### 入力窓の長さ

```bash
bash cluster/submit_train.sh configs/exp/s3_*.yml
sbatch --export=ALL,SUITES="mvsec_dt1 mvsec_dt4",RUNS="step3" cluster/eval.sbatch
```

`evflownet-cnt`、`evflownet-voxel5`、`gru-evflownet-cnt` について、順伝播1回あたりのイベント数を2,500、5,000、10,000に増やします(1,000は上の実験に含まれます)。損失の窓は10,000イベントで固定です。「どれだけ過去を入れると良いか」の曲線になり、後でHebbian機構が狙う対象です。

### AffineObjects(初期設定のみ)

```bash
bash cluster/submit_train.sh configs/exp/s4_affine_*.yml
sbatch --export=ALL,SUITES="affine",RUNS="step4" cluster/eval.sbatch
```

- 640×480は128×128の約19倍の画素数なので、窓もその比で広げています(再帰モデルは20,000イベントごとに順伝播、損失は200,000イベントでEV-LayerSegNetの学習窓と同じ)。**私が置いた初期値で、根拠は画素数の比だけです。**
- フル解像度で10回分を逆伝播するため、GPUメモリが足りるかは未確認です。再帰モデルはバッチ2にしてあります。
- 配布データに正解フローは入っていないようです(変換スクリプトがイベントしか書き出していません)。評価はFWLとRSATだけです。
- EV-LayerSegNet本体はこのリポジトリに入れていません(元リポジトリにライセンス表記がないため)。`umetyu/event_segmentation` 側で、同じテストデータ・同じ50,000イベントの窓・`events` モードでFWL/RSATを出して並べてください。

### 結果の見方

```bash
python experiments/collect.py                 # results/runs.csv と results/summary.md を更新
python experiments/collect.py --runs step2
```

- AEE(px)とRSATは小さいほど良く、FWLは大きいほど良い(1より大きければ、フローで補正したほうがイベント画像が鮮明)。
- 実験の一覧は `python experiments/matrix.py --list`。条件を変えるときは生成物ではなく `experiments/matrix.py` を編集して再生成します。

## 元のevent_flowからの変更点

挙動(モデル、損失、データの読み方、評価指標)は変えていません。変えたのは周辺だけです。

- **チェックポイント**: `mlflow.pytorch.log_model` をやめ、同じ場所に `state_dict` を直接保存。MLflowやPyTorchの版に依存しません。元の形式(モジュールごとpickle)も読めます。
- **評価の安全装置**: 重みが見つからなければエラーで止まります。
- **乱数**: `loader.seed` で `random` と `numpy` も固定(元はPyTorchのみ。シャッフルと拡張は固定されていませんでした)。データ系列は名前順に並べてからシャッフルします。MLflowが内部で乱数を消費するため、MLflowの初期化後にもう一度シードを設定しています。同じシードなら損失が一致することをスモークテストで確認しています(CPU上。GPUでのビット単位の一致は未確認)。
- **パス**: `data.path` で環境変数が使えます。データが無い・足りないときは分かるエラーを出します。
- **ログ**: エポックごとに損失と所要時間を1行出力。学習可能パラメータ数を記録。`--run_name` を追加。
- **MLflow**: ローカルのファイル保存に固定(`mlflow<3`)。

## 注意点

- 記憶なし・カウント入力のモデルは、1,000イベントの画像1枚からフローを当てることになり、ほぼ情報がありません。下限として置いています。
- MVSECは系列が少なく差が小さく出ます。seed間のばらつきより小さい差は差として扱わないでください。
