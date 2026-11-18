# event-flow-baselines

イベントカメラの光フロー推定で、**「過去のイベントをどこに持たせるか(入力表現か、ネットワークの状態か)」**を比べるためのベースライン実験一式です。Hebbian機構を入れる前の参照表を作ることが目的です。

- 土台: [tudelft/event_flow](https://github.com/tudelft/event_flow)(Hagenaars et al., NeurIPS 2021、MITライセンス)。履歴ごと取り込み、取り込み時点を `upstream-base` ブランチとして残しています。`git diff origin/upstream-base` でこのリポジトリでの変更が全部見えます。元のREADMEは [docs/README_event_flow.md](docs/README_event_flow.md) です。
- 損失: contrast maximization(Gallego et al., CVPR 2018)に基づく自己教師あり損失。元コードのまま変更していません。
- 対象モデルは次の3つの出典に限定しています: Hagenaars 2021、Gallego 2018、EV-LayerSegNet([論文](https://arxiv.org/abs/2506.06596) / [コード](https://github.com/Fulmen67/event_segmentation))。

## 現在の状態

| 段階 | 内容 | 状態 |
|---|---|---|
| 0 | コードが今の環境(PyTorch 2系)で動くか | Izar(PyTorch 2.5.1 + CUDA)で学習・保存・評価が通ることを確認 |
| 1 | 再帰の6モデル: 配布済みの重みを評価して論文Table 1と比較(再学習しない) | **実行済み(2026-11-18)。5モデルは論文と一致、RNN-EV-FlowNetだけ不一致。** [結果](docs/results_pretrained_2026-11-18.md) |
| 2 | [56]型の1モデル: UZH-FPVで学習して評価 | **学習中**(2026-11-18投入、1エポック約145秒) |
| 追加 | 記憶なしの4条件、追加シード、入力窓の長さ、AffineObjects(論文にない。再現の後) | configのみ。**未実行** |

数値があるのは段階1だけです。それ以外はまだ結果がありません。

## 対象(7モデル)

論文のTable 1から、**過去の持たせ方が違うANN**に絞っています。Hebbian機構と直接比べる相手になるものです。

**過去をネットワークの状態に持つ6モデル**(論文の著者が学習したもの。入力は時間情報のないカウント):

| 記憶の持ち方 | EV-FlowNet系 | FireNet系(軽量) | コード上のクラス |
|---|---|---|---|
| ConvGRU(論文の主モデル) | EV-FlowNet | FireNet | `RecEVFlowNet` / `FireNet` |
| 素のRNN | RNN-EV-FlowNet | RNN-FireNet | `RNNRecEVFlowNet` / `RNNFireNet` |
| リーク(状態を持つニューロン) | Leaky-EV-FlowNet | Leaky-FireNet | `LeakyRecEVFlowNet` / `LeakyFireNet` |

**過去を入力に持つ1モデル**(論文では文献値として載っている行):

| 論文の行 | 中身 | ここでの設定 |
|---|---|---|
| EV-FlowNet\* [56](Zhu et al., CVPR 2019) | 再帰のないEV-FlowNet。入力は時間方向9ビンのvoxel | `evflownet-voxel9`(クラス `EVFlowNet`) |

- 番号なしの「EV-FlowNet」は著者がConvGRUを足した再帰版で、[56]とは別物です。両方を対象に入れています。
- **[56]の行は再現ではなく「同じ土俵での対応物」です。** 論文によれば\*付きの文献モデルはMVSECの系列で学習されており、ここではUZH-FPVとこのリポジトリの損失で学習します。数値が[56]の値に一致することは期待できません。
- [56]に合わせたのは入力だけです: 9ビン、順伝播1回で損失1回、1画素あたりのイベント数(原論文は256×256で30,000イベント → 128×128では7,500)。
- [56]そのものの再現(MVSECの `outdoor_day2` で学習、イベント数固定での評価)は用意していません。`outdoor_day2` が配布データにあるかも未確認です。
- [54](Zhu et al., RSS 2018)は学習に輝度画像を使う別の方式なので対象外です。参照値だけ記録してあります。
- SNNの8モデル(LIF、ALIF、PLIF、XLIF)は対象外です。配布済みの重みは手元に来るので、評価だけなら `MODELS` にクラス名を足せば加えられます。再学習の設定は用意していません。
- さらに減らすなら、EV-FlowNet系の3つと `evflownet-voxel9` の4つでも比較は成立します。

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

方針: **再帰の6モデルは再学習せず、論文の配布済みの重みを評価します。学習するのは[56]型(`evflownet-voxel9`)の1本だけで、UZH-FPVで学習します。**

### コードをIzarに置く

リポジトリは非公開なので、Izarでは `git clone` の代わりに、Macから送ったバンドル(1ファイル)を使います。

```bash
# Macで
scp -i ~/.ssh/id_ed25519 ~/codes_github/event-flow-baselines.bundle umetsu@izar.hpc.epfl.ch:~/
ssh -i ~/.ssh/id_ed25519 umetsu@izar.hpc.epfl.ch
# Izarで(初回)
git clone ~/event-flow-baselines.bundle ~/projects/event-flow-baselines
# Izarで(更新時。新しいバンドルを同じ場所に送り直してから)
cd ~/projects/event-flow-baselines && git pull
```

リポジトリを公開にすれば、バンドルなしで `git clone https://github.com/umetyu/event-flow-baselines.git` と `git pull` が使えます。

### 初回の一括実行

```bash
cd ~/projects/event-flow-baselines && bash cluster/bootstrap_izar.sh
```

このコマンドは順に次のことをします。

1. **データの配置**: `/scratch/izar/$USER/uzh_mvsec_partition_seed0/data` の `author_training/` をリンクし、`MVSEC.zip` を展開します。置き場所は `/scratch/izar/$USER/event-flow-baselines-data` で、元のフォルダには書き込みません。場所が違う場合は引数で元フォルダを渡します。
2. **環境**: `umetyu/event_segmentation` の `benchmarks/flow_comparison` と同じvenv(`$HOME/venvs/event-flow-torch251-cu121`、PyTorch 2.5.1+cu121)を再利用し、不足パッケージだけ追加します。
3. **データの中身の報告**: ファイル数、解像度、GTの形、`mlruns/` にある重みを `results/data_report.txt` に書き出します。想定と違う点があれば、学習も評価も投入せずに止まります。
4. **ジョブの投入**: スモークテスト。それが通った場合だけ、配布済みの重みの評価(重みが `mlruns/` にあるとき)と、`evflownet-voxel9` の学習、学習後の評価が続けて動きます。

進み具合は `bash cluster/status.sh` で確認できます。出力をそのまま貼ってもらえれば状況が分かります。

- スモークテストは合成データで全モデルの「学習 → 保存 → 評価」を通します。**動くことの確認であり、精度とは無関係です。**
- もう一度実行すると、学習ジョブは二重に投入されます。先に `squeue -u $USER` を確認してください。

### 配布済みの重み(再帰の6モデル)

論文の学習済みモデルは [docs/README_event_flow.md](docs/README_event_flow.md) の "Download models" から入手し、実行フォルダが `mlruns/<実験>/<モデル>/` になるように展開します。初回の一括実行より後に置いた場合は、次で評価します。

```bash
sbatch --export=ALL,SUITES="mvsec_dt1 mvsec_dt4",MODELS=core,REFERENCE=1 cluster/eval.sbatch
cat results/summary.md
```

対象のモデルをMVSECで評価し、[experiments/reference/hagenaars2021_table1.yml](experiments/reference/hagenaars2021_table1.yml) の論文値との差を表にします。論文は小数2桁なので、差が0.01 px程度に収まれば再現できています。

- 配布元はOneDriveの共有リンクです。このリポジトリを準備した環境からは到達できず、**リンクが現在も有効かは未確認**です。
- 参照値はPDFから機械的に書き起こしたものです。引用する前に論文と照合してください。
- 配布モデルは古いMLflow・PyTorchで保存されています。メタデータが読めない場合でも重みを探して読むようにし、**重みが見つからないときは評価を中止します**(元コードは黙って未学習の重みで評価を続けます)。この経路は合成データで模擬して確認しましたが、実物では未確認です。
- [56]型の重みが配布物に含まれるかは未確認です。

### [56]型の学習(1本だけ)

初回の一括実行が投入します。手で流す場合:

```bash
SBATCH_ARGS="--time=1-00:00:00" bash cluster/submit_train.sh configs/exp/s2_evflownet-voxel9_w7500_seed0.yml
sbatch --export=ALL,SUITES="mvsec_dt1 mvsec_dt4",RUNS="s2_evflownet-voxel9_w7500_seed0",REFERENCE=1 cluster/eval.sbatch
```

- 学習データはUZH-FPVの全部、損失と最適化はこのリポジトリのもの(128×128、バッチ8、100エポック、Adam 2e-4)。入力だけ[56]に合わせています(9ビン、順伝播1回で損失1回、7,500イベント)。
- 1エポックの順伝播は約1,170回(再帰モデルは8,750回)です。所要時間は未計測で、ログの `Epoch 0000 done | ... s` の秒数×100が全体の時間です。制限時間は1日にしてあります。**途中から再開する機能はありません。**
- 比較表では[56]の行に `*` が付き、文献値として表示されます。学習データが違うので一致は期待できず、一致の判定にも含めません。

### 再帰の6モデルを再学習する場合(今は行わない)

設定は残してあります。1つ約10時間(6〜20時間、未実測)の見積もりです。

```bash
bash cluster/submit_train.sh configs/exp/s2_{gru,rnn,leaky}-evflownet-cnt_w1000_seed0.yml \
                             configs/exp/s2_{gru,rnn,leaky}-firenet-cnt_w1000_seed0.yml
```

条件は論文と同じです(UZH-FPVの学習データ全部、1,000イベントごとに順伝播、10,000イベントごとに損失、128×128、バッチ8、100エポック、Adam 2e-4)。`rnn-evflownet` だけ元コードのコメントに従い学習率5e-5です。再学習の結果は論文値と小数2桁まで一致するとは限りません。

### そのほかのデータ

| データ | 置き場所 | 入手元 |
|---|---|---|
| ECD、HQF(GTなしの評価。任意) | `$EVFLOW_DATA` の下に `ECD/` `HQF/` | [docs/README_event_flow.md](docs/README_event_flow.md) の "Download datasets" |
| AffineObjects(追加の実験用) | `$AFFINE_DATA`(既定 `$HOME/projects/event_segmentation/dataset`)の下に `train/h5/` `test/recordings_h5/` | 既にIzarにあるものを使用 |

## 追加の実験(論文にない。再現の後)

### 記憶なしの条件と追加シード

```bash
bash cluster/submit_train.sh configs/exp/s2_{evflownet,fireflownet}-{cnt,voxel5}_w1000_seed0.yml
```

| 入力 | EV-FlowNet系 | FireNet系 | 役割 |
|---|---|---|---|
| カウント | `evflownet-cnt` | `fireflownet-cnt` | 下限(入力にも状態にも過去がない) |
| voxel 5ビン | `evflownet-voxel5` | `fireflownet-voxel5` | 過去を手設計で入力に入れる(1,000イベント窓。`evflownet-voxel9` との違いは窓の長さとビン数) |

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
- **乱数**: `loader.seed` で `random` と `numpy` も固定(元はPyTorchのみ。シャッフルと拡張は固定されていませんでした)。データ系列は名前順に並べてからシャッフルします。MLflowが内部で乱数を消費するため、MLflowの初期化後にもう一度シードを設定しています。同じシードなら損失が一致することをスモークテストで確認しています(CPU上)。**GPUでは一致しません**: Izarで同じシードの小さな学習を3回行うと、最初の2エポックの損失が1〜2%ずれました。原因は未特定で、損失の中のイベント加算(GPUでは再現可能でない演算)が有力です。GPUでの比較は、シードを固定しても複数回の学習のばらつきを見る必要があります。
- **パス**: `data.path` で環境変数が使えます。データが無い・足りないときは分かるエラーを出します。
- **ログ**: エポックごとに損失と所要時間を1行出力。学習可能パラメータ数を記録。`--run_name` を追加。
- **MLflow**: ローカルのファイル保存に固定(`mlflow<3`)。

## 注意点

- 記憶なし・カウント入力のモデルは、1,000イベントの画像1枚からフローを当てることになり、ほぼ情報がありません。下限として置いています。
- MVSECは系列が少なく差が小さく出ます。seed間のばらつきより小さい差は差として扱わないでください。
