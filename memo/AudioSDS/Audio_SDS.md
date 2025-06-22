# Score Distillation Sampling for Audio: Source Separation, Synthesis, and Beyond

## 参考
* [Score Distillation Sampling for Audio: Source Separation, Synthesis, and Beyond](https://arxiv.org/abs/2505.04621)

## 1. 概要

学習済の音声ドメイン拡散モデルの知識をうまく使うことで、特殊なデータセットを用いることなく、音声レンダリングモデルのパラメータを調整しようというもの。

論文では特に、衝撃音シミュレーション、FM合成パラメータ、音源分離タスクについて検証している。

注意したいこととして、今回は汎用的なモデルパラメータの学習ではなく、特定のプロンプトに対応するパラメータの最適化を行う。
これにより、テキスト入力によってモデルパラメータをうまく調整できる。
例えば「あらゆる音に対応できるFMシンセサイザー」を作るのではなく「特定の音（プロンプト）に特化したFMシンセサイザー設定」を自動で見つける手法ということ。

### モチベーション

#### 人力でのパラメータ調整工数の削減

従来は音響エンジニアが手動で何時間もかけて調整していたパラメータを、プロンプト一つで自動最適化できる。

* プロンプトドリブン: テキスト入力だけで音響パラメータが自動決定
* 専門知識不要: FMシンセなどの複雑なパラメータを手動調整する必要がない
* カスタマイズ: 各プロンプトに完全に特化したパラメータが得られる

#### 特殊なデータセットの準備を不要に

従来の教師あり学習では、特定の音を学習させようと思うとそれに対応したデータセットを作る必要があったが、Audio-SDSであれば、事前学習済拡散モデルに条件文としてテキストを入力してデノイジングさせるだけで済む。


#### より創造的な探索を可能に

SDSでは、同じテキストプロンプトでもノイズの違いで毎回異なる理想音声が生成される。そのため、従来の教師あり学習と比較して、既存データにない新しい音の組み合わせも可能でより創造的な探索が可能となる。


### ユースケース

1. 映画音響制作
   * 特定のシーンに合わせたインパクト音の生成
   * 環境音の細かな調整
   * プロンプトによる直感的な音響デザイン

2. ゲーム開発
   * キャラクター固有の効果音生成
   * 環境音のバリエーション作成
   * プロンプトによる一貫性のある音響設計

3. 音楽制作
   * FMシンセサイザーのパラメータ最適化
   * 特定の音色への特化
   * プロンプトによる音色の微調整

### 制限事項

* リアルタイム生成の不可能性
* 計算リソースの要件
* プロンプトの品質への依存性



## 2. 技術的詳細

### 基本的な仕組み

次のような流れで音声モデルのパラメータチューニングが進む。

* プロンプト入力: "kick drum, bass, reverb"
* パラメータ初期化: 音声モデルのパラメータをランダム初期化
* 最適化: 事前学習済み拡散モデルを使って、そのプロンプトにマッチする音になるようパラメータを調整
* 結果: そのプロンプト専用にチューニングされた音声モデルパラメータを得る


### プロセスの詳細

#### SDS

SDSでは、事前訓練された拡散モデルの「知識」を、全く異なるパラメトリック表現に効率的に転移する。

信号のレンダリングモデルを $\boldsymbol{g}: \Theta \times \mathcal{C} \to\mathcal{X}$ とする。 $\boldsymbol{\theta}\in\Theta$ をモデルパラメータ（学習対象）、$\boldsymbol{c}\in \mathcal{C}$ をサンプリングされたレンダリングパラメータ、$\boldsymbol{x}\in \mathcal{X}$ をレンダリングされた信号とする。

まず、パラメータ $\boldsymbol{\theta}$ およびレンダリングパラメータ $\boldsymbol{c}$ から信号をレンダリングする。

$$
\begin{align*}
    \boldsymbol{x}=\boldsymbol{g}(\boldsymbol{\theta},\boldsymbol{c})\,.
\end{align*}
$$ 

この信号に対してノイズを付加する。

$$
\begin{align*}
    \boldsymbol{z} &= \alpha_{t^\prime} \boldsymbol{g}(\boldsymbol{\theta},\boldsymbol{c}) + \sigma_{t^\prime}\boldsymbol{\epsilon}\,,\\
    \boldsymbol{\epsilon}&\sim \mathcal{N}(\boldsymbol{0},\boldsymbol{1})\,.
\end{align*}
$$

ここで、 $\alpha_{t^\prime}$ と $\sigma_{t^\prime}$ はそれぞれシグナルとノイズのスケールで、ノイズスケジュールとして与えられる。 $t^\prime$ はタイムステップで $t\in[t^\prime_{\rm min}, t^\prime_{\rm max}]\approx[0,1]$ の値をとる。この値が小さいほどノイズスケールも小さい。

$\boldsymbol{z}$ はランダムサンプリングした $t^\prime$、$\boldsymbol{\epsilon}$ について得られる。

ノイズ予測モデルは $\boldsymbol{\epsilon}_\phi(\boldsymbol{z}, t^\prime, \boldsymbol{p})$ である。
ここで、 $\phi$ は（学習済）拡散モデルパラメータでここでは固定値をとる。 $\boldsymbol{p}$ は拡散モデルへの入力プロンプトを表す。

実際は、より高品質の生成のために、分類器フリーガイダンス (CFG) が用いられる。
$$
\begin{align*}
    \hat{\boldsymbol{\epsilon}}(\boldsymbol{z}, t^\prime, \boldsymbol{p}) 
    = (1+\tau)\boldsymbol{\epsilon}_\phi(\boldsymbol{z}, t^\prime, \boldsymbol{p})
    -\tau\boldsymbol{\epsilon}_\phi(\boldsymbol{z},t^\prime)\,.
\end{align*}
$$
$\tau$ はガイダンスパラメータ。CFGで「プロンプトあり」と「プロンプトなし」の予測の差を拡大することで、テキスト条件の影響を強化している。

レンダリングモデルのパラメータ更新のための損失関数は以下で与えられる: 
$$
\begin{align*}
    \mathcal{L}_{\text{SDS}}(\boldsymbol{\theta}; \boldsymbol{p}) 
    = 
    \mathbb{E}_{t', \boldsymbol{\epsilon}, \boldsymbol{c}}
    \left[\omega(t') \|\hat{\boldsymbol{\epsilon}}_\phi(\boldsymbol{z}(\boldsymbol{\theta},\boldsymbol{c}), t', \boldsymbol{p}) - \boldsymbol{\epsilon}\|^2\right]\,.
\end{align*}
$$

$\omega(t^\prime)$ は時間依存の重みパラメータ。

この損失の勾配は、チェーンルールを用いて次のように表せる:

$$
\begin{align*}
    \nabla_{\boldsymbol{\theta}} \mathcal{L}_{\text{SDS}}(\boldsymbol{\theta}; \boldsymbol{p}) = \mathbb{E}_{t', \boldsymbol{\epsilon}, \boldsymbol{c}}[\omega(t')(\hat{\boldsymbol{\epsilon}}_\phi(\boldsymbol{z}(\boldsymbol{\theta},\boldsymbol{c}), t', \boldsymbol{p}) - \boldsymbol{\epsilon}) \boldsymbol{J}_{\hat{\boldsymbol{\epsilon}}_\phi}(\boldsymbol{z}) \nabla_{\boldsymbol{\theta}} \boldsymbol{z}(\boldsymbol{\theta}, \boldsymbol{c})]\,.
\end{align*}
$$

ここで、 $\boldsymbol{J}$ は拡散モデルのU-Net部分についてのヤコビアン。
しかしながら、このヤコビアン部分のback-propagationは計算コストが高いことや数値的な不安定性を持つことが知られており、計算を単純化するために単位行列で置き換えを行う。なお理論的な正当性もあるらしい([参考](https://arxiv.org/abs/2209.14988)):

$$
\begin{align*}
    u_{\text{SDS}}(\boldsymbol{\theta}; \boldsymbol{p}) = \mathbb{E}_{t', \boldsymbol{\epsilon}, \boldsymbol{c}}[\omega(t')(\hat{\boldsymbol{\epsilon}}_\phi(\boldsymbol{z}(\boldsymbol{\theta},\boldsymbol{c}), t', \boldsymbol{p}) - \boldsymbol{\epsilon}) \nabla_{\boldsymbol{\theta}} \boldsymbol{z}(\boldsymbol{\theta}, \boldsymbol{c})]
\end{align*}
$$

SDSではこの $u_{\text{SDS}}$ をパラメータ更新則としている。

#### Audio-SDS

Audio-SDSでもSDSと同様のアプリーチがなされるが、いくつか変更点が存在する。


##### パラメータ

まず、音声はステレオ（2チャンネル）あることから、オーディオレンダリング関数は

$$
    \begin{align*}
        \boldsymbol{g}_{\text{audio}}: \Theta\to\mathbb{R}^{2\times T}
    \end{align*}\,,
$$

と表され、
出力オーディオ信号は

$$
    \boldsymbol{x}_{\text{audio}}
    =
    \boldsymbol{g}_{\text{audio}}(\theta)\,,
$$

となる。ここで、$T$ はオーディオサンプルの総数を表す。
また、SDSにあったパラメータ $\boldsymbol{c}$ は今回のタスクでは必要ないため使用されない。

##### エンコーダ不安定性

また、音声拡散モデル特有の問題として、エンコーダーの微分が不安定であることがある。
そのため、Audio-SDSではSDSに一部改良を加えたDecorder-SDSを提案、使用している。

その更新則は次の $u^{\text{dec}}_{\text{SDS}}$ で与えられる:

$$
\begin{align*}
    u^{\text{dec}}_{\text{SDS}}(\boldsymbol{\theta}; \boldsymbol{p})
    &=
    \mathbb{E}_{t', \boldsymbol{\epsilon}}[\hat{\boldsymbol{x}}_\phi(\boldsymbol{\theta}, t', \boldsymbol{\epsilon}, \boldsymbol{p})] - \boldsymbol{x}(\boldsymbol{\theta}) \nabla_{\boldsymbol{\theta}} \boldsymbol{x}(\boldsymbol{\theta})\,,\\
    \hat{\boldsymbol{x}}_\phi(\boldsymbol{\theta}, t', \boldsymbol{\epsilon}, \boldsymbol{p})
    & =
    \text{dec}_\phi(\text{denoise}_\phi(\text{noise}(\text{enc}_\phi(\boldsymbol{x}(\boldsymbol{\theta})), t', \boldsymbol{\epsilon}), \boldsymbol{p}))\,.
\end{align*}
$$

ここで、${\hat{\boldsymbol{x}}}_{\phi}$ はデノイズ処理後にデコードされた音声を表す。

また、
$$
\begin{align*}
    \text{enc}_{\phi}(\cdot)&: \text{エンコーダ関数}\,,
    \text{dec}_{\phi}(\cdot)&: \text{デコーダ関数}\,,
    \text{denoise}_{\phi}(\cdot)&: \text{デノイジング関数}\,,
    \text{noise}(\cdot)&: \text{ノイジング関数}\,.
\end{align*}
$$


これにより、潜在音声拡散モデルのエンコーダーを通じた微分を回避し、代わりにデコーダー空間で音声領域での不一致を計算する。

##### スペクトログラムの強調

通常のSDSでは、生成された波形 $x$ と拡散モデルがデコードした波形 $\hat{x}$ の間でノイズ予測誤差を比較し、その差分をパラメータに逆伝播する。
しかし、音声や効果音ではアタック音や急激なスペクトル変化が知覚上とても重要で、時間領域の $\ell_2$ 誤差ではそれらが埋もれてしまいがちになる。
そこでAudio-SDSでは、短時間フーリエ変換（STFT）の振幅スペクトログラム $\mathcal{S}_m(\cdot)$ を、異なる窓幅 ($m=1,\dots,M$) で複数計算し、それらを合算したマルチスケールスペクトログラムを用いて誤差を測っている。

$$
\begin{align*}
    \boldsymbol{u}^{\text{dec}}_{\text{SDS}}(\boldsymbol{\theta}; \boldsymbol{p})
    = \sum_{m=1}^{M}
    \Bigl(
        \mathbb{E}_{t^\prime,\boldsymbol{\epsilon}}
        \bigl[
            \mathcal{S}_{m}\bigl(\hat{x}_{\phi}(\theta,t^\prime,\boldsymbol{\epsilon},\boldsymbol{p})\bigr)
        \bigr]
    - \mathcal{S}_{m}\bigl(\boldsymbol{x}(\boldsymbol{\theta})\bigr)
    \Bigr)\,
    \nabla_{\theta}\mathcal{S}_{m}\bigl(\boldsymbol{x}(\boldsymbol{\theta})\bigr)\,.
\end{align*}
$$

ここで、$S_m(\hat{x})$ は各スケールのスペクトログラムで ${\mathcal{S}_{m}}(x)$ はレンダリング波形を表す。

つまり、各スケールにおけるスペクトログラムとレンダリング波形の差分について全スケール合計を計算し、パラメータに対する勾配 ${\nabla_{\theta}} {\mathcal{S}_{m}}(x)$ と掛け合わせて更新方向を得ている。

この多重スケールの比較により、短い窓では時間分解能を活かして鋭いトランジェントを捉え、高い窓では周波数分解能を活かして細かなスペクトル構造を同時に捉えられるため、
結果として人間が聴いて「自然」で「立体感のある」音響が得られるようになる。



##### マルチステップデノイジング

従来のSDSは、レンダリングされた潜在表現に一度だけノイズを加え、その後に拡散モデルのデノイジングを一回適用して得られた予測ノイズとの誤差をもとにパラメータを更新していた。
しかし、音声信号では短時間に急峻な振幅変化や位相のずれが生じやすいため、一歩のデノイジングだけでは拡散モデルの本来の分布に十分近づかず、更新時にチラつきや不自然なアーチファクト（たとえばチリチリした高周波ノイズ）が生じることがある。

そこでAudo-SDSでは、各イテレーションでサンプリングしたノイズを加えた後に、部分的なDDIMスタイルのサンプリングチェーンを数ステップ（たとえば２～１０ステップ）実行し、段階的に潜在表現をデノイズしながら拡散分布に近づける手法を用いている。
その後、最終ステップで得られたデノイズ済み潜在表現をデコーダに通し、スペクトログラム強調のもとでSDS更新を計算する。
こうすることで、従来の単一ステップに比べて拡散モデルの学習した分布により忠実に沿った更新が可能になり、トランジェントや高周波ディテールがより自然に保持されるとともに、最適化の途中で現れる不安定なノイズ発生が大幅に抑えられる。


## 3. 今回のタスク



「FM合成」「インパクト合成」「音源分離」の3タスクに対してAudioSDSを利用。

### 対象の音声系タスクの詳細

#### FM合成 (FM Synthesis)

モデルの構造:
基本的な古典的FMシンセサイザーを使用。
* 複数のサイン波オシレーター（発振器）で構成。
* 各オシレーターには、音の時間的な変化（立ち上がりや減衰）を制御するためのエンベロープ（Attack/Decay）が存在。

最適化されるパラメータ ( $\boldsymbol{\theta}$ ):
* FMマトリクス $A$
  * 各オシレーターが他のオシレーターの周波数を変調させる際の変調の強さを特徴づける
* 各オシレーターの基本周波数 $\omega$
* 各オシレーターのアタック $\alpha$ とディケイ $\delta$

#### インパクト合成 (Impact Synthesis)

モデルの構造:
物理法則に基づいたモーダル合成モデルを使用。
* オブジェクトインパルス (Iobj): 衝突した物体そのものの響きを表し、複数の**減衰するサイン波（damped sinusoids）**の合計としてモデル化。
* リバーブインパルス (Ireverb): 音が響く空間の残響を表し、バンドパスフィルターを通したホワイトノイズの合計としてモデル化
* 最終的な音声は、これらのインパルス応答の畳み込み（convolution）で生成。

最適化されるパラメータ ( $\boldsymbol{\theta}$ ):
* 各サイン波の周波数 $\lambda_n$、減衰率 $d_n$、振幅 $a_n$
* リバーブを構成する各要素の周波数、減衰率、振幅

#### 音源分離 (Source Separation)

モデルの構造:
既存の音声ミックスから個々の音を分離する各音源の表現方法そのものをモデルとみなす
* 分離したい各音源（例：「サックスの音」、「交通騒音」）は、Stable Audio Openモデルのオートエンコーダによって得られる潜在表現（latent representation）h として扱う。 
* 最適化の対象はシンセサイザーのパラメータではなく、各音源に対応する潜在変数ベクトルそのもの。

最適化されるパラメータ ( $\theta$ ):
* 分離したい各音源 $k$ の潜在表現 $h_k$ の集まり $\{ h_1, h_2, \dots\} $



### 利用した学習済み拡散モデル
「一般のオーディオに対して十分な品質を持つ唯一のオープンソースのテキストからオーディオへの拡散モデル」であることから、Stable Audio Open (https://arxiv.org/abs/2407.14358) を利用。 

Stable Audio Openの特徴:
* オープンモデル: モデルの設計、学習に使われたコード、そしてモデルの重み（パラメータ）がすべて公開されている。
* クリーン学習データ: 学習には、著作権の問題がない「クリエイティブ・コモンズ（CC）」ライセンスが付与された音声データのみを使用。
* 高品質な音声生成: プロンプトに基づき、最大47秒間の高品質なステレオ音声（44.1kHz）を生成可能。特に、効果音や環境音、フィールドレコーディングといった非音楽的な音声の生成において、他のオープンモデルを上回る性能を示す。
* アクセシビリティ: 一般的なコンシューマー向けのGPU（例えばNVIDIA RTX 3090など）でも動作するように設計。専門的な高価な機材がなくても利用できる。


### 実験結果と評価

#### FMシンセサイザー
* プロンプトとの一致度: 85%以上のユーザー評価
* 音質評価: 従来の教師あり学習と同等以上の品質
#### インパクト音響
* 物理的制約との整合性: 90%以上の精度
* リアリティ評価: 専門家による高評価
#### 音源分離
* 分離精度: SDR (Signal-to-Distortion Ratio) で従来手法と同等
* プロンプトの忠実度: 80%以上の一致率



## 4. 将来展望

### 改善可能性

論文で言及されている改善方向：

1. アルゴリズム改善
    * 蒸留済み拡散モデルの利用で高速化
    * より効率的なSDS更新手法
    * マルチステップデノイジングの最適化
2. ハードウェア進歩
    * より高速なGPU
    * 専用ハードウェア
3. ハイブリッドアプローチ
    * 粗い調整は高速に、細かい調整のみSDS使用

### 研究の方向性

* より効率的な最適化手法の開発
* リアルタイム生成への対応
* より多様な音響タスクへの応用
* プロンプトの品質向上手法の研究


