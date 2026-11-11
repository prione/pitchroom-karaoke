# PITCHROOM

PITCHROOMは、音源ファイルやYouTube動画から主旋律を解析し、音程バー・マイク入力・採点を提供するWindows向けElectronカラオケアプリです。解析とキャッシュはPC内で動作します。

## 主な機能

- MP3 / WAV / M4AとYouTube動画の読み込み
- ボーカル曲とインスト曲の自動判定
- 音源分離と主旋律の音高追跡
- 動的な音域表示と音程バー
- マイク機器の選択、感度調整、オクターブ自動補正
- 主旋律ガイド、キー・テンポ・ガイド音量の調整
- 解析済み楽曲ライブラリ、キャッシュ最適化、個別削除
- ライト / ダークテーマ

ボーカル曲は分離した歌声を中心に解析します。インスト曲は複数の候補楽器から、音域・連続性・スペクトル強度・フレーズ構造を組み合わせて主旋律1本を選択します。編曲上の主旋律が曖昧な曲では、人の知覚と完全には一致しない場合があります。

## 必要環境

- Windows 10 / 11（x64）
- Node.js 22.13以降
- pnpm
- Python 3.12
- FFmpeg（開発実行時。配布パッケージには同梱可能）

MuScriptorを使う高精度なインスト解析には、使用するモデル（[small](https://huggingface.co/MuScriptor/muscriptor-small) / [medium](https://huggingface.co/MuScriptor/muscriptor-medium) / [large](https://huggingface.co/MuScriptor/muscriptor-large)）の利用条件への同意とHugging Face認証が必要です。モデルの重みは非商用ライセンスです。

### MuScriptorの精度設定

通常はGPU（CUDA）があれば `medium`、なければCPUの `small` を自動選択します。インスト曲では従来の楽器構成による解析経路の切替を維持しています。速度より精度を優先するときは、PowerShellで起動前に以下を設定します。

```powershell
$env:PITCHROOM_MUSCRIPTOR_DEVICE = "cuda"
$env:PITCHROOM_MUSCRIPTOR_WEIGHTS = "large"
$env:PITCHROOM_MUSCRIPTOR_QUALITY = "high"
$env:PITCHROOM_MUSCRIPTOR_ROUTING = "always"
.\start-local.ps1
```

- `PITCHROOM_MUSCRIPTOR_DEVICE`: `auto`（既定）、`cuda`、`cpu`。CUDAの利用には対応したPyTorchが必要です。
- `PITCHROOM_MUSCRIPTOR_WEIGHTS`: `small` / `medium` / `large`。未指定時はGPUで `medium`、CPUで `small`。
- `PITCHROOM_MUSCRIPTOR_QUALITY`: `balanced`（既定）は複数チャンク並列処理で速度優先、`high` は境界をまたぐノーツを引き継ぐ採譜で精度優先。
- `PITCHROOM_MUSCRIPTOR_ROUTING`: `adaptive`（既定）は既存方式との自動切替、`always` はインスト曲全体でMuScriptorを優先、`off` はMuScriptorを使わない。
- `PITCHROOM_MUSCRIPTOR_DTYPE`: 省略推奨（MuScriptor側の既定動作）。必要なら `float32` / `float16` / `bfloat16` を指定できます。

Windowsで `torch.cuda.is_available()` が `False` の場合は、CPU版PyTorchが入っていないか確認してください。NVIDIA GPU用CUDA対応PyTorchの導入は、仮想環境内で次のように行えます（GPU・ドライバ環境に合った版を選んでください）。

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade torch torchaudio --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

設定を変更した場合はアプリを再起動してください。解析キャッシュはモデルと品質設定ごとに区別され、別設定の古いノーツが再利用されないようにしています。MuScriptorの重みは初回取得が必要です。GPU利用でも音源の種類によっては主旋律の推定が誤るため、精度保証はありません。

## セットアップ

```powershell
pnpm install
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r local_backend\requirements.txt
.\.venv\Scripts\hf.exe auth login
```

開発用Electronアプリを起動します。

```powershell
.\start-local.ps1
```

または、画面とElectronを個別に起動できます。

```powershell
pnpm run electron:dev
```

## テスト

```powershell
pnpm test
pnpm run test:python
```

## Windowsインストーラーの作成

```powershell
pnpm run package:win
```

生成物は`release/PITCHROOM-Setup-<version>-x64.exe`です。Python解析環境、FFmpeg、Basic Pitchモデルを同梱するため、初回ビルドには時間と空き容量が必要です。インストーラーや同梱ランタイムはGit管理対象外です。

## データとプライバシー

- 開発実行時の解析データは`.pitchroom`、インストール版は`%APPDATA%\pitchroom-karaoke\data`に保存されます。
- 音源、ダウンロード動画、解析キャッシュ、モデル、認証情報はリポジトリに含めません。
- yt-dlpはインストール版の起動時に更新を確認し、失敗時は同梱版へフォールバックします。`PITCHROOM_DISABLE_YTDLP_UPDATE=1`で自動更新を無効化できます。
- YouTube機能は、権利者が許可しているコンテンツや自身が利用権を持つコンテンツに限って使用してください。

## 技術構成

- Electron / React / Vite
- FastAPI / Python
- Demucs / Basic Pitch / MuScriptor / librosa
- Web Audio API

このリポジトリにはライセンスを付与していません。第三者による利用・再配布については、著作権者の許可が必要です。
