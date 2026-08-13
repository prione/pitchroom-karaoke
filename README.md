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

MuScriptorを使う高精度なインスト解析には、[MuScriptor small](https://huggingface.co/MuScriptor/muscriptor-small)の利用条件への同意とHugging Face認証が必要です。

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
