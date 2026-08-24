# CLAUDE.md

## プロジェクト概要

imgrit は、k-means とボロノイ図で画像をモザイクアート（`voronoi_mosaic`）や
ウォーホル風（`warhol_effect`）に変換する小さな画像処理ライブラリ。
PyPI で公開している: https://pypi.org/project/imgrit/

- 本体は [src/imgrit/imgrit.py](src/imgrit/imgrit.py) の1ファイル
- 依存: Pillow / NumPy / SciPy。scikit-learn があれば高速な k-means を使う
  （`HAVE_SKL` フラグで分岐。`pip install "imgrit[sklearn]"`）
- 座標系は NumPy 配列準拠で (row, col) = (y, x)。scipy の Voronoi や PIL の
  描画に渡すときに座標を反転している箇所が多いので、変更時は要注意

## 開発

```bash
pytest tests/          # 回帰テスト（tests/test_imgrit.py）
python -m build        # sdist + wheel のビルド確認
```

`tests/sample.py` はテストではなく、実画像で出力を確認するためのスクリプト。

## リリース手順

1. `pyproject.toml` の `version` を上げて main にコミット
2. `v*` タグを push（例: `git tag v0.2.2 && git push origin main v0.2.2`）
3. GitHub Actions の [publish.yml](.github/workflows/publish.yml) が自動で
   ビルドして PyPI にアップロードする

- PyPI 側は Trusted Publishing (OIDC) を登録済み
  （Repository: `tsjshg/imgrit` / Workflow: `publish.yml` / Environment: `pypi`）。
  API トークンや GitHub Secrets は不要
- PyPI は同一バージョンの再アップロードを拒否するので、タグの前に必ず version を上げる
- ワークフローが失敗しても、原因を直せば「Re-run all jobs」で再実行できる
  （タグの打ち直しは不要）

## 経緯

### 2026-08-24: v0.2.1 — コードレビューによるバグ修正と自動リリース整備

Claude によるコードレビューで以下のバグを発見し、修正した（PR #5、
コミット `0519b4e` / `edcf534`）。各バグは実行して再現確認済みで、
[tests/test_imgrit.py](tests/test_imgrit.py) に回帰テストがある。

修正したバグ:

1. `find_edge_point` の垂直2等分線の分岐に typo があり
   （`(vv[1], y_max)` → 正しくは `(vv[0], y_max)`）、垂直であるべき
   ボロノイ境界線が斜めに描画されていた
2. `isinstance(sites, (list, tuple, np.array))` の `np.array` は型ではなく
   関数のため、NumPy 配列の sites を渡すと TypeError になっていた
   （`np.ndarray` に修正）
3. グレースケール（L モード）画像で shape の unpack に失敗してクラッシュ。
   入り口で `convert("RGB")` に正規化して解決（RGBA・パレットも統一的に処理）
4. `KMeansImage` は入力を `data.max(axis=0)` でスケールするのに、逆変換は
   一律 `×255` / `×(h, w)` だったため、(a) warhol_effect で暗い画像の色が
   明るく歪む、(b) サイト座標が 1 ピクセル画像外に出て ValueError、の
   2つのバグがあった。スケール係数を `self.scale` に保持して同じ係数で
   逆変換するよう修正。random モードの `margin = 0.03` は (b) の回避策
   だったので削除
5. 真っ黒画像などの定数チャンネルでゼロ除算 → NaN で kmeans が失敗
6. 丸め後の重複サイトで scipy の Voronoi が QhullError になり得た（重複除去を追加）
7. `pyproject.toml` が `requires-python >= 3.9` なのに `numpy>=2.3.1` は
   Python 3.11+ 必須で、3.9/3.10 ではインストール不能だった（3.11+ に統一）

あわせて実施:

- 裸の `except: print` を logging に置換、`mode` 引数のバリデーション追加
- `__init__.py` の `version()` が未インストールのソース実行で落ちる問題に
  フォールバックを追加
- pyproject に license（BSD-3-Clause）と `sklearn` extra を追加
- タグ push で PyPI に自動公開する publish.yml を新設（上記リリース手順）
- 初回リリース時、GitHub Actions の障害（ランナー割り当て遅延）で run が
  cancelled になったが、復旧後の Re-run で成功。設定の問題ではなかった

未着手の改善候補（レビューで指摘済み・意図的に見送り）:

- `create_Voronoi_image` のピクセルごとの Python ループ（`Pixel` オブジェクトを
  全ピクセル分生成）が最大のボトルネック。`kd_tree.query(...)[1].reshape(h, w)`
  でベクトル化すれば大幅に高速化でき、`Pixel` / `followers` も不要になる。
  ソース冒頭の「kmeans がボトルネック」コメントは実態と合っていない
- 無限リッジの向きを「母点に近い交点」で選ぶヒューリスティック
  （作者コメント「これでいいのか？」の箇所）は幾何によっては逆側を選ぶ。
  scipy の `voronoi_plot_2d` と同じ中点+法線方向の方式にすれば
  `solve` / `find_edge_point` ごと簡素化できる
- 乱数シード（`random_state`）を渡す口がなく、結果の再現ができない
- フル解像度の全ピクセルを KMeans に渡しており大きい画像で遅い
  （MiniBatchKMeans かダウンサンプリングで改善可能）
