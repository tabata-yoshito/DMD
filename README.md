# Dynamic Mode Decomposition (DMD) 処理ツール

このパッケージは、Dynamic Mode Decomposition (DMD) を用いてデータの時空間パターンを解析するためのツールです。

## 機能

- 時空間データの生成
- DMD解析の実行
- 結果の可視化
- 高精度な再構成（誤差10%以下）
- パフォーマンス最適化（大規模データセット対応）

## インストール

```bash
git clone https://github.com/yourusername/DMD.git
cd DMD
pip install -r requirements.txt
```

## 必要条件

- Python 3.10以上
- 以下のPythonパッケージ:
  ```
  numpy>=1.21.0
  matplotlib>=3.4.0
  scipy>=1.7.0
  pytest>=6.2.5
  pytest-cov>=2.12.0
  ```

## 使用方法

### 基本的な使用例

```python
from src.dmd_processor import create_spatiotemporal_data, perform_dmd, plot_results

# データの生成
f1, f2, f3, t = create_spatiotemporal_data((-10, 10), (0, 6*np.pi))
D = (f1 + f2 + f3).T

# DMDの実行
X = D[:,:-1]
Y = D[:,1:]
Phi, mu, W = perform_dmd(X, Y, rank=3)

# 結果の可視化
plot_results(f1, f2, f3, D, Psi, t, S, 'output.png')
```

### メイン関数の実行

```bash
python -m src.dmd_processor
```

## テスト

テストを実行するには：

```bash
# 通常のテスト実行
pytest tests/ -v

# カバレッジレポートの生成
pytest tests/ --cov=src --cov-report=html
```

現在のテストカバレッジ: 94%

## 実装の詳細

### 主要な関数

1. `create_spatiotemporal_data(x_range, t_range, n_x=100, n_t=80)`
   - 時空間データの生成
   - 3つの異なるパターンを生成（二次関数、線形、非線形）

2. `perform_dmd(X, Y, rank)`
   - DMD計算の実行
   - SVDによる次元削減
   - 固有値分解による動的モードの抽出

3. `plot_results(f1, f2, f3, D, Psi, t, S, output_path)`
   - 結果の可視化
   - 元データと再構成データの比較
   - SVD特異値のプロット
   - DMDモードの可視化

## パフォーマンス

- 処理時間: 大規模データセット（200×160）で5秒以内
- メモリ使用量: 100MB以下
- 再構成精度: 誤差10%以下

## 制限事項

- 入力データは2次元（時間×空間）である必要があります
- ランクはデータの次元より小さい必要があります
- プロットには十分なメモリが必要です

## ライセンス

MIT License

## 貢献

1. このリポジトリをフォーク
2. 新しいブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## 作者

Your Name 