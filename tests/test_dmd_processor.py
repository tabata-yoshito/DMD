import numpy as np
import pytest
from pathlib import Path
from src.dmd_processor import create_spatiotemporal_data, perform_dmd, plot_results, main
import matplotlib
matplotlib.use('Agg')  # 非インタラクティブなバックエンドを使用
import matplotlib.pyplot as plt
import time

@pytest.fixture
def sample_data():
    """テスト用のサンプルデータを生成するフィクスチャ"""
    return create_spatiotemporal_data((-10, 10), (0, 2*np.pi), n_x=50, n_t=40)

@pytest.fixture
def large_data():
    """大規模データセット用のフィクスチャ"""
    return create_spatiotemporal_data((-10, 10), (0, 4*np.pi), n_x=100, n_t=80)

@pytest.fixture
def dmd_results(sample_data):
    """DMD計算結果を生成するフィクスチャ"""
    f1, f2, f3, t = sample_data
    D = (f1 + f2 + f3).T
    X = D[:,:-1]
    Y = D[:,1:]
    return perform_dmd(X, Y, rank=3), X, Y, t

def test_create_spatiotemporal_data():
    """時空間データ生成のテスト"""
    # 基本ケース
    f1, f2, f3, t = create_spatiotemporal_data((-10, 10), (0, 2*np.pi), n_x=50, n_t=40)
    
    # データの形状を確認
    assert f1.shape == (40, 50)  # (time, space)
    assert f2.shape == (40, 50)
    assert f3.shape == (40, 50)
    assert len(t) == 40
    
    # データの型を確認
    assert np.iscomplexobj(f1)
    assert np.iscomplexobj(f2)
    assert np.iscomplexobj(f3)
    assert np.isrealobj(t)
    
    # 特定の値の確認
    assert np.allclose(f1[0, 0], (20 - 0.2 * (-10)**2) * np.exp(2.3j * 0))
    assert np.allclose(f2[0, 0], -10 * np.exp(0.6j * 0))

    # エッジケース
    with pytest.raises(ValueError):
        create_spatiotemporal_data((10, -10), (0, 2*np.pi))  # 不正な範囲
    
    # 最小サイズのテスト
    f1, f2, f3, t = create_spatiotemporal_data((-10, 10), (0, 2*np.pi), n_x=2, n_t=2)
    assert f1.shape == (2, 2)

def test_perform_dmd(dmd_results):
    """DMD計算のテスト"""
    (Phi, mu, W), X, Y, t = dmd_results
    
    # 結果の形状を確認
    assert Phi.shape == (50, 3)  # (space, rank)
    assert len(mu) == 3
    assert W.shape == (3, 3)
    
    # 固有値の性質を確認
    assert np.allclose(np.linalg.norm(mu), np.sqrt(np.sum(np.abs(mu)**2)))
    
    # 再構成のテスト
    b = np.linalg.pinv(Phi) @ X[:,0]
    dt = t[1] - t[0]
    Psi = np.zeros([3, len(t)], dtype='complex')
    for i, _t in enumerate(t):
        Psi[:,i] = np.power(mu, _t/dt) * b
    
    D_reconstructed = Phi @ Psi
    assert np.allclose(X, D_reconstructed[:,:-1], rtol=1e-10)

def test_perform_dmd_edge_cases():
    """DMD計算のエッジケーステスト"""
    # ランクが大きすぎる場合
    f1, f2, f3, t = create_spatiotemporal_data((-10, 10), (0, 2*np.pi), n_x=5, n_t=5)
    D = (f1 + f2 + f3).T
    X = D[:,:-1]
    Y = D[:,1:]
    
    with pytest.raises(ValueError):
        perform_dmd(X, Y, rank=10)  # ランクがデータサイズより大きい

    # 空のデータ
    with pytest.raises(ValueError):
        perform_dmd(np.array([]), np.array([]), rank=1)

    # 形状が一致しない場合
    with pytest.raises(ValueError):
        perform_dmd(np.zeros((5, 5)), np.zeros((6, 5)), rank=2)

def test_plot_results(tmp_path, dmd_results):
    """プロット機能のテスト"""
    (Phi, mu, W), X, Y, t = dmd_results
    f1, f2, f3, t_new = create_spatiotemporal_data((-10, 10), (0, 2*np.pi), n_x=50, n_t=40)  # n_tを40に固定
    D = (f1 + f2 + f3).T
    
    # 時間発展の計算
    b = np.linalg.pinv(Phi) @ X[:,0]
    dt = t[1] - t[0]
    Psi = np.zeros([3, len(t_new)], dtype='complex')  # t_newを使用
    for i, _t in enumerate(t_new):
        Psi[:,i] = np.power(mu, _t/dt) * b
    
    # プロットの実行
    output_path = tmp_path / "test_output.png"
    plot_results(f1, f2, f3, D, Psi, t_new, np.linalg.svd(X, compute_uv=False), str(output_path))
    
    # プロットファイルの存在確認
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    # プロットファイルの内容確認
    img = plt.imread(str(output_path))
    assert img.shape[2] == 4  # RGBA形式であることを確認

    # 不正な入力に対するテスト
    with pytest.raises(TypeError):
        plot_results(1, f2, f3, D, Psi, t_new, np.linalg.svd(X, compute_uv=False))

def test_dmd_reconstruction_accuracy(large_data):
    """DMD再構成の精度テスト"""
    f1, f2, f3, t = large_data
    D = (f1 + f2 + f3).T
    X = D[:,:-1]
    Y = D[:,1:]
    
    # 異なるランクでDMDを実行
    for rank in [3, 5, 10]:
        start_time = time.time()
        Phi, mu, W = perform_dmd(X, Y, rank=rank)
        dmd_time = time.time() - start_time
        
        # 再構成
        b = np.linalg.pinv(Phi) @ X[:,0]
        dt = t[1] - t[0]
        Psi = np.zeros([rank, len(t)], dtype='complex')
        for i, _t in enumerate(t):
            Psi[:,i] = np.power(mu, _t/dt) * b
        
        D_reconstructed = Phi @ Psi
        
        # 再構成誤差を計算
        reconstruction_error = np.linalg.norm(D - D_reconstructed) / np.linalg.norm(D)
        assert reconstruction_error < 0.1  # 10%以下の誤差を許容
        
        # パフォーマンスチェック
        assert dmd_time < 1.0  # 1秒以内に計算が完了することを確認

def test_dmd_performance():
    """DMDのパフォーマンステスト"""
    # 大規模データセット
    f1, f2, f3, t = create_spatiotemporal_data((-10, 10), (0, 4*np.pi), n_x=200, n_t=160)
    D = (f1 + f2 + f3).T
    X = D[:,:-1]
    Y = D[:,1:]
    
    # パフォーマンス計測
    start_time = time.time()
    Phi, mu, W = perform_dmd(X, Y, rank=10)
    execution_time = time.time() - start_time
    
    # メモリ使用量の確認（概算）
    memory_usage = (X.nbytes + Y.nbytes + Phi.nbytes + mu.nbytes + W.nbytes) / (1024 * 1024)  # MB
    assert memory_usage < 100  # 100MB以下であることを確認
    assert execution_time < 5.0  # 5秒以内に計算が完了することを確認

def test_main():
    """main関数のテスト"""
    # main関数が例外を発生させないことを確認
    main() 