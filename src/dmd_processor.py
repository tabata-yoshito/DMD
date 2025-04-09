import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from numpy.linalg import inv, eig, pinv
from scipy.linalg import svd
from typing import Tuple, Optional
import warnings

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from numpy import dot, multiply, diag, power
from numpy import pi, exp, sin, cos, cosh, tanh, real, imag
from scipy.linalg import svdvals
from scipy.integrate import odeint, ode, complex_ode
from warnings import warn

def create_spatiotemporal_data(x_range: Tuple[float, float], t_range: Tuple[float, float], 
                             n_x: int = 100, n_t: int = 80) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """時空間データを生成する関数"""
    if x_range[0] >= x_range[1] or t_range[0] >= t_range[1]:
        raise ValueError("範囲の指定が不正です。開始値は終了値より小さくする必要があります。")
    
    if n_x < 2 or n_t < 2:
        raise ValueError("データポイント数は2以上である必要があります。")
    
    x = np.linspace(x_range[0], x_range[1], n_x)
    t = np.linspace(t_range[0], t_range[1], n_t)
    Xm, Tm = np.meshgrid(x, t)
    
    # 3つの時空間パターンを生成
    f1 = (20 - 0.2 * Xm**2) * np.exp(2.3j * Tm)
    f2 = Xm * np.exp(0.6j * Tm)
    f3 = 5 * (1/np.cosh(Xm/2)) * np.tanh(Xm/2) * 2 * np.exp((0.1 + 2.8j) * Tm)
    
    return f1, f2, f3, t

def perform_dmd(X: np.ndarray, Y: np.ndarray, rank: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """DMDを実行する関数"""
    if X.size == 0 or Y.size == 0:
        raise ValueError("入力行列が空です。")
    
    if X.shape != Y.shape:
        raise ValueError("XとYの形状が一致しません。")
    
    if rank <= 0:
        raise ValueError("ランクは正の整数である必要があります。")
    
    if rank > min(X.shape):
        raise ValueError(f"ランク({rank})がデータの次元({min(X.shape)})を超えています。")
    
    # SVD
    U, S, Vh = svd(X, full_matrices=False)
    
    # ランク削減
    U_r = U[:, :rank]
    S_r = np.diag(S[:rank])
    V_r = Vh.conj().T[:, :rank]
    
    # Aチルダの構築
    Atil = U_r.conj().T @ Y @ V_r @ inv(S_r)
    mu, W = eig(Atil)
    
    # DMDモードの構築
    Phi = Y @ V_r @ inv(S_r) @ W
    
    return Phi, mu, W

def plot_results(f1: np.ndarray, f2: np.ndarray, f3: np.ndarray, 
                D: np.ndarray, Psi: np.ndarray, t: np.ndarray, 
                S: np.ndarray, output_path: str = 'output.png') -> None:
    """結果をプロットする関数"""
    if not all(isinstance(arr, np.ndarray) for arr in [f1, f2, f3, D, Psi, t, S]):
        raise TypeError("すべての入力はnumpy配列である必要があります。")
    
    fig = plt.figure(figsize=(15, 40))
    fig.subplots_adjust(hspace=1)
    
    # プロットするデータのインデックス
    indx = min(10, f1.shape[1] - 1)  # インデックスが範囲外にならないように
    
    # 各データのプロット
    for i, (data, title) in enumerate([
        (f1.T, 'f1_raw'),
        (f2.T, 'f2_raw'),
        (f3.T, 'f3_raw'),
        (D, 'fusion')
    ]):
        ax = fig.add_subplot(10, 1, i+1)
        ax.plot(t, data.real[indx,:], 'bx-', label='Real')
        ax.plot(t, data.imag[indx,:], 'gx-', label='Complex')
        ax.set_xlabel('t')
        ax.set_ylabel('y')
        ax.set_title(title)
        ax.legend()
    
    # SVDのプロット
    ax = fig.add_subplot(10, 1, 5)
    ax.scatter(range(len(S)), S, label="SVD")
    ax.set_xlabel('Index')
    ax.set_ylabel('Singular Value')
    ax.set_title('SVD')
    
    # DMDモードのプロット
    for i in range(min(3, Psi.shape[0])):  # Psiの次元に合わせて調整
        ax = fig.add_subplot(10, 1, 6+i)
        ax.plot(t, Psi.real[i,:], 'bx-', label='Real')
        ax.plot(t, Psi.imag[i,:], 'gx-', label='Complex')
        ax.set_xlabel('t')
        ax.set_ylabel('y')
        ax.set_title(f'DMD (m{i+1})')
        ax.legend()
    
    plt.savefig(output_path)
    plt.close()

def main() -> None:
    """メイン関数"""
    try:
        # データ生成
        f1, f2, f3, t = create_spatiotemporal_data((-10, 10), (0, 6*np.pi))
        D = (f1 + f2 + f3).T
        
        # DMDの入力行列
        X = D[:,:-1]
        Y = D[:,1:]
        
        # DMD実行
        Phi, mu, W = perform_dmd(X, Y, rank=3)
        
        # 時間発展の計算
        b = pinv(Phi) @ X[:,0]
        dt = t[1] - t[0]
        Psi = np.zeros([3, len(t)], dtype='complex')
        for i, _t in enumerate(t):
            Psi[:,i] = np.power(mu, _t/dt) * b
        
        # 結果のプロット
        plot_results(f1, f2, f3, D, Psi, t, np.linalg.svd(X, compute_uv=False))
        
    except Exception as e:
        warnings.warn(f"エラーが発生しました: {str(e)}")
        raise

if __name__ == "__main__":
    main()

