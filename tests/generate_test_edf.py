import numpy as np
from pathlib import Path

def generate_test_edf(file_path, num_signals=3, num_samples=1000, sampling_rate=200):
    """テスト用のEDFファイルを生成する"""
    # 信号データの生成
    data = []
    t = np.linspace(0, num_samples/sampling_rate, num_samples)
    
    # 基本周波数（1Hz）の信号
    signal1 = np.sin(2 * np.pi * 1.0 * t)
    data.append(signal1)
    
    # 高周波（5Hz）の信号
    signal2 = 0.5 * np.sin(2 * np.pi * 5.0 * t)
    data.append(signal2)
    
    # 複合波（1Hz + 2Hz + 3Hz）の信号
    signal3 = (np.sin(2 * np.pi * 1.0 * t) + 
              0.5 * np.sin(2 * np.pi * 2.0 * t) + 
              0.3 * np.sin(2 * np.pi * 3.0 * t))
    data.append(signal3)
    
    # EDFヘッダーの作成
    header = bytearray(256)
    
    # バージョン
    header[0:8] = b'0       '
    
    # 患者情報
    patient_id = b'X X X X'
    header[8:8+len(patient_id)] = patient_id
    
    # 記録情報
    recording_id = b'Startdate X X X'
    header[88:88+len(recording_id)] = recording_id
    
    # 開始日時
    header[168:176] = b'01.01.00'  # 日付
    header[176:184] = b'00.00.00'  # 時刻
    
    # ヘッダーサイズ
    header_bytes = 256 * (1 + num_signals)  # メインヘッダー + 信号ヘッダー
    header[184:192] = f"{header_bytes:8d}".encode('ascii')
    
    # 予約領域
    header[192:236] = b'EDF+C' + b' ' * 39
    
    # データレコード数と長さ
    num_records = 1
    header[236:244] = f"{num_records:8d}".encode('ascii')  # データレコード数
    header[244:252] = f"{1.0:8.3f}".encode('ascii')  # レコードの長さ（秒）
    
    # 信号数
    header[252:256] = f"{num_signals:4d}".encode('ascii')
    
    # 信号ヘッダーの作成
    signal_headers = []
    for i in range(num_signals):
        signal_header = bytearray(256)
        
        # ラベル
        label = f"Signal_{i:02d}"
        signal_header[0:16] = f"{label:16}".encode('ascii')
        
        # トランスデューサー
        signal_header[16:80] = b' ' * 64
        
        # 物理的単位
        signal_header[80:88] = b'uV      '
        
        # 物理的最小値/最大値
        signal_header[88:96] = b'-1000   '
        signal_header[96:104] = b'1000    '
        
        # デジタル最小値/最大値
        signal_header[104:112] = b'-32768  '
        signal_header[112:120] = b'32767   '
        
        # プリフィルタリング
        signal_header[120:184] = b' ' * 64
        
        # サンプル数/レコード
        signal_header[184:192] = f"{num_samples:8d}".encode('ascii')
        
        # 予約領域
        signal_header[192:256] = b' ' * 64
        
        signal_headers.append(signal_header)
    
    # ファイルに書き込み
    with open(file_path, 'wb') as f:
        # ヘッダーの書き込み
        f.write(header)
        
        # 信号ヘッダーの書き込み
        for signal_header in signal_headers:
            f.write(signal_header)
        
        # データの書き込み
        for signal in data:
            # 16ビット整数に変換（-1000から1000の範囲にスケーリング）
            scaled_signal = signal * 1000
            digital_signal = (scaled_signal * 32.767).astype(np.int16)
            f.write(digital_signal.tobytes())

if __name__ == "__main__":
    # テストファイルの生成
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)
    
    file_path = output_dir / "test_generator.edf"
    generate_test_edf(file_path)
    print(f"テストファイルを生成しました: {file_path}") 