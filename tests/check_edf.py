import struct
from pathlib import Path

def read_edf_header(file_path):
    """EDFファイルのヘッダーを読み込む"""
    with open(file_path, 'rb') as f:
        # ヘッダー全体を読み込む
        header = f.read(256)
        
        # 各フィールドの位置と長さ
        fields = {
            'version': (0, 8),
            'patient_id': (8, 80),
            'recording_id': (88, 80),
            'start_date': (168, 8),
            'start_time': (176, 8),
            'header_bytes': (184, 8),
            'reserved': (192, 44),
            'num_records': (236, 8),
            'record_duration': (244, 8),
            'num_signals': (252, 4)
        }
        
        # 各フィールドの値を表示
        print("EDF Header Information:")
        for field, (start, length) in fields.items():
            value = header[start:start+length].decode('ascii').strip()
            print(f"{field}: {value}")
        
        # 信号ヘッダーを読み込む
        num_signals = int(header[252:256].decode('ascii'))
        print(f"\nNumber of signals: {num_signals}")
        
        # 信号ヘッダーの各フィールドを表示
        signal_headers = f.read(256 * num_signals)
        for i in range(num_signals):
            start = i * 256
            signal_header = signal_headers[start:start+256]
            
            print(f"\nSignal {i+1} Header:")
            print(f"Label: {signal_header[0:16].decode('ascii').strip()}")
            print(f"Transducer: {signal_header[16:80].decode('ascii').strip()}")
            print(f"Physical dimension: {signal_header[80:88].decode('ascii').strip()}")
            print(f"Physical minimum: {signal_header[88:96].decode('ascii').strip()}")
            print(f"Physical maximum: {signal_header[96:104].decode('ascii').strip()}")
            print(f"Digital minimum: {signal_header[104:112].decode('ascii').strip()}")
            print(f"Digital maximum: {signal_header[112:120].decode('ascii').strip()}")
            print(f"Prefiltering: {signal_header[120:184].decode('ascii').strip()}")
            print(f"Samples per record: {signal_header[184:192].decode('ascii').strip()}")
            print(f"Reserved: {signal_header[192:256].decode('ascii').strip()}")

if __name__ == "__main__":
    file_path = Path(__file__).parent / "data" / "test_generator.edf"
    read_edf_header(file_path) 