"""
EDFファイルのDMD処理パッケージ
"""

from .dmd_processor import (
    create_spatiotemporal_data,
    perform_dmd,
    plot_results,
    main
)

__all__ = [
    'create_spatiotemporal_data',
    'perform_dmd',
    'plot_results',
    'main'
]

__version__ = "0.1.0" 