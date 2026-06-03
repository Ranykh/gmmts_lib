"""
GMM_TS: Gating-based Multimodal Time Series Forecasting

A gating mechanism for multimodal time series forecasting that dynamically 
combines predictions from multiple expert models.

Built on top of MM-TSFlib: https://github.com/AdityaLab/MM-TSFlib
"""

__version__ = "0.1.0"
__author__ = "Kathy Razmadze, Yoli Shavit"

from gmm_ts.gating.Gating import GatingNet
from gmm_ts.exp.exp_online_gating_long_term_forecasting import Exp_Online_Gating_Long_Term_Forecast
from gmm_ts.exp.exp_offline_gating_long_term_forecasting import Exp_Offline_Gating_Long_Term_Forecast

__all__ = [
    'GatingNet',
    'Exp_Online_Gating_Long_Term_Forecast',
    'Exp_Offline_Gating_Long_Term_Forecast',
]

