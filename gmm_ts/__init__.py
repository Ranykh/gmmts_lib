"""GMM-TS package bootstrap and public exports."""

__version__ = "0.1.0"
__author__ = "Kathy Razmadze, Yoli Shavit"

from gmm_ts.mm_tsflib_bootstrap import bootstrap_mm_tsflib

bootstrap_mm_tsflib()

from gmm_ts.gating.Gating import GatingNet
from gmm_ts.exp.exp_online_gating_long_term_forecasting import Exp_Online_Gating_Long_Term_Forecast
from gmm_ts.exp.exp_offline_gating_long_term_forecasting import Exp_Offline_Gating_Long_Term_Forecast

__all__ = [
    'GatingNet',
    'Exp_Online_Gating_Long_Term_Forecast',
    'Exp_Offline_Gating_Long_Term_Forecast',
]

