"""A user seed controls sampling and splitting as well as the booster."""
import importlib
import numpy as np
import pandas as pd


def test_configured_seed_controls_all_training_randomness(tmp_path, monkeypatch):
    train = importlib.import_module('pycwb.modules.postprocess.train_xgboost')
    config = importlib.import_module('pycwb.modules.cwb_xgboost.config')
    data = importlib.import_module('pycwb.modules.cwb_xgboost.read_data')
    balance = importlib.import_module('pycwb.modules.cwb_xgboost.utils_extended')
    seen = {}
    monkeypatch.setattr(config, 'xgb_config', lambda *a: ({'seed': 150914}, ['rho0'], {}, {'tail(training)': True}, {}))
    monkeypatch.setattr(train, '_xgb_required_input_columns', lambda *a, **k: None)
    monkeypatch.setattr(train, '_read_and_concat', lambda *a, **k: pd.DataFrame({'rho0': np.arange(20)+8.}))
    monkeypatch.setattr(data, 'preprocess_events', lambda frame, *a: frame)
    def tail(frame, caps, seed):
        seen['tail'] = seed
        return frame
    monkeypatch.setattr(balance, 'get_balanced_tail', tail)
    original_split = train.train_test_split
    def split(*a, **k):
        seen['split'] = k['random_state']
        return original_split(*a, **k)
    monkeypatch.setattr(train, 'train_test_split', split)
    class Model:
        best_score = .5
        best_iteration = 0
        def __init__(self, **kw): seen['booster'] = kw['seed']
        def fit(self, *a, **kw): pass
        def save_model(self, path):
            from pathlib import Path
            Path(path).write_bytes(b'test')
    monkeypatch.setattr(train.xgb, 'XGBClassifier', Model)
    cfg = tmp_path/'config.py'
    cfg.write_text('def update_config(params, *args):\n    params["seed"] = 98765\n')
    train.train_xgboost(str(tmp_path), bkg_catalog='b.parquet', sim_catalog='s.parquet', nifo=2, config_file=str(cfg))
    assert seen == {'tail': 98765, 'split': 98765, 'booster': 98765}
