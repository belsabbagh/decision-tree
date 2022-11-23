import numpy as np
import pandas as pd


def _eqn(val):
    return val * np.log10(val)


def _e_feat(feat_val_grp: pd.DataFrame, label: str):
    total = len(feat_val_grp)
    return - sum([_eqn(len(i) / total) for n, i in feat_val_grp.groupby(label)])


def _e(feat_grp, label, total_count):
    return sum([(len(val) / total_count) * _e_feat(val, label) for n, val in feat_grp])


def min_e_feature(df, label):
    entropy_values = dict(sorted({f: _e(df.groupby(f), label, len(df)) for f in df}.items(), key=lambda x: x[1]))
    del entropy_values[label]
    feature = min(entropy_values, key=entropy_values.get)
    return feature
