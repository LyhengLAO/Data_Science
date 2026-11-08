import pytest

from src.data import FEATURES, prepare


def test_prepare_columns(prepared):
    assert set(FEATURES + ["treatment", "y"]) == set(prepared.columns)
    assert not prepared.isna().any().any()


def test_prepare_rejects_unknown_arm(synth, cfg):
    bad = synth.copy()
    bad.loc[0, "segment"] = "Carrier Pigeon"
    with pytest.raises(ValueError):
        prepare(bad, cfg)


def test_split_no_overlap_and_balanced(splits, prepared):
    train, test = splits
    assert len(train) + len(test) == len(prepared)
    for arm in prepared.treatment.unique():  # ratio de traitement préservé (tolérance 1 pt)
        full = (prepared.treatment == arm).mean()
        assert abs((train.treatment == arm).mean() - full) < 0.01
        assert abs((test.treatment == arm).mean() - full) < 0.01
    assert abs(train.y.mean() - test.y.mean()) < 0.005


def test_randomization_balance(prepared):
    # test randomisé : la moyenne de recency doit être similaire entre groupes
    means = prepared.groupby("treatment").recency.mean()
    assert means.max() - means.min() < 0.2
