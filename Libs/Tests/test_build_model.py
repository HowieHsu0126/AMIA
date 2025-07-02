from Libs.run_pipeline import build_model
from Libs.Models import cMLP, cRNN, cLSTM


def _cfg(backbone: str):
    return {"backbone": backbone, "lag": 2, "hidden": [8, 4]}


def test_build_cmlp():
    model = build_model(_cfg("cmlp"), num_series=5)
    assert isinstance(model, cMLP.cMLP)


def test_build_crnn():
    model = build_model(_cfg("crnn"), num_series=5)
    assert isinstance(model, cRNN.cRNN)


def test_build_clstm():
    model = build_model(_cfg("clstm"), num_series=5)
    assert isinstance(model, cLSTM.cLSTM) 