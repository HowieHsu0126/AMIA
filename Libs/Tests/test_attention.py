import torch
from Libs.Models import cMLP, cRNN, cLSTM


def _rand_tensor(p=4, lag=3, hidden=[8, 4]):
    model = cMLP.cMLP(p, lag, hidden)
    return model


def test_cmlp_attention_row_sum():
    p = 5
    m = cMLP.cMLP(p, lag=2, hidden=[4])
    attn = m.attention()
    assert attn.shape == (p, p)
    # Row sums ~1
    rs = attn.sum(dim=1)
    assert torch.allclose(rs, torch.ones_like(rs), atol=1e-5)


def test_crnn_attention_shape():
    p = 6
    m = cRNN.cRNN(p, hidden=4)
    attn = m.attention()
    assert attn.shape == (p, p)


def test_clstm_attention_nonneg():
    p = 4
    m = cLSTM.cLSTM(p, hidden=3)
    attn = m.attention()
    assert (attn >= 0).all() 