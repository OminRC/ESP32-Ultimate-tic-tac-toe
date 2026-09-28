import numpy as np
import pytest

torch = pytest.importorskip("torch")

import net
import net_torch


def test_forward_output_shapes_and_softmax():
    model = net_torch.PolicyValueNet()
    x = torch.randn(5, net_torch.INPUT_DIM)
    policy, value = model(x)

    assert policy.shape == (5, net_torch.POLICY_DIM)
    assert value.shape == (5,)
    torch.testing.assert_close(policy.sum(dim=1), torch.ones(5), atol=1e-5, rtol=0)
    assert torch.all(value >= -1.0) and torch.all(value <= 1.0)


def test_save_load_npz_roundtrip(tmp_path):
    model = net_torch.PolicyValueNet()
    path = tmp_path / "params.npz"
    net_torch.save_npz(model, str(path))
    assert path.exists()

    model2 = net_torch.PolicyValueNet()
    net_torch.load_npz_into(model2, str(path))

    x = torch.randn(3, net_torch.INPUT_DIM)
    with torch.no_grad():
        p1, v1 = model(x)
        p2, v2 = model2(x)
    torch.testing.assert_close(p1, p2)
    torch.testing.assert_close(v1, v2)


def test_net_torch_checkpoint_is_loadable_by_plain_numpy_net(tmp_path):
    """Cross-compatibility: a checkpoint saved by the GPU/Kaggle training
    path (net_torch) must be usable by the plain-NumPy path (net.py), since
    export_weights.py and eval_vs_random.py are written against net.py.
    """
    model = net_torch.PolicyValueNet()
    path = tmp_path / "params.npz"
    net_torch.save_npz(model, str(path))

    numpy_params = net.load_params(str(path))
    x = np.random.default_rng(0).normal(size=net.INPUT_DIM).astype(np.float32)
    policy, value = net.predict_single(numpy_params, x)

    assert policy.shape == (net.POLICY_DIM,)
    assert -1.0 <= value <= 1.0
    np.testing.assert_allclose(policy.sum(), 1.0, atol=1e-5)

    # and check it numerically matches the torch model on the same input
    with torch.no_grad():
        t_policy, t_value = model(torch.from_numpy(x).unsqueeze(0))
    np.testing.assert_allclose(policy, t_policy.numpy()[0], atol=1e-5)
    assert value == pytest.approx(float(t_value.item()), abs=1e-5)
