import numpy as np
import pytest

import net


def test_forward_output_shapes_and_softmax():
    params = net.init_params(seed=0)
    x = np.random.default_rng(0).normal(size=(5, net.INPUT_DIM)).astype(np.float32)
    cache = net.forward(params, x)

    assert cache["policy"].shape == (5, net.POLICY_DIM)
    assert cache["value"].shape == (5, 1)
    np.testing.assert_allclose(cache["policy"].sum(axis=1), 1.0, atol=1e-5)
    assert np.all(cache["value"] >= -1.0) and np.all(cache["value"] <= 1.0)


def test_predict_single_matches_batch_forward():
    params = net.init_params(seed=1)
    x = np.random.default_rng(1).normal(size=net.INPUT_DIM).astype(np.float32)

    policy_single, value_single = net.predict_single(params, x)
    cache = net.forward(params, x[None, :])

    np.testing.assert_allclose(policy_single, cache["policy"][0])
    assert value_single == pytest.approx(float(cache["value"][0, 0]), abs=1e-6)


def test_save_load_roundtrip(tmp_path):
    params = net.init_params(seed=2)
    path = tmp_path / "params.npz"  # np.savez keeps the .npz suffix as-is
    net.save_params(params, path)
    assert path.exists()

    loaded = net.load_params(str(path))
    x = np.random.default_rng(3).normal(size=net.INPUT_DIM).astype(np.float32)
    p1, v1 = net.predict_single(params, x)
    p2, v2 = net.predict_single(loaded, x)
    np.testing.assert_allclose(p1, p2)
    assert abs(v1 - v2) < 1e-6


def test_loss_and_grads_decrease_loss_after_one_step():
    params = net.init_params(seed=4)
    rng = np.random.default_rng(4)
    x = rng.normal(size=(16, net.INPUT_DIM)).astype(np.float32)
    pi = np.full((16, net.POLICY_DIM), 1.0 / net.POLICY_DIM, dtype=np.float32)
    z = rng.uniform(-1, 1, size=16).astype(np.float32)

    loss0, _, _, grads = net.loss_and_grads(params, x, pi, z)
    opt = net.Adam(params, lr=1e-2)
    opt.step(params, grads)
    loss1, _, _, _ = net.loss_and_grads(params, x, pi, z)

    assert loss1 < loss0
