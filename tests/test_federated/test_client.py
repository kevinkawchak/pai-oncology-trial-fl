"""Tests for federated/client.py — FederatedClient.

Covers creation, local training (FedAvg/FedProx/SCAFFOLD), get_update,
parameter shapes, training without data, and history tracking.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from tests.conftest import PROJECT_ROOT, load_module


@pytest.fixture(scope="module")
def client_mod():
    return load_module("federated.client", "federated/client.py")


@pytest.fixture(scope="module")
def model_mod():
    return load_module("federated.model", "federated/model.py")


def _default_config(model_mod):
    return model_mod.ModelConfig(input_dim=4, hidden_dims=[8], output_dim=2, learning_rate=0.01, seed=0)


def _make_client(client_mod, model_mod, client_id="hospital_A", with_data=True):
    cfg = _default_config(model_mod)
    client = client_mod.FederatedClient(client_id=client_id, model_config=cfg)
    if with_data:
        rng = np.random.default_rng(42)
        x = rng.standard_normal((50, 4)).astype(np.float32)
        y = rng.integers(0, 2, size=50)
        client.set_data(x, y)
    return client


class TestFederatedClientCreation:
    """Tests for client instantiation and basic attributes."""

    def test_creation_default(self, client_mod, model_mod):
        client = _make_client(client_mod, model_mod, with_data=False)
        assert client.client_id == "hospital_A"
        assert client.x_train is None
        assert client.y_train is None

    def test_creation_with_data(self, client_mod, model_mod):
        client = _make_client(client_mod, model_mod)
        assert client.x_train is not None
        assert client.y_train is not None
        assert client.x_train.shape == (50, 4)

    def test_set_data(self, client_mod, model_mod):
        client = _make_client(client_mod, model_mod, with_data=False)
        x = np.zeros((10, 4))
        y = np.zeros(10, dtype=int)
        client.set_data(x, y)
        assert client.x_train is not None
        assert len(client.x_train) == 10

    def test_get_sample_count_no_data(self, client_mod, model_mod):
        client = _make_client(client_mod, model_mod, with_data=False)
        assert client.get_sample_count() == 0

    def test_get_sample_count_with_data(self, client_mod, model_mod):
        client = _make_client(client_mod, model_mod)
        assert client.get_sample_count() == 50


class TestLocalTraining:
    """Tests for local FedAvg/FedProx training."""

    def test_train_local_basic(self, client_mod, model_mod):
        client = _make_client(client_mod, model_mod)
        global_params = client.get_parameters()
        result = client.train_local(global_params, epochs=1, batch_size=16)
        TrainingResult = client_mod.TrainingResult
        assert isinstance(result, TrainingResult)
        assert result.client_id == "hospital_A"
        assert result.epochs_completed == 1
        assert result.num_samples == 50
        assert isinstance(result.final_loss, float)

    def test_train_local_changes_parameters(self, client_mod, model_mod):
        client = _make_client(client_mod, model_mod)
        params_before = [p.copy() for p in client.get_parameters()]
        client.train_local(params_before, epochs=2, batch_size=16)
        params_after = client.get_parameters()
        differs = any(not np.allclose(a, b) for a, b in zip(params_before, params_after))
        assert differs, "Parameters should change after training"

    def test_train_local_fedprox_with_mu(self, client_mod, model_mod):
        client = _make_client(client_mod, model_mod)
        global_params = client.get_parameters()
        result = client.train_local(global_params, epochs=1, mu=0.1)
        assert result.epochs_completed == 1

    def test_train_local_no_data_raises(self, client_mod, model_mod):
        client = _make_client(client_mod, model_mod, with_data=False)
        with pytest.raises(RuntimeError, match="no training data"):
            client.train_local([np.zeros((4, 8))], epochs=1)

    def test_training_history_accumulates(self, client_mod, model_mod):
        client = _make_client(client_mod, model_mod)
        params = client.get_parameters()
        client.train_local(params, epochs=1)
        client.train_local(params, epochs=1)
        history = client.get_training_history()
        assert len(history) == 2

    def test_update_model_sets_parameters(self, client_mod, model_mod):
        client = _make_client(client_mod, model_mod)
        new_params = [np.ones_like(p) * 0.5 for p in client.get_parameters()]
        client.update_model(new_params)
        current = client.get_parameters()
        for p in current:
            npt.assert_allclose(p, np.ones_like(p) * 0.5)


class TestScaffoldTraining:
    """Tests for SCAFFOLD local training."""

    def test_train_local_scaffold_basic(self, client_mod, model_mod):
        client = _make_client(client_mod, model_mod)
        params = client.get_parameters()
        server_ctrl = [np.zeros_like(p) for p in params]
        client_ctrl = [np.zeros_like(p) for p in params]
        result, delta = client.train_local_scaffold(params, server_ctrl, client_ctrl, epochs=1, batch_size=16)
        TrainingResult = client_mod.TrainingResult
        assert isinstance(result, TrainingResult)
        assert isinstance(delta, list)
        assert len(delta) == len(params)

    def test_train_local_scaffold_no_data_raises(self, client_mod, model_mod):
        client = _make_client(client_mod, model_mod, with_data=False)
        params = client.get_parameters()
        sc = [np.zeros_like(p) for p in params]
        cc = [np.zeros_like(p) for p in params]
        with pytest.raises(RuntimeError, match="no training data"):
            client.train_local_scaffold(params, sc, cc, epochs=1)

    def test_scaffold_control_delta_shapes(self, client_mod, model_mod):
        client = _make_client(client_mod, model_mod)
        params = client.get_parameters()
        sc = [np.zeros_like(p) for p in params]
        cc = [np.zeros_like(p) for p in params]
        _, delta = client.train_local_scaffold(params, sc, cc, epochs=1)
        for d, p in zip(delta, params):
            assert d.shape == p.shape


class TestParameterShapes:
    """Tests for parameter extraction shapes."""

    def test_parameter_count(self, client_mod, model_mod):
        client = _make_client(client_mod, model_mod)
        params = client.get_parameters()
        # Architecture: 4 -> [8] -> 2  =>  (W1, b1, W2, b2) = 4 arrays
        assert len(params) == 4

    def test_parameter_shapes(self, client_mod, model_mod):
        client = _make_client(client_mod, model_mod)
        params = client.get_parameters()
        assert params[0].shape == (4, 8)  # W1
        assert params[1].shape == (8,)  # b1
        assert params[2].shape == (8, 2)  # W2
        assert params[3].shape == (2,)  # b2
