"""Tests for federated/coordinator.py — FederationCoordinator.

Covers creation, FedAvg/FedProx/SCAFFOLD aggregation, round metrics,
control variate access, division-by-zero guards, empty client updates,
and convergence detection.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from tests.conftest import PROJECT_ROOT, load_module


@pytest.fixture(scope="module")
def coordinator_mod():
    return load_module("federated.coordinator", "federated/coordinator.py")


@pytest.fixture(scope="module")
def model_mod():
    return load_module("federated.model", "federated/model.py")


def _default_config(model_mod):
    ModelConfig = model_mod.ModelConfig
    return ModelConfig(input_dim=4, hidden_dims=[8], output_dim=2, learning_rate=0.01, seed=0)


def _make_coordinator(coordinator_mod, model_mod, strategy="fedavg", **kwargs):
    cfg = _default_config(model_mod)
    return coordinator_mod.FederationCoordinator(model_config=cfg, strategy=strategy, min_clients=1, **kwargs)


def _fake_updates(params, n_clients=3, noise=0.01):
    rng = np.random.default_rng(99)
    return [[p + rng.normal(0, noise, size=p.shape) for p in params] for _ in range(n_clients)]


class TestFederationCoordinatorCreation:
    """Tests for coordinator instantiation."""

    def test_creation_defaults(self, coordinator_mod, model_mod):
        coord = _make_coordinator(coordinator_mod, model_mod)
        assert coord.strategy == coordinator_mod.AggregationStrategy.FEDAVG
        assert coord.num_rounds == 10
        assert coord.current_round == 0
        assert coord.global_model is None

    def test_creation_with_string_strategy(self, coordinator_mod, model_mod):
        coord = _make_coordinator(coordinator_mod, model_mod, strategy="fedprox")
        assert coord.strategy == coordinator_mod.AggregationStrategy.FEDPROX

    def test_creation_scaffold_enum(self, coordinator_mod, model_mod):
        coord = _make_coordinator(
            coordinator_mod,
            model_mod,
            strategy=coordinator_mod.AggregationStrategy.SCAFFOLD,
        )
        assert coord.strategy == coordinator_mod.AggregationStrategy.SCAFFOLD

    def test_creation_invalid_strategy_raises(self, coordinator_mod, model_mod):
        with pytest.raises(ValueError):
            _make_coordinator(coordinator_mod, model_mod, strategy="unknown_strategy")

    def test_initialize_returns_parameters(self, coordinator_mod, model_mod):
        coord = _make_coordinator(coordinator_mod, model_mod)
        params = coord.initialize()
        assert isinstance(params, list)
        assert all(isinstance(p, np.ndarray) for p in params)
        assert coord.global_model is not None
        assert coord.current_round == 0

    def test_get_global_parameters_before_init_raises(self, coordinator_mod, model_mod):
        coord = _make_coordinator(coordinator_mod, model_mod)
        with pytest.raises(RuntimeError, match="not initialized"):
            coord.get_global_parameters()


class TestFedAvgAggregation:
    """Tests for FedAvg weighted aggregation."""

    def test_fedavg_equal_weights(self, coordinator_mod):
        FederationCoordinator = coordinator_mod.FederationCoordinator
        updates = [
            [np.array([1.0, 2.0]), np.array([3.0])],
            [np.array([3.0, 4.0]), np.array([5.0])],
        ]
        weights = [0.5, 0.5]
        result = FederationCoordinator._fedavg(updates, weights)
        npt.assert_allclose(result[0], [2.0, 3.0])
        npt.assert_allclose(result[1], [4.0])

    def test_fedavg_unequal_weights(self, coordinator_mod):
        FederationCoordinator = coordinator_mod.FederationCoordinator
        updates = [
            [np.array([0.0, 0.0])],
            [np.array([10.0, 10.0])],
        ]
        weights = [0.8, 0.2]
        result = FederationCoordinator._fedavg(updates, weights)
        npt.assert_allclose(result[0], [2.0, 2.0])

    def test_fedavg_single_client(self, coordinator_mod):
        FederationCoordinator = coordinator_mod.FederationCoordinator
        updates = [[np.array([5.0, 6.0])]]
        weights = [1.0]
        result = FederationCoordinator._fedavg(updates, weights)
        npt.assert_allclose(result[0], [5.0, 6.0])

    def test_run_round_fedavg(self, coordinator_mod, model_mod):
        RoundResult = coordinator_mod.RoundResult
        coord = _make_coordinator(coordinator_mod, model_mod, strategy="fedavg")
        params = coord.initialize()
        updates = _fake_updates(params, n_clients=3)
        rr = coord.run_round(updates, client_sample_counts=[100, 100, 100])
        assert isinstance(rr, RoundResult)
        assert rr.round_number == 1
        assert rr.num_clients == 3
        assert coord.current_round == 1


class TestFedProxAggregation:
    """Tests for FedProx strategy coordinator-level behavior."""

    def test_fedprox_coordinator_mu(self, coordinator_mod, model_mod):
        coord = _make_coordinator(coordinator_mod, model_mod, strategy="fedprox", mu=0.1)
        assert coord.mu == 0.1
        assert coord.strategy == coordinator_mod.AggregationStrategy.FEDPROX

    def test_fedprox_round_executes(self, coordinator_mod, model_mod):
        coord = _make_coordinator(coordinator_mod, model_mod, strategy="fedprox")
        params = coord.initialize()
        updates = _fake_updates(params, n_clients=2)
        rr = coord.run_round(updates)
        assert rr.round_number == 1


class TestScaffoldStrategy:
    """Tests for SCAFFOLD control variates."""

    def test_scaffold_init_creates_server_control(self, coordinator_mod, model_mod):
        coord = _make_coordinator(coordinator_mod, model_mod, strategy="scaffold")
        coord.initialize()
        sc = coord.get_server_control()
        assert sc is not None
        for arr in sc:
            npt.assert_allclose(arr, np.zeros_like(arr))

    def test_scaffold_round_with_control_deltas(self, coordinator_mod, model_mod):
        coord = _make_coordinator(coordinator_mod, model_mod, strategy="scaffold")
        params = coord.initialize()
        updates = _fake_updates(params, n_clients=2)
        deltas = [[np.ones_like(p) * 0.01 for p in params] for _ in range(2)]
        rr = coord.run_round(
            updates,
            client_ids=["c0", "c1"],
            client_control_deltas=deltas,
        )
        assert rr.round_number == 1
        sc = coord.get_server_control()
        assert sc is not None
        assert not all(np.allclose(c, 0.0) for c in sc)

    def test_get_client_control_returns_copy(self, coordinator_mod, model_mod):
        coord = _make_coordinator(coordinator_mod, model_mod, strategy="scaffold")
        coord.initialize()
        cc = coord.get_client_control("new_client")
        cc[0].flat[0] = 999.0
        cc2 = coord.get_client_control("new_client")
        assert cc2[0].flat[0] != 999.0

    def test_get_server_control_returns_copy(self, coordinator_mod, model_mod):
        coord = _make_coordinator(coordinator_mod, model_mod, strategy="scaffold")
        coord.initialize()
        sc = coord.get_server_control()
        sc[0].flat[0] = 999.0
        sc2 = coord.get_server_control()
        assert sc2[0].flat[0] != 999.0

    def test_get_client_control_without_scaffold_raises(self, coordinator_mod, model_mod):
        coord = _make_coordinator(coordinator_mod, model_mod, strategy="fedavg")
        coord.initialize()
        with pytest.raises(RuntimeError, match="SCAFFOLD not enabled"):
            coord.get_client_control("c0")

    def test_get_server_control_non_scaffold_returns_none(self, coordinator_mod, model_mod):
        coord = _make_coordinator(coordinator_mod, model_mod, strategy="fedavg")
        coord.initialize()
        assert coord.get_server_control() is None


class TestRoundEdgeCases:
    """Tests for edge cases in run_round."""

    def test_run_round_before_init_raises(self, coordinator_mod, model_mod):
        coord = _make_coordinator(coordinator_mod, model_mod)
        with pytest.raises(RuntimeError, match="not initialized"):
            coord.run_round([[np.array([1.0])]])

    def test_too_few_clients_raises(self, coordinator_mod, model_mod):
        coord = _make_coordinator(coordinator_mod, model_mod)
        coord.initialize()
        coord.min_clients = 3
        updates = _fake_updates(coord.get_global_parameters(), n_clients=2)
        with pytest.raises(ValueError, match="minimum is 3"):
            coord.run_round(updates)

    def test_division_by_zero_sample_counts(self, coordinator_mod, model_mod):
        coord = _make_coordinator(coordinator_mod, model_mod)
        params = coord.initialize()
        updates = _fake_updates(params, n_clients=2)
        rr = coord.run_round(updates, client_sample_counts=[0, 0])
        assert rr.num_clients == 2

    def test_none_sample_counts_uses_equal_weights(self, coordinator_mod, model_mod):
        coord = _make_coordinator(coordinator_mod, model_mod)
        params = coord.initialize()
        updates = _fake_updates(params, n_clients=2)
        rr = coord.run_round(updates, client_sample_counts=None)
        assert rr.num_clients == 2

    def test_round_with_eval_data(self, coordinator_mod, model_mod):
        coord = _make_coordinator(coordinator_mod, model_mod)
        params = coord.initialize()
        updates = _fake_updates(params, n_clients=2)
        x_eval = np.random.randn(10, 4).astype(np.float32)
        y_eval = np.random.randint(0, 2, size=10)
        rr = coord.run_round(updates, eval_data=(x_eval, y_eval))
        assert "accuracy" in rr.global_metrics
        assert "loss" in rr.global_metrics


class TestConvergenceDetection:
    """Tests for convergence checking."""

    def test_no_convergence_without_accuracy(self, coordinator_mod, model_mod):
        coord = _make_coordinator(coordinator_mod, model_mod, convergence_window=3)
        coord.initialize()
        params = coord.get_global_parameters()
        updates = _fake_updates(params, n_clients=2)
        rr = coord.run_round(updates)
        assert rr.converged is False

    def test_convergence_disabled_when_window_zero(self, coordinator_mod, model_mod):
        coord = _make_coordinator(coordinator_mod, model_mod, convergence_window=0)
        coord.initialize()
        params = coord.get_global_parameters()
        x_eval = np.random.randn(20, 4).astype(np.float32)
        y_eval = np.random.randint(0, 2, size=20)
        for _ in range(10):
            updates = _fake_updates(params, n_clients=2, noise=0.0001)
            coord.run_round(updates, eval_data=(x_eval, y_eval))
        assert all(r.converged is False for r in coord.round_history)

    def test_convergence_triggers_with_plateau(self, coordinator_mod, model_mod):
        RoundResult = coordinator_mod.RoundResult
        coord = _make_coordinator(
            coordinator_mod,
            model_mod,
            convergence_window=3,
            convergence_threshold=0.5,
        )
        coord.initialize()
        for i in range(5):
            coord.round_history.append(RoundResult(round_number=i + 1, num_clients=2, global_metrics={"accuracy": 0.8}))
        result = coord._check_convergence({"accuracy": 0.8})
        assert result is True

    def test_no_convergence_with_improvement(self, coordinator_mod, model_mod):
        RoundResult = coordinator_mod.RoundResult
        coord = _make_coordinator(
            coordinator_mod,
            model_mod,
            convergence_window=3,
            convergence_threshold=0.001,
        )
        coord.initialize()
        coord.round_history.append(RoundResult(round_number=1, num_clients=2, global_metrics={"accuracy": 0.5}))
        coord.round_history.append(RoundResult(round_number=2, num_clients=2, global_metrics={"accuracy": 0.6}))
        result = coord._check_convergence({"accuracy": 0.9})
        assert result is False


class TestTrainingSummary:
    """Tests for reporting."""

    def test_get_training_summary_empty(self, coordinator_mod, model_mod):
        coord = _make_coordinator(coordinator_mod, model_mod)
        coord.initialize()
        summary = coord.get_training_summary()
        assert summary["total_rounds"] == 0
        assert summary["strategy"] == "fedavg"
        assert summary["rounds"] == []

    def test_get_training_summary_after_rounds(self, coordinator_mod, model_mod):
        coord = _make_coordinator(coordinator_mod, model_mod)
        params = coord.initialize()
        updates = _fake_updates(params, n_clients=2)
        coord.run_round(updates)
        coord.run_round(updates)
        summary = coord.get_training_summary()
        assert summary["total_rounds"] == 2
        assert len(summary["rounds"]) == 2
        assert summary["rounds"][0]["round"] == 1
