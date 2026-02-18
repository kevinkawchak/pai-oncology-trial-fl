"""Tests for the federation coordinator.

RESEARCH USE ONLY — Not intended for clinical deployment without validation.
"""

import numpy as np
import pytest

from federated.coordinator import AggregationStrategy, FederationCoordinator, RoundResult
from federated.model import FederatedModel, ModelConfig


@pytest.fixture()
def model_config():
    return ModelConfig(input_dim=10, hidden_dims=[8], output_dim=2, seed=42)


@pytest.fixture()
def coordinator(model_config):
    return FederationCoordinator(model_config=model_config, num_rounds=5, min_clients=2)


class TestFederationCoordinator:
    def test_initialize(self, coordinator):
        params = coordinator.initialize()
        assert len(params) > 0
        assert coordinator.global_model is not None
        assert coordinator.current_round == 0

    def test_get_global_parameters(self, coordinator):
        coordinator.initialize()
        params = coordinator.get_global_parameters()
        assert isinstance(params, list)
        assert all(isinstance(p, np.ndarray) for p in params)

    def test_get_global_parameters_before_init_raises(self, coordinator):
        with pytest.raises(RuntimeError):
            coordinator.get_global_parameters()

    def test_run_round_basic(self, coordinator, model_config):
        global_params = coordinator.initialize()

        client_updates = []
        for _ in range(2):
            m = FederatedModel.from_config(model_config)
            m.set_parameters(global_params)
            updated = [p + np.random.randn(*p.shape) * 0.01 for p in m.get_parameters()]
            client_updates.append(updated)

        result = coordinator.run_round(client_updates)
        assert isinstance(result, RoundResult)
        assert result.round_number == 1
        assert result.num_clients == 2

    def test_run_round_with_eval(self, coordinator, model_config):
        global_params = coordinator.initialize()

        client_updates = []
        for _ in range(2):
            m = FederatedModel.from_config(model_config)
            m.set_parameters(global_params)
            client_updates.append(m.get_parameters())

        x_eval = np.random.randn(20, 10)
        y_eval = np.random.randint(0, 2, 20)
        result = coordinator.run_round(client_updates, eval_data=(x_eval, y_eval))
        assert "accuracy" in result.global_metrics

    def test_run_round_too_few_clients(self, coordinator):
        coordinator.initialize()
        with pytest.raises(ValueError, match="minimum"):
            coordinator.run_round([])

    def test_weighted_aggregation(self, coordinator, model_config):
        global_params = coordinator.initialize()
        p1 = [p + 0.1 for p in global_params]
        p2 = [p - 0.1 for p in global_params]

        coordinator.run_round([p1, p2], client_sample_counts=[100, 100])
        new_params = coordinator.get_global_parameters()
        for orig, new in zip(global_params, new_params):
            assert np.allclose(orig, new, atol=0.01)

    def test_training_summary(self, coordinator, model_config):
        global_params = coordinator.initialize()
        updates = [global_params, global_params]
        coordinator.run_round(updates)
        coordinator.run_round(updates)

        summary = coordinator.get_training_summary()
        assert summary["total_rounds"] == 2
        assert len(summary["rounds"]) == 2
        assert summary["strategy"] == "fedavg"


class TestCoordinatorWithDP:
    def test_dp_enabled(self, model_config):
        coord = FederationCoordinator(
            model_config=model_config,
            use_differential_privacy=True,
            dp_epsilon=1.0,
            dp_delta=1e-5,
        )
        global_params = coord.initialize()
        updates = [global_params, global_params]
        coord.run_round(updates)
        assert coord.dp is not None
        assert coord.dp.rounds_completed == 1


class TestCoordinatorWithSecureAgg:
    def test_secure_agg_enabled(self, model_config):
        coord = FederationCoordinator(
            model_config=model_config,
            use_secure_aggregation=True,
        )
        global_params = coord.initialize()
        updates = [global_params, global_params]
        result = coord.run_round(updates)
        assert result.num_clients == 2


class TestAggregationStrategies:
    def test_fedprox_strategy(self, model_config):
        coord = FederationCoordinator(
            model_config=model_config,
            strategy="fedprox",
            mu=0.01,
        )
        assert coord.strategy == AggregationStrategy.FEDPROX
        global_params = coord.initialize()
        updates = [global_params, global_params]
        result = coord.run_round(updates)
        assert result.round_number == 1

    def test_scaffold_strategy(self, model_config):
        coord = FederationCoordinator(
            model_config=model_config,
            strategy="scaffold",
        )
        assert coord.strategy == AggregationStrategy.SCAFFOLD
        global_params = coord.initialize()

        # SCAFFOLD should initialise server control
        assert coord.get_server_control() is not None
        control = coord.get_client_control("client_0")
        assert all(np.allclose(c, 0) for c in control)

        # Run a round with control deltas
        updates = [global_params, global_params]
        deltas = [[np.zeros_like(p) for p in global_params] for _ in range(2)]
        result = coord.run_round(
            updates,
            client_ids=["client_0", "client_1"],
            client_control_deltas=deltas,
        )
        assert result.round_number == 1


class TestConvergenceDetection:
    def test_converges_when_stable(self, model_config):
        coord = FederationCoordinator(
            model_config=model_config,
            convergence_window=3,
            convergence_threshold=0.01,
        )
        global_params = coord.initialize()
        updates = [global_params, global_params]
        x_eval = np.random.randn(20, 10)
        y_eval = np.random.randint(0, 2, 20)

        for _ in range(5):
            result = coord.run_round(updates, eval_data=(x_eval, y_eval))

        # After several rounds with identical updates, should converge
        assert result.converged is True

    def test_does_not_converge_early(self, model_config):
        coord = FederationCoordinator(
            model_config=model_config,
            convergence_window=10,
            convergence_threshold=0.001,
        )
        global_params = coord.initialize()
        updates = [global_params, global_params]
        x_eval = np.random.randn(20, 10)
        y_eval = np.random.randint(0, 2, 20)

        result = coord.run_round(updates, eval_data=(x_eval, y_eval))
        # Only one round — not enough for convergence window
        assert result.converged is False
