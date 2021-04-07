from datetime import datetime, timedelta

import numpy as np
import pytest

import hela.generation.hmm as gen
import hela.hmm as hmm
from hela.hmm.graphical_models import DynamicBayesianNetwork as dbn
from hela.hmm.graphical_models import structured_inference as dbn_inf
from hela.hmm.graphical_models.ContinuousFactor import ContinuousFactor


@pytest.fixture(scope="module")
def generative_model():
    gen_model = gen.FactoredHMMGenerativeModel(
        ns_hidden_states=[3, 2],
        n_categorical_features=2,
        n_gaussian_features=1)

    factored_hidden_states = gen_model.generate_hidden_state_sequence(
        n_observations=100)
    dataset = gen_model.generate_observations(factored_hidden_states)

    fhmm_training_spec = gen.data_to_fhmm_training_spec(
        factored_hidden_states,
        gen_model.ns_hidden_states,
        dataset,
        categorical_features=list(gen_model.categorical_values.columns),
        gaussian_features=list(gen_model.gaussian_values.columns))

    model_config = hmm.FactoredHMMConfiguration.from_spec(fhmm_training_spec)
    model = model_config.to_model()
    graph = dbn.fhmm_model_to_graph(model)

    return {
        "model": model,
        "graph": graph
    }

@pytest.fixture(scope="module")
def generative_categorical_model():
    gen_model = gen.FactoredHMMGenerativeModel(
        ns_hidden_states=[3, 2],
        n_categorical_features=2,
        n_gaussian_features=0)

    factored_hidden_states = gen_model.generate_hidden_state_sequence(
        n_observations=100)
    dataset = gen_model.generate_observations(factored_hidden_states)

    fhmm_training_spec = gen.data_to_fhmm_training_spec(
        factored_hidden_states,
        gen_model.ns_hidden_states,
        dataset,
        categorical_features=list(gen_model.categorical_values.columns),
        gaussian_features=[])

    model_config = hmm.FactoredHMMConfiguration.from_spec(fhmm_training_spec)
    model = model_config.to_model()
    graph = dbn.fhmm_model_to_graph(model)

    return {
        "model": model,
        "graph": graph,
        "dataset": dataset
    }


def test_continuous_factors(generative_model):
    model = generative_model['model']
    graph = generative_model['graph']

    continuous_factors0 = [
        factor for factors in graph.get_factors(time_slice=0)
        for factor in factors if isinstance(factor, ContinuousFactor)
    ]
    continuous_factors1 = [
        factor for factors in graph.get_factors(time_slice=1)
        for factor in factors if isinstance(factor, ContinuousFactor)
    ]

    # weights are the same across edges
    assert continuous_factors0[0].weights.all() == continuous_factors1[
        0].weights.all()
    assert continuous_factors0[1].weights.all() == continuous_factors1[
        1].weights.all()

    # weights and covariance match with model
    assert continuous_factors0[0].weights.all() == model.gaussian_model.means[
        0].all()
    assert continuous_factors0[1].weights.all() == model.gaussian_model.means[
        1].all()
    assert continuous_factors0[
        0].covariance.all() == model.gaussian_model.covariance.all()

    # cardinality of latent nodes associated with continuous_factors is equal to hidden states
    evidence_cards = [factor.cardinality[1] for factor in continuous_factors0]
    variable_cards = [factor.cardinality[0] for factor in continuous_factors0]
    assert evidence_cards == model.ns_hidden_states

    # cardinality of continuous observation nodes is equal to gaussian features
    assert variable_cards[0] == variable_cards[1] == len(
        model.gaussian_features)


def test_discrete_factors(generative_model):
    graph = generative_model['graph']
    # Built in method for testing TabularCPD factors
    assert graph.check_model()


def test_forward_backward(generative_categorical_model):
    fhmm_graph = generative_categorical_model["graph"]
    model = generative_categorical_model["model"]
    dataset = generative_categorical_model["dataset"]
    system_graphs = fhmm_graph.generate_system_graphs()
    # Correct number of graphs are generated
    assert len(system_graphs) == len(model.ns_hidden_states)

    inference = dbn_inf.DBNInference(system_graphs[0])
    probability = inference.forward_backward(dataset)

    assert probability.shape[0] == dataset.shape[0]-1




