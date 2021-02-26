from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from dask.distributed import Client
from scipy.special import logsumexp

import hela.generation.hmm as gen
import hela.hmm as hmm
from hela.walk_forward_concordance import WalkForwardConcordance


@pytest.fixture(scope="module")
def random():
    return np.random.RandomState(0)


@pytest.fixture(scope="module")
def n_hidden_states(random):
    return random.choice(range(2, 6))


@pytest.fixture(scope="module")
def generative_model(random, n_hidden_states):
    gen_model = gen.FactoredHMMGenerativeModel(
        ns_hidden_states=[3, 2],
        n_categorical_features=2,
        n_gaussian_features=1)

    factored_hidden_states = gen_model.generate_hidden_state_sequence(
        n_observations=500)
    dataset = gen_model.generate_observations(factored_hidden_states)

    fhmm_training_spec = gen.data_to_fhmm_training_spec(
        factored_hidden_states,
        gen_model.ns_hidden_states,
        dataset,
        categorical_features=list(gen_model.categorical_values.columns),
        gaussian_features=list(gen_model.gaussian_values.columns))

    return {
        "gen_model": gen_model,
        "dataset": dataset,
        "factored_hidden_states": factored_hidden_states,
        "fhmm_training_spec": fhmm_training_spec
    }


def test_model_loads_from_spec(generative_model):

    fhmm_training_spec = generative_model["fhmm_training_spec"]

    model_config = hmm.FactoredHMMConfiguration.from_spec(fhmm_training_spec)
    model = model_config.to_model()

    assert model.ns_hidden_states == fhmm_training_spec['hidden_state']['count']


def test_model_sampling_and_inference(generative_model):

    fhmm_training_spec = generative_model["fhmm_training_spec"]
    dataset = generative_model["dataset"]
    factored_hidden_states = generative_model["factored_hidden_states"]

    model_config = hmm.FactoredHMMConfiguration.from_spec(fhmm_training_spec)
    model = model_config.to_model()
    inf = model.to_inference_interface(dataset)

    Gamma, Xi, gibbs_states = inf.gibbs_sampling(
        dataset, iterations=1, burn_down_period=1)

    assert np.all(gibbs_states.index == dataset.index)

    Gamma, Xi, gibbs_states = inf.gibbs_sampling(
        dataset,
        iterations=5,
        burn_down_period=1,
        gather_statistics=True,
        hidden_state_vector_df=gibbs_states)

    assert Gamma.shape[0] == dataset.shape[0]

    # Make sure that the subdiagonal terms of Gamma sum to 1.
    csum = np.concatenate(([0],np.cumsum(model.ns_hidden_states)))
    Gamma_sum = np.array([[g.diagonal()[csum[i]:csum[i+1]] for i in range(len(model.ns_hidden_states))] for g in Gamma])
    assert np.all([[np.sum(d) == 1 for d in g] for g in Gamma_sum])
    
    # Make sure that each block of Xi sums to 1 (i.e. for any system, m, and
    # timestamp, t, the full entries of Xi[m][t] sum to 1).
    assert np.all(np.sum(np.sum(Xi, axis = 3), axis = 2) == 1)
