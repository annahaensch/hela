from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from dask.distributed import Client
from scipy.special import logsumexp

import hela.generation.hmm as gen
import hela.hmm as hmm

@pytest.fixture(scope="module")
def random():
    return np.random.RandomState(0)


@pytest.fixture(scope="module")
def n_hidden_states(random):
    return random.choice(range(2, 6))


@pytest.fixture(scope="module")
def generative_model(random, n_hidden_states):
    gen_model = gen.FactoredHMMGenerativeModel(
        n_hidden_states=[3, 2], n_categorical_features=2, n_gaussian_features=1)

    factored_hidden_states = gen_model.generate_hidden_state_sequence(
        n_observations=500)
    dataset = gen_model.generate_observations(factored_hidden_states)

    fhmm_training_spec = gen.data_to_fhmm_training_spec(
        factored_hidden_states,
        gen_model.n_hidden_states,
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

    assert model.n_hidden_states == fhmm_training_spec['n_hidden_states']


def test_model_sampling_and_inference(generative_model):

    fhmm_training_spec = generative_model["fhmm_training_spec"]
    dataset = generative_model["dataset"]
    factored_hidden_states = generative_model["factored_hidden_states"]

    model_config = hmm.FactoredHMMConfiguration.from_spec(fhmm_training_spec)
    model = model_config.to_model()
    inf = model.load_inference_interface()

    Gamma, Xi, gibbs_states = inf.gibbs_sampling(
        dataset, iterations=1, burn_down_period=1)

    assert np.all(gibbs_states.index == dataset.index)

    Gamma, Xi, gibbs_states = inf.gibbs_sampling(
        dataset,
        iterations=2,
        burn_down_period=0,
        gather_statistics=True,
        hidden_state_vector_df=gibbs_states)

    assert Gamma.shape[0] == dataset.shape[0]

    # Make sure that the subdiagonal terms of Gamma sum to 1.
    csum = np.concatenate(([0], np.cumsum(model.n_hidden_states)))
    Gamma_sum = np.array([[
        g.diagonal()[csum[i]:csum[i + 1]]
        for i in range(len(model.n_hidden_states))
    ]
                          for g in Gamma])
    assert np.all([[np.sum(d) == 1 for d in g] for g in Gamma_sum])

    # Make sure that each block of Xi sums to 1 (i.e. for any system, m, and
    # timestamp, t, the full entries of Xi[m][t] sum to 1).
    assert np.all(np.sum(np.sum(Xi, axis=3), axis=2) == 1)


def test_learning_with_gibbs(generative_model):

    fhmm_training_spec = generative_model["fhmm_training_spec"]
    dataset = generative_model["dataset"]
    factored_hidden_states = generative_model["factored_hidden_states"]

    model_config = hmm.FactoredHMMConfiguration.from_spec(fhmm_training_spec)
    untrained_model = model_config.to_model()
    inf = untrained_model.load_inference_interface()
    alg = untrained_model.load_learning_interface()

    model = alg.run(
        data=dataset,
        method='gibbs',
        training_iterations=5,
        gibbs_iterations=5,
        burn_down_period=2)

    # Check that (up to floating point errors) the transition and emission
    # probabilities sum to the proper value.
    assert np.all(
        np.sum(np.sum(model.transition_matrix, axis=2), axis=1) -
        model.n_hidden_states < 1e-08)

    assert np.all(
        (np.sum(model.categorical_model.emission_matrix, axis=0) - 1) < 1e-08)

    # Check that complete data likelihood is increasing with each iteration.
    likelihood = []
    for m in alg.model_results:
        spec = hmm._factored_hmm_to_discrete_hmm(m)
        hmm_config = hmm.DiscreteHMMConfiguration.from_spec(spec)
        hmm_model = hmm_config.to_model()
        hmm_inf = hmm_model.load_inference_interface()
        log_prob = hmm_inf.predict_hidden_state_log_probability(dataset)
        likelihood.append(
            logsumexp(hmm_inf._compute_forward_probabilities(log_prob)[-1]))

    # Check that cummulative sum of negative log likelihoods is
    # concave down by checking sign of approx. second derivative.
    csum = np.cumsum([-l for l in likelihood])
    concavity = [
        csum[i] - 2 * csum[i + 1] + csum[i - 2]
        for i in range(2,
                       len(csum) - 1)
    ]

    assert np.all(np.array(concavity) < 0)


def test_learning_with_distributed_gibbs(generative_model):

    fhmm_training_spec = generative_model["fhmm_training_spec"]
    dataset = generative_model["dataset"]
    factored_hidden_states = generative_model["factored_hidden_states"]

    model_config = hmm.FactoredHMMConfiguration.from_spec(fhmm_training_spec)
    untrained_model = model_config.to_model()
    inf = untrained_model.load_inference_interface()
    alg = untrained_model.load_learning_interface()

    model = alg.run(
        data=dataset,
        method='gibbs',
        training_iterations=5,
        gibbs_iterations=5,
        burn_down_period=2,
        distributed=True,
        n_workers=2)

    # Check that (up to floating point errors) the transition and emission
    # probabilities sum to the proper value.
    assert np.all(
        np.sum(np.sum(model.transition_matrix, axis=2), axis=1) -
        model.n_hidden_states < 1e-08)

    assert np.all(
        (np.sum(model.categorical_model.emission_matrix, axis=0) - 1) < 1e-08)

    # Check that complete data likelihood is increasing with each iteration.
    likelihood = []
    for m in alg.model_results:
        spec = hmm._factored_hmm_to_discrete_hmm(m)
        hmm_config = hmm.DiscreteHMMConfiguration.from_spec(spec)
        hmm_model = hmm_config.to_model()
        hmm_inf = hmm_model.load_inference_interface()
        log_prob = hmm_inf.predict_hidden_state_log_probability(dataset)
        likelihood.append(
            logsumexp(hmm_inf._compute_forward_probabilities(log_prob)[-1]))

    # Check that cummulative sum of negative log likelihoods is
    # concave down by checking sign of approx. second derivative.
    csum = np.cumsum([-l for l in likelihood])
    concavity = [
        csum[i] - 2 * csum[i + 1] + csum[i - 2]
        for i in range(2,
                       len(csum) - 1)
    ]

    assert np.all(np.array(concavity) < 0)


def test_learning_with_structured_vi(generative_model):

    fhmm_training_spec = generative_model["fhmm_training_spec"]
    dataset = generative_model["dataset"]
    factored_hidden_states = generative_model["factored_hidden_states"]

    model_config = hmm.FactoredHMMConfiguration.from_spec(fhmm_training_spec)
    untrained_model = model_config.to_model()
    inf = untrained_model.load_inference_interface()
    alg = untrained_model.load_learning_interface()

    model = alg.run(data=dataset, method='structured_vi', training_iterations=5)

    # Check that complete data likelihood is increasing with each iteration.
    likelihood = []
    for m in alg.model_results:
        spec = hmm._factored_hmm_to_discrete_hmm(m)
        hmm_config = hmm.DiscreteHMMConfiguration.from_spec(spec)
        hmm_model = hmm_config.to_model()
        hmm_inf = hmm_model.load_inference_interface()
        log_prob = hmm_inf.predict_hidden_state_log_probability(dataset)
        likelihood.append(
            logsumexp(hmm_inf._compute_forward_probabilities(log_prob)[-1]))

    # Check that cummulative sum of negative log likelihoods is
    # concave down by checking sign of approx. second derivative.
    csum = np.cumsum([-l for l in likelihood])
    concavity = [
        csum[i] - 2 * csum[i + 1] + csum[i - 2]
        for i in range(2,
                       len(csum) - 1)
    ]

    assert np.all(np.array(concavity) < 0)
