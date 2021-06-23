import itertools
import logging
import sys
import time
# Utility Libraries
from datetime import datetime

import numpy as np
import pandas as pd
from dask.distributed import Client
from scipy import stats
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture

import hela
import hela.generation.hmm as hmm_gen
# Hela ML libraries
from hela import hmm

logging.basicConfig(level=logging.INFO)

LOG_ZERO = -1e8

RANDOM_STATES = [9, 10, 11]
ITERATIONS = 100


def get_true_ll(gen, train_data, test_data):
    """ Returns negative log likelihood of true generative model.

    Arguments: 
        gen: FactoredHMMGenerativeModel instance
        train_data: (dataframe) training data
        test_data: (dataframe) test data

    Returns:
        Training data and testing data log likelihood under the 
        true generative model, divided by the total number of 
        observations.
    """
    true_fhmm_spec = gen.generative_model_to_fhmm_training_spec()

    true_fhmm_config = hmm.FactoredHMMConfiguration.from_spec(true_fhmm_spec)
    true_fhmm_model = true_fhmm_config.to_model()

    true_hmm_spec = hmm._factored_hmm_to_discrete_hmm(true_fhmm_model)
    true_hmm_config = hmm.DiscreteHMMConfiguration.from_spec(true_hmm_spec)
    true_hmm_model = true_hmm_config.to_model()

    true_hmm_inf = true_hmm_model.load_inference_interface()
    train_log_prob = true_hmm_inf.predict_hidden_state_log_probability(
        train_data)
    test_log_prob = true_hmm_inf.predict_hidden_state_log_probability(test_data)

    true_train_ll = (-1 / train_data.shape[0]) * logsumexp(
        true_hmm_inf._compute_forward_probabilities(train_log_prob)[-1])
    true_test_ll = (-1 / test_data.shape[0]) * logsumexp(
        true_hmm_inf._compute_forward_probabilities(test_log_prob)[-1])

    return true_train_ll, true_test_ll


def train_hmm_em(gen, random_state, train_data):
    """ Trains hmm using em algorithm

    Arguments: 
        gen: FactoredHMMGenerativeModel instance
        random_State: (int) randomizing state
        train_data: (dataframe) training data

    Returns:
        EM training algorithm output and trained hmm model.
    """

    n_gmm_components = 1
    n_hidden_states = np.prod(gen.ns_hidden_states)

    r = np.random.RandomState(random_state)

    covariances = [np.identity(len(gen.gaussian_values.columns))]
    means = r.rand(1, len(gen.gaussian_values.columns))
    weights = np.array([1])

    observations = [{
        'name': col,
        'type': 'continuous',
        'dist': 'gaussian',
        'dims': 1
    } for col in list(gen.gaussian_values.columns)]

    hmm_spec = {
        'hidden_state': {
            'type': 'Finite',
            'count': n_hidden_states
        },
        'observations': observations,
        'model_parameter_constraints': {
            'gmm_parameter_constraints': {
                'n_gmm_components': 1,
                'component_weights': np.array(n_hidden_states * [weights]),
                'means': np.array(n_hidden_states * [means]),
                'covariances': np.array(n_hidden_states * [covariances])
            }
        }
    }

    hmm_model_config = hmm.DiscreteHMMConfiguration.from_spec(hmm_spec)
    untrained_hmm_model = hmm_model_config.to_model(
        set_random_state=random_state)

    em_alg = hmm.LearningAlgorithm()

    em_hmm_model = em_alg.run(
        untrained_hmm_model, train_data, n_em_iterations=ITERATIONS)

    return em_alg, em_hmm_model


def train_fhmm_vi(gen, random_state, train_data, train_fact_hidden_states):
    """ Trains hmm using vi em algorithm

    Arguments: 
        gen: FactoredHMMGenerativeModel instance
        random_State: (int) randomizing state
        train_data: (dataframe) training data
        train_fact_hidden_states: (dataframe) training hidden state sequence

    Returns:
        Variational Inference EM training algorithm output and trained fhmm model.
    """
    fhmm_training_spec = hmm_gen.data_to_fhmm_training_spec(
        train_fact_hidden_states,
        gen.ns_hidden_states,
        train_data,
        categorical_features=[],
        gaussian_features=list(gen.gaussian_values.columns))

    fhmm_config = hmm.FactoredHMMConfiguration.from_spec(fhmm_training_spec)
    untrained_fhmm_model = fhmm_config.to_model(set_random_state=random_state)

    vi_alg = untrained_fhmm_model.load_learning_interface()
    vi_model = vi_alg.run(
        data=train_data, method='structured_vi', training_iterations=ITERATIONS)

    return vi_alg, vi_model


def train_fhmm_gibbs(gen, random_state, train_data, train_fact_hidden_states):
    """ Trains hmm using gibbs approximate em algorithm

    Arguments: 
        gen: FactoredHMMGenerativeModel instance
        random_State: (int) randomizing state
        train_data: (dataframe) training data
        train_fact_hidden_states: (dataframe) training hidden state sequence
        
    Returns:
        Gibbs sampling approximate em training algorithm output and trained fhmm model.
    """
    fhmm_training_spec = hmm_gen.data_to_fhmm_training_spec(
        train_fact_hidden_states,
        gen.ns_hidden_states,
        train_data,
        categorical_features=[],
        gaussian_features=list(gen.gaussian_values.columns))

    fhmm_config = hmm.FactoredHMMConfiguration.from_spec(fhmm_training_spec)
    untrained_fhmm_model = fhmm_config.to_model(set_random_state=random_state)

    gibbs_alg = untrained_fhmm_model.load_learning_interface()
    gibbs_model = gibbs_alg.run(
        data=train_data,
        method='gibbs',
        training_iterations=ITERATIONS,
        gibbs_iterations=20)

    return gibbs_alg, gibbs_model


def compute_learning_ll(learning_alg, train_data, model_type):
    """ Returns log likelihood acroos iterations of EM.

    Arguments: 
        learning alg: algorithm class used for model training
        train_data: (dataframe) training data
        model_type: (str) "fhmm" or "hmm"

    Returns: 
        List of normalized negative log likelihoods.
    """
    learning_ll = []
    for m in learning_alg.model_results:

        if model_type == "fhmm":
            spec = hmm._factored_hmm_to_discrete_hmm(m)
            hmm_config = hmm.DiscreteHMMConfiguration.from_spec(spec)
            hmm_model = hmm_config.to_model()

        if model_type == "hmm":
            hmm_model = m

        hmm_inf = hmm_model.load_inference_interface()
        log_prob = hmm_inf.predict_hidden_state_log_probability(train_data)
        learning_ll.append(-1 * logsumexp(
            hmm_inf._compute_forward_probabilities(log_prob)[-1]))

    return np.array(learning_ll) / train_data.shape[0]


def compute_model_ll(model, train_data, test_data, model_type):
    """ Returns log likelihood of trained model.

    Arguments: 
        model: trained model instance
        train_data: (dataframe) training data
        test_data: (dataframe) test data
        model_type: (str) "fhmm" or "hmm"
        
    Returns: 
        normalized negative log likelihood for training and test data.
    """
    if model_type == "fhmm":
        fhmm_spec = hmm._factored_hmm_to_discrete_hmm(model)
        hmm_config = hmm.DiscreteHMMConfiguration.from_spec(fhmm_spec)
        hmm_model = hmm_config.to_model()

    if model_type == "hmm":
        hmm_model = model

    hmm_inf = hmm_model.load_inference_interface()
    train_log_prob = hmm_inf.predict_hidden_state_log_probability(train_data)
    test_log_prob = hmm_inf.predict_hidden_state_log_probability(test_data)

    train_ll = (-1 / train_data.shape[0]) * logsumexp(
        hmm_inf._compute_forward_probabilities(train_log_prob)[-1])
    test_ll = (-1 / test_data.shape[0]) * logsumexp(
        hmm_inf._compute_forward_probabilities(test_log_prob)[-1])

    return train_ll, test_ll


def main(argv):
    """ Prints summary dataframes to parque.

    Note: argv should be the pair m,n
    """
    m = int(str(argv[0]).split(",")[0])
    n = int(str(argv[0]).split(",")[1])

    ns_hidden_states = [n for i in range(m)]

    time_df = pd.DataFrame(
        columns=['systems', 'states', 'random', 'trial', 'em', 'vi', 'gibbs'])
    ll_df = pd.DataFrame(columns=[
        'systems', 'states', 'trial', 'random', 'train_true', 'test_true',
        'train_em', 'test_em', 'train_vi', 'test_vi', 'train_gibbs',
        'test_gibbs'
    ])
    em_df = pd.DataFrame(columns=[str(t) for t in RANDOM_STATES])
    vi_df = pd.DataFrame(columns=[str(t) for t in RANDOM_STATES])
    gibbs_df = pd.DataFrame(columns=[str(t) for t in RANDOM_STATES])

    for t in range(len(RANDOM_STATES)):

        random_state = RANDOM_STATES[t]

        logging.info("\n\n random state: {}".format(random_state))

        time_df.loc[t, 'systems'] = m
        time_df.loc[t, 'states'] = n
        time_df.loc[t, 'trial'] = t
        time_df.loc[t, 'random'] = random_state

        ll_df.loc[t, 'systems'] = m
        ll_df.loc[t, 'states'] = n
        ll_df.loc[t, 'trial'] = t
        ll_df.loc[t, 'random'] = random_state

        gen = hmm_gen.FactoredHMMGenerativeModel(
            ns_hidden_states=ns_hidden_states,
            n_gaussian_features=4,
            n_categorical_features=0,
            random_state=random_state)

        train_fact_hidden_states = gen.generate_hidden_state_sequence(
            n_observations=400)
        test_fact_hidden_states = gen.generate_hidden_state_sequence(
            n_observations=400)

        vec_list = list(
            itertools.product(*[[t
                                 for t in range(n)]
                                for n in gen.ns_hidden_states]))

        flattened_state_dict = {i: vec_list[i] for i in range(len(vec_list))}
        flattened_state_dict_inv = {
            str(list(v)): k
            for k, v in flattened_state_dict.items()
        }

        train_flat_hidden_states = pd.Series(
            [
                flattened_state_dict_inv[str(list(v))]
                for v in train_fact_hidden_states.values
            ],
            index=train_fact_hidden_states.index)
        test_flat_hidden_states = pd.Series(
            [
                flattened_state_dict_inv[str(list(v))]
                for v in test_fact_hidden_states.values
            ],
            index=test_fact_hidden_states.index)

        train_data = gen.generate_observations(train_fact_hidden_states)
        test_data = gen.generate_observations(test_fact_hidden_states)

        # Compute true log likelihood
        train_true, test_true = get_true_ll(gen, train_data, test_data)

        logging.info("\n True:")
        logging.info("...train_ll: {}".format(train_true))
        logging.info("...test_ll: {}".format(test_true))

        ll_df.loc[t, 'train_true'] = train_true
        ll_df.loc[t, 'test_true'] = test_true

        # Train flattened HMM with EM
        em = "failed"
        em_random_state = random_state
        while em == "failed":
            em_start = time.time()

            try:
                em_alg, em_hmm_model = train_hmm_em(gen, em_random_state,
                                                    train_data)
                time_df.loc[t, 'em'] = (time.time() - em_start) / ITERATIONS

                em = "successful"
                logging.info("\n EM: {} - {}".format(em, em_random_state))

                em_learning_ll = compute_learning_ll(em_alg, train_data, "hmm")
                em_df[str(random_state)] = em_learning_ll

                em_train_ll, em_test_ll = compute_model_ll(
                    em_hmm_model, train_data, test_data, "hmm")
                ll_df.loc[t, 'train_em'] = em_train_ll
                ll_df.loc[t, 'test_em'] = em_test_ll

                logging.info("...train_ll: {}".format(em_train_ll))
                logging.info("...test_ll: {}".format(em_test_ll))

            except:
                em_random_state = em_random_state + 1
                logging.info("EM: {}".format(em))

        # Train with Variational Inference
        vi = "failed"
        vi_random_state = random_state
        while vi == "failed":
            vi_start = time.time()

            try:
                vi_alg, vi_fhmm_model = train_fhmm_vi(
                    gen, vi_random_state, train_data, train_fact_hidden_states)
                time_df.loc[t, 'vi'] = (time.time() - vi_start) / ITERATIONS

                vi = "successful"
                logging.info("\n VI: {} - {}".format(vi, vi_random_state))

                vi_learning_ll = compute_learning_ll(vi_alg, train_data, "fhmm")
                vi_df[str(random_state)] = vi_learning_ll

                vi_train_ll, vi_test_ll = compute_model_ll(
                    vi_fhmm_model, train_data, test_data, "fhmm")
                ll_df.loc[t, 'train_vi'] = vi_train_ll
                ll_df.loc[t, 'test_vi'] = vi_test_ll

                logging.info("...train_ll: {}".format(vi_train_ll))
                logging.info("...test_ll: {}".format(vi_test_ll))

            except:
                vi_random_state = vi_random_state + 1
                logging.info("VI: {}".format(vi))

        # Train with Gibbs sampling
        gibbs = "failed"
        gibbs_random_state = random_state
        while gibbs == "failed":
            gibbs_start = time.time()

            try:
                gibbs_alg, gibbs_fhmm_model = train_fhmm_gibbs(
                    gen, gibbs_random_state, train_data,
                    train_fact_hidden_states)

                time_df.loc[t, 'gibbs'] = (
                    time.time() - gibbs_start) / ITERATIONS

                gibbs = "successful"
                logging.info("\n Gibbs: {} - {}".format(gibbs,
                                                        gibbs_random_state))

                gibbs_learning_ll = compute_learning_ll(gibbs_alg, train_data,
                                                        "fhmm")
                gibbs_df[str(random_state)] = gibbs_learning_ll

                gibbs_train_ll, gibbs_test_ll = compute_model_ll(
                    gibbs_fhmm_model, train_data, test_data, "fhmm")
                ll_df.loc[t, 'train_gibbs'] = gibbs_train_ll
                ll_df.loc[t, 'test_gibbs'] = gibbs_test_ll

                logging.info("...train_ll: {}".format(gibbs_train_ll))
                logging.info("...test_ll: {}".format(gibbs_test_ll))

            except:
                gibbs_random_state = gibbs_random_state + 1
                logging.info("Gibbs: {}".format(gibbs))

    time_df.to_parquet("training_table_data/{}_{}_time_df.pq".format(m, n))
    ll_df.to_parquet("training_table_data/{}_{}_ll_df.pq".format(m, n))
    em_df.to_parquet("training_table_data/{}_{}_em_df.pq".format(m, n))
    vi_df.to_parquet("training_table_data/{}_{}_vi_df.pq".format(m, n))
    gibbs_df.to_parquet("training_table_data/{}_{}_gibbs_df.pq".format(m, n))


if __name__ == "__main__":
    main(sys.argv[1:])
