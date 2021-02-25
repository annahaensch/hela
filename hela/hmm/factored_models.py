""" Factored hidden Markov model implementation for discrete hidden state fhmms.
"""

import itertools
from abc import ABC

import numpy as np
import pandas as pd
from scipy import stats, linalg
from scipy.special import logsumexp

from .base_models import (LOG_ZERO, HiddenMarkovModel, HMMConfiguration,
                          HMMForecasting, HMMValidationMetrics)
from .utils import *


class FactoredHMMConfiguration(ABC):
    """ Abstract base class for fHMM configuration. """

    def __init__(self,
                 ns_hidden_states,
                 random_state=0,
                 hidden_state_vectors=None,
                 hidden_state_vector_to_enum={},
                 hidden_state_enum_to_vector={},
                 categorical_features=[],
                 categorical_values=None,
                 categorical_vector_to_enum={},
                 categorical_enum_to_vector={},
                 gaussian_features=[],
                 gaussian_values=None,
                 model_parameter_constraints=None):

        self.random_state = random_state

        self.ns_hidden_states = ns_hidden_states
        self.hidden_state_vectors = hidden_state_vectors
        self.hidden_state_vector_to_enum = hidden_state_vector_to_enum
        self.hidden_state_enum_to_vector = hidden_state_enum_to_vector

        self.categorical_features = categorical_features
        self.categorical_values = categorical_values
        self.categorical_vector_to_enum = categorical_vector_to_enum
        self.categorical_enum_to_vector = categorical_enum_to_vector

        self.gaussian_features = gaussian_features
        self.gaussian_values = gaussian_values

        self.model_parameter_constraints = model_parameter_constraints

    @classmethod
    def from_spec(cls, spec, random_state=0):
        """ Factored HMM specific implementation of `from_spec`. """
        model_config = cls(ns_hidden_states=spec['hidden_state']['count'])
        model_config.model_type = 'FactoredHMM'
        model_config.random_state = random_state

        # Get mappings between hidden state vectors and enumerations,
        hidden_state_values = [
            [t for t in range(i)] for i in model_config.ns_hidden_states
        ]
        hidden_state_vectors = [
            list(t) for t in itertools.product(*hidden_state_values)
        ]
        model_config.hidden_state_vectors = hidden_state_vectors
        model_config.hidden_state_vector_to_enum = {
            str(hidden_state_vectors[i]): i
            for i in range(len(hidden_state_vectors))
        }
        model_config.hidden_state_enum_to_vector = {
            i: hidden_state_vectors[i]
            for i in range(len(hidden_state_vectors))
        }

        categorical_observations = {
            obs['name']: obs['values']
            for obs in spec['observations']
            if obs['type'] == 'finite'
        }
        if len(categorical_observations) > 0:
            categorical_observations = {
                k: categorical_observations[k]
                for k in sorted(categorical_observations.keys())
            }
            values = [sorted(v) for k, v in categorical_observations.items()]
            categorical_vectors = [list(t) for t in itertools.product(*values)]
            categorical_features = [
                k for k, v in categorical_observations.items()
            ]
            categorical_values = pd.DataFrame(
                categorical_vectors, columns=categorical_features)

            model_config.categorical_features = categorical_features
            model_config.categorical_values = categorical_values
            model_config.categorical_vector_to_enum = {
                str([t
                     for t in np.array(categorical_values.loc[i, :])]): i
                for i in categorical_values.index
            }
            model_config.categorical_enum_to_vector = {
                i: [t for t in np.array(categorical_values.loc[i, :])]
                for i in categorical_values.index
            }

        continuous_features = [
            obs for obs in spec['observations'] if obs['type'] == 'continuous'
        ]
        gaussian_features = [
            obs['name']
            for obs in continuous_features
            if obs['dist'].lower() == 'gaussian'
        ]
        model_config.gaussian_features = sorted(gaussian_features)
        model_config.gaussian_values = pd.DataFrame(
            columns=model_config.gaussian_features)

        model_config.model_parameter_constraints = spec[
            'model_parameter_constraints']

        return model_config

    def to_model(self):
        """ Factored HMM specific implementation of `to_model`. """

        return FactoredHMM.from_config(self)


class FactoredHMM(ABC):
    """ Model class for factored hidden Markov models """

    def __init__(
            self,
            model_config,
            random_state=None,
            trained=False,
            ns_hidden_states=None,
            hidden_state_vectors=None,
            hidden_state_vector_to_enum=None,
            hidden_state_enum_to_vector=None,
            hidden_state_delta_vector=None,
            hidden_state_delta_enum=None,
            categorical_features=None,
            gaussian_features=None,
            log_transition=None,
            log_initial_state=None,
            categorical_model=None,
            # TODO (isalju): incorporate gaussian mixture models
            gaussian_model=None):

        self.random_state = random_state
        self.trained = trained

        self.ns_hidden_states = ns_hidden_states
        self.hidden_state_vectors = hidden_state_vectors
        self.hidden_state_vector_to_enum = hidden_state_vector_to_enum
        self.hidden_state_enum_to_vector = hidden_state_enum_to_vector
        self.hidden_state_delta_vector = hidden_state_delta_vector
        self.hidden_state_delta_enum = hidden_state_delta_enum

        self.categorical_features = categorical_features
        self.gaussian_features = gaussian_features

        self.log_transition = log_transition
        self.log_initial_state = log_initial_state

        self.categorical_model = categorical_model
        # TODO (isalju): incorporate gaussian mixture models
        self.gaussian_model = gaussian_model

    @classmethod
    def from_config(cls, model_config):
        model = cls(model_config=model_config)
        model.random_state = model_config.random_state

        # Get mappings between hidden state vectors and enumerations
        model.ns_hidden_states = model_config.ns_hidden_states
        model.hidden_state_vectors = model_config.hidden_state_vectors
        model.hidden_state_vector_to_enum = model_config.hidden_state_vector_to_enum
        model.hidden_state_enum_to_vector = model_config.hidden_state_enum_to_vector

        model.hidden_state_delta_vector = {
            m: {
                str(v): model.hidden_state_vectors_matching_away_from_m(m, v)
                for v in model.hidden_state_vectors
            }
            for m in range(len(model.ns_hidden_states))
        }

        model.hidden_state_delta_enum = {
            m: {
                model.hidden_state_vector_to_enum[str(v)]: sorted([
                    model.hidden_state_vector_to_enum[str(w)]
                    for w in model.hidden_state_vectors_matching_away_from_m(
                        m, v)
                ])
                for v in model.hidden_state_vectors
            }
            for m in range(len(model.ns_hidden_states))
        }

        # Get categorical features from model_config.
        model.categorical_features = model_config.categorical_features
        if len(model.categorical_features) > 0:
            model.categorical_model = CategoricalModel.from_config(model_config)

        # Get continuous features from model_config.
        model.gaussian_features = model_config.gaussian_features
        if len(model.gaussian_features) > 0:
            model.gaussian_model = GaussianModel.from_config(model_config)

        transition = model_config.model_parameter_constraints[
            'transition_constraints']
        trans_mask = transition.mask
        zero_mask = transition == 0
        temp_masked_trans = np.ma.masked_array(
            np.where(transition != 0, transition, 1),
            trans_mask).filled(fill_value=1)
        log_transition = np.log(temp_masked_trans)
        log_transition[zero_mask] = LOG_ZERO
        model.log_transition = np.ma.masked_array(log_transition, trans_mask)

        initial_state = model_config.model_parameter_constraints[
            'initial_state_constraints']
        initial_state_mask = initial_state.mask
        zero_mask = initial_state == 0
        temp_masked_initial_state = np.ma.masked_array(
            np.where(initial_state != 0, initial_state, 1),
            initial_state_mask).filled(fill_value=1)
        log_initial_state = np.log(temp_masked_initial_state)
        log_initial_state[zero_mask] = LOG_ZERO
        model.log_initial_state = np.ma.masked_array(log_initial_state,
                                                     initial_state_mask)

        # TODO: (AH) add option to randomly seed transition and initial state.

        return model

    def to_inference_interface(self, data):
        """ Returns FactoredHMMInference object

        Arguments: 
            data: timeseries data for which to perform inference

        Returns:
            Initialized FactoredHMMInference object
        """
        return FactoredHMMInference(self, data)

    def hidden_state_vectors_matching_away_from_m(self, m, vector):
        """ Returns a list of vectors

        Arguments: 
            m: (int) index indicating one of the fHMM Markov systems.
            vector: (array) hidden state vector

        Returns: 
            List of all hidden state vectors agreeing with vector in all
            but the mth component.
        """

        mask = [1 if i == m else 0 for i in range(len(self.ns_hidden_states))]
        masked_vec = np.ma.masked_array(vector, mask)

        return [
            v for v in self.hidden_state_vectors
            if np.all(np.ma.masked_array(v, mask) == masked_vec)
        ]

    def vector_to_column_vectors(self, hidden_state_vector):
        """ Returns column vectors associated to hidden_state_vector

        Arguments: 
            hidden_state_vector: (array) hidden state vector.
        Returns:
            List of column vectors with 0 and 1 entries.
        """
        return [
            np.array([
                1 if j == hidden_state_vector[i] else 0
                for j in range(np.max(self.ns_hidden_states))
            ]).reshape(-1, 1)
            for i in range(len(self.ns_hidden_states))
        ]


class CategoricalModel(FactoredHMM):

    def __init__(self,
                 model_config,
                 ns_hidden_states=None,
                 hidden_state_vector_to_enum=None,
                 hidden_state_enum_to_vector=None,
                 n_hidden_states=None,
                 categorical_features=None,
                 categorical_values=None,
                 categorical_vector_to_enum=None,
                 categorical_enum_to_vector=None,
                 log_emission_matrix=None):
        super().__init__(
            model_config=model_config,
            ns_hidden_states=ns_hidden_states,
            hidden_state_vector_to_enum=hidden_state_vector_to_enum,
            hidden_state_enum_to_vector=hidden_state_enum_to_vector)
        self.categorical_features = categorical_features
        self.categorical_values = categorical_values
        self.categorical_vector_to_enum = categorical_vector_to_enum
        self.categorical_enum_to_vector = categorical_enum_to_vector
        self.log_emission_matrix = log_emission_matrix

    @classmethod
    def from_config(cls, model_config):
        """ Return instantiated CategoricalModel object)
        """
        categorical_model = cls(
            model_config=model_config,
            ns_hidden_states=model_config.ns_hidden_states,
            hidden_state_vector_to_enum=model_config.
            hidden_state_vector_to_enum,
            hidden_state_enum_to_vector=model_config.hidden_state_enum_to_vector
        )
        categorical_model.categorical_features = model_config.categorical_features
        categorical_model.categorical_values = model_config.categorical_values
        categorical_model.categorical_vector_to_enum = model_config.categorical_vector_to_enum
        categorical_model.categorical_enum_to_vector = model_config.categorical_enum_to_vector

        # Get log emission matrix, masking to prevent log(0) errors.
        emission_matrix = model_config.model_parameter_constraints[
            'emission_constraints']
        zero_mask = emission_matrix == 0
        log_emission_matrix = np.where(emission_matrix != 0, emission_matrix, 1)
        log_emission_matrix = np.log(log_emission_matrix)
        log_emission_matrix[zero_mask] = LOG_ZERO
        categorical_model.log_emission_matrix = log_emission_matrix

        # TODO: (AH) Add option to randomly seed emission matrix.

        return categorical_model

    def emission_log_probabilities(self, data):
        """ Returns emission log_probabilities for categorical data

        Arguments: 
            data: dataframe of observed categorical data

        Returns: 
            Dataframe where entry [t,i] is log P(x_t | h_i) (i.e. the conditional 
            probability of the categorical emission, x_t, observed at time t, 
            given hidden state h_i at time t).  Here hidden states are 
            enumerated in the "flattened" sense.  
        """
        flattened_observations = [
            self.categorical_vector_to_enum[str(list(v))]
            for v in np.array(data.loc[:, self.categorical_features])
        ]
        log_emission = self.log_emission_matrix

        return pd.DataFrame(
            [log_emission[v] for v in flattened_observations],
            columns=[k for k in self.hidden_state_enum_to_vector.keys()],
            index=data.index)


class GaussianModel(FactoredHMM):

    def __init__(self,
                 model_config,
                 ns_hidden_states=None,
                 hidden_state_vector_to_enum=None,
                 hidden_state_enum_to_vector=None,
                 gaussian_features=None,
                 dims=None,
                 means=None,
                 covariance=None):
        super().__init__(
            model_config=model_config,
            ns_hidden_states=ns_hidden_states,
            hidden_state_vector_to_enum=hidden_state_vector_to_enum,
            hidden_state_enum_to_vector=hidden_state_enum_to_vector)
        self.ns_hidden_states = ns_hidden_states
        self.hidden_state_vector_to_enum = hidden_state_vector_to_enum
        self.hidden_state_enum_to_vector = hidden_state_enum_to_vector
        self.gaussian_features = gaussian_features
        self.dims = dims
        self.means = means
        self.covariance = covariance

    @classmethod
    def from_config(cls, model_config):
        """ Return instantiated GaussianModel object)
        """
        gaussian_model = cls(
            model_config=model_config,
            ns_hidden_states=model_config.ns_hidden_states,
            hidden_state_vector_to_enum=model_config.
            hidden_state_vector_to_enum,
            hidden_state_enum_to_vector=model_config.hidden_state_enum_to_vector
        )

        gaussian_features = model_config.gaussian_features
        gaussian_values = model_config.gaussian_values
        gaussian_params = model_config.model_parameter_constraints[
            'gaussian_parameter_constraints']

        # Gather gaussian features and values.
        gaussian_model.gaussian_features = model_config.gaussian_features
        gaussian_model.dims = len(gaussian_features)
        gaussian_model.means = gaussian_params['means']
        gaussian_model.covariance = gaussian_params['covariance']

        # TODO: (AH) Add option to randomly seed gaussian parameters.

        return gaussian_model

    def mean_for_hidden_state_vector(self, hidden_state_vector):
        """ Computes the mean for a give hidden state vector
        
        Arguments: 
            hidden_state_vector: (array) vector indicating a hidden state for 
                each fHMM Markov system

        Returns:
            Mean vector of dimension equal to the dimension of the gaussian data.
        """
        ns_hidden_states = self.ns_hidden_states
        means = self.means

        # Rewrite hidden_state_vector as a list of column vectors.
        column_vectors = self.vector_to_column_vectors(hidden_state_vector)

        return np.sum(
            np.array([
                means[m].data @ column_vectors[m]
                for m in range(len(ns_hidden_states))
            ]),
            axis=0)

    def emission_log_probabilities(self, data):
        """ Returns emission log_probabilities for gaussian data

        Arguments: 
            data: dataframe of observed categorical data

        Returns: 
            Dataframe where entry [t,i] is log P(x_t | h_i) (i.e. the conditional 
            log probability of the gaussian emission, x_t, observed at time t, 
            given hidden state h_i at time t).  Here hidden states are 
            enumerated in the "flattened" sense.
        """
        vectors = [v for k, v in self.hidden_state_enum_to_vector.items()]
        means = {
            k: self.mean_for_hidden_state_vector(v)
            for k, v in self.hidden_state_enum_to_vector.items()
        }
        cov = self.covariance

        # Initialize dataframe that will hold log probablites for observations
        # (rows) conditioned on hidden states (columns)
        log_prob = pd.DataFrame(
            index=data.index, columns=[i for i in range(len(means))])
        for k, m in means.items():
            log_prob.loc[:, k] = stats.multivariate_normal.logpdf(
                np.array(data.loc[:, self.gaussian_features]),
                m.reshape(1, -1)[0], cov)

        return log_prob


class FactoredHMMInference(ABC):

    def __init__(self, model, data):
        self.model = model
        self.data = data

    def emission_log_probabilities(self, data):
        """ Returns emission log_probabilities for observed data
        Arguments: 
            data: dataframe of observed categorical data
        Returns: 
            Dataframe where entry [t,i] is log P(x_t | h_i) (i.e. the conditional 
            log probability of the observation, x_t, given hidden state h_i at 
            time t).  Here hidden states are enumerated in the "flattened" sense.
        """
        log_prob = np.zeros((data.shape[0],
                             np.prod(self.model.ns_hidden_states)))

        if self.model.categorical_model:
            log_prob += np.array(
                self.model.categorical_model.emission_log_probabilities(data))

        if self.model.gaussian_model:
            log_prob += np.array(
                self.model.gaussian_model.emission_log_probabilities(data))

        return pd.DataFrame(
            log_prob,
            columns=[k for k in self.model.hidden_state_enum_to_vector.keys()],
            index=data.index)

    def probability_distribution_across_hidden_states(
            self,
            data,
            current_hidden_state,
            idx,
            system,
            emission_log_probabilities,
            next_hidden_state=None):
        """ Returns probability distribution across hidden states.

        Arguments:
            data: (dataframe) observations.
            current_hidden_state: (array) hidden state vector corresponding to idx.
            idx: (int) iloc in index of data
            system: (int) indicates fHMM Markov system under consideration
            emission_log_probabilities: (df) row t and column j correspond to the log
                probability of the emission observed at time t given hidden state j (
                where hidden state is taken in the "flattened" sense).
            next_hidden_state: (array) hidden state vector corresponding to idx.

        Returns: 
            Array of probability distributions given data. The ith entry of this 
            array will be the probability of hidden state i given the observations
            at data.iloc[idx,:] and hidden states in systems away from m.  
            
            (Note: this is exactly the function "f" referenced on line 6 of Algorithm 1
            of the technical note.)
        """
        model = self.model
        ns_hidden_states = model.ns_hidden_states
        log_prob = np.zeros(model.ns_hidden_states[system])

        # Get list of eligible flattened hidden states.
        eligible_states = model.hidden_state_delta_enum[system][
            model.hidden_state_vector_to_enum[str(current_hidden_state)]]

        # Add emission log probabilities.
        log_prob += np.array(
            emission_log_probabilities.loc[data.index[idx], eligible_states])

        # Add initial state log probabilities if idx  is 0.
        if idx == 0:
            log_prob += model.log_initial_state[system][:ns_hidden_states[
                system]]

        # Add transition log probabilities untill idx corresponds to the last observed data.
        if idx < data.shape[0] - 1:
            log_prob += np.array(
                model.log_transition[system][:, next_hidden_state[
                    system]])[:ns_hidden_states[system]]

        return np.exp(log_prob)

    def gibbs_sampling(self,
                       data,
                       iterations,
                       burn_down_period=100,
                       gather_statistics=False,
                       hidden_state_vector_df=None):
        """ Samples one timestep and fHMM system

        Arguments: 
            data: (dataframe) observed timeseries data.
            iterations: (int) number of rounds of sampling to carry out.
            burn_down_period: (int) number of iterations for burn down before 
                gathering statistics.
            gather_statistics: (bool) indicates whether to gather statistics while 
                iterating.
            hidden_state_vector_df: (dataframe) timeseries of hidden state vectors
                with the same index as "data".  If default "None" is given, then
                this dataframe will be seeded randomly.

        Returns: The arrays Gamma and Xi containing statistics, and an updated hidden 
            state vector
        """
        model = self.model

        # Initialize dataframe of hidden state vectors if none is given.
        if hidden_state_vector_df is None:
            hidden_state_enum_df = np.random.choice(
                list(model.hidden_state_enum_to_vector.keys()), data.shape[0])
            hidden_state_vector_df = pd.DataFrame(
                [
                    model.hidden_state_enum_to_vector[v]
                    for v in hidden_state_enum_df
                ],
                index=data.index,
                columns=[i for i in range(len(model.ns_hidden_states))])

        # Collect Statistics in Gamma and Xi, with Gibbs sampling at each stage.
        Gamma = None
        Xi = None
        if gather_statistics == True:
            Gamma = np.zeros((data.shape[0], np.sum(model.ns_hidden_states),
                              np.sum(model.ns_hidden_states)))
            Xi = np.zeros((len(model.ns_hidden_states), data.shape[0] - 1,
                           np.sum(model.ns_hidden_states),
                           np.sum(model.ns_hidden_states)))

        for r in range(iterations + burn_down_period):
            sample_times = np.random.choice(
                [i for i in range(data.shape[0])], data.shape[0], replace=False)
            sample_systems = np.random.choice(
                [i for i in range(len(self.model.ns_hidden_states))],
                len(self.model.ns_hidden_states),
                replace=False)
            sample_parameter = np.random.uniform(0, 1, data.shape[0])

            emission = self.emission_log_probabilities(data)

            for t in sample_times:
                h_current = (hidden_state_vector_df.iloc[t, :]).to_list()
                n_next = None
                column_vectors = self.model.vector_to_column_vectors(h_current)

                if t < data.shape[0] - 1:
                    h_next = np.array(hidden_state_vector_df.iloc[t + 1, :])
                    next_column_vectors = self.model.vector_to_column_vectors(
                        h_next)

                if r >= burn_down_period:
                    if gather_statistics == True:
                        for i in range(len(model.ns_hidden_states)):

                            skip_rows = int(np.sum(model.ns_hidden_states[:i]))
                            N = model.ns_hidden_states[i]

                            count = (column_vectors[i]
                                     @ column_vectors[i].reshape(1, -1))[:N, :N]
                            Gamma[t][skip_rows:skip_rows + N, skip_rows:
                                     skip_rows + N] += (
                                         column_vectors[i] @ column_vectors[
                                             i].reshape(1, -1))[:N, :N]

                            if t < data.shape[0] - 1:
                                count = (column_vectors[i] @ column_vectors[i]
                                         .reshape(1, -1))[:N, :N]
                                Xi[i][t][:N, :N] += count

                            for j in range(i):

                                skip_columns = int(
                                    np.sum(model.ns_hidden_states[:j]))
                                M = model.ns_hidden_states[j]

                                count = (column_vectors[i] @ column_vectors[j]
                                         .reshape(1, -1))[:N, :M]
                                Gamma[t][skip_rows:skip_rows + N, skip_columns:
                                         skip_columns + M] += count

                for m in sample_systems:

                    updated_state_prob = self.probability_distribution_across_hidden_states(
                        data,
                        current_hidden_state=h_current,
                        idx=t,
                        system=m,
                        emission_log_probabilities=emission,
                        next_hidden_state=h_next)
                    hidden_state_vector_df.iloc[t, m] = _sample(
                        updated_state_prob, sample_parameter[t])

        # Normalize gathered statistics
        if gather_statistics == True:
            Gamma = Gamma.astype(np.float32) / iterations
            Xi = Xi.astype(np.float32)
            Xi = Xi + (np.sum(Xi, axis=2).reshape(
                Xi.shape[0], Xi.shape[1], Xi.shape[2], -1) == 0).astype(int)
            Xi = Xi / np.sum(
                Xi, axis=3).reshape(Xi.shape[0], Xi.shape[1], Xi.shape[2], -1)

        return Gamma, Xi, hidden_state_vector_df


def _sample(probability_distribution, sample_parameter):
    """ Returns a sample using discrete inverse transform.

    Arguments:
        probability_distribution: array of probabilities
        sample_parameter: (int between 0 and 1)

    Returns: 
        Sample with prescribed probability distribution
    """
    if np.sum(probability_distribution) == 0:
        probability_distribution = np.full(len(probability_distribution), 1)
    probability_distribution = probability_distribution / np.sum(
        probability_distribution)
    cumulative_prob = np.cumsum(probability_distribution)
    updated_state = np.where(cumulative_prob >= sample_parameter)[0][0]
    return updated_state
