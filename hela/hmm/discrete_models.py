""" Hidden Markov model implementation for discrete hidden state hmms.
"""

import itertools
from abc import ABC

import numpy as np
import pandas as pd
from scipy import linalg, stats
from scipy.special import logsumexp

from .base_models import (LOG_ZERO, HiddenMarkovModel, HMMConfiguration,
                          HMMForecasting, HMMValidationMetrics)
from .utils import *


class DiscreteHMMConfiguration(HMMConfiguration):
    """ Intilialize HMM configuration from specification dictionary. """

    def __init__(self, hidden_state_count=None):
        super().__init__(hidden_state_count)
        self.n_hidden_states = None

    def _from_spec(self, spec):
        """ Discrete HMM specific implementation of `from_spec`. """
        self.n_hidden_states = spec['hidden_state_count']
        self.model_type = 'DiscreteHMM'
        return self

    def _to_model(self, random_state):
        """ Discrete HMM specific implementation of `to_model`. """
        return DiscreteHMM.from_config(self, random_state)


class DiscreteHMM(HiddenMarkovModel):
    """ Model class for dicrete hidden Markov models """

    def __init__(self, model_config=None):
        super().__init__(model_config)
        self.random_state = None
        self.trained = False
        # TODO: @annahaensch deal with other continuous observation types.

        self.n_hidden_states = None
        self.seed_parameters = {}

        self.finite_features = None
        self.finite_values = None
        self.continuous_features = None
        self.continuous_values = None

        self.log_transition = None
        self.log_initial_state = None

        self.categorical_model = None
        self.gaussian_mixture_model = None

    @classmethod
    def from_config(cls, model_config, random_state):
        model = cls(model_config=model_config)
        model.n_hidden_states = model_config.n_hidden_states
        model.random_state = random_state

        # Get finite features from model_config.
        model.finite_features = model_config.finite_features
        model.finite_values = model_config.finite_values
        if len(model.finite_features) > 0:
            model.categorical_model = CategoricalModel.from_config(
                model_config, random_state)

        # Get continuous features from model_config.
        model.continuous_values = model_config.continuous_values
        model.continuous_features = model_config.continuous_features
        gaussian_features = [
            i for i in model.continuous_features if model.continuous_values.loc[
                'distribution', i].lower() == 'gaussian'
        ]
        if len(gaussian_features) > 0:
            model.gaussian_mixture_model = GaussianMixtureModel.from_config(
                model_config, random_state)

        # Check that there are no remaining features.
        if len(model.continuous_features) > 0:
            non_gaussian_features = [
                i for i in model.continuous_values.columns
                if model.continuous_values.loc['distribution', i].lower() !=
                'gaussian'
            ]
            non_gaussian_distributions = [
                model.continuous_values.loc['distribution', i]
                for i in non_gaussian_features
            ]
            if len(non_gaussian_features) > 0:
                raise NotImplementedError(
                    "Curent DiscreteHMM implementation is "
                    "not equipped to deal with continuous variables with "
                    "distribution type: {}".format(
                        " ,".join(non_gaussian_distributions)))

        model.log_transition = model.log_transition_from_constraints(
            model_config.model_parameter_constraints['transition_constraints'])

        model.log_initial_state = model.log_initial_state_from_constraints(
            model_config.model_parameter_constraints[
                'initial_state_constraints'])

        return model

    def _load_inference_interface(self, use_jax):
        """ Loads DiscreteHMM specific inference interface."""
        return DiscreteHMMInferenceResults(self, use_jax)

    def _load_validation_interface(self, actual_data, use_jax):
        """ Loads DiscreteHMM specific validation interface.
        """
        return DiscreteHMMValidationMetrics(self, actual_data, use_jax)

    def _load_forecasting_interface(self, use_jax):
        """ Loads DiscreteHMM specific forecasting interface."""
        return DiscreteHMMForecasting(self, use_jax)

    def log_transition_from_constraints(self, transition_constraints=None):
        """ Return log transition matrix with fixed transition constraints"""
        if transition_constraints is None:
            transition = np.full((self.n_hidden_states, self.n_hidden_states),
                                 np.nan)
        else:
            transition = np.array(transition_constraints)
        for t in range(transition.shape[0]):
            if 1 in transition[t]:
                transition[t] = (transition[t] == 1).astype(int)
            else:
                if np.sum(np.isnan(transition[t]).astype(int)) != 0:
                    random_init = self.random_state.rand(
                        transition[t].shape[0]) * np.isnan(
                            transition[t]).astype(int)
                    random_init = (1 - np.nansum(transition[t])) * (
                        random_init) / np.sum(random_init)
                    transition[t] = random_init + np.nan_to_num(transition[t])
        log_transition = np.log(
            transition,
            out=np.zeros_like(transition) + LOG_ZERO,
            where=(transition != 0))

        return log_transition

    def log_initial_state_from_constraints(self,
                                           initial_state_constraints=None):
        """ Return initial state log probability vector with fixed constraints"""
        if initial_state_constraints is None:
            initial_state = np.full(self.n_hidden_states, np.nan)
        else:
            initial_state = np.array(initial_state_constraints)
        if np.sum(np.isnan(initial_state).astype(int)) != 0:
            random_init = self.random_state.rand(
                len(initial_state)) * np.isnan(initial_state).astype(int)
            random_init = (1 - np.nansum(initial_state)) * (
                random_init) / np.sum(random_init)
            initial_state = random_init + np.nan_to_num(initial_state)
        log_initial_state = np.log(
            initial_state,
            out=np.zeros_like(initial_state) + LOG_ZERO,
            where=(initial_state != 0))
        return log_initial_state

    def update_model_parameters(self, finite_states_data, gaussian_data,
                                expectation):

        gamma = expectation.gamma
        xi = expectation.xi
        gamma_by_component = expectation.gamma_by_component

        new_model = self.model_config.to_model()

        if self.model_config.model_parameter_constraints.get(
                'initial_state_constraints', None) is None:
            new_model.log_initial_state = gamma[0]

        new_model.log_transition = logsumexp(
            xi, axis=0) - logsumexp(
                gamma[:-1], axis=0).reshape(-1, 1)

        if self.categorical_model is not None:
            new_model.categorical_model.log_emission_matrix = self.categorical_model.update_log_emission_matrix(
                gamma, finite_states_data)
        if self.gaussian_mixture_model is not None:
            new_model.gaussian_mixture_model = self.gaussian_mixture_model.update_gmm_parameters(
                gaussian_data, gamma, gamma_by_component)

        return new_model


class CategoricalModel(DiscreteHMM):

    def __init__(self,
                 n_hidden_states=None,
                 finite_features=None,
                 finite_values=None,
                 finite_values_dict=None,
                 finite_values_dict_inverse=None,
                 log_emission_matrix=None):
        self.n_hidden_states = n_hidden_states
        self.finite_features = finite_features
        self.finite_values = finite_values
        self.finite_values_dict = finite_values_dict
        self.finite_values_dict_inverse = finite_values_dict_inverse
        self.log_emission_matrix = log_emission_matrix

    @classmethod
    def from_config(cls, model_config, random_state):
        """ Return instantiated CategoricalModel object)
        """
        categorical_model = cls(n_hidden_states=model_config.n_hidden_states)
        categorical_model.random_state = random_state
        categorical_model.finite_features = model_config.finite_features
        categorical_model.finite_values = model_config.finite_values
        categorical_model.finite_values_dict = model_config.finite_values_dict
        categorical_model.finite_values_dict_inverse = model_config.finite_values_dict_inverse
        categorical_model.log_emission_matrix = categorical_model.get_log_emission_matrix_from_config(
            model_config.model_parameter_constraints['emission_constraints'])
        return categorical_model

    def get_log_emission_matrix_from_config(self, emission_constraints=None):
        """ Return emission matrix subject to fixed constraints """
        if emission_constraints is None:
            n_obs_states = self.finite_values.shape[0]
            n_hidden_states = self.n_hidden_states
            emission = np.full((n_obs_states, n_hidden_states), np.nan)
        else:
            emission = np.array(emission_constraints)
        new_emission = []
        for t in range(emission.shape[1]):
            column = np.array([e[t] for e in emission])
            if 1 in column:
                column = (column == 1).astype(int)
                new_emission.append(column)
            else:
                if np.sum(np.isnan(column).astype(int)) != 0:
                    random_init = self.random_state.rand(
                        len(column)) * np.isnan(column).astype(int)
                    random_init = (1 - np.nansum(column)) * (
                        random_init) / np.sum(random_init)
                    column = random_init + np.nan_to_num(column)
                new_emission.append(column)
        emission = np.array(new_emission).transpose()
        log_emission = np.log(
            emission,
            out=np.zeros_like(emission) + LOG_ZERO,
            where=(emission != 0))

        return log_emission

    def log_probability(self, finite_data):
        """ Return log probability of finite observation given hidden state

        Arguments:
            finite_data: observed finite data as Series.

        Returns:
            np.array where entry [t,i] is the log probability of emitting finite observation t in hidden state i.
        """
        n_observations = finite_data.shape[0]
        log_emission = np.array(self.log_emission_matrix)

        return pd.DataFrame(
            [
                list(prob) for prob in np.array(
                    finite_data.map(lambda x: log_emission[x]))
            ],
            index=finite_data.index)

    def update_log_emission_matrix(self, gamma, finite_state_data):
        """ Update log emission matrix for categorical model

        Arguments:
            gamma: output of DiscreteHMMInferenceResults
            finite_states_data: series of finite state data

        Returns:
            Updated log emission matrix
        """
        log_emission_matrix = np.full(
            (np.array(self.log_emission_matrix).shape), LOG_ZERO)
        gamma_df = pd.DataFrame(gamma, index=finite_state_data.index)
        for l in self.finite_values.index:
            if l in finite_state_data.unique():
                l_index = finite_state_data[finite_state_data == l].index
                l_gamma_df = gamma_df.loc[l_index]
                log_emission_matrix[l] = logsumexp(np.array(l_gamma_df), axis=0)
        log_emission_matrix -= logsumexp(gamma, axis=0)
        return log_emission_matrix

    def probability_of_hidden_state_from_discrete_obs(
            self, partial_finite_observation):
        """ Return probabilites associated to each hidden state given the partial observation.

        Arguments:
            partial_partial_observation: single row of a dataframe of finite observations.

        Returns:
            Array of probabilities where the ith entry is the probability of hidden state i given the partial finite observation.

        TODO @annahaensch: this probabilty needs to be refined to include known sequential data.
        """
        finite_values = self.finite_values
        emission_matrix = np.exp(np.array(self.log_emission_matrix))
        known_features = partial_finite_observation.columns[
            ~partial_finite_observation.isna().any()].tolist()
        if len(known_features) == 0:
            return np.ones(self.n_hidden_states)
        elif len(known_features) == partial_finite_observation.shape[1]:
            observation_tuple = str(
                list(partial_finite_observation.loc[:, self.finite_features]
                     .iloc[0]))
            observation_state = self.finite_values_dict_inverse[
                observation_tuple]
            return emission_matrix[observation_state]
        else:
            eligible_states = []
            for feat in known_features:
                eligible_values = finite_values[finite_values[
                    feat] == partial_finite_observation[feat][0]]
                eligible_states.append([
                    self.finite_values_dict_inverse[str(list(x))]
                    for x in np.array(eligible_values)
                ])

            possible_states = list(
                set.intersection(*[set(x) for x in eligible_states]))
            return np.sum(emission_matrix[possible_states], axis=0)


class GaussianMixtureModel(DiscreteHMM):

    def __init__(self,
                 n_hidden_states=None,
                 gaussian_features=None,
                 gaussian_values=None,
                 dims=None,
                 n_gmm_components=None,
                 component_weights=None,
                 means=None,
                 covariances=None):
        self.n_hidden_states = n_hidden_states
        self.gaussian_features = gaussian_features
        self.dims = dims
        self.n_gmm_components = n_gmm_components
        self.component_weights = component_weights
        self.means = means
        self.covariances = covariances

    @classmethod
    def from_config(cls, model_config, random_state):
        """ Return instantiated Gaussian MixtureModel object)
        """
        gmm = cls(n_hidden_states=model_config.n_hidden_states)
        continuous_values = model_config.continuous_values

        # Gather gaussian features and values.
        raw_gaussian_features = [
            c for c in continuous_values
            if continuous_values.loc['distribution', c] == 'gaussian'
        ]
        raw_gaussian_values = continuous_values[raw_gaussian_features]

        gaussian_features = []
        for c in raw_gaussian_values.columns:
            dim = raw_gaussian_values.loc['dimension', c]
            if dim == 1:
                gaussian_features.append(c)
            else:
                for i in range(dim):
                    gaussian_features.append('{}_{}'.format(c, i))
        gaussian_features.sort()
        gmm.gaussian_features = gaussian_features

        gmm.n_hidden_states = model_config.n_hidden_states
        gmm.dims = np.sum([d for d in raw_gaussian_values.loc['dimension', :]])

        if model_config.model_parameter_constraints['gmm_parameter_constraints'] is not None:
            gmm_params = model_config.model_parameter_constraints[
                'gmm_parameter_constraints']
            if 'n_gmm_components' in gmm_params:
                gmm.n_gmm_components = gmm_params['n_gmm_components']
            if 'means' in gmm_params:
                gmm.means = gmm_params['means']
            if 'covariances' in gmm_params:
                gmm.covariances = gmm_params['covariances']
            if 'component_weights' in gmm_params:
                gmm.component_weights = gmm_params['component_weights']

        if gmm.n_gmm_components is None:
            # set n_gmm_components equal to n_hidden_states if no value is given
            gmm.n_gmm_components = gmm.n_hidden_states
        if gmm.means is None:
            means = np.zeros((gmm.n_hidden_states, gmm.n_gmm_components,
                              gmm.dims))
            gmm.means = means
        if gmm.covariances is None:
            covariances = np.array(
                gmm.n_hidden_states *
                [np.array(gmm.n_gmm_components * [np.identity(gmm.dims)])])
            gmm.covariances = covariances
        if gmm.component_weights is None:
            weights = np.empty((gmm.n_hidden_states, gmm.n_gmm_components))
            for i in range(weights.shape[0]):
                rand_init = random_state.rand(gmm.n_gmm_components)
                rand_init = rand_init / np.sum(rand_init)
                weights[i,:] = rand_init
            
            gmm.component_weights = weights

        return gmm

    def log_probability_by_component(self, gaussian_data):
        """ Return log probability of gaussian observation given hidden state and gmm component

        Arguments:
            model: HiddenMarkovModel object.
            gaussian_data: observed gaussian data as DataFrame

        Returns:
            np.array where entry [i,t,m] is the log probability of emitting continuous observation t in hidden state i and gaussian component m.
        """
        n_hidden_states = self.n_hidden_states
        n_gmm_components = self.n_gmm_components
        n_observations = gaussian_data.shape[0]
        means = self.means
        covariances = self.covariances

        log_emission_by_component = np.full(
            (n_hidden_states, n_observations, n_gmm_components), np.nan)
        for i in range(n_hidden_states):
            log_emission_by_component_i = np.full(
                (n_gmm_components, n_observations), np.nan)
            for m in range(n_gmm_components):
                log_emission_by_component_i[
                    m] = stats.multivariate_normal.logpdf(
                        gaussian_data,
                        means[i][m],
                        covariances[i][m],
                        allow_singular=True)
            log_emission_by_component[
                i] = log_emission_by_component_i.transpose()
        return log_emission_by_component

    def log_probability(self, gaussian_data):
        """ Return log probability of observations for each hidden state

        Arguments:
            gaussian_data: observed gaussian data as DataFrame

        Returns:
            DataFrame of log probabilties of observations for each hidden state
        """
        n_hidden_states = self.n_hidden_states
        n_observations = gaussian_data.shape[0]
        weights = np.array(self.component_weights)
        log_weights = np.log(
            weights,
            out=np.zeros_like(weights) + LOG_ZERO,
            where=(weights != 0))
        log_emission_by_component = self.log_probability_by_component(
            gaussian_data)

        log_emission = np.full((n_hidden_states, n_observations), np.nan)
        for i in range(len(log_weights)):
            log_emission[i] = logsumexp(
                log_emission_by_component[i] + log_weights[i], axis=1)

        return pd.DataFrame(log_emission.transpose(), index=gaussian_data.index)

    def update_means(self, gaussian_data, gamma_by_component):
        """ Return updated means for current hmm parameters.

        Arguments:
            gaussian_data: observed gaussian data as DataFrame
            gamma_by_component: array where entry [i,t,m] if the probability of being in hidden state i and gmm component m at time t.

        Returns:
            Array of updated means.
        """
        n_hidden_states = self.n_hidden_states
        n_gmm_components = self.n_gmm_components
        means = np.zeros_like(np.array(self.means))
        for i in range(n_hidden_states):
            for m in range(n_gmm_components):
                means[i, m] = np.sum(
                    np.array([g[m] for g in np.exp(gamma_by_component)[i]]
                            ).reshape(-1, 1) * np.asarray(gaussian_data),
                    axis=0) / np.sum(
                        np.array([g[m] for g in np.exp(gamma_by_component)[i]
                                 ]).reshape(-1, 1))
        return means

    def update_covariances(self, gaussian_data, gamma_by_component):
        """ Return updated covarinaces for current hmm parameters.

        Arguments:
            gaussian_data: observed gaussian data as DataFrame
            gamma_by_component: array where entry [i,t,m] if the probability of being in hidden state i and gmm component m at time t.

        Returns:
            Array of updated covariance matrices.
        """
        n_hidden_states = self.n_hidden_states
        n_gmm_components = self.n_gmm_components
        means = self.means
        covariances = np.zeros_like(np.array(self.covariances))
        gamma_by_component = np.exp(gamma_by_component)
        for i in range(n_hidden_states):
            for m in range(n_gmm_components):
                error = np.array(gaussian_data) - means[i][m]
                error_prod = np.array([
                    error[t].reshape(-1, 1) @ error[t].reshape(1, -1)
                    for t in range(len(error))
                ])
                gamma = np.array([g[m] for g in gamma_by_component[i]]).reshape(
                    -1, 1, 1)
                covariances[i][m] = np.sum(
                    gamma * error_prod, axis=0) / np.sum(
                        gamma, axis=0)

        return covariances

    def update_component_weights(self, gamma, gamma_by_component):
        """ Return updated component weights for current hmm parameters.

        Arguments:
            gamma: array where entry [t,i] is the probability of being in hidden state i at time t.
            gamma_by_component: array where entry [i,t,m] if the probability of being in hidden state i and gmm component m at time t.


        Returns:
            Array of updated component weights matrices.
        """
        log_weights = logsumexp(
            gamma_by_component, axis=1) - logsumexp(
                gamma, axis=0).reshape(-1, 1)
        weights = np.exp(log_weights)

        return weights

    def update_gmm_parameters(self, gaussian_data, gamma, gamma_by_component):
        """ Return gmm with updated parameters

        Arguments:
            gaussian_data: observed gaussian data as DataFrame
            gamma: array where entry [t,i] is the probability of being in hidden state i at time t.
            gamma_by_component: array where entry [i,t,m] if the probability of being in hidden state i and gmm component m at time t.

        Returns:
            GaussianMixtureModel object with updated parameters
        """
        new_gmm = GaussianMixtureModel(
            n_hidden_states=self.n_hidden_states,
            n_gmm_components=self.n_gmm_components,
            dims=self.dims,
            gaussian_features=self.gaussian_features)

        new_gmm.means = self.update_means(gaussian_data, gamma_by_component)
        new_gmm.covariances = self.update_covariances(gaussian_data,
                                                      gamma_by_component)
        new_gmm.component_weights = self.update_component_weights(
            gamma, gamma_by_component)

        return new_gmm

    def marginal_probability_of_gaussian_observation_by_hidden_state(
            self, partial_gaussian_observation):
        """ Return marginal probabilty of observation by hidden state

        Arguments:
            partial_gaussian_obervation: single row of an observation dataframe containing guassian obsevations or nan.

        Returns:
            Array of probabilities of the given observation by hidden state
        """
        obs = np.array(partial_gaussian_observation.iloc[0])
        nan_index = np.argwhere(np.isnan(obs)).flatten()

        if len(nan_index) == len(obs):
            marginal_prob = np.full(self.n_hidden_states,
                                    1 / self.n_hidden_states)

        elif len(nan_index) == 0:
            marginal_prob = np.exp(
                np.array(self.log_probability(partial_gaussian_observation)))

        else:
            marginal_prob = np.empty(self.n_hidden_states)
            for i in range(self.n_hidden_states):
                marginal_prob[
                    i] = compute_marginal_probability_gaussian_mixture(
                        partial_gaussian_observation, self.means[i],
                        self.covariances[i], self.component_weights[i])

        return marginal_prob


class DiscreteHMMInferenceResults(ABC):
    """ Abstract base class for HMM inference results """

    def __init__(self, model, use_jax):
        self.model = model
        self.log_probability = None
        self.gamma = None
        self.alpha = None
        self.beta = None
        self.xi = None
        self.gamma_by_component = None
        self.use_jax = use_jax

    def compute_sufficient_statistics(self, data):
        """ Compute inference results for model.

        Returns:
            Statistics from HMMInferenceResults
        """
        model = self.model
        self.log_prob = self.predict_hidden_state_log_probability(data)
        self.gamma = self._gamma(data)
        self.xi = self._xi(data)
        self.gamma_by_component = None
        if model.gaussian_mixture_model is not None:
            self.gamma_by_component = self._gamma_by_component(data)

    def predict_hidden_state_log_probability(self, data):
        """ Return log probabilities of hidden states

        Arguments:
            data: dataframe of mixed data types

        Returns:
            array of log probabilites of hidden states where entry [t,i]
            is the probably of observing the emission at time t given
            hidden state i.
        """
        log_probability = np.zeros((data.shape[0], self.model.n_hidden_states))
        if self.model.categorical_model is not None:
            finite_state_data = get_finite_observations_from_data_as_states(
                self.model, data)
            log_probability += np.array(
                self.model.categorical_model.log_probability(finite_state_data))
        if self.model.gaussian_mixture_model is not None:
            gaussian_data = get_gaussian_observations_from_data(
                self.model, data)
            log_probability += np.array(
                self.model.gaussian_mixture_model.log_probability(
                    gaussian_data))

        return pd.DataFrame(log_probability, index=data.index)

    def predict_hidden_states(self, data):
        """ Predict most likely hidden state

        Arguments:
            data: dataframe of mixed data types

        Returns:
            Series of most likely hidden states
        """
        log_probability = self.predict_hidden_state_log_probability(data)
        return log_probability.idxmax(axis=1)

    def predict_hidden_states_viterbi(self, data):
        """ Predict most likely hidden states with Viterbi algorithm

        Arguments:
            data: dataframe of mixed data types

        Returns:
            Series of most likely hidden states
        """
        initial_state = np.exp(self.model.log_initial_state)
        transition = np.exp(self.model.log_transition)
        log_probability = np.array(
            self.predict_hidden_state_log_probability(data))
        probability = np.exp(
            log_probability - logsumexp(log_probability, axis=1).reshape(-1, 1))

        viterbi_matrix = np.empty((data.shape[0], len(transition)))
        backpoint_matrix = np.empty((data.shape[0], len(transition)))

        viterbi_matrix[0] = initial_state * probability[0]
        backpoint_matrix[0] = 0

        for t in range(1, data.shape[0]):
            step = viterbi_matrix[t - 1].reshape(
                -1, 1) * transition * probability[t]
            viterbi_matrix[t] = np.max(step, axis=0)
            backpoint_matrix[t] = np.argmax(step, axis=0)

        best_score = np.max(viterbi_matrix[-1])
        back_trace_start = int(np.max(backpoint_matrix[-1]))

        back_trace = [back_trace_start]
        for t in range(1, data.shape[0]):
            back_trace.append(
                int(backpoint_matrix[(data.shape[0] - 1) - t][int(
                    back_trace[t - 1])]))

        back_trace = np.flip(back_trace, axis=0)

        return pd.Series(
            back_trace,
            index=data.index,
            name='viterbi_score ' + str(best_score))

    def predict_hidden_states_gibbs(self,
                                    data,
                                    n_iterations=5,
                                    seed_inference=None):
        """ Returns most likely sequence of hidden states using Gibbs sampling.

        Arguments:
            data: dataframe of observed data
            n_iterations: number of sampling iterations
            seed_inference: sequence of hidden states to seed sampling algorithm.
                Default is None in which case hidden states are seeded randomly.

        Returns:
            Most likely sequence of hidden states determined by n_iterations of Gibbs sampling.
        """
        model = self.model
        initial_state_prob = np.exp(model.log_initial_state)
        emission_prob = np.exp(
            np.array(self.predict_hidden_state_log_probability(data)))
        emission_prob = emission_prob / np.sum(
            emission_prob, axis=1).reshape(-1, 1)
        transition_prob = np.exp(model.log_transition)

        if seed_inference is None:
            hidden_states = pd.Series(
                np.random.choice(range(model.n_hidden_states), data.shape[0]),
                index=data.index)
            hidden_states.iloc[0] = np.argmax(initial_state_prob)
        else:
            hidden_states = seed_inference.copy()
            na_index = seed_inference[seed_inference.isna()]
            if na_index.shape[0] > 0:
                hidden_states.loc[na_index] = np.random.choice(
                    range(model.n_hidden_states), na_index.shape[0])

        for i in range(n_iterations):
            sampling_parameter = np.random.uniform(0, 1, data.shape[0])
            sampling_order = np.random.choice(
                data.index.shape[0], len(data.index), replace=False)
            for t in sampling_order:
                if t == 0:
                    next_state = hidden_states.iloc[1]
                    updated_state_prob = initial_state_prob * emission_prob[
                        t] * transition_prob[:, next_state]

                elif t == hidden_states.shape[0] - 1:
                    previous_state = hidden_states.iloc[t - 1]
                    current_state = hidden_states.iloc[t]
                    updated_state_prob = transition_prob[
                        previous_state] * emission_prob[t]

                else:
                    previous_state = hidden_states.iloc[t - 1]
                    current_state = hidden_states.iloc[t]
                    next_state = hidden_states.iloc[t + 1]
                    updated_state_prob = transition_prob[
                        previous_state] * emission_prob[
                            t] * transition_prob[:, next_state]

                if np.sum(updated_state_prob) == 0:
                    updated_state_prob = np.full(len(updated_state_prob), 1)
                updated_state_prob = updated_state_prob / np.sum(
                    updated_state_prob)
                cumulative_prob = np.cumsum(updated_state_prob)
                updated_state = np.where(
                    cumulative_prob >= sampling_parameter[t])[0][0]
                hidden_states.iloc[t] = updated_state

        return hidden_states

    def conditional_probability_of_partial_observation(self, observation):
        """ Returns probability of partial observation by hidden state.

        Arguments:
            observation: single row of a dataframe of mixed partial data

        Returns:
            Array of length (number of hidden states) were entry i is the  conditional probability of the partial observation given hidden state i.

        """
        partial_finite_observation = get_finite_observations_from_data(
            self.model, observation)
        partial_gaussian_observation = get_gaussian_observations_from_data(
            self.model, observation)

        prob = []
        if self.model.categorical_model is not None:
            prob.append(
                self.model.categorical_model.
                probability_of_hidden_state_from_discrete_obs(
                    partial_finite_observation))
        if self.model.gaussian_mixture_model is not None:
            prob.append(
                self.model.gaussian_mixture_model.
                marginal_probability_of_gaussian_observation_by_hidden_state(
                    partial_gaussian_observation))
        prob = np.prod(prob, axis=0)

        return prob

    def conditional_probability_of_hidden_states(self, data):
        """ Returns dataframe of conditional probability of hidden state given observation.

        Arguments:
            data: dataframe with possible NaN entries

        Returns:
            dataframe where iloc[t,i] is the conditional probability of hidden state i
            given the complete/partial/missing observation at time t.  For complete observations,
            this is done using the gamma_chunked method, for partial/missing observations, this is
            done with a modified forward algorithm, computing p(z_t = i | x_t, x_(t-1)).
        """
        log_transition = self.model.log_transition
        nan_index = data[data.isna().any(axis=1)].index
        # Do forward backward for complete chunks of data
        if len(nan_index) < data.shape[0]:
            cond_prob = self._gamma_chunked(data)

        nan_index_total = data[data.isna().all(axis=1)].index
        nan_index_partial = [
            idx for idx in nan_index if not idx in nan_index_total
        ]

        for idx in nan_index:
            if idx in nan_index_total:
                if idx == data.index[0]:
                    cond_prob.loc[idx] = np.log(
                        np.full(self.model.n_hidden_states,
                                1 / self.model.n_hidden_states))
                else:
                    i = data.index.get_loc(idx)
                    cond_prob.loc[idx] = logsumexp(
                        np.array(cond_prob.iloc[i - 1]).reshape(-1, 1) +
                        log_transition,
                        axis=1)
            else:
                i = data.index.get_loc(idx)
                p = self.conditional_probability_of_partial_observation(
                    (data.iloc[[i - 1]]))
                log_probability = np.log(
                    p, out=np.full(p.shape, LOG_ZERO), where=(p != 0))
                alpha = self._compute_forward_probabilities(
                    pd.DataFrame(log_probability.reshape(1, -1), index=[idx]))
                cond_prob_of_partial = \
                    self.conditional_probability_of_partial_observation(
                    data.loc[[idx]])
                log_cond_prob_of_partial = [
                    np.log(p) if p > 0 else LOG_ZERO
                    for p in cond_prob_of_partial.flatten()
                ]
                joint_prob = log_cond_prob_of_partial + logsumexp(
                    alpha.reshape(-1, 1) + log_transition, axis=1)
                cond_prob.loc[idx] = joint_prob - logsumexp(joint_prob)

        return np.exp(cond_prob.loc[nan_index])

    def impute_missing_data_single_observation(self,
                                               observation,
                                               hidden_state_prob,
                                               method='argmax'):
        """ Return most observation with missing data imputed

        Arguments:
            observation: single row of a dataframe of incomplete mixed data.
            hidden_state_prob: vector of relative probabilities of hidden states
            given observation.
            method: method of imputing Gaussian data, can be either 'average'
                (which imputes the weighted average of the means) or 'maximal'
                (which imputes the means of the most probable state and
                component).

        Returns:
            Observation with missing data replaced by most data most likely to
            be observed given the current model.
        """
        new_observation = observation.copy()

        if self.model.categorical_model is not None:
            partial_finite_observation = get_finite_observations_from_data(
                self.model, observation)
            known_features = partial_finite_observation.columns[
                ~(partial_finite_observation.isna().any())].tolist()
            emission_matrix = np.exp(
                np.array(self.model.categorical_model.log_emission_matrix))

            if len(known_features) < len(self.model.finite_features):
                eligible_values_index_list = []
                finite_values = self.model.finite_values
                if len(known_features) == 0:
                    # Impute missing values when all data is missing.
                    eligible_values_index_list = list(finite_values.index)
                else:
                    # Impute missing values when partial data is missing.
                    for feat in known_features:
                        val = partial_finite_observation.loc[:, feat][0]
                        eligible_values_index_list.append(
                            set(finite_values[finite_values[feat] == val]
                                .index))
                    eligible_values_index_list = [
                        i for i in set.intersection(*eligible_values_index_list)
                    ]
                eligible_values_index_list.sort()
                eligible_emissions = np.array(
                    [emission_matrix[i] for i in eligible_values_index_list])
                maximum_index = np.argmax(
                    np.sum(hidden_state_prob * eligible_emissions, axis=1))
                most_likely_observation = self.model.finite_values.iloc[[
                    eligible_values_index_list[maximum_index]
                ]]
                for col in most_likely_observation.columns:
                    new_observation.loc[new_observation.index[
                        0], col] = most_likely_observation.loc[
                            most_likely_observation.index[0], col]

        if self.model.gaussian_mixture_model is not None:
            partial_gaussian_observation = get_gaussian_observations_from_data(
                self.model, observation)
            obs = np.array(partial_gaussian_observation.iloc[0])
            index_nan = np.argwhere(np.isnan(obs)).flatten()
            unknown_values = partial_gaussian_observation.columns[index_nan]
            if len(unknown_values) > 0:
                means = np.array(self.model.gaussian_mixture_model.means)
                covariances = np.array(
                    self.model.gaussian_mixture_model.covariances)
                component_weights = np.array(
                    self.model.gaussian_mixture_model.component_weights)
                if method == 'average':
                    conditional_means_of_missing_values = np.empty(
                        (self.model.n_hidden_states, len(index_nan)))
                    for i in range(self.model.n_hidden_states):
                        conditional_means_of_missing_values[
                            i] = compute_mean_of_conditional_probability_gaussian_mixture(
                                partial_gaussian_observation, means[i],
                                covariances[i], component_weights[i])
                    conditional_means_of_missing_values = np.sum(
                        np.array(hidden_state_prob).reshape(-1, 1) *
                        conditional_means_of_missing_values,
                        axis=0)

                if method == 'maximal':
                    maximal_hidden_state = np.argmax(hidden_state_prob)
                    conditional_means_of_missing_values = compute_mean_of_conditional_probability_gaussian_mixture(
                        partial_gaussian_observation,
                        means[maximal_hidden_state],
                        covariances[maximal_hidden_state],
                        component_weights[maximal_hidden_state])

                if method == 'argmax':
                    new_covariances = np.empty((np.array(covariances).shape[0],
                                                np.array(covariances).shape[1],
                                                len(index_nan), len(index_nan)))
                    new_means = np.empty((np.array(means).shape[0],
                                          np.array(means).shape[1],
                                          len(index_nan)))
                    for i in range(new_covariances.shape[0]):
                        for j in range(new_covariances.shape[1]):
                            new_covariances[i][
                                j] = compute_covariance_of_conditional_probability_gaussian(
                                    partial_gaussian_observation, means[i][j],
                                    covariances[i][j])
                            new_means[i][
                                j] = compute_mean_of_conditional_probability_gaussian(
                                    partial_gaussian_observation, means[i][j],
                                    covariances[i][j])

                    covariances = np.array(
                        self.model.gaussian_mixture_model.covariances)
                    means = np.array(self.model.gaussian_mixture_model.means)
                    component_weights = np.array(
                        self.model.gaussian_mixture_model.component_weights)
                    probabilities = np.zeros((means.shape[0], means.shape[1]))
                    for i in range(probabilities.shape[0]):
                        for j in range(probabilities.shape[1]):
                            pdf = np.exp(
                                stats.multivariate_normal.logpdf(
                                    means[i][j],
                                    means[i][j],
                                    covariances[i][j],
                                    allow_singular=True))
                            probabilities[i][j] += hidden_state_prob[
                                i] * component_weights[i][j] * pdf

                    max_index = np.where(
                        probabilities == np.amax(probabilities))

                    conditional_means_of_missing_values = new_means[max_index[
                        0][0]][max_index[1][0]]

            for i in range(len(index_nan)):
                new_observation.loc[new_observation.index[0], unknown_values[
                    i]] = conditional_means_of_missing_values[i]

        return new_observation

    def impute_missing_data(self, data, method='argmax'):
        """ Return dataframe with missing data imputed

        Arguments:
            data: dataframe of observations with some entries as NaN.
            method: method of imputing Gaussian data, can be either 'average' (imputes the weighted average of the means) or 'maximal' (imputes the means of the most probable state and component) or 'argmax' (imputes the mean with highest probability).
        Returns:
            Observation with missing data replaced by most data most likely to be observed given the current model.
        """
        imputed_data = data.copy()
        # Make sure that integer columns are being cast as integers.
        float_to_int = {
            feature: "Int64"
            for feature in imputed_data[self.model.finite_features]
            .select_dtypes("float")
        }
        imputed_data = imputed_data.astype(float_to_int, errors='ignore')

        incomplete_observations = data[data.isna().any(axis=1)]
        complete_data_chunks = get_complete_data_chunks(data)

        for idx in incomplete_observations.index:
            if idx < complete_data_chunks.iloc[0, 0]:
                if idx == data.index[0]:
                    cond_prob = np.full(self.model.n_hidden_states,
                                        1 / self.model.n_hidden_states)
                    hidden_state_probabilities = pd.DataFrame(
                        cond_prob.reshape(1, -1), index=[idx])
                else:
                    hidden_state_probabilities = self.conditional_probability_of_hidden_states(
                        imputed_data.loc[:idx])
            else:
                start = complete_data_chunks[
                    complete_data_chunks['end'] < idx].iloc[-1, 0]
                hidden_state_probabilities = self.conditional_probability_of_hidden_states(
                    imputed_data.loc[start:idx])

            imputed_data.loc[[
                idx
            ]] = self.impute_missing_data_single_observation(
                incomplete_observations.loc[[idx]],
                np.array(hidden_state_probabilities.loc[idx]), method)

        return imputed_data

    def _compute_forward_probabilities(self, log_probability):
        """ Compute forward probabilities.

        Arguments:
            log_probability: dataframe of log probability of hidden state

        Returns:
            Array where entry [t,i] is the log probability of observations o_0,...,o_t and h_t = i under the current model parameters.

        """
        log_transition = self.model.log_transition
        log_initial_state = self.model.log_initial_state
        log_prob = np.array(log_probability)
        if self.use_jax == True:
            alpha = jax_compute_forward_probabilities(
                log_initial_state, log_transition, log_probability)
        else:
            alpha = np.empty((log_probability.shape))
            alpha[0] = log_initial_state + log_prob[0]
            for t in range(1, len(alpha)):
                alpha_t = alpha[t - 1, None].reshape(
                    -1, 1) + log_transition + log_prob[t]
                alpha[t] = logsumexp(alpha_t, axis=0)

        return alpha

    def _compute_backward_probabilities(self, log_probability):
        """ compute backard probabilities.

        Arguments:
            log_probability: dataframe of log probability of hidden state

        Returns:
            Array where entry [t,i] is the log probability of observations o_{t+1},...,o_T given h_t = i under the current model parameters.

        """
        log_transition = self.model.log_transition

        log_prob = np.array(log_probability)

        if self.use_jax == True:
            beta = jax_compute_backward_probabilities(log_transition,
                                                      log_probability)
        else:
            beta = np.empty((log_probability.shape))
            beta[0] = 0
            for t in range(1, len(beta)):
                beta_t = beta[t - 1, None] + log_transition + log_prob[len(beta)
                                                                       - t]
                beta[t] = logsumexp(beta_t, axis=1)

            beta = np.flip(beta, axis=0)
        return beta

    def _gamma(self, data):
        """Auxiliary function for EM.

        Arguments:
            model: HiddenMarkovModel object.

        Returns:
            List of gamma values where gamma[t,i] is the log proability of
            hidden state i occuring at time t, given observations and current HMM parameters.
        """
        log_probability = np.asarray(
            self.predict_hidden_state_log_probability(data))
        alpha = self._compute_forward_probabilities(log_probability)
        beta = self._compute_backward_probabilities(log_probability)
        gamma = np.asarray(alpha) + np.asarray(beta)
        gamma -= logsumexp(gamma, axis=1).reshape(-1, 1)
        return gamma

    def _gamma_by_component(self, data):
        """Auxiliary function for EM.

        Arguments:
            data: dataframe of mixed data types

        Returns:
            List of gamma values where gamma[i][t][l] gives the log probability of being in hidden states i and component l at time t, given observations and current HMM parameters.
        """
        n_hidden_states = self.model.n_hidden_states
        n_components = self.model.gaussian_mixture_model.n_gmm_components
        weights = self.model.gaussian_mixture_model.component_weights
        log_weights = np.log(
            weights,
            out=np.zeros_like(weights) + LOG_ZERO,
            where=(weights != 0))

        gaussian_data = get_gaussian_observations_from_data(self.model, data)
        log_probability = np.array(
            self.model.gaussian_mixture_model.log_probability(gaussian_data))
        log_probability_by_component = np.array(
            self.model.gaussian_mixture_model.log_probability_by_component(
                gaussian_data))

        gamma = self._gamma(data)
        gamma_by_component = np.empty((n_hidden_states, data.shape[0],
                                       n_components))
        for i in range(n_hidden_states):
            gamma_by_component[i] = np.array([g[i] for g in gamma]).reshape(
                -1, 1
            ) + log_probability_by_component[i] + log_weights[i] - np.array(
                [l[i] for l in log_probability]).reshape(-1, 1)

        return gamma_by_component

    def _gamma_chunked(self, data):
        """ Return gamma for chunks of non-NaN data.

        Arguments:
            data: dataframe with possible NaN entries

        Returns:
            Dataframe where iloc[t,i] is the probability of hidden state i
            given the observation at time t, computed using the forward backward
            algorithm; for rows with NaN entries, iloc[,i] is NaN.
        """
        gamma_chunk = pd.DataFrame()
        data_chunks = get_complete_data_chunks(data)

        def convert_chunk(data, chunks, i):
            chunk = data.loc[chunks.loc[i, 'start']:chunks.loc[i, 'end']]
            return pd.DataFrame(self._gamma(chunk), index=chunk.index)

        gamma_chunk = pd.concat(
            [convert_chunk(data, data_chunks, i) for i in data_chunks.index])

        nan_entries = pd.DataFrame(
            index=data[data.isna().any(axis=1)].index,
            columns=gamma_chunk.columns)
        gamma_chunk = pd.concat((gamma_chunk, nan_entries))
        gamma_chunk.sort_index(inplace=True)

        return gamma_chunk

    def _xi(self, data):
        """Auxiliary function for EM.

        Arguments:
            data: dataframe of mixed data types

        Returns:
            List of xi values where xi[t][i,j] gives the log probability
            of hidden state i occuring at time t and j occuring at time t+1,
            given observations and current HMM parameters.
        """
        n_hidden_states = self.model.n_hidden_states
        log_probability = np.asarray(
            self.predict_hidden_state_log_probability(data))
        gamma = self._gamma(data)
        beta = self._compute_backward_probabilities(log_probability)
        log_transition = self.model.log_transition

        xi = np.empty((data.shape[0] - 1, n_hidden_states, n_hidden_states))
        for t in range(data.shape[0] - 1):
            xi[t] = gamma[t].reshape(
                -1, 1
            ) + log_transition + log_probability[t +
                                                 1] + beta[t +
                                                           1] - beta[t].reshape(
                                                               -1, 1)

        return xi


class DiscreteHMMForecasting(HMMForecasting):
    """ Forecasting class specific to discrete HMM
    """

    def __init__(self, model, use_jax=False):
        super().__init__(model)
        self.inf = model.load_inference_interface(use_jax)

    def hidden_state_probability_at_conditioning_date(self, data,
                                                      conditioning_date):
        """ Compute probability of hidden state given data up to conditioning date.

        Arguments:
            data: dataframe with complete data
            conditioning_date: entry from `data` index, forecast will
                consider only the data up to this date.

        Returns:
            Probability distribution of hidden state at condtioning date.
        """
        data_restricted = data.loc[:conditioning_date]

        log_prob = self.inf.predict_hidden_state_log_probability(
            data_restricted)
        joint_prob = self.inf._compute_forward_probabilities(log_prob)

        return np.exp(joint_prob[-1] - logsumexp(joint_prob, axis=1)[-1])

    def hidden_state_probability_at_horizon(self, data, horizon_timestep,
                                            conditioning_date):
        """ Compute hidden state probability at horizon.

        Argument:
            data: dataframe with complete data
            horizon_timestep: timestep to consider for horizon.
                It is assumed that the rows of `data` have a uniform timedelta; this uniform timedelta is 1 timestep
            conditioning_date: entry from `data` index, forecast will
                consider only the data up to this date.

        Returns:
            An array with dimention (1 x n_hidden_states) where the ith entry is the conditional probability of hidden state i at horizon.
        """
        conditioning_date_prob = self.hidden_state_probability_at_conditioning_date(
            data, conditioning_date)
        transition_matrix = np.exp(self.model.log_transition)

        return conditioning_date_prob @ np.linalg.matrix_power(
            transition_matrix, horizon_timestep)

    def hidden_state_probability_at_horizons(self, data, horizon_timesteps,
                                             conditioning_date):
        """ Compute hidden state probability at horizons.

        Argument:
            data: dataframe with complete data
            horizon_timesteps: list of timesteps to consider for horizons.
                It is assumed that the rows of `data` have a uniform timedelta; this uniform timedelta is 1 timestep
            conditioning_date: entry from `data` index, forecast will
                consider only the data up to this date.

        Returns:
            Dataframe with hidden state probabilities for horizon dates.
        """
        conditioning_date_prob = self.hidden_state_probability_at_conditioning_date(
            data, conditioning_date)

        transition_matrix = np.exp(self.model.log_transition)

        delta = data.index[-1] - data.index[-2]
        horizon_date = [
            conditioning_date + (t * delta) for t in horizon_timesteps
        ]
        horizon_prediction = np.array([
            conditioning_date_prob @ np.linalg.matrix_power(
                transition_matrix, t) for t in horizon_timesteps
        ])

        forecast = pd.DataFrame(
            horizon_prediction,
            index=horizon_date,
            columns=[i for i in range(len(transition_matrix))])

        return forecast

    def _forecast_hidden_state_at_horizons(self, data, horizon_timesteps,
                                           conditioning_date):
        """ Returns series with most likely hidden states at horizons.

        Arguments:
            data: dataframe with complete data
            horizon_timesteps: list of timesteps to consider for horizons.
                It is assumed that the rows of `data` have a uniform timedelta; this uniform timedelta is 1 timestep
            conditioning_date: entry from `data` index, forecast will
                consider only the data up to this date.

        Returns:
            Series with hidden state prections with the conditioning date as the first entry.
        """
        forecast = self.hidden_state_probability_at_horizons(
            data, horizon_timesteps, conditioning_date)

        return pd.Series(
            np.array(forecast).argmax(axis=1), index=forecast.index)

    def forecast_observation_at_horizon(self,
                                        data,
                                        horizon_timestep,
                                        conditioning_date,
                                        imputation_method='average'):
        """ Returns dataframe with most likely observations at horizon.

        Arguments:
            data: dataframe with complete data
            horizon_timestep: timestep to consider for horizon.
                It is assumed that the rows of `data` have a uniform timedelta; this uniform timedelta is 1 timestep
            conditioning_date: entry from `data` index, forecast will
                consider only the data up to this date.

        Returns:
            dataframe with forecast observations at horizon
        """
        delta = data.index[-1] - data.index[-2]
        new_time = conditioning_date + (horizon_timestep * delta)
        observation = data.loc[[conditioning_date]].copy()
        observation.loc[new_time, data.columns] = np.nan

        hidden_state_prob = self.hidden_state_probability_at_horizon(
            data, horizon_timestep, conditioning_date)

        forecast = self.inf.impute_missing_data_single_observation(
            observation.loc[[new_time]], hidden_state_prob, imputation_method)

        return forecast

    def _forecast_observation_at_horizons(self,
                                          data,
                                          horizon_timesteps,
                                          conditioning_date,
                                          imputation_method='average'):
        """ Returns dataframe with most likely observations at horizons.

        Arguments:
            data: dataframe with complete data
            horizon_timesteps: list of timesteps to consider for horizons.
                It is assumed that the rows of `data` have a uniform timedelta; this uniform timedelta is 1 timestep
            conditioning_date: entry from `data` index, forecast will
                consider only the data up to this date.

        Returns:
            Dataframe with forecast observations for horizon_timestep dates. The first row of the dataframe is the conditioning date.
        """
        forecast = data.loc[[conditioning_date]]
        for horizon in horizon_timesteps:
            forecast = pd.concat((forecast,
                                  self.forecast_observation_at_horizon(
                                      data,
                                      horizon,
                                      conditioning_date,
                                      imputation_method,
                                  )))

        return forecast

    def steady_state(self):
        """ Return steady state for model.
        """

        transition = np.exp(self.model.log_transition)
        val, left_eig, right_eig = linalg.eig(transition, left=True)
        idx = np.argmax(np.array([abs(v) for v in val]))
        vec = left_eig[:, idx]

        return vec / np.sum(vec)

    def _steady_state_and_horizon(self, data, conditioning_date, atol):
        """ Returns dictionary with steady state information.

        Arguments:
            data: dataframe with complete data
            conditioning_date: entry from `data` index, forecast will
                consider only the data up to this date.
            atol: tolerance for determining whether steady state has been
                reached.

        Returns:
            Dictionary with 'steady_state' for model and 'steady_state_horizon_timesteps', the timestep horizon at which the steady state has been achieved up to tolerance atol, and 'steady_state_horizon_date', the date at which the steady state has been achieved up to tolerance atol.
        """
        if conditioning_date is None:
            conditioning_date = data.index[-1]
        vec = self.steady_state()
        transition = np.exp(self.model.log_transition)
        initial_prob = self.hidden_state_probability_at_conditioning_date(
            data, conditioning_date)

        i = 1
        while np.max(
                np.abs(initial_prob @ np.linalg.matrix_power(transition, i) -
                       vec)) > atol:
            i += 1

        delta = data.index[-1] - data.index[-2]
        horizon_date = conditioning_date + (i * delta)

        return {
            'steady_state': vec,
            'steady_state_horizon_timesteps': i,
            'steady_state_horizon_date': horizon_date
        }


class DiscreteHMMValidationMetrics(HMMValidationMetrics):
    """ Validation class specific to discrete HMM
    """

    def __init__(self, model, actual_data, use_jax=False):
        super().__init__(model, actual_data)
        self.inf = model._load_inference_interface(use_jax)
        self.actual_gaussian_data = get_gaussian_observations_from_data(
            self.model, actual_data)
        self.actual_categorical_data = get_finite_observations_from_data(
            self.model, actual_data)

    def _validate_imputation(self, redacted_data, imputed_data):
        """ Return DiscreteHMM specific dictionary of validation metrics for imputation.

        Arguments:
            redacted_data: dataframe with values set to nan.
            imputed_data: dataframe with missing values imputed.

        Returns:
            Dictionary with validation metrics for imputed data against actual data.
        """
        cond_prob_of_hidden_states = self.inf.conditional_probability_of_hidden_states(
            redacted_data)
        val_dict = {}

        if self.model.categorical_model:
            redacted_categorical_data = get_finite_observations_from_data(
                self.model, redacted_data)
            imputed_categorical_data = get_finite_observations_from_data(
                self.model, imputed_data)

            val_dict[
                'accuracy_of_imputed_categorical_data'] = self.accuracy_of_predicted_categorical_data(
                    redacted_categorical_data, imputed_categorical_data)

            val_dict[
                'relative_accuracy_of_imputed_categorical_data'] = self.relative_accuracy_of_predicted_categorical_data(
                    redacted_categorical_data, imputed_categorical_data)

            val_dict[
                'best_possible_accuracy_of_categorical_imputation'] = best_possible_accuracy_of_categorical_prediction(
                    self.actual_categorical_data, redacted_categorical_data)

        if self.model.gaussian_mixture_model:
            redacted_gaussian_data = get_gaussian_observations_from_data(
                self.model, redacted_data)
            imputed_gaussian_data = get_gaussian_observations_from_data(
                self.model, imputed_data)

            val_dict[
                'average_relative_log_likelihood_of_imputed_gaussian_data'] = self.average_relative_log_likelihood_of_predicted_gaussian_data(
                    redacted_gaussian_data, imputed_gaussian_data,
                    cond_prob_of_hidden_states)

            val_dict[
                'average_z_score_of_imputed_gaussian_data'] = self.average_z_score_of_predicted_gaussian_data(
                    redacted_gaussian_data, cond_prob_of_hidden_states)

        return val_dict

    def _validate_forecast(self, forecast_data):
        """ Return DiscreteHMM specific dictionary of validation metrics for imputation.

        Arguments:
            forecast_data: dataframe with forecast data where the first row
                of the dataframe is actual observed conditioning date data.

        Returns:
            Dictionary with validation metrics for forecast data against actual data.
        """
        conditioning_date = forecast_data.index[0]
        delta = self.actual_data.index[-1] - self.actual_data.index[-2]

        horizon_timesteps = [
            int(t) for t in (forecast_data.index - conditioning_date) / delta
        ]

        cond_prob_of_hidden_states = DiscreteHMMForecasting(
            self.model).hidden_state_probability_at_horizons(
                self.actual_data, horizon_timesteps, conditioning_date)

        val_dict = {}

        if self.model.categorical_model:
            forecast_categorical_data = get_finite_observations_from_data(
                self.model, forecast_data)

            redacted_categorical_data = forecast_categorical_data.copy()
            redacted_categorical_data.loc[:, :] = np.nan

            val_dict[
                'accuracy_of_forecast_categorical_data'] = self.accuracy_of_predicted_categorical_data(
                    redacted_categorical_data, forecast_categorical_data)

            val_dict[
                'relative_accuracy_of_forecast_categorical_data'] = self.relative_accuracy_of_predicted_categorical_data(
                    redacted_categorical_data, forecast_categorical_data)

            val_dict[
                'best_possible_accuracy_of_categorical_forecast'] = best_possible_accuracy_of_categorical_prediction(
                    self.actual_categorical_data, redacted_categorical_data)

        if self.model.gaussian_mixture_model:
            forecast_gaussian_data = get_gaussian_observations_from_data(
                self.model, forecast_data)

            redacted_gaussian_data = forecast_gaussian_data.copy()
            redacted_gaussian_data.loc[:, :] = np.nan

            val_dict[
                'average_relative_log_likelihood_of_forecast_gaussian_data'] = self.average_relative_log_likelihood_of_predicted_gaussian_data(
                    redacted_gaussian_data, forecast_gaussian_data,
                    cond_prob_of_hidden_states)

            val_dict[
                'average_z_score_of_forecast_gaussian_data'] = self.average_z_score_of_predicted_gaussian_data(
                    redacted_gaussian_data, cond_prob_of_hidden_states)

        return val_dict

    def average_relative_log_likelihood_of_predicted_gaussian_data(
            self, redacted_gaussian_data, imputed_gaussian_data,
            conditional_probability_of_hidden_states):
        """Returns the difference between the log likelihood of the actual
        data and the log likelihood of the imputed data.  This is done
        using the probability density function for the conditional probability
        of the unknown part of the observation given the known part of the
        observation.  This metric is intended to be a measure of how surprised
        you should be to see the actual value relative to the imputed value.

        Arguments:
            redacted_gaussian_data: dataframe if Gaussian observations
                with values set to nan.
            imputed_gaussian_data: dataframe with missing values imputed.
            conditional_probability_of_hidden_states: dataframe with
                conditional probability of hidden states given partial
                observations at all timesteps with redacted data.

        Returns:
            float
        """
        actual_gaussian_data = self.actual_gaussian_data
        means = self.model.gaussian_mixture_model.means
        covariances = self.model.gaussian_mixture_model.covariances
        component_weights = self.model.gaussian_mixture_model.component_weights

        redacted_index = redacted_gaussian_data[redacted_gaussian_data.isnull()
                                                .any(axis=1)].index
        imputed_likelihood = np.empty(len(redacted_index))
        actual_likelihood = np.empty(len(redacted_index))
        for i in range(len(redacted_index)):
            idx = redacted_index[i]
            p = np.float64(
                np.array(conditional_probability_of_hidden_states.loc[idx]))
            log_cond_prob = np.log(
                p, np.full(p.shape, LOG_ZERO), where=(p != 0))

            actual_gaussian_observation = actual_gaussian_data.loc[[idx]]
            imputed_gaussian_observation = imputed_gaussian_data.loc[[idx]]
            partial_gaussian_observation = redacted_gaussian_data.loc[[idx]]

            actual_prob = compute_log_likelihood_with_inferred_pdf(
                actual_gaussian_observation, partial_gaussian_observation,
                means, covariances, component_weights)
            actual_likelihood[i] = logsumexp(actual_prob + log_cond_prob)

            imputed_prob = compute_log_likelihood_with_inferred_pdf(
                imputed_gaussian_observation, partial_gaussian_observation,
                means, covariances, component_weights)
            imputed_likelihood[i] = logsumexp(imputed_prob + log_cond_prob)

        total_actual_log_likelihood = logsumexp(actual_likelihood)
        total_imputed_log_likelihood = logsumexp(imputed_likelihood)

        return total_actual_log_likelihood - total_imputed_log_likelihood

    def average_z_score_of_predicted_gaussian_data(
            self, redacted_gaussian_data,
            conditional_probability_of_hidden_states):
        """ Computes z score of gaussian data averaged over observations.

        Arguments:
            redacted_gaussian_data: dataframe if Gaussian observations
                with values set to nan.
            imputed_gaussian_data: dataframe with missing values imputed.
            conditional_probability_of_hidden_states: dataframe with
                conditional probability of hidden states given partial
                observations at all timesteps with redacted data.

        Returns:
            float
        """
        means = self.model.gaussian_mixture_model.means
        covariances = self.model.gaussian_mixture_model.covariances
        component_weights = self.model.gaussian_mixture_model.component_weights

        return average_z_score(
            means, covariances, component_weights, self.actual_gaussian_data,
            redacted_gaussian_data, conditional_probability_of_hidden_states)

    def accuracy_of_predicted_categorical_data(self, redacted_categorical_data,
                                               imputed_categorical_data):
        """ Returns ratio of correctly imputed categorical values to total imputed categorical values.

        Arguments:
            redacted_categorical_data: dataframe of categorical data with
                values set to nan.
            imputed_categorical_data: dataframe of categorical data with missing values fill in.

        Returns:
            float
        """
        redacted_index = redacted_categorical_data[
            redacted_categorical_data.isnull().any(axis=1)].index

        total_correct = np.sum(
            (self.actual_categorical_data.loc[redacted_index] ==
             imputed_categorical_data.loc[redacted_index]).all(axis=1))

        return total_correct / len(redacted_index)

    def relative_accuracy_of_predicted_categorical_data(
            self, redacted_categorical_data, imputed_categorical_data):
        """ Returns ratio of rate of accuracy in imputed data to expected rate of accuracy with random guessing.

        Arguments:
            redacted_categorical_data: dataframe of categorical data with
                values set to nan.
            imputed_categorical_data: dataframe of categorical data with missing values fill in.

        Returns:
            float
        """
        expected_accuracy = expected_proportional_accuracy(
            self.actual_categorical_data, redacted_categorical_data)
        imputed_accuracy = self.accuracy_of_predicted_categorical_data(
            redacted_categorical_data, imputed_categorical_data)

        return imputed_accuracy / expected_accuracy

    def precision_recall_df_for_predicted_categorical_data(
            self, redacted_data, imputed_data):
        """ Return DataFrame with precision, recall, and proportion of categorical values

        Arguments:
            redacted_data: dataframe with values set to nan.
            imputed_data: dataframe with missing values imputed.

        Returns:
            Dataframe with precision, recall, and proportion of imputed data against actual data.
        """
        if len(self.model.finite_features) == 0:
            return None
        else:
            redacted_categorical_data = get_finite_observations_from_data(
                self.model, redacted_data)
            redacted_index = redacted_categorical_data[
                redacted_categorical_data.isnull().any(axis=1)].index

            df = self.actual_data.copy()
            df['tuples'] = list(
                zip(*[
                    self.actual_data[c]
                    for c in self.actual_categorical_data.columns
                ]))
            proportion = (df['tuples'].value_counts() / df.shape[0]).to_dict()

            df_imputed = imputed_data.copy()
            df_imputed['tuples'] = list(
                zip(*[
                    imputed_data[c]
                    for c in self.actual_categorical_data.columns
                ]))

            state = df['tuples'].unique()

            precision_recall = pd.DataFrame(
                np.full((df['tuples'].nunique(), 3), np.nan),
                index=state,
                columns=['precision', 'recall', 'proportion'])

            for n, idx in enumerate(precision_recall.index):
                # pandas cannot use loc with tuples in the index
                precision_recall.iloc[n]['proportion'] = proportion[idx]

            actual = df.loc[redacted_index, 'tuples']

            imputed = df_imputed.loc[redacted_index, 'tuples']

            true_pos = ((np.array(actual)[:, None] == state) &
                        (np.array(imputed)[:, None] == state)).sum(axis=0)
            false_pos = ((np.array(actual)[:, None] != state) &
                         (np.array(imputed)[:, None] == state)).sum(axis=0)
            false_neg = ((np.array(actual)[:, None] == state) &
                         (np.array(imputed)[:, None] != state)).sum(axis=0)

            precision_recall.loc[state, 'precision'] = [
                x for x in true_pos / np.
                array([np.nan if x == 0 else x for x in true_pos + false_pos])
            ]
            precision_recall.loc[state, 'recall'] = [
                x for x in true_pos / np.
                array([np.nan if x == 0 else x for x in true_pos + false_neg])
            ]

            precision_recall.sort_values(
                by=['proportion'], ascending=False, inplace=True)
            return precision_recall
