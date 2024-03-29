""" Hidden Markov model implementation for discrete hidden state hmms.
"""

import itertools
from abc import ABC

import numpy as np
import pandas as pd
from scipy import linalg, stats
from scipy.special import logsumexp

from .base_models import (LOG_ZERO, HiddenMarkovModel, HMMConfiguration,
                          HMMLearningAlgorithm)
from .utils import *


class DiscreteHMMConfiguration(HMMConfiguration):
    """ Intilialize HMM configuration from specification dictionary. """

    def __init__(self, n_hidden_states=None):
        super().__init__(n_hidden_states)

    def _from_spec(self, spec):
        """ Discrete HMM specific implementation of `from_spec`. """
        self.n_hidden_states = spec['n_hidden_states']
        self.model_type = 'DiscreteHMM'
        return self

    def _to_model(self, set_random_state):
        """ Discrete HMM specific implementation of `to_model`. """
        return DiscreteHMM.from_config(self, set_random_state)


class DiscreteHMM(HiddenMarkovModel):
    """ Model class for dicrete hidden Markov models """

    def __init__(self, model_config=None):
        super().__init__(model_config)
        self.set_random_state = None
        self.random_state = None
        self.trained = False

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
    def from_config(cls, model_config, set_random_state):
        model = cls(model_config=model_config)
        model.n_hidden_states = model_config.n_hidden_states
        model.set_random_state = set_random_state

        # Set random state.
        random_state = np.random.RandomState(set_random_state)
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
                model_config,random_state)

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

    def _load_learning_interface(self):
        """ Loads DiscreteHMM specific learning interface."""
        return DiscreteHMMLearningAlgorithm()

    def _load_inference_interface(self, use_jax):
        """ Loads DiscreteHMM specific inference interface."""
        return DiscreteHMMInferenceResults(self, use_jax)

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

class CategoricalModel(DiscreteHMM):

    def __init__(self,
                 random_state=None,
                 n_hidden_states=None,
                 finite_features=None,
                 finite_values=None,
                 finite_values_dict=None,
                 finite_values_dict_inverse=None,
                 log_emission_matrix=None):
        self.random_state = random_state
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

    def categorical_log_probability(self, finite_data_enum):
        """ Return log conditional prob of observation given hidden state.

        Arguments:
            finite_data_enum: (Series) finite observations by observation 
                vector enumeration (i.e. 0 = (a,a), 1 = (a,b), etc.).

        Returns:
            Dataframe where entry [t,i] is the log conditional probability of 
            emitting the discrete observation x_t given hidden state i, more 
            formally, log[ p(x_t | z_t = i) ].
        """
        n_observations = finite_data_enum.shape[0]
        log_emission = np.array(self.log_emission_matrix)

        return pd.DataFrame(
            [
                list(prob) for prob in np.array(
                    finite_data_enum.map(lambda x: log_emission[x]))
            ],
            index=finite_data_enum.index)

    def update_log_emission_matrix(self, gamma, finite_data_enum):
        """ Update log emission matrix for categorical model

        Arguments:
            gamma: output of DiscreteHMMInferenceResults
            finite_data_enum: (Series) finite observations by observation 
                vector enumeration (i.e. 0 = (a,a), 1 = (a,b), etc.).

        Returns:
            Updated log emission matrix.
        """
        log_emission_matrix = np.full(
            (np.array(self.log_emission_matrix).shape), LOG_ZERO)
        gamma_df = pd.DataFrame(gamma, index=finite_data_enum.index)
        for l in self.finite_values.index:
            if l in finite_data_enum.unique():
                l_index = finite_data_enum[finite_data_enum == l].index
                l_gamma_df = gamma_df.loc[l_index]
                log_emission_matrix[l] = logsumexp(np.array(l_gamma_df), axis=0)
        log_emission_matrix -= logsumexp(gamma, axis=0)
        return log_emission_matrix


class GaussianMixtureModel(DiscreteHMM):

    def __init__(self,
                 random_state=None,
                 n_hidden_states=None,
                 gaussian_features=None,
                 gaussian_values=None,
                 dims=None,
                 n_gmm_components=None,
                 component_weights=None,
                 means=None,
                 covariances=None):
        self.random_state = random_state
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
        gmm.random_state = random_state
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
                #gmm.covariances = gmm_params['covariances']
                covariances = np.array(gmm.n_hidden_states *
                    [np.array(gmm.n_gmm_components * [np.identity(gmm.dims)])])
                gmm.covariances = covariances.copy()
            if 'component_weights' in gmm_params:
                gmm.component_weights = gmm_params['component_weights']

        if gmm.n_gmm_components is None:
            # set n_gmm_components equal to n_hidden_states if no value is given
            gmm.n_gmm_components = gmm.n_hidden_states
        if gmm.means is None:
            means = random_state.uniform(0,1,
                (gmm.n_hidden_states, gmm.n_gmm_components, 
                    len(gaussian_features)))
            gmm.means = means.copy()
        if gmm.covariances is None:
            covariances = np.array(
                gmm.n_hidden_states *
                [np.array(gmm.n_gmm_components * [np.identity(gmm.dims)])])
            gmm.covariances = covariances.copy()
        if gmm.component_weights is None:
            weights = random_state.uniform(0,1,
                (gmm.n_hidden_states, gmm.n_gmm_components))
            weights = weights/weights.sum(axis = 1, keepdims = True)
            gmm.component_weights = weights.copy()

        return gmm

    def gaussian_log_probability_by_component(self, gaussian_data):
        """ Return log conditional prob of obs given state and component.

        Arguments:
            gaussian_data: observed gaussian data as DataFrame

        Returns:
            Numpy array where entry [i,t,m] is the log conditional probability 
            of emitting continuous observation x_t in hidden state i and gmm 
            component m, formally log[ p(x_t | z_t = h_i, c_t = m) ].
        """
        n_hidden_states = self.n_hidden_states
        n_gmm_components = self.n_gmm_components
        n_observations = gaussian_data.shape[0]
        means = self.means
        covariances = self.covariances

        log_emission_by_component = np.full(
            (n_hidden_states, n_observations, n_gmm_components), np.nan)
        for i in range(n_hidden_states):
            for m in range(n_gmm_components):
                try:
                    log_emission_by_component[i,:,m
                                        ] = stats.multivariate_normal.logpdf(
                                            x = gaussian_data,
                                            mean = means[i][m],
                                            cov = covariances[i][m],
                                            allow_singular=False)

                except:
                    raise ValueError("Oh no, it looks like you've arrived at "+
                        "a singular covariance matrix.  This probably means "+
                        "that you are using too many gmm components.  Try "+
                        "reinitializing your model with fewer gmm components " +
                        "or try seeding it with difference initial parameters.")

        return log_emission_by_component

    def gaussian_log_probability(self, gaussian_data):
        """ Return log conditional prob of observation given hidden state.

        Arguments:
            gaussian_data: observed gaussian data as DataFrame

        Returns:
            DataFrame of log conditional probabilities of observations given 
            hidden states.  Entry in row t and column i corresponds to the 
            log conditional probability log[ p(x_t | z_t = i) ] 
        """
        n_hidden_states = self.n_hidden_states
        n_observations = gaussian_data.shape[0]
        n_gmm_components = self.n_gmm_components
        weights = np.array(self.component_weights)
        weights = weights.reshape(n_hidden_states, -1, n_gmm_components)

        log_emission_by_component = self.gaussian_log_probability_by_component(
            gaussian_data)
        prob = (weights * np.exp(log_emission_by_component)).sum(axis = 2)
        prob = np.where(prob != 0, prob, 1e-08)
        log_prob = np.log(prob.transpose())

        return pd.DataFrame(log_prob, index=gaussian_data.index)

    def update_means(self, gaussian_data, gamma_by_component):
        """ Return updated means for current hmm parameters.

        Arguments:
            gaussian_data: observed gaussian data as DataFrame
            gamma_by_component: output of `_gamma_by_component`

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
        return means.copy()

    def update_covariances(self, gaussian_data, gamma_by_component,r):
        """ Return updated covariances for current hmm parameters.

        Arguments:
            gaussian_data: observed gaussian data as DataFrame
            gamma_by_component: output of `_gamma_by_component`
            r: (float) regularization constant.

        Returns:
            Array of updated covariance matrices.
        """
        n_hidden_states = self.n_hidden_states
        n_gmm_components = self.n_gmm_components
        means = self.means
        covariances = np.empty(self.covariances.shape)
        for i in range(n_hidden_states):
            for m in range(n_gmm_components):
                gamma_exp = np.exp(gamma_by_component[i,:,m])
                error = (gaussian_data - means[i][m]).values
                error_prod = np.array([e.reshape(-1, 1) @ e.reshape(1, -1)
                                            for e in error])
                new_covariance = np.array([gamma_exp[i] * error_prod[i] for 
                    i in range(error_prod.shape[0])]).sum(axis = 0)
                new_covariance = new_covariance / gamma_exp.sum(axis = 0)

                covariances[i][m] = new_covariance + (r * np.identity(
                        new_covariance.shape[0]))

        return covariances.copy()

    def update_component_weights(self, gamma, gamma_by_component):
        """ Return updated component weights for current hmm parameters.

        Arguments:
            gamma: output of `_gamma`
            gamma_by_component: output of `_gamma_by_component`

        Returns:
            Array of updated component weights matrices.
        """
        log_weights = logsumexp(
            gamma_by_component, axis=1) - logsumexp(
                gamma, axis=0).reshape(-1, 1)
        weights = np.exp(log_weights)

        return weights.copy()


class DiscreteHMMLearningAlgorithm(HMMLearningAlgorithm):
    """ Discrete model class for HMM learning algorithms """

    def __init__(self):
        self.data = None
        self.finite_data_enum = None
        self.gaussian_data = None
        self.other_data = None
        self.sufficient_statistics = []
        self.model_results = []

    def run(self, model, data, training_iterations, r = 0, method="em", use_jax=False):
        """ Base class for EM learning methods.

        Arguments:
            model: instance of DiscreteHMM
            data: dataframe with hybrid data for training.
            training_iterations: number of training iterations to carry out.
            r: (float) regularization constant, default 0.
            method: "em"
            use_jax: (bool) If True, run distributed training using Jax. 

        Returns:
            Trained instance of DiscreteHMM.  Also returns em training results.
        """
        col = list(data.columns.copy())
        col.sort()
        msg = "Check that your data columns are ordered correctly."
        assert col == list(data.columns), msg
        self.data = data

        new_model = model.model_config.to_model(
            set_random_state=model.set_random_state)
        new_model.random_state = model.random_state
        
        # Make sure initial and transition parameters are up to date.
        transition = model.log_transition.copy()
        initial = model.log_initial_state.copy()
        new_model.log_transition = transition
        new_model.log_initial_state = initial

        if len(model.finite_features) > 0:
            self.finite_data_enum = get_finite_observations_from_data_as_enum(
                model, data)

            # Make sure that finite model parameters are up to date.
            emissions = model.categorical_model.log_emission_matrix.copy()
            new_model.categorical_model.log_emission_matrix = emissions

        if len(model.continuous_features) > 0:
            if model.gaussian_mixture_model:
                self.gaussian_data = get_gaussian_observations_from_data(
                    model, data)

                # Make sure that gmm parameters are up to date.
                means = model.gaussian_mixture_model.means.copy()
                weights = model.gaussian_mixture_model.component_weights.copy()
                cov = model.gaussian_mixture_model.covariances.copy()
                new_model.gaussian_mixture_model.means = means
                new_model.gaussian_mixture_model.component_weights = weights
                new_model.gaussian_mixture_model.covariances = cov

        for _ in range(training_iterations):
            new_model, ss = self.get_updated_model_parameters(new_model, data, 
                r, use_jax=False)
            hmm_inf = new_model.load_inference_interface()
            log_prob = hmm_inf.observation_log_probability(data)
            ll = logsumexp(hmm_inf._compute_forward_probabilities(log_prob))
            self.model_results.append(new_model)
            self.sufficient_statistics.append(ss)

        return new_model

    def get_updated_model_parameters(self, model, data, r, use_jax):
        """  Update model parameters using EM.
        
        Arguments: 
            model: starting instance of DiscreteHMM
            data: dataframe with hybrid data for training.
            r: (float) regularization constant.
            use_jax: (bool) if True, use jax.
            
        Returns: 
            Updated instances of Discrete HMM.
            
        """
        # Parameterize new model.
        new_model = model.model_config.to_model(
                set_random_state=model.set_random_state)
        new_model.log_transition = model.log_transition.copy()
        new_model.log_initial_state = model.log_initial_state.copy()
        if model.categorical_model:
            emission = model.categorical_model.log_emission_matrix.copy()
            new_model.categorical_model.log_emission_matrix = emission
        if model.gaussian_mixture_model:
            means = model.gaussian_mixture_model.means.copy()
            cov = model.gaussian_mixture_model.covariances.copy()
            weights = model.gaussian_mixture_model.component_weights.copy()
            new_model.gaussian_mixture_model.means = means
            new_model.gaussian_mixture_model.covariances = cov
            new_model.gaussian_mixture_model.component_weights = weights
        
        # Compute initial expectation.
        expectation = new_model.load_inference_interface(use_jax)
        expectation.compute_sufficient_statistics(data)
        gamma = expectation.gamma
        xi = expectation.xi
        new_model.log_initial_state = gamma[0].copy()
        new_model.log_transition = logsumexp(
            xi, axis=0) - logsumexp(
                gamma[:-1], axis=0).reshape(-1, 1).copy()
        
        expectation.compute_sufficient_statistics(data)
        gamma = expectation.gamma
        gamma_by_component = expectation.gamma_by_component

        if model.categorical_model:
            new_emission = new_model.categorical_model.update_log_emission_matrix(
                gamma, self.finite_data_enum)
            new_model.log_emission_matrix = new_emission.copy()
            
            # Recompute expectation
            expectation.compute_sufficient_statistics(data)
            gamma = expectation.gamma
            gamma_by_component = expectation.gamma_by_component
        
        if model.gaussian_mixture_model:
            new_means = new_model.gaussian_mixture_model.update_means(
                self.gaussian_data, gamma_by_component)
            new_model.gaussian_mixture_model.means = new_means.copy()
            
            # Recompute expectation
            expectation.compute_sufficient_statistics(data)
            gamma = expectation.gamma
            gamma_by_component = expectation.gamma_by_component
            new_covariances = new_model.gaussian_mixture_model.update_covariances(
                self.gaussian_data, gamma_by_component,r)
            new_model.gaussian_mixture_model.covariances = new_covariances.copy()

            # Recompute expectation
            expectation.compute_sufficient_statistics(data)
            gamma = expectation.gamma
            gamma_by_component = expectation.gamma_by_component

            new_weights = new_model.gaussian_mixture_model.update_component_weights(
                gamma, gamma_by_component)
            new_model.gaussian_mixture_model.component_weights = new_weights.copy()
            
            # Recompute expectation
            ss = expectation.compute_sufficient_statistics(data)

        return new_model, ss

class DiscreteHMMInferenceResults(ABC):
    """ Abstract base class for HMM inference results """

    def __init__(self, model, use_jax):
        self.model = model
        self.log_prob = None
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
        self.log_prob = self.observation_log_probability(data)
        self.gamma = self._gamma(data)
        self.xi = self._xi(data)

        if model.gaussian_mixture_model is not None:
            self.gamma_by_component = self._gamma_by_component(data)

        return self.log_prob

    def observation_log_probability(self, data):
        """ Return log probabilities of observations and hidden states

        Arguments:
            data: dataframe of mixed data types

        Returns:
            Dataframe where entry [t,i] is the log conditional probabilites of 
            observing the emission at time t given hidden state i.
        """
        log_probability = np.zeros((data.shape[0], self.model.n_hidden_states))
        if self.model.categorical_model is not None:
            finite_data_enum = get_finite_observations_from_data_as_enum(
                self.model, data)
            log_probability += np.array(
                self.model.categorical_model.categorical_log_probability(finite_data_enum))
        if self.model.gaussian_mixture_model is not None:
            gaussian_data = get_gaussian_observations_from_data(
                self.model, data)
            log_probability += np.array(
                self.model.gaussian_mixture_model.gaussian_log_probability(
                    gaussian_data))

        return pd.DataFrame(log_probability, index=data.index)

    def predict_hidden_states(self, data):
        """ Predict most likely hidden state

        Arguments:
            data: dataframe of mixed data types

        Returns:
            Series of most likely hidden states
        """
        log_probability = self.observation_log_probability(data)
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
            self.observation_log_probability(data))
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
            seed_inference: sequence of hidden states to seed sampling 
                algorithm.  Default is None in which case hidden states are 
                seeded randomly.

        Returns:
            Most likely sequence of hidden states determined by n_iterations of 
            Gibbs sampling.
        """
        model = self.model
        initial_state_prob = np.exp(model.log_initial_state)
        emission_prob = np.exp(
            np.array(self.observation_log_probability(data)))
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

    def _compute_forward_probabilities(self, log_probability):
        """ Compute forward probabilities.

        Arguments:
            log_probability: dataframe of log probability of hidden state

        Returns:
            Array where entry [t,i] is the log probability of observations 
            o_0,...,o_t and h_t = i under the current model parameters.

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
            Array where entry [t,i] is the log probability of observations 
            o_{t+1},...,o_T given h_t = i under the current model parameters.

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
            data: dataframe of mixed data types.

        Returns:
            Array where the [t,i] entry is the log conditional probability of 
            hidden state i at time t given ALL observations and current model 
            parameters, more formally, log[ p(z_t = i | x_1,...,x_T) ]
        """
        log_prob = self.log_prob
        alpha = self._compute_forward_probabilities(log_prob)
        beta = self._compute_backward_probabilities(log_prob)
        gamma = np.asarray(alpha) + np.asarray(beta)
        gamma -= logsumexp(gamma, axis=1).reshape(-1, 1)

        assert np.all(np.exp(gamma).sum(axis = 1) - 1 < 1e-08)

        return gamma

    def _gamma_by_component(self, data):
        """Auxiliary function for EM.

        Arguments:
            data: dataframe of mixed data types.

        Returns:
            Array where the [i,t,m] entry given the log conditional probability 
            of hidden state i and gmm component m at time t given ALL 
            observations and current model parameters, more formally, 
            log[ (p(z_t = i, c_t = m | x_1,...,X_T) ].
        """
        n_hidden_states = self.model.n_hidden_states
        n_components = self.model.gaussian_mixture_model.n_gmm_components
        weights = self.model.gaussian_mixture_model.component_weights
        log_weights = np.log(
            weights,
            out=np.zeros_like(weights) + LOG_ZERO,
            where=(weights != 0)).reshape(n_hidden_states, -1, n_components)

        gaussian_data = get_gaussian_observations_from_data(self.model, data)
        log_probability_by_component = np.array(
            self.model.gaussian_mixture_model.gaussian_log_probability_by_component(
                gaussian_data))

        gamma = self._gamma(data)
        gamma_by_component = log_probability_by_component + log_weights
        gamma_by_component = gamma_by_component - logsumexp(
                                gamma_by_component, axis = 2, keepdims = True)
        gamma_by_component = gamma_by_component + gamma.transpose().reshape(
                                n_hidden_states, data.shape[0],-1)
        
        assert np.all(np.exp(gamma_by_component).sum(axis = 0
                                    ).sum(axis = 1) -1 < 1e-08)

        return gamma_by_component

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
            self.observation_log_probability(data))
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
