""" Factored hidden Markov model implementation for discrete hidden state fhmms.
"""

import itertools
from abc import ABC

import numpy as np
import pandas as pd
from dask.distributed import Client
from scipy import linalg, stats
from scipy.special import logsumexp

from .base_models import (LOG_ZERO, HiddenMarkovModel, HMMConfiguration,
                          HMMForecasting, HMMValidationMetrics)
from .graphical_models.DynamicBayesianNetwork import fhmm_model_to_graph
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
        """ Factored HMM specific implementation of `from_spec`. 

        Arguments: 
            spec: (dict) model specification.
            random_state: (int) value to set random state.

        Returns: 
            Model configuration which can be used to instantiate 
            an instance of FactoredHMM.
        """
        model_config = cls(ns_hidden_states=spec['hidden_state']['count'])
        model_config.model_type = 'FactoredHMM'
        model_config.random_state = random_state

        # Get mappings between hidden state vectors and enumerations.
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

        # Add categorical observation data to model config.
        if len(categorical_observations) > 0:
            categorical_observations = {
                k: categorical_observations[k]
                for k in sorted(categorical_observations.keys())
            }
            values = [sorted(v) for k, v in categorical_observations.items()]
            categorical_vectors = [list(t) for t in itertools.product(*values)]
            categorical_features = sorted(list(categorical_observations.keys()))
            categorical_values = pd.DataFrame(
                categorical_vectors, columns=categorical_features)

            model_config.categorical_features = categorical_features
            model_config.categorical_values = categorical_values
            model_config.categorical_vector_to_enum = {
                str(list(row)): i
                for i, row in model_config.categorical_values.iterrows()
            }
            model_config.categorical_enum_to_vector = {
                i: list(row)
                for i, row in model_config.categorical_values.iterrows()
            }

        # Add gaussian observation data to model config.
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

    def to_model(self, set_random_state=0):
        """ Factored HMM specific implementation of `to_model`. """
        random_state = np.random.RandomState(set_random_state)
        return FactoredHMM.from_config(self, random_state)


class FactoredHMM(ABC):
    """ Model class for factored hidden Markov models """

    def __init__(
            self,
            model_config,
            random_state=None,
            ns_hidden_states=None,
            hidden_state_vectors=None,
            hidden_state_vector_to_enum=None,
            hidden_state_enum_to_vector=None,
            categorical_features=None,
            gaussian_features=None,
            transition_matrix=None,
            initial_state_matrix=None,
            categorical_model=None,
            # TODO (isalju): incorporate gaussian mixture models
            gaussian_model=None,
            training_dict={},
            graph=None):

        self.model_config = model_config
        self.random_state = random_state

        self.ns_hidden_states = ns_hidden_states
        self.hidden_state_vectors = hidden_state_vectors
        self.hidden_state_vector_to_enum = hidden_state_vector_to_enum
        self.hidden_state_enum_to_vector = hidden_state_enum_to_vector

        self.categorical_features = categorical_features
        self.gaussian_features = gaussian_features

        self.transition_matrix = transition_matrix
        self.initial_state_matrix = initial_state_matrix

        self.categorical_model = categorical_model
        # TODO (isalju): incorporate gaussian mixture models
        self.gaussian_model = gaussian_model

        self.training_dict = training_dict
        # self.graph = graph

    @classmethod
    def from_config(cls, model_config, random_state):
        """ Instantiates FactoredHMM instance.

        Arguments: 
            model_config: output of `from_spec`.
            random_state: (int) set random state.

        Returns: 
            Instantiated FactoredHMM instace.
        """
        model = cls(model_config=model_config)
        model.random_state = random_state
        model.training_data = None
        model.training_dict = {"trained": False}

        # Get mappings between hidden state vectors and enumerations
        model.ns_hidden_states = model_config.ns_hidden_states
        model.hidden_state_vectors = model_config.hidden_state_vectors
        model.hidden_state_vector_to_enum = model_config.hidden_state_vector_to_enum
        model.hidden_state_enum_to_vector = model_config.hidden_state_enum_to_vector

        # Get categorical features from model_config.
        model.categorical_features = model_config.categorical_features
        if len(model.categorical_features) > 0:
            model.categorical_model = CategoricalModel.from_config(model_config)

        # Get continuous features from model_config.
        model.gaussian_features = model_config.gaussian_features
        if len(model.gaussian_features) > 0:
            model.gaussian_model = GaussianModel.from_config(model_config)

        # TODO: (AH) These matrix copies are here because model training was
        # chaning the underlying model config. This is probably easy to fix.
        model.transition_matrix = model_config.model_parameter_constraints[
            'transition_constraints'].copy()

        model.initial_state_matrix = model_config.model_parameter_constraints[
            'initial_state_constraints'].copy()

        # model.graph = fhmm_model_to_graph(model)

        return model

    def load_inference_interface(self):
        """ Returns FactoredHMMInference object

        Returns:
            Initialized FactoredHMMInference object
        """
        return FactoredHMMInference(self)

    def load_learning_interface(self):
        """ Returns FactoredHMMLearning object

        Returns:
            Initialized FactoredHMMLearning object
        """
        return FactoredHMMLearningAlgorithm(self)

    def vector_to_column_vectors(self, hidden_state_vector):
        """ Returns column vectors associated to hidden_state_vector

        Arguments: 
            hidden_state_vector: (array) hidden state vector.
        Returns:
            Array of one-hot column vectors with 0 and 1 entries.
        """
        column_list = [
            np.array([
                1 if j == hidden_state_vector[i] else 0
                for j in range(np.max(self.ns_hidden_states))
            ])
            for i in range(len(self.ns_hidden_states))
        ]

        return np.array(column_list).transpose()

    def get_update_statistics(self, gamma):
        ns_hidden_states = self.ns_hidden_states
        systems = len(ns_hidden_states)
        csum = np.concatenate(([0], np.cumsum(ns_hidden_states)))

        Gamma = np.zeros((gamma.shape[0], csum[-1], csum[-1]))
        Xi = np.zeros((len(ns_hidden_states), gamma.shape[0] - 1,
                       np.max(ns_hidden_states), np.max(ns_hidden_states)))

        for m, hidden_state in enumerate(ns_hidden_states):
            for t in range(gamma.shape[0]):
                blocks = []
                other_systems = [
                    system for system in range(systems)
                    if system != m and system > m
                ]
                padding = np.zeros((csum[m], hidden_state))
                blocks.append(padding)
                if t > 0:
                    Xi[m, t - 1, :, :] = np.tensordot(
                        gamma[t - 1, m, :, np.newaxis],
                        gamma[t, m, :, np.newaxis],
                        axes=((1, 1)))

                blocks.append(np.diag(gamma[t, m, :hidden_state]))

                for m_prime in other_systems:
                    blocks.append(
                        (gamma[t, m, :hidden_state].reshape(-1, 1) @ gamma[
                            t, m_prime, :ns_hidden_states[m_prime]].reshape(
                                1, -1)).transpose())

                Gamma[t, :, csum[m]:csum[m + 1]] = np.vstack(blocks)

        update_statistics = {
            "Gamma": Gamma,
            "Xi": Xi,
        }

        return update_statistics

    def update_model_parameters(self, data, update_statistics):
        """ Returns updated model

        Arguments: 
            data: (df) observed timeseries data
            update_statistics: (arrays) Gamma and Xi update arrays 
                typically obtained from running `gibbs_sampling`.
        """
        new_model = self.model_config.to_model()
        ns_hidden_states = self.ns_hidden_states
        csum = np.concatenate(([0], np.cumsum(ns_hidden_states)))

        Gamma = update_statistics["Gamma"]
        Xi = update_statistics["Xi"]

        # Update and verify initial state matrix.
        new_model.initial_state_matrix = np.array([
            Gamma[0].diagonal()[csum[i]:csum[i + 1]]
            for i in range(len(ns_hidden_states))
        ])

        msg = "Initial state update returns invalid array: {}.".format(
            new_model.initial_state_matrix)
        assert np.all([
            np.abs(np.sum(t) - 1) < 1e-08
            for t in new_model.initial_state_matrix
        ]), msg

        # Update and verify transition matrices.
        Xi_sum = np.sum(Xi, axis=1)
        Gamma_sum = [
            np.sum((Gamma)[:-1],
                   axis=0).diagonal()[csum[i]:csum[i + 1]].reshape(-1, 1)
            for i in range(len(ns_hidden_states))
        ]
        for m in range(len(Xi_sum)):
            new_model.transition_matrix[m][:ns_hidden_states[
                m], :ns_hidden_states[m]] = Xi_sum[m][:ns_hidden_states[
                    m], :ns_hidden_states[m]] / Gamma_sum[m]

        msg = "Transition update returns invalid array: {}".format(
            new_model.transition_matrix)
        assert np.abs(
            np.sum(new_model.transition_matrix) -
            np.sum(new_model.ns_hidden_states)) < 1e-08, msg

        # Update and verify categorical emission parameters.
        if self.categorical_model:
            new_model.categorical_model.emission_matrix = self.categorical_model.update_emission_matrix(
                data, Gamma, Xi)

            msg = "Emission update returns invalid array: {}".format(
                new_model.categorical_model.emission_matrix)
            assert np.all(
                np.abs(
                    np.sum(new_model.categorical_model.emission_matrix, axis=0)
                    - 1) < 1e-08), msg

        # Update and verify gaussian emission parameters.
        if self.gaussian_model:

            # Update means
            new_model.gaussian_model.means = self.gaussian_model.update_means(
                data, Gamma)

            # Update covariance.
            new_model.gaussian_model.covariance = self.gaussian_model.update_covariance(
                new_model.gaussian_model.means, data, Gamma)

            msg = "Covariance update is not positive definite: {}".format(
                new_model.gaussian_model.covariance)
            assert np.all(
                np.linalg.eigvals(new_model.gaussian_model.covariance) > 0), msg

        return new_model


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
                 emission_matrix=None):
        super().__init__(
            model_config=model_config,
            ns_hidden_states=ns_hidden_states,
            hidden_state_vector_to_enum=hidden_state_vector_to_enum,
            hidden_state_enum_to_vector=hidden_state_enum_to_vector)
        self.categorical_features = categorical_features
        self.categorical_values = categorical_values
        self.categorical_vector_to_enum = categorical_vector_to_enum
        self.categorical_enum_to_vector = categorical_enum_to_vector
        self.emission_matrix = emission_matrix

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

        categorical_model.emission_matrix = model_config.model_parameter_constraints[
            'emission_constraints']

        # TODO: (AH) Add option to randomly seed emission matrix.

        return categorical_model

    def emission_probabilities(self, data):
        """ Returns emission probabilities for categorical data

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
        emission = self.emission_matrix

        return pd.DataFrame(
            [emission[v] for v in flattened_observations],
            columns=[k for k in self.hidden_state_enum_to_vector.keys()],
            index=data.index)

    def update_emission_matrix(self, data, Gamma, Xi):
        """ Returns updated emission matrix 

        Arguments: 
            data: (df) dataframe of timeseries observations.
            Gamma: (array) update array typically obtained 
                from running `gibbs_sampling`.
            Xi: (array) update array typically obtained from 
                running `gibbs_sampling`.

        Returns: 
            Updated emission matrix.
        """
        csum = np.concatenate(([0], np.cumsum(self.ns_hidden_states)))
        emission_matrix = np.zeros_like(self.emission_matrix)
        for i, h_vec in self.hidden_state_enum_to_vector.items():
            Gamma_split = [[
                g.diagonal()[csum[i]:csum[i + 1]]
                for i in range(len(self.ns_hidden_states))
            ]
                           for g in Gamma]
            Gamma_at_v = [
                np.prod([g[i][h_vec[i]]
                         for i in range(len(h_vec))])
                for g in Gamma_split
            ]
            for d, cat_vec in self.categorical_enum_to_vector.items():
                data_at_d = np.all(
                    np.array(data.loc[:, self.categorical_features]) == cat_vec,
                    axis=1).astype(int)
                emission_matrix[d][i] = np.sum(
                    data_at_d * Gamma_at_v) / np.sum(Gamma_at_v)

        return emission_matrix


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
            [
                means[i].data @ column_vectors[:, i].reshape(-1, 1)
                for i in range(len(means))
            ],
            axis=0)

    def emission_probabilities(self, data):
        """ Returns emission probabilities for gaussian data

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

        # Initialize dataframe that will hold probablites for observations
        # (rows) conditioned on hidden states (columns)
        prob = pd.DataFrame(
            index=data.index, columns=[i for i in range(len(means))])
        for k, m in means.items():
            prob.loc[:, k] = stats.multivariate_normal.pdf(
                np.array(data.loc[:, self.gaussian_features]),
                m.reshape(1, -1)[0], cov)

        return prob

    def update_means(self, data, Gamma):
        """ Returns updated means.

        Arguments: 
            data: (df) dataframe of timeseries observations.
            Gamma: (array) update array typically obtained 
                from running `gibbs_sampling`.
        
        Returns: 
            Updated means for gaussian model.
        """
        csum = np.concatenate(([0], np.cumsum(self.ns_hidden_states)))
        means = np.ma.masked_array(
            np.zeros_like(self.means.data), self.means.mask,
            self.means.fill_value)

        ns_hidden_states = self.ns_hidden_states
        gauss_data = np.array(data.loc[:, self.gaussian_features])

        Gamma_sum = np.sum(
            [
                gauss_data[i].reshape(-1, 1) @ Gamma[i].diagonal().reshape(
                    1, -1) for i in range(len(Gamma))
            ],
            axis=0)
        Gamma_inv = np.linalg.pinv(np.sum(Gamma, axis=0))

        means_concat = Gamma_sum @ Gamma_inv
        for i in range(len(ns_hidden_states)):
            means[i, :, :ns_hidden_states[i]] = [
                w[csum[i]:csum[i + 1]] for w in means_concat
            ]

        return means

    def update_covariance(self, means, data, Gamma):
        """ Update covariance matrix.

        Arguments: 
            means: (masked array) updated means
            data: (df) dataframe of timeseries observations.
            Gamma: (array) update array typically obtained 
                from running `gibbs_sampling`.

        Returns:
            Covariance matrix relative to input means and data.
        """
        ns_hidden_states = self.ns_hidden_states
        csum = np.concatenate(([0], np.cumsum(ns_hidden_states)))
        gauss_data = np.array(data.loc[:, self.gaussian_features])

        new_cov = np.zeros((gauss_data.shape[1], gauss_data.shape[1]))

        for t in range(len(Gamma)):
            error = np.zeros((gauss_data.shape[1], 1))

            for m in range(len(ns_hidden_states)):
                W = means[m].data[:, :ns_hidden_states[m]]
                G = np.array([
                    Gamma[t][c][c] for c in range(csum[m], csum[m + 1])
                ]).reshape(-1, 1)

                error += W @ G

            new_cov += (gauss_data[t].reshape(-1, 1) - error) @ ((
                gauss_data[t].reshape(-1, 1) - error)).reshape(1, -1)

        new_cov = new_cov / gauss_data.shape[0]

        return new_cov


class FactoredHMMLearningAlgorithm(ABC):
    """ Abstract base class for HMM learning algorithms """

    def __init__(self, model):
        self.model = model
        self.model_results = []
        self.sufficient_statistics = []

    def run(self,
            data,
            method="gibbs",
            training_iterations=2,
            gibbs_iterations=2,
            burn_down_period=1,
            distributed=False,
            n_workers=8):
        """ Trains model using Gibbs sampling methods.

        Arguments: 
            model: FactoredHMM object
            data: dataframe of timeseries observations
            method: (str) training method to use(i.e. "gibbs").
            training_iterations: (int) number of update iterations to carry 
                out.
            gibbs_iterations: (int) number of recorded rounds of Gibbs sampling 
                to carry out for each round of training.
            burn_down_period: (int) number of iterations of Gibbs sampling to 
                carry out before gathering statistics.
            distributed: (bool) if True the training will be distributed via 
                Dask
            n_workers: (int) number of works to use for distributed Dask training

        Returns:
            Trained instance of FactoredHMM object and gathered training statistics.
        """
        if method.lower() == "gibbs":
            return self.train_model_with_gibbs_sampling(
                data, training_iterations, gibbs_iterations, burn_down_period,
                distributed, n_workers)

        if method.lower() == "structured_vi":
            return self.train_model_with_structured_vi(data,
                                                       training_iterations)

        else:
            raise NotImplementedError(
                "Other learning methods haven't been implemented yet.")

    def train_model_with_gibbs_sampling(self,
                                        data,
                                        training_iterations,
                                        gibbs_iterations,
                                        burn_down_period,
                                        distributed=False,
                                        n_workers=8):
        """ Trains model using Gibbs sampling methods.

        Arguments: 
            data: dataframe of timeseries observations
            training_iterations: (int) number of update iterations to carry 
                out.
            gibbs_iterations: (int) number of recorded rounds of Gibbs sampling 
                to carry out for each round of training.
            burn_down_period: (int) number of iterations of Gibbs sampling to 
                carry out before gathering statistics.
            distributed: (bool) if True the training will be distributed via 
                Dask
            n_workers: (int) number of works to use for distributed Dask training

        Returns:
            Trained instance of FactoredHMM object and gathered training statistics.
        """
        model = self.model

        new_model = model.model_config.to_model()
        ns_hidden_states = model.ns_hidden_states
        hidden_state_vector_df = None
        if distributed == True:
            client = Client(
                processes=True, n_workers=n_workers, threads_per_worker=1)
        for r in range(training_iterations):
            inf = new_model.load_inference_interface()
            if distributed == True:
                Gamma, Xi, hidden_state_vector_df = inf.distributed_gibbs_sampling(
                    data,
                    iterations=gibbs_iterations,
                    burn_down_period=burn_down_period,
                    gather_statistics=True,
                    hidden_state_vector_df=hidden_state_vector_df,
                    client=client,
                    n_workers=n_workers)
            else:
                Gamma, Xi, hidden_state_vector_df = inf.gibbs_sampling(
                    data,
                    iterations=gibbs_iterations,
                    burn_down_period=burn_down_period,
                    gather_statistics=True,
                    hidden_state_vector_df=hidden_state_vector_df,
                    distributed=False)

            update_statistics = {
                "Gamma": Gamma,
                "Xi": Xi,
                "hidden_state_vector_df": hidden_state_vector_df
            }

            new_model = new_model.update_model_parameters(
                data, update_statistics)

            self.sufficient_statistics.append(update_statistics)

            self.model_results.append(new_model)

        new_model.training_dict = {
            "trained": True,
            "training_method": "gibbs learning",
            "training_data": data
        }

        return new_model

    def train_model_with_structured_vi(self, data, training_iterations):

        model = self.model

        new_model = model.model_config.to_model()
        ns_hidden_states = model.ns_hidden_states

        h_t = np.random.rand(
            len(data), len(ns_hidden_states), np.max(ns_hidden_states))

        for r in range(training_iterations):
            inf = new_model.load_inference_interface()

            for i in range(5):
                gamma = inf.log_forward_backward(data, h_t)
                h_t = inf.h_t_update(gamma, data)

            update_statistics = new_model.get_update_statistics(gamma)
            new_model = new_model.update_model_parameters(
                data, update_statistics)
            self.sufficient_statistics.append(update_statistics)
            self.model_results.append(new_model)

        new_model.training_dict = {
            "trained": True,
            "training_method": "structured_vi learning",
            "training_data": data
        }

        return new_model


class FactoredHMMInference(ABC):

    def __init__(self, model):
        self.model = model

    def emission_probabilities(self, data):
        """ Returns emission probabilities for observed data
        Arguments: 
            data: dataframe of observed categorical data
        Returns: 
            Dataframe where entry [t,i] is log P(x_t | h_i) (i.e. the conditional 
            probability of the observation, x_t, given hidden state h_i at 
            time t).  Here hidden states are enumerated in the "flattened" sense.
        """
        prob = np.full((data.shape[0], np.prod(self.model.ns_hidden_states)),
                       1).astype(np.float64)

        if self.model.categorical_model:
            prob *= np.array(
                self.model.categorical_model.emission_probabilities(data))

        if self.model.gaussian_model:
            prob *= np.array(
                self.model.gaussian_model.emission_probabilities(data))

        return pd.DataFrame(
            prob,
            columns=[k for k in self.model.hidden_state_enum_to_vector.keys()],
            index=data.index)

    def probability_distribution_across_hidden_states(self,
                                                      data,
                                                      current_hidden_state,
                                                      idx,
                                                      system,
                                                      emission_probabilities,
                                                      next_hidden_state=None):
        """ Returns probability distribution across hidden states.

        Arguments:
            data: (dataframe) observations.
            current_hidden_state: (array) hidden state vector corresponding to idx.
            idx: (int) iloc in index of data
            system: (int) indicates fHMM Markov system under consideration
            emission_probabilities: (df) row t and column j correspond to the log
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
        prob = np.full(model.ns_hidden_states[system], 1).astype(np.float64)

        # Get list of eligible flattened hidden states.
        eligible_states = np.array(
            ns_hidden_states[system] * [current_hidden_state])
        val = list(range(model.ns_hidden_states[system]))
        eligible_states[val, system] = val
        eligible_states = [
            model.hidden_state_vector_to_enum[str(list(v))]
            for v in eligible_states
        ]

        # Add emission probabilities.
        prob *= np.array(
            emission_probabilities.loc[data.index[idx], eligible_states])

        # Add initial state probabilities if idx  is 0.
        if idx == 0:
            prob *= model.initial_state_matrix[system][:ns_hidden_states[
                system]]

        # Add transition probabilities untill idx corresponds to the last observed data.
        if idx < data.shape[0] - 1:
            prob *= np.array(
                model.transition_matrix[system][:, next_hidden_state[
                    system]])[:ns_hidden_states[system]]

        return prob

    def gibbs_sampling(self,
                       data,
                       iterations,
                       burn_down_period=10,
                       gather_statistics=False,
                       hidden_state_vector_df=None,
                       distributed=False):
        """ Samples hidden state sequence for given data

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
            distributed: (bool) if True the training will return unnormalized
                 statistics.

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

        Gamma = None
        Xi = None
        full_sample = np.empty((iterations, data.shape[0],
                                len(model.ns_hidden_states)))

        for r in range(iterations + burn_down_period):

            # Carry out sampling
            sample_times = np.random.choice(
                [i for i in range(data.shape[0])], data.shape[0], replace=False)
            sample_systems = np.random.choice(
                [i for i in range(len(model.ns_hidden_states))],
                len(self.model.ns_hidden_states),
                replace=False)
            sample_parameter = np.random.uniform(0, 1, data.shape[0])

            emission = self.emission_probabilities(data)

            csum = np.concatenate(([0], np.cumsum(model.ns_hidden_states)))

            for t in sample_times:
                h_current = (hidden_state_vector_df.iloc[t, :]).to_list()
                h_next = None

                if t < data.shape[0] - 1:
                    h_next = np.array(hidden_state_vector_df.iloc[t + 1, :])

                for m in sample_systems:

                    updated_state_prob = self.probability_distribution_across_hidden_states(
                        data,
                        current_hidden_state=h_current,
                        idx=t,
                        system=m,
                        emission_probabilities=emission,
                        next_hidden_state=h_next)

                    hidden_state_vector_df.iloc[t, m] = _sample(
                        updated_state_prob, sample_parameter[t])

            # Gather statistics
            if r >= burn_down_period:

                full_sample[
                    r - burn_down_period, :, :] = hidden_state_vector_df.values

                if gather_statistics == True:
                    Gamma, Xi = self.gather_statistics(hidden_state_vector_df,
                                                       Gamma, Xi)

        if distributed == True:
            return Gamma, Xi, full_sample

        # Compute mode of full sample
        if iterations > 0:
            for i in range(len(model.ns_hidden_states)):
                hidden_state_vector_df.iloc[:, i] = stats.mode(
                    full_sample[:, :, i].transpose(), axis=1).mode.astype(int)

        # Normalize gathered statistics
        if gather_statistics == True:
            Gamma = Gamma / iterations
            Xi = Xi / iterations

        return Gamma, Xi, hidden_state_vector_df

    def distributed_gibbs_sampling(self,
                                   data,
                                   iterations,
                                   burn_down_period=10,
                                   gather_statistics=False,
                                   hidden_state_vector_df=None,
                                   client=None,
                                   n_workers=9):
        """ Samples hidden state sequence for given data in a distributed way using Dask

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
            client: Dask client which connects to Dask cluster.  If "None", one is
                initialized
            n_workers: (int) number of works to use for distributed Dask training

        Returns: The arrays Gamma and Xi containing statistics, and an updated hidden 
            state vector
        """

        model = self.model

        # Carry out initial burn down period
        Gamma, Xi, hidden_state_vector_df = self.gibbs_sampling(
            data,
            iterations=0,
            burn_down_period=burn_down_period,
            gather_statistics=False,
            hidden_state_vector_df=hidden_state_vector_df,
            distributed=False)
        if client is None:
            client = Client(
                processes=True, n_workers=n_workers, threads_per_worker=1)
        # Initialize workers
        local_iterations = [
            iterations // n_workers
            if i >= (iterations % n_workers) else (iterations // n_workers) + 1
            for i in range(n_workers)
        ]

        partition_labels = list(client.scheduler_info()["workers"].keys())
        partitions = {
            partition_label: (data, hidden_state_vector_df, local_iterations[i])
            for i, partition_label in enumerate(partition_labels)
        }

        scattered = client.scatter(list(partitions.values()))

        # Sample and gather statistics
        partition_states = {
            partition: client.submit(_distributed_gibbs_statistics, self, state,
                                     gather_statistics)
            for partition, state in zip(partitions.keys(), scattered)
        }

        update_statistics = client.gather(
            [state for partition, state in partition_states.items()])

        # Compute mode of full sample
        full_sample = np.concatenate(
            [update_statistics[i][2] for i in range(len(update_statistics))],
            axis=0)
        for i in range(len(model.ns_hidden_states)):
            hidden_state_vector_df.iloc[:, i] = stats.mode(
                full_sample[:, :, i].transpose(), axis=1).mode.astype(int)

        if gather_statistics == True:
            Gamma = np.zeros_like(update_statistics[0][0])
            Xi = np.zeros_like(update_statistics[0][1])
            for local_gamma, local_xi, local_samples in update_statistics:
                Gamma += local_gamma
                Xi += local_xi

            Gamma = Gamma / iterations
            Xi = Xi / iterations

        return Gamma, Xi, hidden_state_vector_df

    def gather_statistics(self, hidden_state_vector_df, Gamma=None, Xi=None):
        """ Compiles Gamma and Xi statistics 
        
        Arguments: 
            hidden_state_df: (df) hidden state vectors
            Gamma: (array) array of dimension hidden_state_vector_df.shape[0]
                x np.sum(self.model.ns_hidden_states) x np.sum(
                self.model.ns_hidden_states)
            Xi: (masked array) array with dimensions len(self.model.ns_hidden_states)
                x hidden_state_vector_df.shape[0] x np.max(self.model.ns_hidden_states)
                x np.max(self.model.ns_hidden_states).
        
        Returns: 
            Arrays Gamma and Xi
        """
        model = self.model
        ns_hidden_states = model.ns_hidden_states
        csum = np.concatenate(([0], np.cumsum(ns_hidden_states)))
        V = np.array(hidden_state_vector_df)

        # Initialize Gamma and/or Xi if None is given.
        if Gamma is None:
            Gamma = np.zeros((hidden_state_vector_df.shape[0], csum[-1],
                              csum[-1]))
        if Xi is None:
            Xi = np.zeros((len(ns_hidden_states),
                           hidden_state_vector_df.shape[0] - 1,
                           np.max(ns_hidden_states), np.max(ns_hidden_states)))

        # Update Gamma
        Gamma_pos = np.concatenate(
            [[(i,) + t
              for t in list(
                  itertools.combinations_with_replacement(
                      [csum[i] + v[i]
                       for i in range(len(v))], 2))]
             for i, v in enumerate(V)])
        times, cols, rows = zip(*Gamma_pos)
        Gamma[times, rows, cols] += 1

        # Update Xi
        Xi_pos = np.concatenate([[(j, i) + (V[i][j], V[i + 1][j])
                                  for j in range(len(V[i]))]
                                 for i in range(len(V[:-1]))])
        systems, times, rows, cols = zip(*Xi_pos)
        Xi[systems, times, rows, cols] += 1

        return Gamma, Xi

    def log_forward_backward(self, data, h_t):
        """  
        Gets log probabilities under current model parameters

        Arguments: 
            data: (dataframe) df on which to peform inference
            h_t: (array) array of dimension T x M X N for current variational parameters
        
        Returns: 
            Arrays for gamma
        """
        time = len(data)
        model = self.model
        systems = len(model.ns_hidden_states)
        beta = np.empty((time, systems, np.max(model.ns_hidden_states)))
        alpha = np.empty((time, systems, np.max(model.ns_hidden_states)))
        gamma = np.empty((time, systems, np.max(model.ns_hidden_states)))
        initial_state = np.zeros(alpha[0].shape)
        for m in range(systems):
            initial_state[m, :model.ns_hidden_states[m]] = np.array(
                model.initial_state_matrix[m])[:model.ns_hidden_states[m]]

        log_initial_state = np.log(
            np.array(initial_state),
            out=np.zeros_like(np.array(initial_state)) + LOG_ZERO,
            where=(np.array(initial_state) != 0))

        log_transition = np.array([
            np.log(
                transition,
                out=np.zeros_like(transition) + LOG_ZERO,
                where=(np.array(transition) != 0))
            for transition in model.transition_matrix
        ])

        log_h_t = np.log(
            h_t, out=np.zeros_like(h_t) + LOG_ZERO, where=(h_t != 0))

        alpha[0][:][:] = log_h_t[0][:][:] + log_initial_state
        beta[time - 1][:][:] = np.zeros((len(model.ns_hidden_states),
                                         np.max(model.ns_hidden_states)))

        for m in range(systems):
            hidden_state = model.ns_hidden_states[m]
            # Forward probabilities
            for t in range(1, time):
                alpha_t = logsumexp(
                    (alpha[t - 1][m][:hidden_state].reshape(-1, 1) +
                     log_transition[m, :hidden_state, :hidden_state]),
                    axis=0)
                alpha[t][
                    m][:hidden_state] = log_h_t[t][m][:hidden_state] + alpha_t
            # Backward probabilities
            for t in range(time - 2, -1, -1):
                beta_t = log_h_t[t +
                                 1][m][:
                                       hidden_state] + log_transition[m, :
                                                                      hidden_state, :
                                                                      hidden_state] + beta[t
                                                                                           +
                                                                                           1][m][:
                                                                                                 hidden_state]
                beta[t][m][:hidden_state] = logsumexp(
                    beta_t[:hidden_state], axis=1)

            gamma[:, m, :hidden_state] = np.asarray(
                alpha[:, m, :hidden_state]) + np.asarray(
                    beta[:, m, :hidden_state])
            gamma[:, m, :hidden_state] = np.exp(
                gamma[:, m, :hidden_state] -
                logsumexp(gamma[:, m, :hidden_state], axis=1).reshape(-1, 1))

        return gamma

    def h_t_update(self, gamma, data):
        """  
        Updates variational parameters under the current expectations

        Arguments: 
            gamma: (array) array of dimension T x M X N
            data: dataframe of gaussian and categorical observations.
        
        Returns: 
            Array of updated variational parameters
        """
        model = self.model

        if model.gaussian_model:
            inv_cov = np.linalg.inv(model.gaussian_model.covariance)
            gauss_data = np.array(data.loc[:, model.gaussian_features])

        if model.categorical_model:
            cat_data_enum = [
                model.categorical_model.categorical_vector_to_enum[str(list(d))]
                for d in np.array(data.loc[:, model.categorical_features])
            ]

        h_t_new = np.zeros((len(data), len(model.ns_hidden_states),
                            np.max(model.ns_hidden_states)))

        systems = len(model.ns_hidden_states)

        for m in range(systems):
            hidden_state = model.ns_hidden_states[m]
            gaussian_upgrade = np.zeros_like(h_t_new[:, m, :hidden_state])
            categorical_upgrade = np.zeros_like(h_t_new[:, m, :hidden_state])

            if len(model.gaussian_features) > 0:
                mean = model.gaussian_model.means[m].data[:, :hidden_state]
                delta = (mean.T @ inv_cov @ mean).diagonal()
                other_systems = [i for i in range(systems) if i != m]
                error = np.zeros(gauss_data.shape)
                for system in other_systems:
                    error += gamma[:, system, :model.ns_hidden_states[
                        system]] @ (model.gaussian_model.means[system]
                                    .data[:, :model.ns_hidden_states[system]].T)

                residual_error = gauss_data - error

                gaussian_upgrade = (residual_error @ inv_cov @ mean) - (
                    delta / 2)

            if len(model.categorical_features) > 0:
                hidden_states_enum = [[
                    k
                    for k, v in model.hidden_state_enum_to_vector.items()
                    if v[m] == i
                ]
                                      for i in range(hidden_state)]
                categorical_upgrade = np.array([[
                    np.sum(model.categorical_model.emission_matrix[i, j])
                    for j in hidden_states_enum
                ]
                                                for i in cat_data_enum])

            h_t_new[:, m, :hidden_state] = np.exp(
                gaussian_upgrade + categorical_upgrade)

        return h_t_new

    def predict_hidden_states_viterbi(self, data):
        """  
        Predicts the most likely series of hidden states using viterbi algorithm

        Arguments: 
            data: (dataframe) df on which to perform inference

        Returns: 
            DataFrame of most likely series of hidden states
        """
        time = len(data)
        model = self.model
        systems = len(model.ns_hidden_states)
        n = np.max(model.ns_hidden_states)

        # randomly initialize variational parameters
        h_t = np.random.rand(
            len(data), len(self.model.ns_hidden_states),
            np.max(self.model.ns_hidden_states))

        # Log forward-backward
        for i in range(5):
            gamma = self.log_forward_backward(data, h_t)
            h_t = self.h_t_update(gamma, data)

        viterbi_matrix = np.zeros((time, systems, n))
        backpoint_matrix = np.zeros((time, systems, n))
        best_path = np.zeros((systems, time))
        viterbi_matrix[0][:][:] = h_t[0][:][:] * model.initial_state_matrix
        for m in range(systems):
            for t in range(1, time):
                step = h_t[t][m][:, np.newaxis] * model.transition_matrix[
                    m] * viterbi_matrix[t - 1][m][:]
                viterbi_matrix[t][m][:] = np.max(step, axis=1)
                backpoint_matrix[t][m][:] = np.argmax(step, axis=1)

        for m in range(systems):
            best_path[m][time - 1] = np.argmax(viterbi_matrix[time - 1][m][:])
            for t in range(time - 2, -1, -1):
                forward_state = int(best_path[m][t + 1])
                best_path[m][t] = backpoint_matrix[t + 1][m][forward_state]

        return pd.DataFrame(best_path.T, index=data.index)

    def predict_hidden_states_log_viterbi(self, data):
        """  
        Predicts the most likely series of hidden states using viterbi algorithm

        Arguments: 
            data: (dataframe) df on which to perform inference

        Returns: 
            DataFrame of most likely series of hidden states
        """
        time = len(data)
        model = self.model
        systems = len(model.ns_hidden_states)
        n = np.max(model.ns_hidden_states)

        # randomly initialize variational parameters
        h_t = np.random.rand(
            len(data), len(self.model.ns_hidden_states),
            np.max(self.model.ns_hidden_states))

        # Log forward-backward
        for i in range(5):
            gamma = self.log_forward_backward(data, h_t)
            h_t = self.h_t_update(gamma, data)

        viterbi_matrix = np.zeros((time, systems, n))
        backpoint_matrix = np.zeros((time, systems, n))
        best_path = np.zeros((systems, time))
        initial_state = np.zeros(h_t[0].shape)
        for m in range(systems):
            initial_state[m, :model.ns_hidden_states[m]] = np.array(
                model.initial_state_matrix[m])[:model.ns_hidden_states[m]]

        log_initial_state = np.log(
            np.array(initial_state),
            out=np.zeros_like(np.array(initial_state)) + LOG_ZERO,
            where=(np.array(initial_state) != 0))

        log_transition = np.array([
            np.log(
                transition,
                out=np.zeros_like(transition) + LOG_ZERO,
                where=(np.array(transition) != 0))
            for transition in model.transition_matrix
        ])

        log_h_t = np.log(
            h_t, out=np.zeros_like(h_t) + LOG_ZERO, where=(h_t != 0))

        viterbi_matrix[0][:][:] = log_h_t[0][:][:] + log_initial_state

        for m in range(systems):
            hidden_state = model.ns_hidden_states[m]
            for t in range(1, time):
                step = log_h_t[t][m][:hidden_state, np.
                                     newaxis] + model.transition_matrix[m, :
                                                                        hidden_state, :
                                                                        hidden_state] + viterbi_matrix[t
                                                                                                       -
                                                                                                       1][m][:
                                                                                                             hidden_state]
                viterbi_matrix[t][m][:hidden_state] = np.max(step, axis=1)
                backpoint_matrix[t][m][:hidden_state] = np.argmax(step, axis=1)

        for m in range(systems):
            hidden_state = model.ns_hidden_states[m]
            best_path[m][time - 1] = np.argmax(
                viterbi_matrix[time - 1][m][:hidden_state])
            for t in range(time - 2, -1, -1):
                forward_state = int(best_path[m][t + 1])
                best_path[m][t] = backpoint_matrix[t + 1][m][forward_state]

        return pd.DataFrame(best_path.T, index=data.index)


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


def _distributed_gibbs_statistics(inference, state, gather_statistics):
    """Update the state of a worker with update statistics and hidden states
        found from gibbs_sampling.

    Args:
        inference: FactoredHMMInference, global copy
        state (tuple): a tuple of (data, hidden_state_vector_df, iterations) where
        data and hidden_state_vector_df are the same across workers. Iterations
        are the number of iterations the worker should sample.
        gather_statistics: (bool) indicates whether to gather statistics while 
                iterating.
    Returns:
        local_gamma, local_xi, local_full_sample where local_gamma and local_xi are unnormalized.
    """
    data, hidden_state_vector_df, iterations = state

    local_gamma, local_xi, local_full_sample = inference.gibbs_sampling(
        data,
        iterations,
        burn_down_period=0,
        gather_statistics=gather_statistics,
        hidden_state_vector_df=hidden_state_vector_df,
        distributed=True)
    return local_gamma, local_xi, local_full_sample


def _factored_hmm_to_discrete_hmm(model):
    """ Returns discrete hmm training spec for flattened HMM
        obtained from a factored HMM model object.
    """
    n_hidden_states = np.prod(model.ns_hidden_states)
    training_spec = {
        'hidden_state': {
            'type': 'finite',
            'count': n_hidden_states
        }
    }

    model_parameter_constraints = {}
    observations = []

    # Assemble categorical features.
    if len(model.categorical_features) > 0:
        categorical_values = model.categorical_model.categorical_values
        for c in sorted(categorical_values.columns):
            observations.append({
                'name': c,
                'type': 'finite',
                'values': sorted(categorical_values[c].unique())
            })
        model_parameter_constraints[
            'emission_constraints'] = model.categorical_model.emission_matrix

    # Assemble gaussian features.
    if len(model.gaussian_features) > 0:
        for g in model.gaussian_features:
            observations.append({
                'name': g,
                'type': 'continuous',
                'dist': 'gaussian',
                'dims': 1
            })
        gmm_parameter_constraints = {'n_gmm_components': 1}
        gmm_parameter_constraints['means'] = [
            model.gaussian_model.mean_for_hidden_state_vector(v).reshape(1, -1)
            for v in model.hidden_state_vectors
        ]
        gmm_parameter_constraints['covariances'] = n_hidden_states * [[
            model.gaussian_model.covariance
        ]]
        gmm_parameter_constraints['component_weights'] = np.array(
            n_hidden_states * [[[1]]])

        model_parameter_constraints[
            'gmm_parameter_constraints'] = gmm_parameter_constraints

    transition_matrix = np.zeros((n_hidden_states, n_hidden_states))

    enum_pairs = list(
        itertools.permutations(model.hidden_state_enum_to_vector.keys(), 2)) + [
            (k, k) for k in model.hidden_state_enum_to_vector.keys()
        ]

    trans_pairs = [
        (v, w,
         np.prod([
             model.transition_matrix[i][model.hidden_state_enum_to_vector[v][i]]
             [model.hidden_state_enum_to_vector[w][i]]
             for i in range(len(model.ns_hidden_states))
         ]))
        for v, w in enum_pairs
    ]

    old, new, transition_prob = zip(*trans_pairs)

    transition_matrix[old, new] = transition_prob

    model_parameter_constraints['transition_constraints'] = transition_matrix

    training_spec['model_parameter_constraints'] = model_parameter_constraints
    training_spec['observations'] = observations

    return training_spec
