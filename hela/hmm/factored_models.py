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
                random_state = 0,
                hidden_state_vectors = None,
                hidden_state_vector_to_enum = {},
                hidden_state_enum_to_vector = {},
                categorical_features = [],
                categorical_values = None,
                categorical_vector_to_enum = {},
                categorical_enum_to_vector = {},
                gaussian_features = [],
                gaussian_values = None,
                model_parameter_constraints = None):
        
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
    def from_spec(cls, spec, random_state = 0):
        """ Factored HMM specific implementation of `from_spec`. """
        model_config = cls(ns_hidden_states = spec['hidden_state']['count'])
        model_config.model_type = 'FactoredHMM'
        model_config.random_state = random_state

        # Get mappings between hidden state vectors and enumerations,
        hidden_state_values = [[t for t in range(i)] for i in model_config.ns_hidden_states]
        hidden_state_vectors = [list(t) for t in itertools.product(*hidden_state_values)]
        model_config.hidden_state_vectors = hidden_state_vectors
        model_config.hidden_state_vector_to_enum = {str(hidden_state_vectors[i]):i for i in range(len(hidden_state_vectors))}
        model_config.hidden_state_enum_to_vector = {i:hidden_state_vectors[i] for i in range(len(hidden_state_vectors))}


        categorical_observations = {obs['name']:obs['values'] for obs in spec['observations'] if obs['type'] == 'finite'}
        if len(categorical_observations) > 0:
            categorical_observations = {k:categorical_observations[k] for k in sorted(categorical_observations.keys())}
            values = [sorted(v) for k,v in categorical_observations.items()]
            categorical_vectors = [list(t) for t in itertools.product(*values)]
            categorical_features = [k for k,v in categorical_observations.items()]
            categorical_values = pd.DataFrame(categorical_vectors, columns = categorical_features)
        
            model_config.categorical_features = categorical_features
            model_config.categorical_values = categorical_values
            model_config.categorical_vector_to_enum = {str([t for t in np.array(categorical_values.loc[i,:])]):i for i in categorical_values.index}
            model_config.categorical_enum_to_vector = {i:[t for t in np.array(categorical_values.loc[i,:])] for i in categorical_values.index}

        continuous_features = [obs for obs in spec['observations'] if obs['type'] == 'continuous']
        gaussian_features = [obs['name'] for obs in continuous_features if obs['dist'].lower() == 'gaussian']
        model_config.gaussian_features = sorted(gaussian_features)
        model_config.gaussian_values = pd.DataFrame(columns = model_config.gaussian_features)

        model_config.model_parameter_constraints = spec['model_parameter_constraints']

        return model_config

    def to_model(self):
        """ Factored HMM specific implementation of `to_model`. """

        return FactoredHMM.from_config(self)

class FactoredHMM(ABC):
    """ Model class for factored hidden Markov models """

    def __init__(self, 
                model_config,
                random_state = None,
                trained = False,
                ns_hidden_states = None,
                hidden_state_vectors = None,
                hidden_state_vector_to_enum = None,
                hidden_state_enum_to_vector = None,
                categorical_features = None,
                gaussian_features = None,
                log_transition = None,
                log_initial_state = None,
                categorical_model = None,
                # TODO (isalju): incorporate gaussian mixture models
                gaussian_model = None):

        self.random_state = random_state
        self.trained = trained

        self.ns_hidden_states = ns_hidden_states
        self.hidden_state_vectors = hidden_state_vectors
        self.hidden_state_vector_to_enum = hidden_state_vector_to_enum
        self.hidden_state_enum_to_vector = hidden_state_enum_to_vector

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


        # Get categorical features from model_config.
        model.categorical_features = model_config.categorical_features
        if len(model.categorical_features) > 0:
                model.categorical_model = CategoricalModel.from_config(
                model_config)

        # Get continuous features from model_config.
        model.gaussian_features = model_config.gaussian_features
        if len(model.gaussian_features) > 0:
            model.gaussian_model = GaussianModel.from_config(
                model_config)

        transition = model_config.model_parameter_constraints['transition_constraints']
        trans_mask = transition.mask
        zero_mask = transition == 0
        temp_masked_trans = np.ma.masked_array(np.where(transition!=0,transition,1),trans_mask).filled(fill_value=1)
        log_transition = np.log(temp_masked_trans)
        log_transition[zero_mask] = LOG_ZERO
        model.log_transition = np.ma.masked_array(log_transition,trans_mask)

        initial_state = model_config.model_parameter_constraints['initial_state_constraints']
        initial_state_mask = initial_state.mask
        zero_mask = initial_state == 0
        temp_masked_initial_state = np.ma.masked_array(np.where(initial_state!=0,initial_state,1),initial_state_mask).filled(fill_value=1)
        log_initial_state = np.log(temp_masked_initial_state)
        log_initial_state[zero_mask] = LOG_ZERO
        model.log_initial_state = np.ma.masked_array(log_initial_state,initial_state_mask)

        # TODO: (AH) add option to randomly seed transition and initial state.

        return model

    def to_inference_interface(self,data):
        """ Returns FactoredHMMInference object

        Arguments: 
            data: timeseries data for which to perform inference

        Returns:
            Initialized FactoredHMMInference object
        """
        return FactoredHMMInference(self,data)

    def hidden_state_vectors_matching_away_from_m(self,m,vector):
        """ Returns a list of vectors

        Arguments: 
            m: (int) index indicating one of the fHMM Markov systems.
            vector: (array) hidden state vector

        Returns: 
            List of all hidden state vectors agreeing with vector in all
            but the mth component.
        """
        
        mask = [1 if i == m else 0 for i in range(len(self.ns_hidden_states))]
        masked_vec = np.ma.masked_array(vector,mask)
        
        return [v for v in self.hidden_state_vectors if np.all(np.ma.masked_array(v,mask) == masked_vec)]



class CategoricalModel(FactoredHMM):
    def __init__(self,
                 hidden_state_vector_to_enum,
                 hidden_state_enum_to_vector,
                 n_hidden_states=None,
                 categorical_features=None,
                 categorical_values=None,
                 categorical_vector_to_enum=None,
                 categorical_enum_to_vector =None,
                 log_emission_matrix=None):
        self.hidden_state_vector_to_enum = hidden_state_vector_to_enum
        self.hidden_state_enum_to_vector = hidden_state_enum_to_vector
        self.categorical_features = categorical_features
        self.categorical_values = categorical_values
        self.categorical_vector_to_enum  = categorical_vector_to_enum 
        self.categorical_enum_to_vector = categorical_enum_to_vector
        self.log_emission_matrix = log_emission_matrix

    @classmethod
    def from_config(cls, model_config):
        """ Return instantiated CategoricalModel object)
        """
        hidden_state_vector_to_enum = model_config.hidden_state_vector_to_enum
        hidden_state_enum_to_vector = model_config.hidden_state_enum_to_vector

        categorical_model = cls(hidden_state_vector_to_enum = hidden_state_vector_to_enum,
            hidden_state_enum_to_vector = hidden_state_enum_to_vector)
        categorical_model.categorical_features = model_config.categorical_features
        categorical_model.categorical_values = model_config.categorical_values
        categorical_model.categorical_vector_to_enum = model_config.categorical_vector_to_enum
        categorical_model.categorical_enum_to_vector = model_config.categorical_enum_to_vector
        
        # Get log emission matrix, masking to prevent log(0) errors.
        emission_matrix = model_config.model_parameter_constraints['emission_constraints']
        zero_mask = emission_matrix == 0
        log_emission_matrix = np.where(emission_matrix!=0,emission_matrix,1)
        log_emission_matrix = np.log(log_emission_matrix)
        log_emission_matrix[zero_mask] = LOG_ZERO
        categorical_model.log_emission_matrix = log_emission_matrix

        # TODO: (AH) Add option to randomly seed emission matrix.
        
        return categorical_model

    def get_emission_log_probabilities(self,data):
        """ Returns emission log_probabilities for categorical data

        Arguments: 
            data: dataframe of observed categorical data

        Returns: 
            Dataframe where entry [t,i] is log P(x_t | h_i) (i.e. the conditional 
            probability of the categorical emission, x_t, observed at time t, 
            given hidden state h_i at time t).  Here hidden states are 
            enumerated in the "flattened" sense.  
        """
        flattened_observations = [self.categorical_vector_to_enum[str(list(v))] for v in np.array(data.loc[:,self.categorical_features])]
        log_emission = self.log_emission_matrix

        return pd.DataFrame([log_emission[v] for v in flattened_observations], 
                            columns = [k for k in self.hidden_state_enum_to_vector.keys()],
                            index = data.index)


class GaussianModel(FactoredHMM):
    def __init__(self,
                 ns_hidden_states,
                 hidden_state_vector_to_enum = None,
                 hidden_state_enum_to_vector = None,
                 gaussian_features=None,
                 dims=None,
                 means=None,
                 covariance=None,
                 model_config = None):
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
        gaussian_model = cls(ns_hidden_states=model_config.ns_hidden_states)
        gaussian_model.hidden_state_vector_to_enum = model_config.hidden_state_vector_to_enum
        gaussian_model.hidden_state_enum_to_vector = model_config.hidden_state_enum_to_vector

        gaussian_features= model_config.gaussian_features
        gaussian_values = model_config.gaussian_values
        gaussian_params = model_config.model_parameter_constraints['gaussian_parameter_constraints']

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
        column_vectors = [np.array([1 if j == hidden_state_vector[i] else 0 for j in range(np.max(ns_hidden_states))]).reshape(-1,1) for i in range(len(ns_hidden_states))]

        return np.sum(np.array([means[m].data @ column_vectors[m] for m in range(len(ns_hidden_states))]), axis = 0)


    def get_emission_log_probabilities(self, data):
        """ Returns emission log_probabilities for categorical data

        Arguments: 
            data: dataframe of observed categorical data

        Returns: 
            Dataframe where entry [t,i] is log P(x_t | h_i) (i.e. the conditional 
            probability of the gaussian emission, x_t, observed at time t, 
            given hidden state h_i at time t).  Here hidden states are 
            enumerated in the "flattened" sense.
        """
        ns_hidden_states = self.ns_hidden_states

        vectors = [v for k,v in self.hidden_state_enum_to_vector.items()]
        means = {k : self.mean_for_hidden_state_vector(v) for k,v in self.hidden_state_enum_to_vector.items()}
        cov = self.covariance
        stats.multivariate_normal.logpdf(np.array(data.loc[:,self.gaussian_features]),[0,0],[[1,0],[0,1]])
        log_prob = pd.DataFrame(index = data.index, columns = [i for i in range(len(means))])
        for k,m in means.items():
            log_prob.loc[:,k] = stats.multivariate_normal.logpdf(np.array(
                data.loc[:,self.gaussian_features]),m.reshape(1,-1)[0],cov)
        
        return log_prob



class FactoredHMMInference(ABC):

    def __init__(self, model, data):
        self.model = model
        self.data = data

    def get_emission_log_probabilities(self, data):
        """ Returns emission log_probabilities for observed data

        Arguments: 
            data: dataframe of observed categorical data

        Returns: 
            Dataframe where entry [t,i] is log P(x_t | h_i) (i.e. the conditional 
            log probability of the observation, x_t, given hidden state h_i at 
            time t).  Here hidden states are enumerated in the "flattened" sense.
        """
        log_prob = np.zeros((data.shape[0], np.prod(self.model.ns_hidden_states)))
        if self.model.categorical_model:
            log_prob += np.array(self.model.categorical_model.get_emission_log_probabilities(data))
        if self.model.gaussian_model:
            log_prob += np.array(self.model.gaussian_model.get_emission_log_probabilities(data))
        return pd.DataFrame(log_prob, 
                            columns = [k for k in self.model.hidden_state_enum_to_vector.keys()],
                            index = data.index)


    def gibbs_sample(self, data, iterations, hidden_state_vector_df = None):
        """ Samples one timestep and fHMM system

        Arguments: 
            data: (dataframe) observed timeseries data.
            iterations: (int) number of rounds of sampling to carry out
            hidden_state_vector_df: (dataframe) timeseries of hidden state vectors
                with the same index as "data".  If default "None" is given, then
                this dataframe will be seeded randomly.

        Returns: The updated hidden state for the given timeseries index and 
            fHMM system.
        """
        model = self.model
        ns_hidden_states = model.ns_hidden_states
        log_initial_state = model.log_initial_state
        log_transition = model.log_transition
        
        # Seed hidden state vector dataframe if none is given.
        if hidden_state_vector_df is None:
            hidden_state_enum_df = np.random.choice(list(self.model.hidden_state_enum_to_vector.keys()),data.shape[0])
            hidden_state_vector_df = pd.DataFrame([self.model.hidden_state_enum_to_vector[v] for v in hidden_state_enum_df],
                                                  index = data.index,
                                                  columns = [i for i in range(len(self.model.ns_hidden_states))])
        # Get emission probability for categorical and gaussian emissions.
        emission_log_prob_df = self.get_emission_log_probabilities(data)
        
        for r in range(iterations):
            sample_times = np.random.choice([i for i in range(data.shape[0])],
                                             data.shape[0], 
                                             replace = False)
            sample_systems = np.random.choice([i for i in range(len(ns_hidden_states))],
                                               len(ns_hidden_states),replace = False)
            sample_parameter = np.random.uniform(0, 1, data.shape[0])
            
            for t in sample_times:
                h_current = np.array(hidden_state_vector_df.iloc[t,:])  
                for m in sample_systems:
                    log_prob = np.zeros(ns_hidden_states[m])
                    
                    # Add emission log probabilities.
                    eligible_vectors = self.model.hidden_state_vectors_matching_away_from_m(m,h_current)
                    eligible_vectors.sort()
                    eligible_states = [self.model.hidden_state_vector_to_enum[str(v)] for v in eligible_vectors]            
                    log_prob += np.array(emission_log_prob_df.iloc[t,eligible_states])
                    
                    # Add initial state log probabilities
                    if t == 0:
                        log_prob += log_initial_state[m][:ns_hidden_states[m]]
                    
                    # Add transition probabilities
                    if t < data.shape[0] -1:
                        h_next = np.array(hidden_state_vector_df.iloc[t+1,:])
                        log_prob += np.array(log_transition[m][:,h_next[m]])[:ns_hidden_states[m]]


                    updated_state_prob = np.exp(log_prob)
                    if np.sum(updated_state_prob) == 0:
                        updated_state_prob = np.full(len(updated_state_prob), 1)
                    updated_state_prob = updated_state_prob / np.sum(
                            updated_state_prob)
                    cumulative_prob = np.cumsum(updated_state_prob)
                    updated_state = np.where(
                            cumulative_prob >= sample_parameter[t])[0][0]
                    hidden_state_vector_df.iloc[t,m] = updated_state
        
        return hidden_state_vector_df