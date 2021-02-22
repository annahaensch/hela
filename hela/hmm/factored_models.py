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

    def __init__(self, ns_hidden_states):
        
        self.random_state = 0
        
        self.ns_hidden_states = []
        self.hidden_state_vectors = None
        self.hidden_state_vector_to_enum = {}
        self.hidden_state_enum_to_vector = {}

        self.categorical_features = []
        self.categorical_values = None
        self.categorical_vector_to_enum = {}
        self.categorical_enum_to_vector = {}

        self.gaussian_features = []
        self.gaussian_values = None

        self.model_parameter_constraints = None
        
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


        categorical_observations = {obs['name']:obs['values'] for obs in spec['observations'] if obs['type'] == 'categorical'}
        if len(categorical_observations) > 0:
            categorical_observations = {k:categorical_observations[k] for k in sorted(categorical_observations.keys())}
            values = [sorted(v) for k,v in categorical_observations.items()]
            categorical_vectors = [list(t) for t in itertools.product(*values)]
            categorical_features = [k for k,v in categorical_observations.items()]
            categorical_values = pd.DataFrame(categorical_vectors, columns = categorical_features)
        
            model_config.categorical_features = categorical_features
            model_config.categorical_values = categorical_values
            model_config.categorical_vector_to_enum = {str([t for t in np.array(categorical_values.loc[i,:])]):i for i in categorical_values.index}
            model_config.categorical_vector_to_enum = {i:[t for t in np.array(categorical_values.loc[i,:])] for i in categorical_values.index}

        continuous_features = [obs for obs in spec['observations'] if obs['type'] == 'continuous']
        gaussian_features = [obs['name'] for obs in continuous_features if obs['dist'].lower() == 'gaussian']
        model_config.gaussian_features = sorted(gaussian_features)
        model_config.gaussian_values = pd.DataFrame(columns = model_config.gaussian_features)

        model_config.model_parameter_constraints = spec['model_parameter_constraints']

        return model_config

    def to_model(self):
        """ Factored HMM specific implementation of `to_model`. """

        return FactoredHMM.from_config(self)

class FactoredHMM(HiddenMarkovModel):
    """ Model class for factored hidden Markov models """

    def __init__(self, model_config=None):
        super().__init__(model_config)
        self.random_state = None
        self.trained = False

        self.ns_hidden_states = None
        self.hidden_state_vectors = None
        self.hidden_state_vector_to_enum = None
        self.hidden_state_enum_to_vector = None

        self.categorical_features = None
        self.gaussian_features = None

        self.log_transitions = None
        self.log_initial_state = None

        self.categorical_model = None
        # TODO (isalju): incorporate gaussian mixture models
        self.gaussian_model = None

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

    # These are abstract methods from the master hmm code, they can eventually
    # be removed, but I'm keeping them here for now as stubs.
    def _load_inference_interface(self):
        """ Loads DiscreteHMM specific inference interface."""
        pass

    def _load_validation_interface(self):
        """ Loads DiscreteHMM specific validation interface.
        """
        pass

    def _load_forecasting_interface(self):
        """ Loads DiscreteHMM specific forecasting interface."""
        pass

# class GaussianMixtureModel(DiscreteHMM):

class CategoricalModel(FactoredHMM):
    def __init__(self,
                 n_hidden_states=None,
                 categorical_features=None,
                 categorical_values=None,
                 categorical_values_dict=None,
                 categorical_values_dict_inverse=None,
                 log_emission_matrix=None):
        self.n_hidden_states = n_hidden_states
        self.categorical_features = categorical_features
        self.categorical_values = categorical_values
        self.categorical_values_dict = categorical_values_dict
        self.categorical_values_dict_inverse = categorical_values_dict_inverse
        self.log_emission_matrix = log_emission_matrix

    @classmethod
    def from_config(cls, model_config):
        """ Return instantiated CategoricalModel object)
        """
        categorical_model = cls(n_hidden_states=np.prod(model_config.ns_hidden_states))
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

class GaussianModel(FactoredHMM):
    def __init__(self,
                 ns_hidden_states=None,
                 gaussian_features=None,
                 dims=None,
                 means=None,
                 covariance=None):
        self.ns_hidden_states = ns_hidden_states
        self.gaussian_features = gaussian_features
        self.dims = dims
        self.means = means
        self.covariance = covariance

    @classmethod
    def from_config(cls, model_config):
        """ Return instantiated GaussianModel object)
        """
        gaussian_model = cls(ns_hidden_states=model_config.ns_hidden_states)
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

class FactoredHMMInference(ABC):

    def __init__(self, model, data):
        self.model = model
        self.data = data


    def gibbs_sample(self,idx,system,hidden_state):
        """ Samples one timestep and fHMM system

        Arguments: 
            idx: (int) timeseries data index value. 
            system: (int) value in [0,...,len(ns_hidden_states)-1]
            hidden_state: (int) hidden state value for idx and system

        Returns: The updated hidden state for the given timeseries index and 
            fHMM system.
        """
        return None
