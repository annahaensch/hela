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


class FactoredHMMConfiguration(HMMConfiguration):
    """ Intilialize FHMM configuration from specification dictionary. """

    def __init__(self, hidden_state_type=None):
        super().__init__(hidden_state_type)
        self.ns_hidden_states = None

    def _from_spec(self, spec):
        """ Factored HMM specific implementation of `from_spec`. """
        self.ns_hidden_states = spec['hidden_state']['count']
        self.n_systems = spec['n_systems']
        self.model_type = 'FactoredHMM'
        return self

    def _to_model(self, random_state):
        """ Factored HMM specific implementation of `to_model`. """

        return FactoredHMM.from_config(self, random_state)

class FactoredHMM(HiddenMarkovModel):
    """ Model class for factored hidden Markov models """

    def __init__(self, model_config=None):
        super().__init__(model_config)
        self.random_state = None
        self.trained = False

        self.ns_hidden_states = None
        self.n_systems = None
        self.seed_parameters = {}

        self.finite_features = None
        self.finite_values = None
        self.continuous_features = None
        self.continuous_values = None

        self.log_transitions = None
        self.log_initial_state = None

        self.categorical_model = None
        # TODO (isalju): incorporate gaussian mixture models
        self.gaussian_model = None

    @classmethod
    def from_config(cls, model_config, random_state):
        model = cls(model_config=model_config)
        model.ns_hidden_states = model_config.ns_hidden_states
        model.random_state = random_state
        
        # Get mappings between hidden state vectors and enumerations,
        hidden_state_values = [[t for t in range(i)] for i in model.ns_hidden_states]
        hidden_state_vectors = [list(t) for t in itertools.product(*hidden_state_values)]
        model.hidden_state_vector_to_enum = {str(hidden_state_vectors[i]):i for i in range(len(hidden_state_vectors))}
        model.hidden_state_enum_to_vector = {i:hidden_state_vectors[i] for i in range(len(hidden_state_vectors))}


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
            i for i in model.continuous_features
            if model.continuous_values.loc['distribution', i].lower() ==
            'gaussian'
        ]
        if len(gaussian_features) > 0:
            model.gaussian_model = GaussianModel.from_config(
                model_config, random_state)

        # Check that there are no remaining features.
        if len(model.continuous_features) > 0:
            non_gaussian_features = [
                i for i in model.continuous_values.columns
                if model.continuous_values.loc['distribution', i].lower() !=
                'gaussian']
            non_gaussian_distributions = [
                model.continuous_values.loc['distribution', i]
                for i in non_gaussian_features
            ]
            if len(non_gaussian_features) > 0:
                raise NotImplementedError(
                    "Curent FactoredHMM implementation is "
                    "not equipped to deal with continuous variables with "
                    "distribution type: {}".format(
                        " ,".join(non_gaussian_distributions)))

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
        categorical_model = cls(n_hidden_states=np.prod(model_config.ns_hidden_states))
        categorical_model.random_state = random_state
        categorical_model.finite_features = model_config.finite_features
        categorical_model.finite_values = model_config.finite_values
        categorical_model.finite_values_dict = model_config.finite_values_dict
        categorical_model.finite_values_dict_inverse = model_config.finite_values_dict_inverse
        
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
    def from_config(cls, model_config, random_state):
        """ Return instantiated GaussianModel object)
        """
        gaussian_model = cls(ns_hidden_states=model_config.ns_hidden_states)
        continuous_values = model_config.continuous_values
        gaussian_params = model_config.model_parameter_constraints['gaussian_parameter_constraints']

        # Gather gaussian features and values.
        gaussian_features = [
            c for c in continuous_values.columns
            if continuous_values.loc['distribution', c] == 'gaussian'
        ]
        gaussian_model.gaussian_features = gaussian_features
        gaussian_model.dims = len(gaussian_features)
        gaussian_model.means = gaussian_params['means']
        gaussian_model.covariance = gaussian_params['covariance']

        # TODO: (AH) Add option to randomly seed gaussian parameters.

        return gaussian_model