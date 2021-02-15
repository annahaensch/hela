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


class DiscreteFHMMConfiguration(HMMConfiguration):
    """ Intilialize FHMM configuration from specification dictionary. """

    def __init__(self, hidden_state_type=None):
        super().__init__(hidden_state_type)
        self.ns_hidden_states = None

    def _from_spec(self, spec):
        """ Discrete FHMM specific implementation of `from_spec`. """
        # self.ns_hidden_states = [states['count'] for states in spec['hidden_states']]
        self.ns_hidden_states = spec['hidden_state']['count']
        self.n_systems = spec['n_systems']
        self.model_type = 'DiscreteFHMM'
        return self

    def _to_model(self, random_state):
        """ Discrete FHMM specific implementation of `to_model`. """
        #TODO (isalju): implement FHMM model
        raise NotImplementedError(
        "FHMM model not yet implemented")
        # return DiscreteFHMM.from_config(self, random_state)
        return None

class DiscreteFHMM(HiddenMarkovModel):
    """ Model class for dicrete hidden Markov models """

    def __init__(self, model_config=None):
        super().__init__(model_config)
        self.random_state = None
        self.trained = False

        self.ns_hidden_states = None
        self.seed_parameters = {}

        self.finite_features = None
        self.finite_values = None
        self.continuous_features = None
        self.continuous_values = None

        self.log_transitions = None
        self.log_initial_state = None

        self.categorical_model = None
        # TODO (isalju): incorporate gaussian mixture models
        self.gaussian_mixture_model = None

    @classmethod
    def from_config(cls, model_config, random_state):
        model = cls(model_config=model_config)
        model.ns_hidden_states = model_config.ns_hidden_states
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
            i for i in model.continuous_features
            if model.continuous_values.loc['distribution', i].lower() ==
            'gaussian'
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