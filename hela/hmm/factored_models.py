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
        self.ns_hidden_states = spec['hidden_state']['count']
        self.n_systems = spec['n_systems']
        self.model_type = 'DiscreteFHMM'
        return self

    def _to_model(self, random_state):
        """ Discrete FHMM specific implementation of `to_model`. """

        return DiscreteFHMM.from_config(self, random_state)

class DiscreteFHMM(HiddenMarkovModel):
    """ Model class for dicrete hidden Markov models """

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
        self.gaussian_mixture_model = None

    @classmethod
    def from_config(cls, model_config, random_state):

        raise NotImplementedError(
         "FHMM model not yet implemented")

# class GaussianMixtureModel(DiscreteHMM):

# class CategoricalModel(DiscreteHMM):