"""Hidden Markov model implementation for multiple data types.

Each HMM type has several associated classes:
1. A *model configuration*, which defines the structure of the HMM: the
   observation space, the space of hidden states, restrictions on the transition
   function, and so forth. A model configuration can be used (with a learning
   algorithm and a dataset) to construct a *model*.
2. A *model* object, which has parameters and inference methods. A model can be
   used (with a sample) to construct an *inference* object.
3. A *learning algorithm* object which carries out model training and parameter
   optimization with expectation-maximization.
4. An *inference* object, which encapsulates the result of the forward-backward
   algorithm, can be used to compute various statistics and predictions. It is
   mostly used as a convenient way to cache the results of inference.
5. A *forecasting* object which allows forecting to predict hidden states and
   forecast observations.
6. A *validation* object which contains the validation metrics for the various
   other objects.
"""

import itertools
import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .utils import *

LOG_ZERO = -1e8

logger = logging.getLogger(name='root')
logger.setLevel(level=logging.INFO)


class HMMConfiguration(ABC):
    """ Abstract base class for HMM configuration """

    def __init__(self, n_hidden_states=None):
        self.n_hidden_states = n_hidden_states
        self.model_parameter_constraints = {
            'transition_constraints': None,
            'initial_state_constraints': None,
            'emission_constraints': None,
            'gmm_parameter_constraints': None
        }
        self.finite_features = []
        self.finite_values = None
        self.finite_values_dict = {}
        self.finite_values_dict_inverse = {}
        self.continuous_features = []
        self.continuous_values = None

    @classmethod
    def from_spec(cls, spec):
        """ Returns model configuration from specification dictionary (`spec`).  
        
        The specification dictionary contains the following key - value pairs:
        
        	* n_hidden_states - int or list of ints indicating number of 
        		discrete hidden states for each Markov system.   
        	* observations - list of observation dictionaries with keys:
        		* name - string name of feature.
        		* type - 'finite' or 'continuous'.
        		* dist - named distribution of 'continuous' else None.
        		* values - list of finite values if 'finite' else None.  
        	* model_parameter_constraints - (optional) dictionary giving 
        		model constraints with keys: 
        		* transition_contraints - array of transition probabilities.
        		* initial_state_constraints -  array of initial state 
        			probabilities.
        		* gmm_parameter_constraints - dictionary with keys:
        			* n_gmm_components - int number of gmm components.
        			* means - array of means
        			* covariances - array of covariance arrays
        			* component_weights - array of component weights.
		
		Optional parameters which are not given will be seeded using random 
		state whenever the spec is used to generate an untrained model. 
        """
        config = cls(n_hidden_states=spec['n_hidden_states'])

        finite_observations = [
            i for i in spec['observations'] if i['type'].lower() == 'finite'
        ]
        continuous_observations = [
            i for i in spec['observations'] if i['type'].lower() == 'continuous'
        ]

        other_observations = [
            i for i in spec['observations']
            if i['type'].lower() not in ['finite', 'continuous']
        ]
        if len(other_observations) > 0:
            raise NotImplementedError(
                "The specification contains the following data types "
                "which are not part of this implementation {}".format(
                	", ".join([i['type'] for i in other_observations])))

        if len(finite_observations) > 0:
            config.add_finite_observations(finite_observations)
            config.set_finite_values_dict()

        if len(continuous_observations) > 0:
            config.add_continuous_observations(continuous_observations)

        if 'model_parameter_constraints' in spec:
            for param in spec['model_parameter_constraints']:
                config.add_model_parameter_constraints(
                    param, spec['model_parameter_constraints'][param])
        return config._from_spec(spec)

    @abstractmethod
    def _from_spec(self, spec):
        """Child class specific implementation of `from_spec`.
        """

    def add_finite_observations(self, finite_observations):
        """ Add a finite observation to a DiscreteHMM object. """
        finite_features_dict = {
            i['name']: i['values']
            for i in finite_observations
        }

        # Add finite features to a list and sort.
        finite_features = sorted(finite_features_dict)
        self.finite_features = finite_features

        values = []
        for f in finite_features:
            # Add values to a list and sort.
            values_f = finite_features_dict[f].copy()
            if any(isinstance(x, float) for x in values_f):
                raise ValueError(
                    "The categorical feature, '{}', contains "
                    "floats which are not a valid categorical datatype.  To "
                    "avoid this error, try casting your floats as strings "
                    "instead.".format(f))
            values_f.sort()
            values.append(values_f)
        value_tuples = [t for t in itertools.product(*values)]
        self.finite_values = pd.DataFrame(value_tuples, columns=finite_features)

    def add_continuous_observations(self, continuous_observations):
        """ Add a continuous observation to a DiscreteHMM object. """
        continuous_features_dict = {
            i['name']: i['dist']
            for i in continuous_observations
        }

        for k,v in continuous_features_dict.items():
        	if v.lower().replace(" ","_") not in [
        	'gaussian', 'gaussian_mixture_model', 'gmm']:
        		raise NotImplementedError(
        			"This implementation only works for continuous values "
        			"drawn from gaussian or gaussian mixture model "
        			"distributions."
        			)

        # Add continuous features to a list and sort.
        continuous_features = list(continuous_features_dict.keys())
        continuous_features.sort()
        self.continuous_features = continuous_features

        continuous_values = pd.DataFrame(index = ["distribution", "dimension"], columns = continuous_features)
        for c in continuous_values.columns:
        	continuous_values.loc["distribution",c] = continuous_features_dict[c]
        	continuous_values.loc["dimension",c] = 1
        self.continuous_values = continuous_values	

    def add_model_parameter_constraints(self, parameter, constraint):
        """ Add constraints for seed parameters. """
        self.model_parameter_constraints[parameter] = constraint

    def set_finite_values_dict(self):
        """Set dictionary mapping int values to observable finite states as tuples, and inverse dictionary doing the inverse. """
        self.finite_values_dict = {
            i: list(self.finite_values.loc[i])
            for i in self.finite_values.index
        }
        self.finite_values_dict_inverse = {
            str(list(self.finite_values.loc[i])): i
            for i in self.finite_values.index
        }

    def to_model(self, set_random_state=0):
        """ Instatiate model from configuration """
        random_state = np.random.RandomState(set_random_state)
        return self._to_model(random_state)

    @abstractmethod
    def _to_model(self, random_state):
        """Child class specific implementation of `to_model`.
        """


class HiddenMarkovModel(ABC):
    """ Abstract base class for HMM model objects. """

    def __init__(self, model_config=None):
        self.model_config = model_config

    @abstractmethod
    def from_config(self, model_config, random_state):
        """ Initilaize HiddenMarkovModel object from configuration.

        Arguments:
            model_config: model configuration object, obtained by running 
            	`from_spec` with a model specifiction dictionary.
            random_state: random seed for randomly initialized model parameters.

        Returns:
            Object in HiddenMarkovModel child class.
        """

    def load_learning_interface(self):
    	"""  Loads LearningAlgorithm interface specific to model type. 
    	"""
    	return self._load_learning_interface()

    def load_inference_interface(self, use_jax=False):
        """ Load HiddenMarkovModel inference interface.  This includes 
        methods that comprise the forward-backward passes of EM as well 
        as Viterbi prediction methods.
        """
        return self._load_inference_interface(use_jax)

    @abstractmethod
    def _load_learning_interface(self):
        """ Load learning interface specific to model type."""

    @abstractmethod
    def _load_inference_interface(self, use_jax):
        """ Load inference interface specific to model type."""


class HMMLearningAlgorithm(ABC):
    """ Abstract base class for HMM learning algorithms """

    def __init__(self):
        self.data = None
        self.finite_state_data = None
        self.gaussian_data = None
        self.other_data = None 
        self.sufficient_statistics = []
        self.model_results = []


    @abstractmethod
    def run(self, model, data, training_iterations, method, use_jax):
        """ Runs specified training method for given data."""

