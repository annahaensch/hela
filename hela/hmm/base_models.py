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

    def __init__(self, hidden_state_count=None):
        self.hidden_state_count = hidden_state_count
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

    @classmethod
    def from_spec(cls, spec):
        """ Returns model configuration from specification dictionary (`spec`).  
        
        The specification dictionary contains the following key - value pairs:
        
        	* hidden_state_count - int or list of ints indicating number of 
        		discrete hidden states for each Markov system.   
        	* observations - list of observation dictionaries with keys:
        		* name - string name of feature.
        		* type - 'finite' or 'continuous'.
        		* dist - named distribution of 'continuous' else None.
        		* values - list of finite values if 'finite' else None.  
        	* model_parameter_constraints - (optional) dictionary giving 
        		model constraints with keys: 
        		* transition_contraints - 
        		* initial_state_constraints - 
        		* gmm_parameter_constraints - 
		
		Optional parameters which are not given will be seeded using random 
		state whenever the spec is used to generate an untrained model. 
        """
        #config = cls(hidden_state_count=spec['hidden_state_count'])
        config = cls(hidden_state_count=0)

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
        self.continuous_features = continuous_features_dist_dict

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


class HiddenMarkovModel(ABC):
    """ Abstract base class for HMM model objects. """

    def __init__(self, model_config=None):
        self.model_config = model_config

    @abstractmethod
    def from_config(self, model_config, random_state):
        """ Initilaize HiddenMarkovModel object from configuration.

        Arguments:
            model_config: model configuration object, obtained by running `from_spec` with a model specifiction dictionary.
            random_state: random seed for randomly initialized model paramters.

        Returns:
            Object in HiddenMarkovModel child class.
        """

    def load_inference_interface(self, use_jax=False):
        """ Load HiddenMarkovModel inference interface.  This includes methods that comprise the forward-backward passes of EM as well as imputation and Viterbi prediction methods.
        """
        return self._load_inference_interface(use_jax)

    def load_forecasting_interface(self, use_jax=False):
        """ Load HiddenMarkovModel forecasting interface.  This includes methods to forecast hidden states and observations at timestep horizons for given conditioning date.
        """
        return self._load_forecasting_interface(use_jax)

    def load_validation_interface(self, actual_data, use_jax=False):
        """Load validation interface for imputation and forecasting.

        This includes the tools the validate hidden state prediction,
        imputation, and forecasting against `actual data`.
        """
        # FIXME(wrvb) The arguments don't line up with the definition below
        return self._load_validation_interface(actual_data, use_jax)

    @abstractmethod
    def _load_inference_interface(self, use_jax):
        """ Load inference interface specific to model type."""

    @abstractmethod
    def _load_forecasting_interface(self, use_jax):
        """ Load forecasting interface specific to model type."""

    @abstractmethod
    def _load_validation_interface(self, actual_data, redacted_data,
                                   imputed_data, forecast_data, use_jax):
        """ Load validation interface specific to model type."""


class LearningAlgorithm(ABC):
    """ Abstract base class for HMM learning algorithms """

    def __init__(self):
        self.data = None
        self.finite_state_data = None
        self.gaussian_data = None
        self.other_data = None  #TODO: @AH incorporate other observation types.
        self.sufficient_statistics = []
        self.model_results = []

    def run(self, model, data, n_em_iterations, use_jax=False):
        """ Base class method for EM learning algorithm.

        Arguments:
            data: dataframe with hybrid data for training.
            n_em_iterations: number of em iterations to perform.

        Returns:
            Trained instance of HiddenMarkovModel belonging to the same child class as model.  Also returns em training results.
        """
        self.data = data
        if len(model.finite_features) > 0:
            self.finite_state_data = get_finite_observations_from_data_as_states(
                model, data)
        if len(model.continuous_features) > 0:
            if model.gaussian_mixture_model:
                self.gaussian_data = get_gaussian_observations_from_data(
                    model, data)

        new_model = model.model_config.to_model()

        for _ in range(n_em_iterations):
            # e_step
            expectation = new_model.load_inference_interface(use_jax)
            expectation.compute_sufficient_statistics(data)
            self.sufficient_statistics.append(expectation)

            # m_step
            new_model = new_model.update_model_parameters(
                self.finite_state_data, self.gaussian_data, expectation)
            self.model_results.append(new_model)

        return new_model


class HMMForecasting(ABC):

    def __init__(self, model, use_jax=False):
        self.model = model
        self.inf = model.load_inference_interface(use_jax)

    def forecast_hidden_state_at_horizons(self,
                                          data,
                                          horizon_timesteps,
                                          conditioning_date=None):
        """ Returns series with most likely hidden states at horizons.

        Arguments:
            data: dataframe with complete data
            horizon_timesteps: list of timesteps to consider for horizons.  It
                is assumed that the rows of `data` have a uniform timedelta; this uniform timedelta is 1 timestep
            conditioning_date: entry from `data` index, forecast will consider
                only the data up to this date.  If `None` the conditioning date is assumed to be the last entry of the index.

        Returns:
            Series with hidden state prections with the conditioning date as the first entry.
        """
        if conditioning_date is None:
            conditioning_date = data.index[-1]
        data_restricted = data.loc[:conditioning_date]

        return self._forecast_hidden_state_at_horizons(
            data_restricted, horizon_timesteps, conditioning_date)

    def forecast_observation_at_horizons(self,
                                         data,
                                         horizon_timesteps,
                                         conditioning_date=None):
        """ Returns dataframe with most likely observations at horizons.

        Arguments:
            data: dataframe with complete data
            horizon_timesteps: list of timesteps to consider for horizons.  It
                is assumed that the rows of `data` have a uniform timedelta; this uniform timedelta is 1 timestep
            conditioning_date: entry from `data` index, forecast will consider
                only the data up to this date.  If `None` the conditioning date is assumed to be the last entry of the index.

        Returns:
            Dataframe with forecast observations for horizon_timestep dates.  The first row of the dataframe is the conditioning date.
        """
        if conditioning_date is None:
            conditioning_date = data.index[-1]
        data_restricted = data.loc[:conditioning_date]

        return self._forecast_observation_at_horizons(
            data_restricted, horizon_timesteps, conditioning_date)

    def steady_state_and_horizon(self, data, conditioning_date=None,
                                 atol=1e-05):
        """ Returns dictionary with steady state information.

        Arguments:
            data: dataframe with complete data
            conditioning_date: entry from `data` index, forecast will consider
                only the data up to this date.  If `None` the conditioning date is assumed to be the last entry of the index.
            atol: tolerance for determining whether steady state has been
                reached.

        Returns:
            Dictionary with 'steady_state' for model and 'steady_state_horizon_timesteps', the timestep horizon at which the steady state has been achieved up to tolerance atol, and 'steady_state_horizon_date', the date at which the steady state has been achieved up to tolerance atol.
        """
        if conditioning_date is None:
            conditioning_date = data.index[-1]
        data_restricted = data.loc[:conditioning_date]

        return self._steady_state_and_horizon(data_restricted,
                                              conditioning_date, atol)


class HMMValidationMetrics(ABC):
    """ Abstract Base Class for HMM validation Metrics """

    def __init__(self, model, actual_data, use_jax=False):
        self.model = model
        self.actual_data = actual_data
        self.inf = self.model.load_inference_interface(use_jax)

    def predicted_hidden_state_log_likelihood_viterbi(self):
        """ Predict most likely hidden states with Viterbi algorithm

        Arguments:
            data: dataframe of mixed data types

        Returns:
            log likelihood of most likely series of hidden states
        """
        inf = self.model.load_inference_interface(use_jax)
        log_likelihood = inf.predict_hidden_states_viterbi(
            self.actual_data).name.split(" ")[-1]

        return float(log_likelihood)

    def validate_imputation(self, redacted_data, imputed_data):
        """ Return dictionary of validation metrics for imputation.

        Arguments:
            redacted_data: dataframe with values set to nan.
            imputed_data: dataframe with missing values imputed.

        Returns:
            Dictionary with validation metrics for imputed data against actual data.
        """
        # Make sure that integers are being cast as integers.
        float_to_int = {
            feature: "Int64"
            for feature in redacted_data[self.model.finite_features]
            .select_dtypes("float")
        }
        redacted_data = redacted_data.astype(float_to_int, errors='ignore')
        imputed_data = imputed_data.astype(float_to_int, errors='ignore')

        return self._validate_imputation(redacted_data, imputed_data)

    def validate_forecast(self, forecast_data):
        """ Return dictionary of validation metrics for imputation.

        Arguments:
            forecast_data: dataframe with forecast data where the first row
                of the dataframe is actual observed conditioning date data.

        Returns:
            Dictionary with validation metrics for forecast data against actual data.
        """
        return self._validate_forecast(forecast_data)
