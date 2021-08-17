""" Forecasting tool for discrete HMMs.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .imputation import HMMImputationTool
from .utils import *


class HMMForecastingTool(ABC):

    def __init__(self, model, data, use_jax=False):
        self.model = model
        self.inf = model.load_inference_interface(use_jax)
        self.data = data
        self.use_jax = use_jax

    def forecast_hidden_state_at_horizons(self,
                                          horizon_timesteps):
        """ Returns series with most likely hidden states at horizons.

        Arguments:
            horizon_timesteps: list of timesteps to consider for horizons.  It
                is assumed that the rows of `data` have a uniform timedelta; this 
                uniform timedelta is 1 timestep

        Returns:
            Series with hidden state prections with the conditioning date as the 
            first entry.
        """
        return self._forecast_hidden_state_at_horizons(horizon_timesteps)

    def forecast_observation_at_horizons(self,
                                         horizon_timesteps):
        """ Returns dataframe with most likely observations at horizons.

        Arguments:
            horizon_timesteps: list of timesteps to consider for horizons.  It
                is assumed that the rows of `data` have a uniform timedelta; this 
                uniform timedelta is 1 timestep

        Returns:
            Dataframe with forecast observations for horizon_timestep dates.  The 
            first row of the dataframe is the conditioning date.
        """
        return self._forecast_observation_at_horizons(horizon_timesteps)

    def steady_state_and_horizon(self, atol=1e-05):
        """ Returns dictionary with steady state information.

        Arguments:
            atol: tolerance for determining whether steady state has been
                reached.

        Returns:
            Dictionary with 'steady_state' for model and 'steady_state_horizon_timesteps', 
            the timestep horizon at which the steady state has been achieved up to 
            tolerance atol, and 'steady_state_horizon_date', the date at which the steady 
            state has been achieved up to tolerance atol.
        """
        return self._steady_state_and_horizon(atol)


    def hidden_state_probability_at_last_observation(self):
        """ Compute probability of hidden state given data up to last observation,

        Returns:
            Probability distribution of hidden state at last observation.
        """
        log_prob = self.inf.predict_hidden_state_log_probability(self.data)
        joint_prob = self.inf._compute_forward_probabilities(log_prob)

        return np.exp(joint_prob[-1] - logsumexp(joint_prob, axis=1)[-1])

    def hidden_state_probability_at_horizon(self, horizon_timestep):
        """ Compute hidden state probability at horizon.

        Argument:
            horizon_timestep: timestep to consider for horizon.
                It is assumed that the rows of `data` have a uniform 
                timedelta; this uniform timedelta is 1 timestep

        Returns:
            An array with dimention (1 x n_hidden_states) where the ith entry is 
            the conditional probability of hidden state i at horizon.
        """
        hidden_state_prob = self.hidden_state_probability_at_last_observation(
            data, conditioning_date)
        transition_matrix = np.exp(self.model.log_transition)

        return hidden_state_prob @ np.linalg.matrix_power(
            transition_matrix, horizon_timestep)

    def hidden_state_probability_at_horizons(self, horizon_timesteps):
        """ Compute hidden state probability at horizons.

        Argument:
            horizon_timesteps: list of timesteps to consider for horizons.
                It is assumed that the rows of `data` have a uniform 
                timedelta; this uniform timedelta is 1 timestep

        Returns:
            Dataframe with hidden state probabilities for horizon dates.
        """
        hidden_state_prob = self.hidden_state_probability_at_last_observation()

        transition_matrix = np.exp(self.model.log_transition)

        delta = self.data.index[-1] - self.data.index[-2]
        horizon_date = [
            self.data.index[-1] + (t * delta) for t in horizon_timesteps
        ]
        horizon_prediction = np.array([
            hidden_state_prob @ np.linalg.matrix_power(
                transition_matrix, t) for t in horizon_timesteps
        ])

        forecast = pd.DataFrame(
            horizon_prediction,
            index=horizon_date,
            columns=[i for i in range(len(transition_matrix))])

        return forecast

    def forecast_observation_at_horizon(self,
                                        horizon_timestep
                                        imputation_method='hmm_average'):
        """ Returns dataframe with most likely observations at horizon.

        Arguments:
            horizon_timestep: timestep to consider for horizon.
                It is assumed that the rows of `data` have a uniform 
                timedelta; this uniform timedelta is 1 timestep

        Returns:
            dataframe with forecast observations at horizon
        """
        delta = self.data.index[-1] - self.data.index[-2]
        new_time = self.data.index[-1] + (horizon_timestep * delta)
        observation = data.iloc[[-1]].copy()
        observation.loc[new_time, self.data.columns] = np.nan

        hidden_state_prob = self.hidden_state_probability_at_horizon(horizon_timestep)

        model = self.model
        imp = HMMImputationTool(model = model, method = imputation_method)
        
        forecast = self.inf.impute_missing_data_single_observation(
            observation.loc[[new_time]], hidden_state_prob, imputation_method)

        return forecast

    def _forecast_observation_at_horizons(self,
                                          horizon_timesteps,
                                          imputation_method='average'):
        """ Returns dataframe with most likely observations at horizons.

        Arguments:
            horizon_timesteps: list of timesteps to consider for horizons.
                It is assumed that the rows of `data` have a uniform 
                timedelta; this uniform timedelta is 1 timestep

        Returns:
            Dataframe with forecast observations for horizon_timestep dates. The 
            first row of the dataframe is the conditioning date.
        """
        forecast = self.data.iloc[[-1]]
        for horizon in horizon_timesteps:
            forecast = pd.concat((forecast,
                                  self.forecast_observation_at_horizon(
                                      horizon,
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
            atol: tolerance for determining whether steady state has been
                reached.

        Returns:
            Dictionary with 'steady_state' for model and 'steady_state_horizon_timesteps', 
            the timestep horizon at which the steady state has been achieved up to 
            tolerance atol, and 'steady_state_horizon_date', the date at which the steady 
            state has been achieved up to tolerance atol.
        """
        vec = self.steady_state()
        transition = np.exp(self.model.log_transition)
        hidden_state_prob = self.hidden_state_probability_at_last_observation()

        i = 1
        while np.max(
                np.abs(hidden_state_prob @ np.linalg.matrix_power(transition, i) -
                       vec)) > atol:
            i += 1

        delta = self.data.index[-1] - self.data.index[-2]
        horizon_date = self.data.index[-1] + (i * delta)

        return {
            'steady_state': vec,
            'steady_state_horizon_timesteps': i,
            'steady_state_horizon_date': horizon_date
        }