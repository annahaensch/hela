""" Forecasting tool for discrete HMMs.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy import linalg

from .imputation import HMMImputationTool
from .utils import *


class HMMForecastingTool(ABC):

    def __init__(self, model, data, use_jax=False):
        self.model = model
        self.inf = model.load_inference_interface(use_jax)
        self.data = data
        self.use_jax = use_jax

    def forecast_hidden_state_at_horizons(self, horizon_timesteps):
        """ Returns series with most likely hidden states at horizons.

        Arguments:
            horizon_timesteps: list of timesteps to consider for horizons.  It
                is assumed that the rows of `data` have a uniform timedelta; this 
                uniform timedelta is 1 timestep

        Returns:
            Series with hidden state predictions with the last true observation 
            corresponding to the first entry.
        """
        return self._forecast_hidden_state_at_horizons(horizon_timesteps)

    def forecast_observation_at_horizons(self, horizon_timesteps):
        """ Returns dataframe with most likely observations at horizons.

        Arguments:
            horizon_timesteps: list of timesteps to consider for horizons.  It
                is assumed that the rows of `data` have a uniform timedelta; this 
                uniform timedelta is 1 timestep

        Returns:
            Dataframe with forecast observations for horizon_timestep dates.  The 
            first row of the dataframe is the last true observation.
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
        log_prob = self.inf.observation_log_probability(self.data)
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
        hidden_state_prob = self.hidden_state_probability_at_last_observation()
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
            hidden_state_prob @ np.linalg.matrix_power(transition_matrix, t)
            for t in horizon_timesteps
        ])

        return pd.DataFrame(
            horizon_prediction,
            index=horizon_date,
            columns=[i for i in range(len(transition_matrix))])

    def forecast_observation_at_horizons(self,
                                         horizon_timesteps,
                                         imputation_method='hmm_average'):
        """ Returns dataframe with most likely observations at horizon.

        Arguments:
            horizon_timesteps: list of timestep to consider for horizon.
                It is assumed that the rows of `data` have a uniform 
                timedelta; this uniform timedelta is 1 timestep

        Returns:
            dataframe with forecast observations at horizon
        """
        # length of timestep.
        delta = self.data.index[-1] - self.data.index[-2]
        model = self.model

        forecast_df = pd.DataFrame(
            index=[
                self.data.index[-1] + delta * (t + 1)
                for t in range(horizon_timesteps[-1])
            ],
            columns=self.data.columns)
        df = self.data.copy()
        imp = HMMImputationTool(model=model)
        for i in forecast_df.index:
            df = imp.impute_missing_data(
                pd.concat([df, forecast_df.loc[[i], :]]),
                method=imputation_method)

        return df.loc[forecast_df.index[[t - 1 for t in horizon_timesteps]]]

    def steady_state(self):
        """ Return steady state for model.
        """

        transition = np.exp(self.model.log_transition)
        val, left_eig, right_eig = linalg.eig(transition, left=True)
        idx = np.argmax(np.array([abs(v) for v in val]))
        vec = left_eig[:, idx]

        return vec / np.sum(vec)

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
        vec = self.steady_state()
        transition = np.exp(self.model.log_transition)
        hidden_state_prob = self.hidden_state_probability_at_last_observation()

        i = 1
        while np.max(
                np.abs(hidden_state_prob @ np.linalg.
                       matrix_power(transition, i) - vec)) > atol:
            i += 1

        delta = self.data.index[-1] - self.data.index[-2]
        horizon_date = self.data.index[-1] + (i * delta)

        return {
            'steady_state': vec,
            'steady_state_horizon_timesteps': i,
            'steady_state_horizon_date': horizon_date
        }
