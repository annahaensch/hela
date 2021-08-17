""" Forecasting tool for discrete HMMs.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .utils import *

class HMMForecastingTool(ABC):

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


    def hidden_state_probability_at_conditioning_date(self, data,
                                                      conditioning_date):
        """ Compute probability of hidden state given data up to conditioning date.

        Arguments:
            data: dataframe with complete data
            conditioning_date: entry from `data` index, forecast will
                consider only the data up to this date.

        Returns:
            Probability distribution of hidden state at condtioning date.
        """
        data_restricted = data.loc[:conditioning_date]

        log_prob = self.inf.predict_hidden_state_log_probability(
            data_restricted)
        joint_prob = self.inf._compute_forward_probabilities(log_prob)

        return np.exp(joint_prob[-1] - logsumexp(joint_prob, axis=1)[-1])

    def hidden_state_probability_at_horizon(self, data, horizon_timestep,
                                            conditioning_date):
        """ Compute hidden state probability at horizon.

        Argument:
            data: dataframe with complete data
            horizon_timestep: timestep to consider for horizon.
                It is assumed that the rows of `data` have a uniform timedelta; this uniform timedelta is 1 timestep
            conditioning_date: entry from `data` index, forecast will
                consider only the data up to this date.

        Returns:
            An array with dimention (1 x n_hidden_states) where the ith entry is the conditional probability of hidden state i at horizon.
        """
        conditioning_date_prob = self.hidden_state_probability_at_conditioning_date(
            data, conditioning_date)
        transition_matrix = np.exp(self.model.log_transition)

        return conditioning_date_prob @ np.linalg.matrix_power(
            transition_matrix, horizon_timestep)

    def hidden_state_probability_at_horizons(self, data, horizon_timesteps,
                                             conditioning_date):
        """ Compute hidden state probability at horizons.

        Argument:
            data: dataframe with complete data
            horizon_timesteps: list of timesteps to consider for horizons.
                It is assumed that the rows of `data` have a uniform timedelta; this uniform timedelta is 1 timestep
            conditioning_date: entry from `data` index, forecast will
                consider only the data up to this date.

        Returns:
            Dataframe with hidden state probabilities for horizon dates.
        """
        conditioning_date_prob = self.hidden_state_probability_at_conditioning_date(
            data, conditioning_date)

        transition_matrix = np.exp(self.model.log_transition)

        delta = data.index[-1] - data.index[-2]
        horizon_date = [
            conditioning_date + (t * delta) for t in horizon_timesteps
        ]
        horizon_prediction = np.array([
            conditioning_date_prob @ np.linalg.matrix_power(
                transition_matrix, t) for t in horizon_timesteps
        ])

        forecast = pd.DataFrame(
            horizon_prediction,
            index=horizon_date,
            columns=[i for i in range(len(transition_matrix))])

        return forecast

    def _forecast_hidden_state_at_horizons(self, data, horizon_timesteps,
                                           conditioning_date):
        """ Returns series with most likely hidden states at horizons.

        Arguments:
            data: dataframe with complete data
            horizon_timesteps: list of timesteps to consider for horizons.
                It is assumed that the rows of `data` have a uniform timedelta; this uniform timedelta is 1 timestep
            conditioning_date: entry from `data` index, forecast will
                consider only the data up to this date.

        Returns:
            Series with hidden state prections with the conditioning date as the first entry.
        """
        forecast = self.hidden_state_probability_at_horizons(
            data, horizon_timesteps, conditioning_date)

        return pd.Series(
            np.array(forecast).argmax(axis=1), index=forecast.index)

    def forecast_observation_at_horizon(self,
                                        data,
                                        horizon_timestep,
                                        conditioning_date,
                                        imputation_method='hmm_average'):
        """ Returns dataframe with most likely observations at horizon.

        Arguments:
            data: dataframe with complete data
            horizon_timestep: timestep to consider for horizon.
                It is assumed that the rows of `data` have a uniform timedelta; this uniform timedelta is 1 timestep
            conditioning_date: entry from `data` index, forecast will
                consider only the data up to this date.

        Returns:
            dataframe with forecast observations at horizon
        """
        delta = data.index[-1] - data.index[-2]
        new_time = conditioning_date + (horizon_timestep * delta)
        observation = data.loc[[conditioning_date]].copy()
        observation.loc[new_time, data.columns] = np.nan

        hidden_state_prob = self.hidden_state_probability_at_horizon(
            data, horizon_timestep, conditioning_date)

        model = self.model
        imp = HMMImputationTool(model = model, method = imputation_method)
        
        forecast = self.inf.impute_missing_data_single_observation(
            observation.loc[[new_time]], hidden_state_prob, imputation_method)

        return forecast

    def _forecast_observation_at_horizons(self,
                                          data,
                                          horizon_timesteps,
                                          conditioning_date,
                                          imputation_method='average'):
        """ Returns dataframe with most likely observations at horizons.

        Arguments:
            data: dataframe with complete data
            horizon_timesteps: list of timesteps to consider for horizons.
                It is assumed that the rows of `data` have a uniform timedelta; this uniform timedelta is 1 timestep
            conditioning_date: entry from `data` index, forecast will
                consider only the data up to this date.

        Returns:
            Dataframe with forecast observations for horizon_timestep dates. The first row of the dataframe is the conditioning date.
        """
        forecast = data.loc[[conditioning_date]]
        for horizon in horizon_timesteps:
            forecast = pd.concat((forecast,
                                  self.forecast_observation_at_horizon(
                                      data,
                                      horizon,
                                      conditioning_date,
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
            data: dataframe with complete data
            conditioning_date: entry from `data` index, forecast will
                consider only the data up to this date.
            atol: tolerance for determining whether steady state has been
                reached.

        Returns:
            Dictionary with 'steady_state' for model and 'steady_state_horizon_timesteps', the timestep horizon at which the steady state has been achieved up to tolerance atol, and 'steady_state_horizon_date', the date at which the steady state has been achieved up to tolerance atol.
        """
        if conditioning_date is None:
            conditioning_date = data.index[-1]
        vec = self.steady_state()
        transition = np.exp(self.model.log_transition)
        initial_prob = self.hidden_state_probability_at_conditioning_date(
            data, conditioning_date)

        i = 1
        while np.max(
                np.abs(initial_prob @ np.linalg.matrix_power(transition, i) -
                       vec)) > atol:
            i += 1

        delta = data.index[-1] - data.index[-2]
        horizon_date = conditioning_date + (i * delta)

        return {
            'steady_state': vec,
            'steady_state_horizon_timesteps': i,
            'steady_state_horizon_date': horizon_date
        }