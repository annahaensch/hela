""" Validation tools for discrete HMMs.
"""
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .imputation import HMMImputationTool
from .utils import *

class HMMValidationTool(ABC):
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

class DiscreteHMMValidationTool(HMMValidationTool):
    """ Validation class specific to discrete HMM
    """

    def __init__(self, model, actual_data, use_jax=False):
        super().__init__(model, actual_data)
        self.inf = model._load_inference_interface(use_jax)
        self.actual_gaussian_data = get_gaussian_observations_from_data(
            self.model, actual_data)
        self.actual_categorical_data = get_finite_observations_from_data(
            self.model, actual_data)

    def _validate_imputation(self, redacted_data, imputed_data):
        """ Return DiscreteHMM specific dictionary of validation metrics for imputation.

        Arguments:
            redacted_data: dataframe with values set to nan.
            imputed_data: dataframe with missing values imputed.

        Returns:
            Dictionary with validation metrics for imputed data against actual data.
        """
        cond_prob_of_hidden_states = self.inf.conditional_probability_of_hidden_states(
            redacted_data)
        val_dict = {}

        if self.model.categorical_model:
            redacted_categorical_data = get_finite_observations_from_data(
                self.model, redacted_data)
            imputed_categorical_data = get_finite_observations_from_data(
                self.model, imputed_data)

            val_dict[
                'accuracy_of_imputed_categorical_data'] = self.accuracy_of_predicted_categorical_data(
                    redacted_categorical_data, imputed_categorical_data)

            val_dict[
                'relative_accuracy_of_imputed_categorical_data'] = self.relative_accuracy_of_predicted_categorical_data(
                    redacted_categorical_data, imputed_categorical_data)

            val_dict[
                'best_possible_accuracy_of_categorical_imputation'] = best_possible_accuracy_of_categorical_prediction(
                    self.actual_categorical_data, redacted_categorical_data)

        if self.model.gaussian_mixture_model:
            redacted_gaussian_data = get_gaussian_observations_from_data(
                self.model, redacted_data)
            imputed_gaussian_data = get_gaussian_observations_from_data(
                self.model, imputed_data)

            val_dict[
                'average_relative_log_likelihood_of_imputed_gaussian_data'] = self.average_relative_log_likelihood_of_predicted_gaussian_data(
                    redacted_gaussian_data, imputed_gaussian_data,
                    cond_prob_of_hidden_states)

            val_dict[
                'average_z_score_of_imputed_gaussian_data'] = self.average_z_score_of_predicted_gaussian_data(
                    redacted_gaussian_data, cond_prob_of_hidden_states)

        return val_dict

    def _validate_forecast(self, forecast_data):
        """ Return DiscreteHMM specific dictionary of validation metrics for imputation.

        Arguments:
            forecast_data: dataframe with forecast data where the first row
                of the dataframe is actual observed conditioning date data.

        Returns:
            Dictionary with validation metrics for forecast data against actual data.
        """
        conditioning_date = forecast_data.index[0]
        delta = self.actual_data.index[-1] - self.actual_data.index[-2]

        horizon_timesteps = [
            int(t) for t in (forecast_data.index - conditioning_date) / delta
        ]

        cond_prob_of_hidden_states = DiscreteHMMForecasting(
            self.model).hidden_state_probability_at_horizons(
                self.actual_data, horizon_timesteps, conditioning_date)

        val_dict = {}

        if self.model.categorical_model:
            forecast_categorical_data = get_finite_observations_from_data(
                self.model, forecast_data)

            redacted_categorical_data = forecast_categorical_data.copy()
            redacted_categorical_data.loc[:, :] = np.nan

            val_dict[
                'accuracy_of_forecast_categorical_data'] = self.accuracy_of_predicted_categorical_data(
                    redacted_categorical_data, forecast_categorical_data)

            val_dict[
                'relative_accuracy_of_forecast_categorical_data'] = self.relative_accuracy_of_predicted_categorical_data(
                    redacted_categorical_data, forecast_categorical_data)

            val_dict[
                'best_possible_accuracy_of_categorical_forecast'] = best_possible_accuracy_of_categorical_prediction(
                    self.actual_categorical_data, redacted_categorical_data)

        if self.model.gaussian_mixture_model:
            forecast_gaussian_data = get_gaussian_observations_from_data(
                self.model, forecast_data)

            redacted_gaussian_data = forecast_gaussian_data.copy()
            redacted_gaussian_data.loc[:, :] = np.nan

            val_dict[
                'average_relative_log_likelihood_of_forecast_gaussian_data'] = self.average_relative_log_likelihood_of_predicted_gaussian_data(
                    redacted_gaussian_data, forecast_gaussian_data,
                    cond_prob_of_hidden_states)

            val_dict[
                'average_z_score_of_forecast_gaussian_data'] = self.average_z_score_of_predicted_gaussian_data(
                    redacted_gaussian_data, cond_prob_of_hidden_states)

        return val_dict

    def average_relative_log_likelihood_of_predicted_gaussian_data(
            self, redacted_gaussian_data, imputed_gaussian_data,
            conditional_probability_of_hidden_states):
        """Returns the difference between the log likelihood of the actual
        data and the log likelihood of the imputed data.  This is done
        using the probability density function for the conditional probability
        of the unknown part of the observation given the known part of the
        observation.  This metric is intended to be a measure of how surprised
        you should be to see the actual value relative to the imputed value.

        Arguments:
            redacted_gaussian_data: dataframe if Gaussian observations
                with values set to nan.
            imputed_gaussian_data: dataframe with missing values imputed.
            conditional_probability_of_hidden_states: dataframe with
                conditional probability of hidden states given partial
                observations at all timesteps with redacted data.

        Returns:
            float
        """
        actual_gaussian_data = self.actual_gaussian_data
        means = self.model.gaussian_mixture_model.means
        covariances = self.model.gaussian_mixture_model.covariances
        component_weights = self.model.gaussian_mixture_model.component_weights

        redacted_index = redacted_gaussian_data[redacted_gaussian_data.isnull()
                                                .any(axis=1)].index
        imputed_likelihood = np.empty(len(redacted_index))
        actual_likelihood = np.empty(len(redacted_index))
        for i in range(len(redacted_index)):
            idx = redacted_index[i]
            p = np.float64(
                np.array(conditional_probability_of_hidden_states.loc[idx]))
            log_cond_prob = np.log(
                p, np.full(p.shape, LOG_ZERO), where=(p != 0))

            actual_gaussian_observation = actual_gaussian_data.loc[[idx]]
            imputed_gaussian_observation = imputed_gaussian_data.loc[[idx]]
            partial_gaussian_observation = redacted_gaussian_data.loc[[idx]]

            actual_prob = compute_log_likelihood_with_inferred_pdf(
                actual_gaussian_observation, partial_gaussian_observation,
                means, covariances, component_weights)
            actual_likelihood[i] = logsumexp(actual_prob + log_cond_prob)

            imputed_prob = compute_log_likelihood_with_inferred_pdf(
                imputed_gaussian_observation, partial_gaussian_observation,
                means, covariances, component_weights)
            imputed_likelihood[i] = logsumexp(imputed_prob + log_cond_prob)

        total_actual_log_likelihood = logsumexp(actual_likelihood)
        total_imputed_log_likelihood = logsumexp(imputed_likelihood)

        return total_actual_log_likelihood - total_imputed_log_likelihood

    def average_z_score_of_predicted_gaussian_data(
            self, redacted_gaussian_data,
            conditional_probability_of_hidden_states):
        """ Computes z score of gaussian data averaged over observations.

        Arguments:
            redacted_gaussian_data: dataframe if Gaussian observations
                with values set to nan.
            imputed_gaussian_data: dataframe with missing values imputed.
            conditional_probability_of_hidden_states: dataframe with
                conditional probability of hidden states given partial
                observations at all timesteps with redacted data.

        Returns:
            float
        """
        means = self.model.gaussian_mixture_model.means
        covariances = self.model.gaussian_mixture_model.covariances
        component_weights = self.model.gaussian_mixture_model.component_weights

        return average_z_score(
            means, covariances, component_weights, self.actual_gaussian_data,
            redacted_gaussian_data, conditional_probability_of_hidden_states)

    def accuracy_of_predicted_categorical_data(self, redacted_categorical_data,
                                               imputed_categorical_data):
        """ Returns ratio of correctly imputed categorical values to total imputed categorical values.

        Arguments:
            redacted_categorical_data: dataframe of categorical data with
                values set to nan.
            imputed_categorical_data: dataframe of categorical data with missing values fill in.

        Returns:
            float
        """
        redacted_index = redacted_categorical_data[
            redacted_categorical_data.isnull().any(axis=1)].index

        total_correct = np.sum(
            (self.actual_categorical_data.loc[redacted_index] ==
             imputed_categorical_data.loc[redacted_index]).all(axis=1))

        return total_correct / len(redacted_index)

    def relative_accuracy_of_predicted_categorical_data(
            self, redacted_categorical_data, imputed_categorical_data):
        """ Returns ratio of rate of accuracy in imputed data to expected rate of accuracy with random guessing.

        Arguments:
            redacted_categorical_data: dataframe of categorical data with
                values set to nan.
            imputed_categorical_data: dataframe of categorical data with missing values fill in.

        Returns:
            float
        """
        expected_accuracy = expected_proportional_accuracy(
            self.actual_categorical_data, redacted_categorical_data)
        imputed_accuracy = self.accuracy_of_predicted_categorical_data(
            redacted_categorical_data, imputed_categorical_data)

        return imputed_accuracy / expected_accuracy

    def precision_recall_df_for_predicted_categorical_data(
            self, redacted_data, imputed_data):
        """ Return DataFrame with precision, recall, and proportion of categorical values

        Arguments:
            redacted_data: dataframe with values set to nan.
            imputed_data: dataframe with missing values imputed.

        Returns:
            Dataframe with precision, recall, and proportion of imputed data against actual data.
        """
        if len(self.model.finite_features) == 0:
            return None
        else:
            redacted_categorical_data = get_finite_observations_from_data(
                self.model, redacted_data)
            redacted_index = redacted_categorical_data[
                redacted_categorical_data.isnull().any(axis=1)].index

            df = self.actual_data.copy()
            df['tuples'] = list(
                zip(*[
                    self.actual_data[c]
                    for c in self.actual_categorical_data.columns
                ]))
            proportion = (df['tuples'].value_counts() / df.shape[0]).to_dict()

            df_imputed = imputed_data.copy()
            df_imputed['tuples'] = list(
                zip(*[
                    imputed_data[c]
                    for c in self.actual_categorical_data.columns
                ]))

            state = df['tuples'].unique()

            precision_recall = pd.DataFrame(
                np.full((df['tuples'].nunique(), 3), np.nan),
                index=state,
                columns=['precision', 'recall', 'proportion'])

            for n, idx in enumerate(precision_recall.index):
                # pandas cannot use loc with tuples in the index
                precision_recall.iloc[n]['proportion'] = proportion[idx]

            actual = df.loc[redacted_index, 'tuples']

            imputed = df_imputed.loc[redacted_index, 'tuples']

            true_pos = ((np.array(actual)[:, None] == state) &
                        (np.array(imputed)[:, None] == state)).sum(axis=0)
            false_pos = ((np.array(actual)[:, None] != state) &
                         (np.array(imputed)[:, None] == state)).sum(axis=0)
            false_neg = ((np.array(actual)[:, None] == state) &
                         (np.array(imputed)[:, None] != state)).sum(axis=0)

            precision_recall.loc[state, 'precision'] = [
                x for x in true_pos / np.
                array([np.nan if x == 0 else x for x in true_pos + false_pos])
            ]
            precision_recall.loc[state, 'recall'] = [
                x for x in true_pos / np.
                array([np.nan if x == 0 else x for x in true_pos + false_neg])
            ]

            precision_recall.sort_values(
                by=['proportion'], ascending=False, inplace=True)
            return precision_recall


def find_risk_at_horizons(data,
                          asset_id,
                          model,
                          label_column,
                          selected_labels,
                          horizons,
                          prediction_dates,
                          event_time_resolution,
                          skip_last_horizon=True):
    """Probabilities at horizons, formatted as input for TTE Walk-forward CI.

    Arguments:
        data (DF): Dataframe, same format as in training, with a column for the
            label(s) to be predicted with the name 'label_column'
        asset_id (str): Asset identifier, for storage in validation_df output
        model: HMM model
        label_column (str): Name of column in 'data' that holds labels listed in
            'selected_labels'
        selected_labels (list): Subset of label identifiers to be predicted
            These will be predicted and tagged separately from each other
        horizons (list): List of horizons as pandas Timedelta values.
        prediction_dates (pd.date_range or list of pd.Timestamps):
            The list of timestamps at which predictions are calculated. This
            can be spaced non-uniformly.
        event_time_resolution (pd.Timedelta): Consolidate events within this
            resolution into one, using the first timestamp as the event time.
        skip_last_horizon: Do not calculate probabilities for times that are
            within the minimum horizon wrt the last data point. Set to zero.

    Returns:
        predictions: an array of predictions with shape
            (n_assets, n_timesteps, n_horizons, n_events), pass to predictions
            arg of wfci in tte/walk_forward_concordance.py .
        prediction_dates (pd.DateRange): The range of prediction timestamps,
            pass to wfci's prediction_dates arg.
        validation_df (pd.DataFrame): a dataframe where each row corresponds
            to true event, with columns asset_id, start_date, event_date,
            event_id. The last, event_id, is the zero-indexed counter of event
            labels as listed in 'selected_labels'. Input for wfci's
            validation_df arg.
    """
    assert label_column in data.columns, (
        f"Column name '{label_column}' not found in data!")
    # check for consistent interval
    diffs = np.unique((data.index[1:] - data.index[:-1]).round(
        pd.Timedelta(minutes=1)))  #ignore offsets of less than 1 minute
    if len(diffs) != 1:
        raise ValueError(
            "The 'data' must be a DataFrame with uniform time index.")
    data_interval = diffs[0]

    prediction_dates = pd.DatetimeIndex(prediction_dates).sort_values()
    start = prediction_dates[0]
    end = prediction_dates[-1]
    n_timesteps = len(prediction_dates)

    validation_df = pd.DataFrame(
        columns=["asset_id", "start_date", "event_date", "event_id"])

    predictions = np.ones((n_timesteps, len(horizons), len(selected_labels)))
    if skip_last_horizon:
        in_last_horizon = np.argwhere(
            prediction_dates > (data.index[-1] - min(horizons))).flatten()
        predictions[in_last_horizon, :, :] = 0
    # Set to zero and don't compute the probabilities that are within the
    # minimum horizon of the end, since CI will exclude these anyway.
    for e, event_id in enumerate(selected_labels):
        idx = data[data[label_column] == event_id].index
        if len(idx) == 0:
            predictions[:, :, e] = 0  # no occurences of this event_id
        episode_start = start
        for i in idx:
            if (i <= start) or (i > end + max(horizons)):
                # ignore events outside prediction window
                continue
            elif (i - episode_start) < event_time_resolution:
                # don't use prediction dates within an ongoing longer event
                within = np.argwhere(
                    np.logical_and(prediction_dates >= episode_start,
                                   prediction_dates <= i)).flatten()
                predictions[within, :, :] = 0
                # use only first event occurrence if any event is separated
                # by less than timestep from the previous occurrence
                episode_start = i + data_interval
                continue
            validation_df = validation_df.append(
                {
                    "asset_id": asset_id,
                    "start_date": episode_start,
                    "event_date": i,
                    "event_id": e
                },
                ignore_index=True)
            # new episode_start for next event
            episode_start = i + data_interval

    fc = model.load_forecasting_interface()
    emission = np.exp(model.categorical_model.log_emission_matrix)

    if label_column not in model.finite_features:
        raise IndexError(f"The label_column value '{label_column}' is not the "
                         f"name of a finite observation in the HMM model!")
    finite_values = model.finite_values
    label_values = finite_values[label_column].unique()
    assert all([
        s in label_values for s in selected_labels
    ]), ("Not all selected_labels are in the known values for column "
         f"{label_column}; unknown values: \n"
         f"{set(selected_labels) - set(label_values)}")

    horizons_di = [int(h / data_interval) for h in horizons]
    for t, prediction_date in enumerate(prediction_dates):
        if np.all(predictions[t, :, :] == 0):
            continue  # in case of dates eliminated with skip_last_horizon
        state_probs = fc.hidden_state_probability_at_horizons(
            data, horizons_di, prediction_date).values.T
        # state_probs has shape n_emission_states x len(horizons)
        weighted_emission = emission @ state_probs
        for e, event_val in enumerate(selected_labels):
            # vectorized emission prob for event e, for each horizons
            finite_values = model.finite_values
            probs = weighted_emission[tuple(
                [finite_values[label_column] == event_val])].sum(axis=0)
            predictions[t, :, e] = probs

    return predictions, prediction_dates, validation_df