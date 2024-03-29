""" Validation tools for discrete HMMs.
"""
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .forecasting import HMMForecastingTool
from .utils import *

LOG_ZERO = -1e8


class HMMValidationTool(ABC):
    """ Abstract Base Class for HMM validation Metrics """

    def __init__(self, model, true_data, use_jax=False):
        self.model = model
        self.true_data = true_data
        self.use_jax = use_jax
        self.inf = self.model.load_inference_interface(use_jax)

        if model.categorical_model is not None:
            self.true_finite_data = get_finite_observations_from_data(
                self.model, self.true_data)
        if model.gaussian_mixture_model is not None:
            self.true_gaussian_data = get_gaussian_observations_from_data(
                self.model, self.true_data)

    def predicted_hidden_state_log_likelihood_viterbi(self):
        """ Predict most likely hidden states with Viterbi algorithm

        Returns:
            log likelihood of most likely series of hidden states
        """
        inf = self.model.load_inference_interface(self.use_jax)
        log_likelihood = inf.predict_hidden_states_viterbi(
            self.true_data).name.split(" ")[-1]

        return float(log_likelihood)

    def validate_imputation(self, incomplete_data, data_to_verify):
        """ Return dictionary of validation metrics for imputation.

        Arguments:
            incomplete_data: dataframe with values set to nan.
            data_to_verify: dataframe with missing values imputed.

        Returns:
            Dictionary with validation metrics for imputed data against actual data.
        """
        # Make sure that integers are being cast as integers.
        float_to_int = {
            feature: "Int64"
            for feature in incomplete_data[self.model.finite_features]
            .select_dtypes("float")
        }
        incomplete_data = incomplete_data.astype(float_to_int, errors='ignore')
        data_to_verify = data_to_verify.astype(float_to_int, errors='ignore')

        cond_prob_of_hidden_states = self.conditional_log_probability_of_hidden_states(
            incomplete_data)
        val_dict = {}

        if self.model.categorical_model:
            incomplete_finite_data = get_finite_observations_from_data(
                self.model, incomplete_data)
            verify_finite_data = get_finite_observations_from_data(
                self.model, data_to_verify)

            if incomplete_finite_data[incomplete_finite_data.isna().any(
                    axis=1)].shape[0] == 0:
                val_dict['accuracy_of_imputed_finite_data'] = 1
                val_dict['relative_accuracy_of_imputed_finite_data'] = 1
                val_dict['best_possible_accuracy_of_finite_imputation'] = 1

            else:
                val_dict[
                    'accuracy_of_imputed_finite_data'] = self.accuracy_of_predicted_finite_data(
                        incomplete_finite_data, verify_finite_data)

                val_dict[
                    'relative_accuracy_of_imputed_finite_data'] = self.relative_accuracy_of_predicted_finite_data(
                        incomplete_finite_data, verify_finite_data)

                val_dict[
                    'best_possible_accuracy_of_finite_imputation'] = best_possible_accuracy_of_categorical_prediction(
                        self.true_finite_data, incomplete_finite_data)

        if self.model.gaussian_mixture_model:
            true_gaussian_data = get_gaussian_observations_from_data(
                self.model, self.true_data)
            incomplete_gaussian_data = get_gaussian_observations_from_data(
                self.model, incomplete_data)
            verify_gaussian_data = get_gaussian_observations_from_data(
                self.model, data_to_verify)

            val_dict[
                'average_relative_log_likelihood_of_imputed_gaussian_data'] = self.average_relative_log_likelihood_of_predicted_gaussian_data(
                    incomplete_gaussian_data, verify_gaussian_data,
                    cond_prob_of_hidden_states)

            val_dict[
                'average_z_score_of_imputed_gaussian_data'] = self.average_z_score_of_predicted_gaussian_data(
                    incomplete_gaussian_data, cond_prob_of_hidden_states)

        return val_dict

    def validate_forecast(self, conditioning_data, forecast):
        """ Return DiscreteHMM specific dictionary of validation metrics for imputation.

        Arguments:
            conditioning_data: observed data from which forecast is taken.
            forecast: dataframe with forecast data.

        Returns:
            Dictionary with validation metrics for forecast data against actual data.
        """
        # Data up to conditioning date and true observations for forecast horizons.
        true_data = pd.concat(
            [conditioning_data, self.true_data.loc[forecast.index]])

        forecast_data = pd.concat([conditioning_data, forecast])
        incomplete_data = forecast_data.copy()
        for idx in forecast.index:
            incomplete_data.loc[idx] = np.nan

        delta = true_data.index[1] - true_data.index[0]
        horizon_timesteps = []
        for idx in forecast.index:
            horizon_timesteps.append(
                int((idx - conditioning_data.index[-1]) / delta))

        forc = HMMForecastingTool(model=self.model, data=conditioning_data)
        cond_prob_of_hidden_states = forc.hidden_state_probability_at_horizons(
            horizon_timesteps=horizon_timesteps)
        val_dict = {}

        if self.model.categorical_model:
            forecast_finite_data = get_finite_observations_from_data(
                self.model, forecast_data)

            incomplete_finite_data = get_finite_observations_from_data(
                self.model, incomplete_data)

            incomplete_index = incomplete_finite_data[
                incomplete_finite_data.isnull().any(axis=1)].index

            val_dict[
                'accuracy_of_forecast_finite_data'] = self.accuracy_of_predicted_finite_data(
                    incomplete_finite_data, forecast_finite_data)

            val_dict[
                'relative_accuracy_of_forecast_finite_data'] = self.relative_accuracy_of_predicted_finite_data(
                    incomplete_finite_data, forecast_finite_data)

            val_dict[
                'best_possible_accuracy_of_finite_forecast'] = best_possible_accuracy_of_categorical_prediction(
                    forecast_finite_data, incomplete_finite_data)

        if self.model.gaussian_mixture_model:
            forecast_gaussian_data = get_gaussian_observations_from_data(
                self.model, forecast_data)

            incomplete_gaussian_data = get_gaussian_observations_from_data(
                self.model, incomplete_data)

            val_dict[
                'average_relative_log_likelihood_of_forecast_gaussian_data'] = self.average_relative_log_likelihood_of_predicted_gaussian_data(
                    incomplete_gaussian_data, forecast_gaussian_data,
                    np.log(cond_prob_of_hidden_states))

            val_dict[
                'average_z_score_of_forecast_gaussian_data'] = self.average_z_score_of_predicted_gaussian_data(
                    incomplete_gaussian_data,
                    np.log(cond_prob_of_hidden_states))

        return val_dict

    def conditional_log_probability_of_hidden_states(self, incomplete_data):
        """

        Arguments: 
            incomplete_data: dataframe with missing values.

        Returns: 
            Dataframe of hidden state log probabilties for timesteps with missing values.

        """

        model = self.model
        prob_df = pd.DataFrame(
            columns=[i for i in range(model.n_hidden_states)])

        # Create copy of dataframe with missing values.
        verify_data = incomplete_data.copy()

        # Get loc and iloc index for missing values.
        red_idx = list(
            incomplete_data[incomplete_data.isna().any(axis=1)].index)
        ired_idx = [0] + [
            list(incomplete_data.index).index(i) for i in red_idx
        ] + [incomplete_data.shape[0] + 1]

        # Deal with missing values in chunks.
        for i in range(len(red_idx)):

            df_pre = incomplete_data.iloc[ired_idx[i] + 1:ired_idx[i + 1]]
            # Dataframe of what is known after the missing value.
            df_post = incomplete_data.iloc[ired_idx[i + 1] + 1:ired_idx[i + 2]]

            unknown_col = list(incomplete_data.loc[red_idx[i]][
                incomplete_data.loc[red_idx[i]].isna()].index)
            known_col = [
                g for g in incomplete_data.columns if not g in unknown_col
            ]

            # Compute bracket Z star.
            inf = self.inf
            if df_pre.shape[0] == 0:
                log_prob_pre = pd.DataFrame([
                    np.log(
                        np.full(model.n_hidden_states,
                                1 / model.n_hidden_states))
                ])
            else:
                log_prob_pre = inf.observation_log_probability(df_pre)

            if df_post.shape[0] == 0:
                log_prob_post = pd.DataFrame([
                    np.log(
                        np.full(model.n_hidden_states,
                                1 / model.n_hidden_states))
                ])
            else:
                log_prob_post = inf.observation_log_probability(
                    df_post)

            alpha = inf._compute_forward_probabilities(log_prob_pre)
            beta = inf._compute_backward_probabilities(log_prob_post)

            log_p_fb = logsumexp(
                alpha[-1].reshape(-1, 1) + model.log_transition,
                axis=0) + beta[0]

            log_p_finite = np.log(
                np.full(model.n_hidden_states, 1 / model.n_hidden_states))
            log_p_gauss = np.log(
                np.full(model.n_hidden_states, 1 / model.n_hidden_states))

            if model.categorical_model:
                # Compute probability of finite observation components.
                finite_obs = incomplete_data.loc[[red_idx[i]],
                                                 list(model.finite_features)]
                known_finite = [
                    c for c in model.finite_features if c in known_col
                ]

                # If all finite observations are known...
                if len(known_finite) == len(model.finite_features):
                    finite_obs_enum = model.categorical_model.finite_values_dict_inverse[
                        str(list(np.array(finite_obs)[0]))]
                    log_p_finite = model.categorical_model.log_emission_matrix[
                        finite_obs_enum]

                # If no finite observations are known...
                elif len(known_finite) == 0:
                    possible_finite_obs_enum = list(model.finite_values.index)
                    log_p_finite = np.log(
                        np.full(model.n_hidden_states,
                                1 / model.n_hidden_states))

                # If some finite observations are known, but not all...
                else:
                    possible_finite_obs_enum = []
                    for c in known_col:
                        if c in model.finite_features:
                            possible_finite_obs_enum += list(
                                model.finite_values[model.finite_values[
                                    c] == finite_obs.loc[red_idx[i], c]].index)
                    log_p_finite = logsumexp(
                        model.categorical_model.log_emission_matrix[
                            possible_finite_obs_enum],
                        axis=0)

            if model.gaussian_mixture_model:

                # Compute probability of Gaussian observation components.
                gaussian_obs = incomplete_data.loc[[red_idx[i]],
                                                   model.continuous_features]
                known_gaussian = [
                    c for c in model.continuous_features if c in known_col
                ]

                means = model.gaussian_mixture_model.means
                covariances = model.gaussian_mixture_model.covariances
                weights = model.gaussian_mixture_model.component_weights

                # If all Gaussian observations are known...
                if len(known_gaussian) == len(model.continuous_features):
                    log_p_gauss = np.array(
                        model.gaussian_mixture_model.gaussian_log_probability(
                            gaussian_obs))[0]

                # If no Gaussian observations are known...
                elif len(known_gaussian) == 0:
                    log_p_gauss = np.log(
                        np.full(model.n_hidden_states,
                                1 / model.n_hidden_states))

                # If some Gaussian observations are known...
                else:
                    known_gauss_dim = [
                        j for j in range(len(model.continuous_features))
                        if model.continuous_features[j] in known_col
                    ]
                    for h in range(model.n_hidden_states):
                        for m in range(
                                model.gaussian_mixture_model.n_gmm_components):

                            p = stats.multivariate_normal.logpdf(
                                gaussian_obs.iloc[0, known_gauss_dim],
                                means[h][m][known_gauss_dim],
                                covariances[h][m][known_gauss_dim, :]
                                [:, known_gauss_dim],
                                allow_singular=True)

                            log_p_gauss[h] += p + np.log(weights[h][m])

            Z_star = log_p_fb + log_p_finite + log_p_gauss - logsumexp(
                log_p_fb + log_p_finite + log_p_gauss)

            prob_df.loc[red_idx[i]] = Z_star

        return prob_df

    def average_relative_log_likelihood_of_predicted_gaussian_data(
            self, incomplete_gaussian_data, verify_gaussian_data,
            conditional_log_probability_of_hidden_states):
        """Returns the difference between the log likelihood of the actual
        data and the log likelihood of the imputed data.  This is done
        using the probability density function for the conditional probability
        of the unknown part of the observation given the known part of the
        observation.  This metric is intended to be a measure of how surprised
        you should be to see the actual value relative to the imputed value.

        Arguments:
            incomplete_gaussian_data: dataframe if Gaussian observations
                with values set to nan.
            verify_gaussian_data: dataframe with missing values imputed.
            conditional_log_probability_of_hidden_states: dataframe with
                conditional log probability of hidden states given partial
                observations at all timesteps with incomplete data.

        Returns:
            float
        """
        means = self.model.gaussian_mixture_model.means
        covariances = self.model.gaussian_mixture_model.covariances
        component_weights = self.model.gaussian_mixture_model.component_weights

        incomplete_index = incomplete_gaussian_data[
            incomplete_gaussian_data.isnull().any(axis=1)].index
        verify_likelihood = np.empty(len(incomplete_index))
        true_likelihood = np.empty(len(incomplete_index))
        for i in range(len(incomplete_index)):
            idx = incomplete_index[i]
            cond_prob = np.array(
                conditional_log_probability_of_hidden_states.loc[idx])

            true_gaussian_observation = self.true_gaussian_data.loc[[idx]]
            verify_gaussian_observation = verify_gaussian_data.loc[[idx]]
            partial_gaussian_observation = incomplete_gaussian_data.loc[[idx]]

            true_prob = compute_log_likelihood_with_inferred_pdf(
                true_gaussian_observation, partial_gaussian_observation, means,
                covariances, component_weights)
            true_likelihood[i] = logsumexp(true_prob + cond_prob)

            verify_prob = compute_log_likelihood_with_inferred_pdf(
                verify_gaussian_observation, partial_gaussian_observation,
                means, covariances, component_weights)
            verify_likelihood[i] = logsumexp(verify_prob + cond_prob)

        total_true_log_likelihood = logsumexp(true_likelihood)
        total_verify_log_likelihood = logsumexp(verify_likelihood)

        return total_true_log_likelihood - total_verify_log_likelihood

    def average_z_score_of_predicted_gaussian_data(
            self, incomplete_gaussian_data,
            conditional_log_probability_of_hidden_states):
        """ Computes z score of gaussian data averaged over observations.

        Arguments:
            incomplete_gaussian_data: dataframe if Gaussian observations
                with values set to nan.
            verify_gaussian_data: dataframe with missing values imputed.
            conditional_log_probability_of_hidden_states: dataframe with
                conditional log probability of hidden states given partial
                observations at all timesteps with incomplete data.

        Returns:
            float
        """
        means = self.model.gaussian_mixture_model.means
        covariances = self.model.gaussian_mixture_model.covariances
        component_weights = self.model.gaussian_mixture_model.component_weights

        return average_z_score(
            means, covariances, component_weights, self.true_gaussian_data,
            incomplete_gaussian_data,
            np.exp(conditional_log_probability_of_hidden_states))

    def accuracy_of_predicted_finite_data(self, incomplete_finite_data,
                                          verify_finite_data):
        """ Returns ratio of correctly imputed finite values to total imputed finite values.

        Arguments:
            incomplete_finite_data: dataframe of finite data with
                some missing values.
            verify_finite_data: dataframe of finite data with missing 
                values filled in.

        Returns:
            float
        """
        true_finite_data = self.true_finite_data.loc[verify_finite_data.index]
        incomplete_index = incomplete_finite_data[
            incomplete_finite_data.isnull().any(axis=1)].index

        total_correct = np.sum(
            (true_finite_data.loc[incomplete_index] == verify_finite_data.loc[
                incomplete_index]).all(axis=1))

        return total_correct / len(incomplete_index)

    def relative_accuracy_of_predicted_finite_data(self, incomplete_finite_data,
                                                   verify_finite_data):
        """ Returns ratio of rate of accuracy in imputed data to expected rate of accuracy with random guessing.

        Arguments:
            incomplete_finite_data: dataframe of finite data with
                some missing values.
            verify_finite_data: dataframe of finite data with missing 
                values filled in.
        Returns:
            float
        """
        expected_accuracy = expected_proportional_accuracy(
            self.true_finite_data, incomplete_finite_data)

        verify_accuracy = self.accuracy_of_predicted_finite_data(
            incomplete_finite_data, verify_finite_data)

        return verify_accuracy / expected_accuracy

    def precision_recall_df_for_predicted_finite_data(self, incomplete_data,
                                                      data_to_verify):
        """ Return DataFrame with precision, recall, and proportion of finite values

        Arguments:
            incomplete_data: dataframe with values set to nan.
            data_to_verify: dataframe with missing values imputed.

        Returns:
            Dataframe with precision, recall, and proportion of imputed data against actual data.
        """
        if len(self.model.finite_features) == 0:
            return None
        else:
            incomplete_finite_data = get_finite_observations_from_data(
                self.model, incomplete_data)
            incomplete_index = incomplete_finite_data[
                incomplete_finite_data.isnull().any(axis=1)].index

            df = self.true_data.copy()
            df['tuples'] = list(
                zip(*[self.true_data[c]
                      for c in self.true_finite_data.columns]))
            proportion = (df['tuples'].value_counts() / df.shape[0]).to_dict()

            df_imputed = data_to_verify.copy()
            df_imputed['tuples'] = list(
                zip(*[data_to_verify[c]
                      for c in self.true_finite_data.columns]))

            state = df['tuples'].unique()

            precision_recall = pd.DataFrame(
                np.full((df['tuples'].nunique(), 3), np.nan),
                index=state,
                columns=['precision', 'recall', 'proportion'])

            for n, idx in enumerate(precision_recall.index):
                # pandas cannot use loc with tuples in the index
                precision_recall.iloc[n]['proportion'] = proportion[idx]

            actual = df.loc[incomplete_index, 'tuples']

            imputed = df_imputed.loc[incomplete_index, 'tuples']

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
