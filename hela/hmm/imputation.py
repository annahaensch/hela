""" Imputation methods for discrete HMMs.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .utils import *


class HMMImputationTool(ABC):

    def __init__(self, model):
        self.model = model

    def impute_missing_data(self, partial_data, method):
        """ Imputes missing values using self.method
        
        Arguments: 
            partial_data: dataframe with missing values.
            method: method for imputation, can be "hmm_argmax",
                "hmm_maximal" or "hmm_average".
        Returns: 
            Complete dataframe with missing values imputed.
        """
        if method.lower().replace(
                " ", "_") in ["hmm_argmax", "hmm_maximal", "hmm_average"]:
            imp = DiscreteHMMImputation(self.model, method)
            data = imp.hmm_imputation(partial_data)

        else:
            raise NotImplementedError("Other imputation methods"
                                      "are not yet implemented.")
        return data


class DiscreteHMMImputation(ABC):
    """ Intilialize HMM imputation object. """

    def __init__(self, model, method):
        self.model = model
        self.method = method

    def hmm_imputation(self, partial_data):

        model = self.model

        # Create copy of dataframe with missing values.
        imputed_data = partial_data.copy()

        # Get loc and iloc index for missing values.
        red_idx = list(partial_data[partial_data.isna().any(axis=1)].index)
        ired_idx = [0] + [list(partial_data.index).index(i) for i in red_idx
                         ] + [partial_data.shape[0] + 1]

        # Deal with missing values in chunks.
        for i in range(len(red_idx)):

            df_pre = partial_data.iloc[ired_idx[i] + 1:ired_idx[i + 1]]
            # Dataframe of what is known after the missing value.
            df_post = partial_data.iloc[ired_idx[i + 1] + 1:ired_idx[i + 2]]

            unknown_col = list(partial_data.loc[red_idx[i]][partial_data.loc[
                red_idx[i]].isna()].index)
            known_col = [
                g for g in partial_data.columns if not g in unknown_col
            ]

            # Compute bracket Z star.
            inf = model.load_inference_interface()
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
                finite_obs = partial_data.loc[[red_idx[i]],
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
                gaussian_obs = partial_data.loc[[red_idx[i]],
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

            for c in unknown_col:

                if c in model.finite_features:
                    # Impute finite observation.
                    log_emission = model.categorical_model.log_emission_matrix[
                        possible_finite_obs_enum, :]
                    y_star_enum = np.sum(
                        log_emission + Z_star.reshape(-1, model.n_hidden_states)
                        - logsumexp(log_emission, axis=0).reshape(
                            -1, model.n_hidden_states),
                        axis=1).argmax()
                    y_star = model.categorical_model.finite_values_dict[
                        y_star_enum][list(model.finite_features).index(c)]

                if c in model.continuous_features:
                    # Impute Gaussian observation.
                    idx = model.continuous_features.index(c)

                    if self.method == "hmm_argmax":
                        hidden_state = Z_star.argmax()
                        component = weights[hidden_state].argmax()
                        y_star = means[hidden_state][component][idx]

                    if self.method == "hmm_maximal":
                        probs = np.empty(means.shape[:2])
                        for n in range(probs.shape[0]):
                            for m in range(probs.shape[1]):
                                mu = means[n][m]
                                probs[n][m] = Z_star[n] + np.log(
                                    model.gaussian_mixture_model.
                                    component_weights[n][m]
                                ) + stats.multivariate_normal.logpdf(
                                    means[n][m], means[n][m], covariances[n][m])

                        maximal_index = list(
                            np.unravel_index(probs.argmax(), probs.shape))
                        y_star = means[maximal_index[0]][maximal_index[1]][idx]

                    if self.method == "hmm_average":
                        mu = means[:, :, idx]
                        y_star = np.sum(
                            mu * weights * np.exp(Z_star).reshape(-1, 1))

                imputed_data.loc[red_idx[i], c] = y_star

        return imputed_data
