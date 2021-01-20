""" Functions that generate factored data using factored HMM.
"""

import numpy as np
import pandas as pd

from .base_model import HMMGenerativeModel


class FactoredHMMGenerativeModel(HMMGenerativeModel):
    """ Class for generative factored HMM """

    def __init__(
            self,
            ns_hidden_states=None,
            random_state=0,
            n_categorical_features=0,
            # TODO @AH: incorporate categorical features
            n_categorical_values=(),
            categorical_values=None,
            n_gaussian_features=None,
            gaussian_values=None,
            n_gmm_components=1
            # TODO @AH: incorporate gaussian mixture models.
    ):
        super().__init__(
            random_state=random_state,
            n_categorical_features=n_categorical_features,
            categorical_values=categorical_values,
            n_categorical_values=n_categorical_values,
            n_gaussian_features=n_gaussian_features,
            gaussian_values=gaussian_values)
        self.ns_hidden_states = ns_hidden_states
        self.transition_matrices = None
        self.initial_state_vector = None
        self.emission_matrix = None
        self.n_gmm_components = n_gmm_components
        self.means = None
        self.covariances = None
        self.component_weights = None

        random = self.random
        transition_matrices = []
        for n in ns_hidden_states:
            transition_matrices.append(self.get_transition_matrix(n))
        self.transition_matrices = np.array(transition_matrices)

        self.initial_state_vector = np.array(
            [random.choice(h) for h in ns_hidden_states])

        # Generate categorical model parameters.
        if self.n_categorical_features > 0:
            raise NotImplementedError("Categorical emissions features are not "
                                      "yet implemented.")

        # Generate Gaussian mixture model parameters.
        if self.n_gaussian_features > 0:
            if n_gmm_components > 0:
                n_gmm_components = random.choice(range(1, 4))
            self.n_gmm_components = n_gmm_components
            self.means = self._generate_means()
            self.covariances = self._generate_covariance()

    def generate_hidden_state_sequence(self, n_observations, frequency="D"):
        """ Generates sequence of hidden states with correct distribution

        Arguments:
            n_observations: (int) number of observations.
            frequency: (str) fixed interval size for timeseries. Can take any
                of the aliases for `freq` in pd.date_range (e.g. "5H"). Default
                value is "D" for one day.

        Returns:
            dataframe of n_observations many hidden states indexed by time.
            The order and distribution of the hidden states is govered by
            'self.transition_matrices'.
        """
        random = self.random
        hidden_states = np.empty((n_observations, len(self.ns_hidden_states)))
        hidden_states[0] = self.initial_state_vector
        for n in range(len(self.ns_hidden_states)):
            transition_matrix = self.transition_matrices[n]
            for i in range(n_observations - 1):
                # Use discrete inverse transform method to sample hidden states
                u = random.uniform()
                current_state = int(hidden_states[i][n])
                cumulative_prob = np.cumsum(transition_matrix[current_state])
                new_state = int(np.argmax(cumulative_prob >= u))
                hidden_states[i + 1][n] = new_state

        df = pd.DataFrame(
            np.array(hidden_states),
            index=pd.date_range(
                "2020-08-01", freq=frequency, periods=n_observations))

        for col in df.columns:
            df[col] = df[col].astype(int)

        return df

    def generate_observations(self, hidden_states):
        """ Returns dataframe of observations with correct distributions

        Arguments:
            hidden_states: (series) of hidden states (typically) generated
                using self.generate_hidden_state_sequence(n_observations)

        Returns:
            Dataframe with observations governed by hidden state dynamics
            and categorical and gaussian feature parameters.
        """
        random = self.random

        # Generated categorical data.
        df_categorical = pd.DataFrame()
        if self.n_categorical_features > 0:
            raise NotImplementedError("Categorical emissions features are not "
                                      "yet implemented.")

        # Generate gaussian data.
        df_gaussian = pd.DataFrame()
        if self.n_gaussian_features > 0:
            means = self.means
            covariances = self.covariances
            observation_sequence = []
            for _, row in hidden_states.iterrows():
                hidden_state = np.array(row)
                mean = np.sum(
                    [[m[hidden_state[i]] for m in means[i]]
                     for i in range(len(hidden_state))],
                    axis=0)
                observation_sequence.append(
                    random.multivariate_normal(mean, covariances))
            df_gaussian = pd.DataFrame(
                observation_sequence,
                index=hidden_states.index,
                columns=self.gaussian_values.columns)

        return df_categorical.join(df_gaussian, how="outer")

    def _generate_means(self):
        """ Returns array of means

        Returns:
            Array where entry [m,d,n] is the contribution of hidden state n
            in hmm system m to gaussian dimension d.
        """
        means = []
        for n in self.ns_hidden_states:
            weights = np.random.uniform(-3, 3, (self.n_gaussian_features, n))
            means.append(weights)

        return np.array(means)

    def _generate_covariance(self):
        """ Returns covariance array with dim. n_gaussian_features x n_gaussian_features
        """
        return np.identity(self.n_gaussian_features) * np.random.uniform(
            0, 2, self.n_gaussian_features)

    def flatten_hidden_state_sequence(self, hidden_states):
        """ Return series of flattened hidden states.

        Arguments:
            hidden_states: (series) of hidden states which are typically
                generated by `self.generate_hidden_state_sequence()`.

        Returns:
            Series of flattened hidden state values corresponding to
            the hidden state vectors in hidden_states.
        """
        hidden_state_tuples = np.array(hidden_states.drop_duplicates())

        hidden_state_dict = {
            str(list(hidden_state_tuples[i])): i
            for i in range(len(hidden_state_tuples))
        }

        return pd.Series(
            [hidden_state_dict[str(list(v))] for v in np.array(hidden_states)],
            index=hidden_states.index)
