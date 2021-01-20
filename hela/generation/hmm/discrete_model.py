""" Functions that generate data for discrete hidden Markov models.
"""

import numpy as np
import pandas as pd

from .base_model import HMMGenerativeModel


class DiscreteHMMGenerativeModel(HMMGenerativeModel):
    """ Class for generative discrete HMM

    Arguments:
        random_state: (int) seed value for RandomState.
        n_hidden_states: (int) number of discrete hidden states.
        n_categorical_features: (int) number of categorical features to
            generate. If argument is left as the default None, then the
            number of categorical features will be chosen randomly.
        n_categorical_values: (list) list of integers with length
            n_categorical_features indicating the number of categorical
            values to consider for each categorical feature. If left as
            default, [], list will be randomly generated using
            n_categorical_features.
        categorical_values: (dataframe) columns corresponding to each of
            the categorical features, rows corresponding to each of the
            categorical observation tuples. If left as default None,
            dataframe will be generated using n_categorical_features and
            n_categorical_values.
        n_gaussian_features: (int) number of gaussian features to generate,
            if argument is left as the default None, then the number of
            gaussian features will be chosen randomly.
        gaussian_features: (dataframe) columns corresponding to each of the
            gaussian features. If left as default None, dataframe will be
            generated using n_gaussian_features.
        n_gmm_components: (int) number of gaussian mixture model components
            to generate, if argument is left as the default None, then the
            number of gaussian mixture model components will be chosen
            randomly.

    Returns:
        DiscreteHMMGenerativeModel object for generating HMM data.
    """

    def __init__(self,
                 random_state=0,
                 n_hidden_states=3,
                 n_categorical_features=None,
                 n_categorical_values=(),
                 categorical_values=None,
                 n_gaussian_features=None,
                 gaussian_values=None,
                 n_gmm_components=None):
        super().__init__(
            random_state=random_state,
            n_categorical_features=n_categorical_features,
            categorical_values=categorical_values,
            n_gaussian_features=n_gaussian_features,
            n_categorical_values=n_categorical_values,
            gaussian_values=None)

        self.n_hidden_states = n_hidden_states
        self.transition_matrix = self.get_transition_matrix(n_hidden_states)

        random = self.random

        # Generate initial state vector.
        initial_state = random.choice(n_hidden_states)
        self.initial_state_vector = [
            0 if i != initial_state else 1 for i in range(n_hidden_states)
        ]

        # Generate categorical model parameters.
        if self.n_categorical_features > 0:
            self.emission_matrix = self._generate_emission_matrix()

        # Generate Gaussian mixture model parameters.
        if self.n_gaussian_features > 0:
            if n_gmm_components is None:
                n_gmm_components = random.choice(range(1, 4))
            self.n_gmm_components = n_gmm_components
            self.means = random.choice(
                range(-5, 5),
                (n_hidden_states, n_gmm_components, self.n_gaussian_features))
            self.covariances = self._generate_covariances()
            self.component_weights = self._generate_component_weights()

    def generative_model_to_discrete_hmm_training_spec(self):
        """ Returns dictionary training spec suitable for HMM training.

        N.B.: This training spec will be suitable input for the hmm
            function `DiscreteHMMConfiguration.from_spec()`.
        """
        training_spec = {
            'hidden_state': {
                'type': 'finite',
                'count': self.n_hidden_states
            }
        }

        model_parameter_constraints = {
            'transition_constraints': self.transition_matrix
        }
        model_parameter_constraints[
            'initial_state_constraints'] = self.initial_state_vector

        observations = []
        if self.n_categorical_features > 0:
            categorical_values = self.categorical_values
            for c in categorical_values.columns:
                observations.append({
                    'name': c,
                    'type': 'finite',
                    'values': categorical_values[c].unique()
                })
            model_parameter_constraints[
                'emission_constraints'] = self.emission_matrix

        if self.n_gaussian_features > 0:
            for g in self.gaussian_values.columns:
                observations.append({
                    'name': g,
                    'type': 'continuous',
                    'dist': 'gaussian',
                    'dims': 1
                })
            gmm_parameter_constraints = {
                'n_gmm_components': self.n_gmm_components
            }
            gmm_parameter_constraints['means'] = self.means
            gmm_parameter_constraints['covariances'] = self.covariances
            gmm_parameter_constraints[
                'component_weights'] = self.component_weights

            model_parameter_constraints[
                'gmm_parameter_constraints'] = gmm_parameter_constraints

        training_spec[
            'model_parameter_constraints'] = model_parameter_constraints
        training_spec['observations'] = observations

        return training_spec

    def generate_hidden_state_sequence(self, n_observations, frequency="D"):
        """ Generates sequence of hidden states with correct distribution

        Arguments:
            n_observations: (int) number of observations.
            frequency: (str) fixed interval size for timeseries. Can take any
                of the aliases for `freq` in pd.date_range (e.g. "5H"). Default
                value is "D" for one day.

        Returns:
            Series of n_observations many hidden states indexed by time.
            The order and distribution of the hidden states is govered by
            'self.transition_matrix'.
        """
        random = self.random
        initial_state_vector = self.initial_state_vector
        transition_matrix = self.transition_matrix

        hidden_states = [int(np.argmax(initial_state_vector))]
        for i in range(n_observations - 1):
            # Use discrete inverse transform method to sample hidden states
            u = random.uniform()
            current_state = hidden_states[i]
            cumulative_prob = np.cumsum(transition_matrix[current_state])
            new_state = np.argmax(cumulative_prob >= u)
            hidden_states.append(new_state)

        return pd.Series(
            hidden_states,
            index=pd.date_range(
                "2020-08-01", freq=frequency, periods=n_observations))

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
        n_observations = hidden_states.shape[0]

        # Generated categorical data.
        df_categorical = pd.DataFrame()
        if self.n_categorical_features > 0:
            emission_matrix = self.emission_matrix
            categorical_values = self.categorical_values
            observation_sequence = []
            for i in range(n_observations):
                # Use discrete inverse transform method to sample hidden states.
                u = random.uniform()
                current_state = hidden_states[i]
                cumulative_prob = np.cumsum(
                    [e[current_state] for e in emission_matrix])
                observation = np.argmax(cumulative_prob >= u)
                observation_sequence.append(
                    list(categorical_values.loc[observation]))

            columns = categorical_values.columns
            df_categorical = pd.DataFrame(
                observation_sequence,
                columns=columns,
                index=hidden_states.index)

        # Generate gaussian data.
        df_gaussian = pd.DataFrame()
        if self.n_gaussian_features > 0:
            means = self.means
            covariances = self.covariances
            component_weights = self.component_weights
            df = pd.DataFrame(hidden_states, columns=['hidden_state'])
            component_sequence = []
            for i in range(n_observations):
                # Use discrete inverse transform method to sample hidden states.
                u = random.uniform()
                current_state = hidden_states[i]
                cumulative_prob = np.cumsum(component_weights[current_state])
                component = np.argmax(cumulative_prob >= u)
                component_sequence.append(component)
            df['gmm_component'] = component_sequence

            observation_sequence = []
            for idx in df.index:
                hidden_state = df.loc[idx, 'hidden_state']
                gmm_component = df.loc[idx, 'gmm_component']
                observation_sequence.append(
                    np.random.multivariate_normal(
                        means[hidden_state][gmm_component],
                        covariances[hidden_state][gmm_component]))

            columns = self.gaussian_values.columns
            df_gaussian = pd.DataFrame(
                observation_sequence,
                columns=columns,
                index=hidden_states.index)

        return df_categorical.join(df_gaussian, how="outer")

    def _generate_emission_matrix(self):
        """ Returns emission matrix with constraints.
        """
        random = self.random
        n_hidden_states = self.n_hidden_states
        categorical_values = self.categorical_values

        values = list(categorical_values.index)

        if len(values) >= n_hidden_states:
            most_likely_emission = random.choice(
                values, n_hidden_states, replace=False)
        else:
            most_likely_emission = random.choice(values, n_hidden_states)
        emission_matrix = np.zeros((n_hidden_states, len(values)))
        for i in range(n_hidden_states):
            emission_prob = random.uniform(0, 1, len(values))
            most_likely_prob = random.uniform(.75, 1)
            emission_prob[most_likely_emission[i]] = 0

            emission_prob = (1 - most_likely_prob) * (
                emission_prob / np.sum(emission_prob))
            emission_prob[most_likely_emission[i]] = most_likely_prob
            emission_matrix[i] = emission_prob

        return emission_matrix.transpose()

    def _generate_covariances(self):
        """ Returns array of symmetric covariance matrices for the gmm.
        """
        n_hidden_states = self.n_hidden_states
        n_gaussian_features = self.n_gaussian_features
        n_gmm_components = self.n_gmm_components

        cov = []
        for _ in range(n_hidden_states):
            cov_i = []
            for _ in range(n_gmm_components):
                cov_i.append(
                    np.identity(n_gaussian_features) * np.random.uniform(
                        0, 2, n_gaussian_features))
            cov.append(cov_i)
        return np.array(cov)

    def _generate_component_weights(self):
        """ Returns array of gaussian mixture model component weights.
        """
        random = self.random
        n_hidden_states = self.n_hidden_states
        n_gmm_components = self.n_gmm_components

        if n_gmm_components == 1:
            component_weights = np.ones((n_hidden_states, n_gmm_components))

        else:
            if n_gmm_components >= n_hidden_states:
                most_likely_component = random.choice(
                    n_gmm_components, n_hidden_states, replace=False)
            else:
                most_likely_component = random.choice(n_gmm_components,
                                                      n_hidden_states)

            component_weights = np.zeros((n_hidden_states, n_gmm_components))
            for i in range(n_hidden_states):
                weights = random.uniform(0, 1, n_gmm_components)
                most_likely_weight = random.uniform(.75, 1)
                weights[most_likely_component[i]] = 0
                weights = (1 - most_likely_weight) * (weights / np.sum(weights))
                weights[most_likely_component[i]] = most_likely_weight
                component_weights[i] = weights

        return component_weights


def model_to_discrete_generative_spec(model):
    """ Creates a generative model specification from DiscreteHMM object.

    Arguments:
        model: DiscreteHMM object

    Returns:
        DiscreteHMMGenerativeModel corresponding to the DiscreteHMM.
    """
    spec = DiscreteHMMGenerativeModel(
        n_hidden_states=model.n_hidden_states,
        n_categorical_features=0,
        n_gaussian_features=0)

    spec.transition_matrix = np.exp(model.log_transition)
    spec.initial_state_vector = np.exp(model.log_initial_state)

    if model.categorical_model:
        categorical_features = model.categorical_model.finite_features
        categorical_features.sort()
        spec.categorical_values = model.categorical_model.finite_values
        spec.n_categorical_values = np.array([
            spec.categorical_values[c].nunique()
            for c in spec.categorical_values.columns
        ])
        spec.n_categorical_features = len(categorical_features)
        spec.emission_matrix = np.exp(
            model.categorical_model.log_emission_matrix)

    if model.gaussian_mixture_model:
        gaussian_features = model.gaussian_mixture_model.gaussian_features
        gaussian_features.sort()
        spec.gaussian_values = pd.DataFrame(columns=gaussian_features)
        spec.n_gaussian_features = len(gaussian_features)
        spec.n_gmm_components = model.gaussian_mixture_model.n_gmm_components
        spec.means = model.gaussian_mixture_model.means
        spec.covariances = model.gaussian_mixture_model.covariances
        spec.component_weights = model.gaussian_mixture_model.component_weights

    return spec
