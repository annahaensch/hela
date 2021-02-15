""" Functions that generate data for HMM training and testing.
"""

import itertools
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn import mixture


class HMMGenerativeModel(ABC):
    """ Abstract base class for generative HMM """

    def __init__(self,
                 random_state=0,
                 n_categorical_features=None,
                 categorical_values=None,
                 n_categorical_values=(),
                 n_gaussian_features=None,
                 gaussian_values=None):

        self.random = np.random.RandomState(random_state)

        self.get_categorical_hyperparameters(
            n_categorical_features=n_categorical_features,
            n_categorical_values=n_categorical_values,
            categorical_values=categorical_values)
        self.get_gaussian_hyperparameters(
            n_gaussian_features=n_gaussian_features,
            gaussian_values=gaussian_values)

    @abstractmethod
    def generate_hidden_state_sequence(self, n_observations, frequency):
        """Generate hidden states with instantiated generative model.
        """

    def get_categorical_hyperparameters(self,
                                        n_categorical_features=None,
                                        n_categorical_values=None,
                                        categorical_values=None):
        """ Returns randomly generated categorical values

        Arguments:
            n_categorical_features: (int) number of categorical features
                to inlcude.
            n_categorical_values: (list) of integer values indicating the
                number of categorical values to consider for each
                categorical feature
            categorical_values: (dataframe) with columns corresponding to
                each of the categorical features, and rows corresponding to
                each of the possible categorical observation vectors.

        Returns:
            Dataframe with columns corresponding to each of the
            categorical features, and rows corresponding to each
            of the possible categorical observation vectors.

        """
        # Check that all input values agree.
        if n_categorical_features:
            if len(n_categorical_values) > 0:
                msg = "n_categorical_features and n_categorical_values disagree"
                assert n_categorical_features == len(n_categorical_values), msg
            if categorical_values:
                msg = "categorical_features and n_categorical_values disagree"
                assert n_categorical_features == categorical_values.shape[
                    1], msg
        else:
            if categorical_values:
                msg = "categorical_values and n_categorical_values disagree"
                assert categorical_values.shape[1] == len(
                    n_categorical_values), msg

        random = self.random
        # Check if categorical_values has already been defined.
        if categorical_values is not None:
            n_categorical_features = categorical_values.shape[1]
            n_categorical_values = [
                categorical_values[c].nunique()
                for c in categorical_values.columns
            ]

        else:
            # Return None if there are no categorical features.
            if n_categorical_features == 0:
                if len(n_categorical_values) > 0:
                    msg = "n_categorical_features and n_categorical_values disagree"
                    assert n_categorical_features == len(
                        n_categorical_values), msg
                categorical_values = None

            else:
                if n_categorical_features is None:
                    if len(n_categorical_values) != 0:
                        n_categorical_features = len(n_categorical_values)
                    else:
                        n_categorical_features = random.choice(range(2, 4))
                        n_categorical_values = random.choice(
                            range(2, 5), n_categorical_features)
                else:
                    n_categorical_values = random.choice(
                        range(2, 5), n_categorical_features)

                msg = "all n_categorical_values entries must be greater than 1"
                assert np.all(np.array(n_categorical_values) > 1), msg

                values = [list(range(v)) for v in n_categorical_values]
                value_tuples = list(itertools.product(*values))
                categorical_features = [
                    "categorical_feature_{}".format(i)
                    for i in range(n_categorical_features)
                ]
                categorical_values = pd.DataFrame(
                    value_tuples, columns=categorical_features)

        self.n_categorical_features = n_categorical_features
        self.n_categorical_values = n_categorical_values
        self.categorical_values = categorical_values

    def get_gaussian_hyperparameters(self,
                                     n_gaussian_features=None,
                                     gaussian_values=None):
        """ Returns randomly generated gaussian values

        Arguments:
            n_gaussian_features: (int) number of gaussian features to
                include.

        Returns:
            Dataframe with columns corresponding to each of the
            gaussian features.
        """
        # Check that all input values agree.
        if n_gaussian_features:
            if gaussian_values:
                msg = "n_gaussian_features and gaussian_values disagree"
                assert n_gaussian_features == gaussian_values.shape[1], msg
        random = self.random

        # Return None if there are no gaussian features.
        if n_gaussian_features == 0:
            gaussian_values = None

        else:
            if n_gaussian_features is None:
                n_gaussian_features = random.choice(range(2, 4))
            gaussian_features = [
                "gaussian_feature_{}".format(i)
                for i in range(n_gaussian_features)
            ]
            gaussian_values = pd.DataFrame(columns=gaussian_features)

        self.n_gaussian_features = n_gaussian_features
        self.gaussian_values = gaussian_values

    def get_transition_matrix(self, n_hidden_states):
        """ Returns transition matrix

        Arguments:
            n_hidden_states: (int) number of hidden states.

        Returns:
            Array with shape (n_hidden_states,n_hidden_states) where
            the ij^th entry is the probability of transitioning from
            hidden state i to hidden state j.
        """
        random = self.random
        transition_matrix = np.zeros((n_hidden_states, n_hidden_states))
        for i in range(n_hidden_states):
            # This forces the transitions to be cyclic with a strong
            # preference for remaining in the current state.
            transition_matrix[i][i] = random.uniform(.9, 1)
            transition_matrix[i][(
                i + 1) % n_hidden_states] = 1 - transition_matrix[i][i]
        return transition_matrix


def data_to_discrete_fhmm_training_spec(hidden_states, ns_hidden_states, data,
                                       categorical_features, gaussian_features):
    """ Returns fhmm training spec from hidden state sequence and data.

    Arguments:
        hidden_states: (series) of hidden states (typically) generated
                using generate_hidden_state_sequence(n_observations).
        ns_hidden_states: (array) number of hidden states per system.
        data: (dataframe) data with the same index as the series
            hidden_states and columns corresponding to categorical_features
            and guassian_features.
        categorical_features: (list) of categorical features as strings.
        gaussian_features: (list) of gaussian features as strings.

    Returns:
        Training spec dictionary informed by the hidden state
        sequence and the data
    """
    spec = {"hidden_state": {"type": "finite", "count": ns_hidden_states}}
    spec["n_systems"] = len(ns_hidden_states)

    hidden_state_tuples = np.array(hidden_states.drop_duplicates())
    hidden_state_dict = {
        str(list(hidden_state_tuples[i])): i
        for i in range(len(hidden_state_tuples))
    }
    flattened_hidden_states = pd.Series(
    [hidden_state_dict[str(list(v))] for v in np.array(hidden_states)],
    index=hidden_states.index)

    observations = []
    if categorical_features:
        categorical_features.sort()
        categorical_values = []
    for feat in categorical_features:
        values = list(data[feat].unique())
        values.sort()
        observations.append({
                    "name": feat,
                    "type": "finite",
                    "values": values
                })
        categorical_values.append(values)
        value_tuples = list(itertools.product(*categorical_values))
    if gaussian_features:
        gaussian_features.sort()
        for feat in gaussian_features:
                observations.append({
                    "name": feat,
                    "type": "continuous",
                    "dist": "gaussian",
                    "dims": 1
                })
    spec["observations"] = observations    

    model_parameter_constraints = {}

    # Construct transition matrices from hidden state sequence.
    transition_matrices = []
    for n in ns_hidden_states:
        transition_matrix = np.zeros((n, n))
        transition_matrices.append(transition_matrix)
    transition_matrices = np.array(transition_matrices)

    for i in hidden_states.columns:
        transition_matrix = transition_matrices[i]
        for j in range(hidden_states[i].shape[0]-1):
            current_state = hidden_states[i][j]
            next_state = hidden_states[i][j+1]
            if ~np.isnan(current_state) and ~np.isnan(next_state):
                transition_matrix[int(current_state)][int(next_state)] += 1
        for k in range(ns_hidden_states[i]):
            if np.sum(transition_matrix[k]) == 0:
                transition_matrix[k] = np.full(ns_hidden_states[k], 1)
        transition_matrix = transition_matrix / np.sum(transition_matrix, axis=1).reshape(-1, 1)
        transition_matrices[i] = transition_matrix
    model_parameter_constraints["transition_constraints"] = transition_matrices
    
    # Construct initial state vector from hidden state sequence.
    initial_state_vector = np.zeros((len(ns_hidden_states),np.max(ns_hidden_states)))

    for i in hidden_states.columns:
        if ~np.isnan(hidden_states.iloc[0][i]):
            initial_state_vector[i][hidden_states.iloc[0][i]] = 1
        model_parameter_constraints[
            "initial_state_constraints"] = initial_state_vector

    if categorical_features:
        value_tuple_dict = {str(list(v)): e for e, v in enumerate(value_tuples)}
        emissions = pd.Series(
            [
                value_tuple_dict[str(list(i))]
                for i in np.array(data[categorical_features])
            ],
            index=data.index)
        emission_matrix = np.zeros((emissions.nunique(), np.prod(ns_hidden_states)))
        
    # emission_matrix
        for i in list(
                set.intersection(
                    set(hidden_states[~hidden_states.isna()].index),
                    set(emissions.index))):
            emission_matrix[emissions[i], int(flattened_hidden_states[i])] += 1

        for i in range(np.prod(ns_hidden_states)):
            if np.sum([e[i] for e in emission_matrix]) == 0:
                emission_matrix[:, i] = np.full(emission_matrix.shape[0], 1)
        emission_matrix = emission_matrix / np.sum(emission_matrix, axis=0)
        model_parameter_constraints["emission_constraints"] = emission_matrix
        spec["model_parameter_constraints"] = model_parameter_constraints
        
    if gaussian_features:
        gaussian_parameter_constraints = {}
        means = np.zeros((len(ns_hidden_states),
                          len(gaussian_features),
                              np.max(ns_hidden_states)))
        covariances = []
        for system in hidden_states.columns:
            for i in hidden_states[:][system].unique():
                df = data.loc[hidden_states[system][hidden_states[system]==i].index,
                          gaussian_features]
                means[system][int(i)] = np.mean(np.array(df), axis = 0)
                covariances.append(np.array(
                    np.cov(np.array(df), rowvar=False)))
                
        covariances = np.mean(np.array(covariances), axis=0)
        means = np.ma.masked_equal([np.transpose(m) for m in means], 0)

    gaussian_parameter_constraints = {
        "means": means,
        "covariances": covariances
    }
    model_parameter_constraints[
                "gaussian_parameter_constraints"] = gaussian_parameter_constraints

    spec["model_parameter_constraints"] = model_parameter_constraints

    return spec

def data_to_discrete_hmm_training_spec(hidden_states, n_hidden_states, data,
                                       categorical_features, gaussian_features,
                                       n_gmm_components):
    """ Returns hmm training spec from hidden state sequence and data.

    Arguments:
        hidden_states: (series) of hidden states (typically) generated
                using generate_hidden_state_sequence(n_observations).
        n_hidden_states: (int) number of hidden states.
        data: (dataframe) data with the same index as the series
            hidden_states and columns corresponding to categorical_features
            and guassian_features.
        categorical_features: (list) of categorical features as strings.
        gaussian_features: (list) of gaussian features as strings.
        n_gmm_components: (int) number of gmm components per hidden state.

    Returns:
        Training spec dictionary informed by the hidden state
        sequence and the data which can be used as input in the
        hmm function `DiscreteHHHConfiguration.from_spec()`.
    """
    spec = {"hidden_state": {"type": "finite", "count": n_hidden_states}}
    observations = []
    if categorical_features:
        categorical_features.sort()
        categorical_values = []
        for feat in categorical_features:
            values = list(data[feat].unique())
            values.sort()
            observations.append({
                "name": feat,
                "type": "finite",
                "values": values
            })
            categorical_values.append(values)
        value_tuples = list(itertools.product(*categorical_values))
    if gaussian_features:
        gaussian_features.sort()
        for feat in gaussian_features:
            observations.append({
                "name": feat,
                "type": "continuous",
                "dist": "gaussian",
                "dims": 1
            })
    spec["observations"] = observations

    model_parameter_constraints = {}

    # Construct transition matrix from hidden state sequence.
    transition_matrix = np.zeros((n_hidden_states, n_hidden_states))
    for i in range(hidden_states.shape[0] - 1):
        current_state = hidden_states[i]
        next_state = hidden_states[i + 1]
        if ~np.isnan(current_state) and ~np.isnan(next_state):
            transition_matrix[int(current_state)][int(next_state)] += 1
    for i in range(n_hidden_states):
        if np.sum(transition_matrix[i]) == 0:
            transition_matrix[i] = np.full(n_hidden_states, 1)
    transition_matrix = transition_matrix / np.sum(
        transition_matrix, axis=1).reshape(-1, 1)
    model_parameter_constraints["transition_constraints"] = transition_matrix

    # Construct initial state vector from hidden state sequence.
    initial_state_vector = np.zeros(n_hidden_states)
    if ~np.isnan(hidden_states[0]):
        initial_state_vector[int(hidden_states[0])] = 1
        model_parameter_constraints[
            "initial_state_constraints"] = initial_state_vector

    # Construct emission matrix from hidden state sequence and data.
    if categorical_features:
        value_tuple_dict = {str(list(v)): e for e, v in enumerate(value_tuples)}
        emissions = pd.Series(
            [
                value_tuple_dict[str(list(i))]
                for i in np.array(data[categorical_features])
            ],
            index=data.index)
        emission_matrix = np.zeros((emissions.nunique(), n_hidden_states))
        for i in list(
                set.intersection(
                    set(hidden_states[~hidden_states.isna()].index),
                    set(emissions.index))):
            emission_matrix[emissions[i], int(hidden_states[i])] += 1
        for i in range(n_hidden_states):
            if np.sum([e[i] for e in emission_matrix]) == 0:
                emission_matrix[:, i] = np.full(emission_matrix.shape[0], 1)
        emission_matrix = emission_matrix / np.sum(emission_matrix, axis=0)
        model_parameter_constraints["emission_constraints"] = emission_matrix
        spec["model_parameter_constraints"] = model_parameter_constraints

    # Determine gmm parameter constraints from data.
    if gaussian_features:
        gmm_parameter_constraints = {}
        means = np.empty((n_hidden_states, n_gmm_components,
                          len(gaussian_features)))
        covariances = np.empty((n_hidden_states, n_gmm_components,
                                len(gaussian_features), len(gaussian_features)))
        component_weights = np.empty((n_hidden_states, n_gmm_components))
        for i in hidden_states[~hidden_states.isna()].unique():
            df = data.loc[hidden_states[hidden_states == i].index,
                          gaussian_features]
            if n_gmm_components == 1:
                means[int(i)] = np.mean(np.array(df), axis=0)
                covariances[int(i)] = np.array(
                    np.cov(np.array(df), rowvar=False))
                component_weights[int(i)] = np.array([1])
            if n_gmm_components > 1:
                gmm = mixture.GaussianMixture(
                    n_components=n_gmm_components, covariance_type="full")
                gmm.fit(df)
                means[int(i)] = gmm.means_
                covariances[int(i)] = gmm.covariances_
                component_weights[int(i)] = gmm.weights_
            gmm_parameter_constraints = {
                "n_gmm_components": n_gmm_components,
                "means": means,
                "covariances": covariances,
                "component_weights": component_weights
            }
        model_parameter_constraints[
            "gmm_parameter_constraints"] = gmm_parameter_constraints

    spec["model_parameter_constraints"] = model_parameter_constraints

    return spec