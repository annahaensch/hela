from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from dask.distributed import Client, wait
from scipy import stats
from scipy.special import logsumexp

from .base_models import LOG_ZERO
from .discrete_models import GaussianMixtureModel
from .utils import *


def partition_tasks(labels, tasks, method=None, weights=None):
    """Divide tasks roughly evenly into partitions.

    Args:
        labels (seq): a sequence of hashable partition labels
        tasks (seq): a sequence of hashable task labels
        method (str): one of {'greedy', 'round_robin'}
          - greedy: use a freedy algorithm to evenly divide work by weight;
            default if weights are given.
          - round_robin: ignore weights and allocate tasks by round robin;
            default if no weights are given
        weights (seq): sequence of weights of the same length as tasks

    Returns:
        (dict) mapping partition labels to a list of assigned task labels
    """
    if len(tasks) == len(labels) == 0:
        return {}
    elif len(tasks) > len(labels) == 0:
        raise ValueError("Cannot assign tasks to 0 partitions")
    if method is None:
        method = "round_robin" if weights is None else "greedy"

    if method == "greedy":
        if weights is None:
            weights = np.ones(len(tasks))
        partition = {label: [] for label in labels}
        queue = list((0, label) for label in labels)
        heapq.heapify(queue)
        top = heapq.heappop(queue)
        for weight, task in sorted(zip(weights, tasks), reverse=True):
            partition[top[1]].append(task)
            top = heapq.heappushpop(queue, (top[0] + weight, top[1]))
        return partition
    elif method == "round_robin":
        return {label: tasks[i::len(labels)] for i, label in enumerate(labels)}
    else:
        raise ValueError("Unknown partition method '{}', expected one of "
                         "{{'greedy', 'round_robin'}}".format(method))


class SyncMaster(object):
    '''Base class for distributing computations that use a fixed list of data.

    This class is heavily based on the design of SyncAdmmMaster in admm.py, but
    since the `run` method is different and the `broadcast` is abstract, and
    since it is not actually an ADMM optimization implemented here, this class
    does not inherit from it.'''

    def __init__(self,
                 data,
                 objective,
                 partition_labels,
                 objective_args=(),
                 objective_kwargs=None):

        self.data = data
        self.partition_count = len(partition_labels)
        self.partitions = partition_tasks(partition_labels, data)

        # Objective is the actual function to be distributed
        self.objective = objective
        self.objective_args = objective_args
        self.objective_kwargs = objective_kwargs or {}

    @abstractmethod
    def run(self, model, n_em_iterations):
        """Run calculation, i.e. loop over iters, broadcast, and combine.

        Arguments:
            model: untrained HMM model
            n_em_iterations: number of EM iterations

        Returns:
            new_model: trained HMM model
        """

    @abstractmethod
    def broadcast(self, inference):
        """Broadcast the global parameters to the workers and collect updates

        Args:
            inference: global inference across all workers

        Returns:
            model_update_statistics (dict):
                1. log_initial_state
                2. log_transition
                3. log_emission_matrix
                4. updated GMM model
        """


class HMMSyncMaster(SyncMaster):

    def run(self, model, n_em_iterations):
        """ Base class method for distributed EM learning algorithm.

        Arguments:
            model: untrained DiscreteHMM object
            n_em_iterations: number of em iterations to perform.

        Returns:
            Trained instance of HiddenMarkovModel belonging to the same child class as model.  Also returns em training results.
        """

        new_model = model.model_config.to_model()

        for _ in range(n_em_iterations):

            inference = new_model.load_inference_interface()
            # E step and M step
            sufficient_statistics = self.broadcast(inference)

            # Updating model
            new_model = self.update_model(sufficient_statistics, new_model)
            self.model_results.append(new_model)

        return new_model

    def update_model(self, sufficient_statistics, model):
        """ Method for updating HiddenMarkovModel

        Arguments:
            sufficient_statistics (dict): updated parameters for:
                1. log_initial_state
                2. log_transition
                3. log_emission_matrix
                4. means, covariances, weights

            model: Current HiddenMarkovModel

        Returns:
            Next instance of HiddenMarkovModel
        """

        new_model = model.model_config.to_model()
        if new_model.model_config.model_parameter_constraints.get(
                'initial_state_constraints', None) is None:
            new_model.log_initial_state = sufficient_statistics[
                'log_initial_state']

        new_model.log_transition = sufficient_statistics['log_transition']

        if new_model.categorical_model is not None:
            new_model.categorical_model.log_emission_matrix = sufficient_statistics[
                'log_emission_matrix']

        if new_model.gaussian_mixture_model is not None:
            new_gmm = GaussianMixtureModel(
                n_hidden_states=model.gaussian_mixture_model.n_hidden_states,
                n_gmm_components=model.gaussian_mixture_model.n_gmm_components,
                dims=model.gaussian_mixture_model.dims,
                gaussian_features=model.gaussian_mixture_model.gaussian_features
            )
            new_gmm.means = sufficient_statistics['means']
            new_gmm.covariances = sufficient_statistics['covariances']
            new_gmm.component_weights = sufficient_statistics['weights']

        new_model.gaussian_mixture_model = new_gmm

        return new_model


class DistributedLearningAlgorithm(HMMSyncMaster):

    def __init__(self, client, data, objective, model, **kwargs):
        # Get the list of workers from the scheduler
        self.client = client
        self.partition_labels = list(client.scheduler_info()["workers"].keys())
        super().__init__(
            data=data,
            objective=objective,
            partition_labels=self.partition_labels,
            **kwargs)

        self.model_results = []

        # Initialize worker state. This is where data loading occurs
        scattered = self.client.scatter(list(self.partitions.values()))
        self.partition_states = {
            partition:
            self.client.submit(hmm_init, self.objective, tasks, model,
                               self.objective_args, self.objective_kwargs)
            for partition, tasks in zip(self.partitions.keys(), scattered)
        }

        self.update_params = {
            partition: None
            for partition, tasks in zip(self.partitions.keys(), scattered)
        }

    def broadcast(self, inference):
        """Broadcast data to workers and collect updates.
        """
        partition_count = self.partition_count
        # Create a new future with updated inference including log probability for each partition
        for partition, state in self.partition_states.items():
            self.partition_states[partition] = self.client.submit(
                calc_hmm_log_prob, inference, state)

        # First worker initializes forward pass of forward-backward algorithm
        worker = self.partition_labels[0]
        state = self.partition_states[worker]
        first_alpha = self.client.submit(forward_init, state, workers=worker)
        wait(first_alpha)

        # Last worker initializes backward pass of forward-backward algorithm
        backward_worker = self.partition_labels[partition_count - 1]
        backward_state = self.partition_states[backward_worker]
        beta_carry = self.client.submit(
            backward_init, backward_state, workers=backward_worker)
        wait(beta_carry)

        # Remaining workers calculate forward and backward probabilities
        for i in range(1, partition_count):

            worker = self.partition_labels[i]
            forward_state = self.partition_states[worker]
            first_alpha = self.client.submit(
                calc_forward_probabilities,
                forward_state,
                first_alpha,
                workers=worker)
            wait(first_alpha)

            backward_worker = self.partition_labels[partition_count - i - 1]
            backward_state = self.partition_states[backward_worker]
            beta_carry = self.client.submit(
                calc_backward_probabilities,
                backward_state,
                beta_carry,
                workers=backward_worker)
            wait(beta_carry)

        # Create new future so inference includes gamma, xi, gamma_by_component
        for partition, state in self.partition_states.items():
            self.partition_states[partition] = self.client.submit(
                calc_sufficient_statistics, state)
        # Create new future to update model parameters
        for partition, state in self.partition_states.items():
            self.update_params[partition] = self.client.submit(
                calc_update_parameters, state)

        results = self.client.gather([
            self.client.submit(lambda s: s[1], state)
            for partition, state in self.update_params.items()
        ])

        # Collect the results into arrays for processing.
        first = results[0]
        log_initial_state = first['initial_state']
        lse_gamma = np.zeros(
            ([partition_count] + list(first['lse_gamma'][0].shape)))
        lse_gamma_transition = np.zeros(
            ([partition_count] + list(first['lse_gamma'][1].shape)))

        lse_xi = np.zeros(([partition_count] + list(first['lse_xi'].shape)))

        lse_gamma_by_component = np.zeros(
            ([partition_count] + list(first['lse_gamma_by_component'].shape)))

        if inference.model.categorical_model is not None:
            log_emissions = np.zeros(
                ([partition_count] + list(first['log_emission_matrix'].shape)))

        if inference.model.gaussian_mixture_model is not None:
            means_numerator = np.zeros(
                ([partition_count] + list(first['means'][0].shape)))
            means_denominator = np.zeros(
                ([partition_count] + list(first['means'][1].shape)))
            covariances_numerator = np.zeros(
                ([partition_count] + list(first['covariances'][0].shape)))
            covariances_denominator = np.zeros(
                ([partition_count] + list(first['covariances'][1].shape)))

        for i in range(len(results)):
            if i != len(results) - 1:
                lse_gamma_transition[i] = results[i]['lse_gamma'][0]
            else:
                lse_gamma_transition[i] = results[i]['lse_gamma'][1]

            lse_gamma[i] = results[i]['lse_gamma'][0]
            lse_xi[i] = results[i]['lse_xi']
            lse_gamma_by_component[i] = results[i]["lse_gamma_by_component"]

            if inference.model.categorical_model is not None:
                log_emissions[i] = results[i]['log_emission_matrix']

            if inference.model.gaussian_mixture_model is not None:
                means_numerator[i] = results[i]['means'][0]
                means_denominator[i] = results[i]['means'][1]
                covariances_numerator[i] = results[i]['covariances'][0]
                covariances_denominator[i] = results[i]['covariances'][1]

        update_gamma = logsumexp(lse_gamma, axis=0)
        log_transition = logsumexp(
            lse_xi, axis=0) - logsumexp(
                lse_gamma_transition, axis=0)
        update_gamma_by_component = logsumexp(lse_gamma_by_component, axis=0)
        model_update_statistics = {
            'log_initial_state': log_initial_state,
            'log_transition': log_transition
        }

        if inference.model.categorical_model is not None:
            update_log_emission = logsumexp(log_emissions, axis=0)
            update_log_emission -= update_gamma.reshape(update_gamma.shape[0])
            model_update_statistics.update({
                'log_emission_matrix':
                update_log_emission
            })

        if inference.model.gaussian_mixture_model is not None:
            means_numerator = np.sum(means_numerator, axis=0)
            means_denominator = np.sum(means_denominator, axis=0)

            covariances_numerator = np.sum(covariances_numerator, axis=0)
            covariances_denominator = np.sum(covariances_denominator, axis=0)

            update_weights = np.exp(update_gamma_by_component - update_gamma)

            means_update = np.zeros_like(means_numerator)
            covariance_update = np.zeros_like(covariances_numerator)

            for i in range(means_update.shape[0]):
                for m in range(means_update.shape[1]):
                    means_update[i, m] = means_numerator[
                        i, m] / means_denominator[i, m]
                    covariance_update[i, m] = covariances_numerator[
                        i, m] / covariances_denominator[i, m]

            model_update_statistics.update({
                'means': means_update,
                'covariances': covariance_update,
                'weights': update_weights
            })

        return model_update_statistics


class EMObjective(object):
    """Each worker in distributed fit does this work which is later gathered
    to update model parameters. Many methods are similar to those found in discrete_models.py
    but take out computations that can't be done in a distributed way and push
    them until after distributed computation is completed.
    """

    def __init__(self, data, model):
        # Dask sends out each dataframe of input data
        # as a list of one dataframe.
        if len(data) != 1:
            raise RuntimeError(
                "Dask workers must be equal to number of partitions of data")
        self.data = data[0]

        if model.categorical_model:
            self.finite_state_data = get_finite_observations_from_data_as_enum(
                model, self.data)
        if model.gaussian_mixture_model:
            self.gaussian_data = get_gaussian_observations_from_data(
                model, self.data)

        self.beta_carry = None

    def find_log_prob(self, inference):
        """ Updates inference with log probabilities of hidden states

        Arguments:
            inference: DiscreteHMMInferenceResults object

        Returns:
            inference updated with array of log probabilites of hidden states
        """

        inference = inference.model.load_inference_interface()
        log_prob = inference.observation_log_probability(self.data)
        inference.log_probability = log_prob

        return inference

    def find_xi(self, inference, beta_carry):
        """ Updates DiscreteHMMInferenceResults with xi

        Arguments:
            inference: DiscreteHMMInferenceResults object
            beta_carry: (beta[t+1], log_prob[t+1]) from next partition of data,
            this is None when calculating xi for the last partition of data.

        Returns:
            inference object updated with array of list of xi values.

            List of xi values where xi[t][i,j] gives the log probability
            of hidden state i occuring at time t and j occuring at time t+1,
            given observations and current HMM parameters.
        """

        n_hidden_states = inference.model.n_hidden_states
        log_probability = np.asarray(inference.log_probability)
        gamma = inference.gamma
        beta = inference.beta
        log_transition = inference.model.log_transition
        data = self.data

        if beta_carry is not None:
            beta_, log_prob_ = beta_carry
            xi = np.empty((data.shape[0], n_hidden_states, n_hidden_states))
            xi[-1] = gamma[-1].reshape(
                -1, 1) + log_transition + log_prob_ + beta_ - beta[-1].reshape(
                    -1, 1)
        else:
            xi = np.empty((data.shape[0] - 1, n_hidden_states, n_hidden_states))
        for t in range(data.shape[0] - 1):
            xi[t] = gamma[t].reshape(
                -1, 1
            ) + log_transition + log_probability[t +
                                                 1] + beta[t +
                                                           1] - beta[t].reshape(
                                                               -1, 1)

        inference.xi = xi
        return inference

    def find_gamma_by_component(self, inference):
        """ Updates DiscreteHMMInferenceResults with gamma by component values

        Arguments:
            inference: DiscreteHMMInferenceResults object

        Returns:
            inference object updated with array of gamma by component values.

            Where gamma by component is array where entry [i,t,m] is the probability
            of being in hidden state i and gmm component m at time t.
        """

        n_hidden_states = inference.model.n_hidden_states
        n_components = inference.model.gaussian_mixture_model.n_gmm_components
        weights = inference.model.gaussian_mixture_model.component_weights
        data = self.data

        log_weights = np.log(
            weights,
            out=np.zeros_like(weights) + LOG_ZERO,
            where=(weights != 0))

        gaussian_data = self.gaussian_data
        log_probability = np.array(
            inference.model.gaussian_mixture_model.gaussian_log_probability(
                gaussian_data))
        log_probability_by_component = np.array(
            inference.model.gaussian_mixture_model.gaussian_log_probability_by_component(
                gaussian_data))

        gamma = inference.gamma
        gamma_by_component = np.empty((n_hidden_states, data.shape[0],
                                       n_components))
        for i in range(n_hidden_states):
            gamma_by_component[i] = np.array([g[i] for g in gamma]).reshape(
                -1, 1
            ) + log_probability_by_component[i] + log_weights[i] - np.array(
                [l[i] for l in log_probability]).reshape(-1, 1)

        inference.gamma_by_component = gamma_by_component

        return inference

    def find_update_params(self, inference):
        """ Sums across axes of inference results as preprocessing for M step of finding update parameters

        Arguments:
            inference: DiscreteHMMInferenceResults object which has values for:
                    1. gamma
                    2. xi
                    3. gamma_by_component

        Returns:
            dictionary of results for calculating update matrices in M step.
        """

        gamma = inference.gamma
        xi = inference.xi
        gamma_by_component = inference.gamma_by_component

        lse_gamma = (logsumexp(gamma, axis=0).reshape(-1, 1),
                     logsumexp(gamma[:-1], axis=0).reshape(-1, 1))
        lse_xi = logsumexp(xi, axis=0)
        lse_gamma_by_component = logsumexp(gamma_by_component, axis=1)

        results = {
            'initial_state': gamma[0],
            'lse_gamma': lse_gamma,
            'lse_xi': lse_xi,
            'lse_gamma_by_component': lse_gamma_by_component
        }

        if inference.model.categorical_model is not None:
            log_emission_matrix = self.update_log_emission_matrix(inference)
            results.update({'log_emission_matrix': log_emission_matrix})

        if inference.model.gaussian_mixture_model is not None:
            means = self.update_means(inference)
            covariances = self.update_covariances(inference)
            results.update({'means': means, 'covariances': covariances})

        return results

    def update_log_emission_matrix(self, inference):
        """ Finds partial log_emission_matrix for each worker

        Arguments:
            inference: DiscreteHMMInferenceResults object

        Returns:
            partial unnormalized log emission matrix
        """

        model = inference.model.categorical_model
        gamma = inference.gamma
        finite_state_data = self.finite_state_data
        n_finite_states = model.finite_values.shape[0]
        log_emission_matrix = np.full(
            (np.array(model.log_emission_matrix).shape), LOG_ZERO)

        for l in range(n_finite_states):
            if l in finite_state_data.unique():
                log_emission_matrix[l] = logsumexp(
                    np.array([
                        gamma[idx]
                        for idx in range(finite_state_data.shape[0])
                        if finite_state_data.iloc[idx] == l
                    ]),
                    axis=0)

        return log_emission_matrix

    def update_means(self, inference):
        """ Finds numerator and denominator used for means update calculation

        Arguments:
            inference: DiscreteHMMInferenceResults object

        Returns:
            means_numerator: (np.array)
            means_denominator: (np.array)
        """

        gaussian_model = inference.model.gaussian_mixture_model
        gaussian_data = self.gaussian_data
        gamma_by_component = inference.gamma_by_component

        n_hidden_states = gaussian_model.n_hidden_states
        n_gmm_components = gaussian_model.n_gmm_components
        means_numerator = np.zeros_like(np.array(gaussian_model.means))
        means_denominator = np.zeros((n_hidden_states, n_gmm_components))

        for i in range(n_hidden_states):
            for m in range(n_gmm_components):
                exp_gbc = np.array(
                    [g[m] for g in np.exp(gamma_by_component)[i]]).reshape(
                        -1, 1)
                means_numerator[i, m] = np.sum(
                    exp_gbc * np.asarray(gaussian_data), axis=0)
                means_denominator[i, m] = np.sum(exp_gbc)

        return means_numerator, means_denominator

    def update_covariances(self, inference):
        """ Finds numerator and denominator used for covariances update calculation

        Arguments:
            inference: DiscreteHMMInferenceResults object

        Returns:
            covariances_numerator: (np.array)
            covariances_denominator: (np.array)
        """

        gaussian_model = inference.model.gaussian_mixture_model
        gaussian_data = self.gaussian_data
        gamma_by_component = inference.gamma_by_component

        n_hidden_states = gaussian_model.n_hidden_states
        n_gmm_components = gaussian_model.n_gmm_components
        means = gaussian_model.means
        covariances_numerator = np.zeros_like(
            np.array(gaussian_model.covariances))
        covariances_denom = np.zeros((n_hidden_states, n_gmm_components))

        for i in range(n_hidden_states):
            for m in range(n_gmm_components):
                error = np.array(gaussian_data) - means[i][m]
                error_prod = np.array([
                    error[t].reshape(-1, 1) @ error[t].reshape(1, -1)
                    for t in range(len(error))
                ])
                gamma = np.array([g[m] for g in gamma_by_component[i]]).reshape(
                    -1, 1, 1)
                covariances_numerator[i][m] = np.sum(gamma * error_prod, axis=0)
                covariances_denom[i][m] = np.sum(gamma, axis=0)
        return covariances_numerator, covariances_denom


def distributed_init(partitioned_data, model, client):
    """ Initialization method for distributed EM learning algorithm.

    Arguments:
        partitioned_data: list of dataframes with hybrid data for training.
        n_em_iterations: number of em iterations to perform.

    Returns:
        Initialized DistributedLearningAlgorithm
    """
    master = DistributedLearningAlgorithm(
        client=client,
        data=partitioned_data,
        objective=EMObjective,
        model=model)

    return master


def hmm_init(objective_factory, data, model, args, kwargs):
    """Initialize the state of an ADMM worker.

    Args:
        objective_factory: construct an objective function for the target
            problem with the provided data partition. For HMM it is EMObjective.
        data: Tasks partitioned by Dask
        args (tuple): extra positional arguments passed to objective factory
        kwargs (dict): extra keyword arguments passed to objective factory
    Returns:
        ADMM state tuple with objective and data.
    """
    objective = objective_factory(data, model, *args, **kwargs)
    return (objective, data)


def calc_hmm_log_prob(inference, state):
    """Update the state of a worker to include log probabilities.

    Args:
        inference: DiscreteHMMInferenceResults, global copy
        state (tuple): a tuple of EMObject, DiscreteHMMInferenceResults, local to worker
    Returns:
        updated state tuple
    """
    data_object, local_inference = state
    if local_inference is None:
        local_inference = inference

    local_inference = data_object.find_log_prob(inference)

    return (data_object, local_inference)


def calc_sufficient_statistics(state):
    """Calculate gamma, xi, gamma_by_component for each DiscreteHMMInferenceResults object

    Args:
        state (tuple): a tuple of EMObject, DiscreteHMMInferenceResults, local to worker
    Returns:
        updated state tuple
    """
    data_object, local_inference = state

    alpha = local_inference.alpha
    beta = local_inference.beta
    gamma = np.asarray(alpha) + np.asarray(beta)
    gamma -= logsumexp(gamma, axis=1).reshape(-1, 1)
    local_inference.gamma = gamma

    beta_carry = data_object.beta_carry
    local_inference = data_object.find_xi(local_inference, beta_carry)

    local_inference = data_object.find_gamma_by_component(local_inference)

    return (data_object, local_inference)


def calc_update_parameters(state):
    """Gets preprocessing parameters for M step

    Args:
        state (tuple): a tuple of EMObject, DiscreteHMMInferenceResults, local to worker
    Returns:
        state tuple, local update parameters
    """

    data_object, local_inference = state

    local_parameters = data_object.find_update_params(local_inference)

    return (data_object, local_parameters, local_inference)


def forward_init(forward_state):
    """Gets alpha for first worker where alpha is an array where entry [t,i]
     is the log probability of observations o_0,...,o_t and h_t = i under the current model parameters.

    Args:
        state (tuple): a tuple of EMObject, DiscreteHMMInferenceResults, local to worker
    Returns:
        last row of alpha, for the next worker to calculate forward probabilities
    """
    data_object, local_inference = forward_state

    log_prob = np.asarray(local_inference.log_probability)
    alpha = local_inference._compute_forward_probabilities(log_prob)
    local_inference.alpha = np.array(alpha)

    return alpha[-1]


def backward_init(backward_state):
    """Gets beta for last worker where beta is an array where entry [t,i]
     is the log probability of observations o_{t+1},...,o_T given h_t = i under the current model parameters.

    Args:
        state (tuple): a tuple of EMObject, DiscreteHMMInferenceResults, local to worker
    Returns:
        first row of beta, for the next worker to calculate backward probabilities
        first row of log probabilities, needed for the next worker to calc xi
    """
    data_object, local_inference = backward_state
    log_prob = np.asarray(local_inference.log_probability)
    beta = local_inference._compute_backward_probabilities(log_prob)
    local_inference.beta = np.array(beta)
    return beta[0], log_prob[0]


def calc_backward_probabilities(backward_state, beta_carry):
    """Gets beta for workers where beta is an array where entry [t,i]
     is the log probability of observations o_{t+1},...,o_T given h_t = i under the current model parameters.

    Args:
        state (tuple): a tuple of EMObject, DiscreteHMMInferenceResults, local to worker
        beta_carry (tuple): first row of beta, first row of log_prob from previous worker
    Returns:
        first row of beta, for the next worker to calculate backward probabilities
        first row of log probabilities, needed for the next worker to calc xi
    """
    data_object, local_inference = backward_state
    data_object.beta_carry = beta_carry
    log_prob = np.asarray(local_inference.log_probability)
    beta = dist_compute_backward_probabilities(local_inference, log_prob,
                                               beta_carry)
    local_inference.beta = np.array(beta)

    return beta[0], log_prob[0]


def calc_forward_probabilities(forward_state, first_alpha):
    """Gets alpha for workers where alpha is an array where entry [t,i]
     is the log probability of observations o_0,...,o_t and h_t = i under the current model parameters.

    Args:
        state (tuple): a tuple of EMObject, DiscreteHMMInferenceResults, local to worker
        first_alpha: last row of alpha from the previous worker
    Returns:
        last row of alpha, for the next worker to calculate forward probabilities
    """
    data_object, local_inference = forward_state
    log_prob = np.asarray(local_inference.log_probability)

    alpha = dist_compute_forward_probabilities(local_inference, log_prob,
                                               first_alpha)
    local_inference.alpha = np.array(alpha)

    return alpha[-1]


def dist_compute_forward_probabilities(inference, log_probability, first_alpha):
    """ Compute forward probabilities.

    Arguments:
        log_probability: dataframe of log probability of hidden state
        first_alpha: last row of alpha from the previous worker

    Returns:
        Array where entry [t,i] is the log probability of observations o_0,...,o_t and h_t = i under the current model parameters.

    """
    log_transition = inference.model.log_transition
    log_initial_state = inference.model.log_initial_state
    log_prob = np.array(log_probability)

    alpha = np.empty((log_probability.shape))
    alpha_t = first_alpha + log_transition + log_prob[0]
    alpha[0] = logsumexp(alpha_t, axis=1)
    for t in range(1, len(alpha)):
        alpha_t = alpha[t - 1, None] + log_transition + log_prob[t]
        alpha[t] = logsumexp(alpha_t, axis=1)

    return alpha


def dist_compute_backward_probabilities(inference, log_probability, beta_carry):
    """ compute backard probabilities.

    Arguments:
        log_probability: dataframe of log probability of hidden state
        beta_carry (tuple): first row of beta, first row of log_prob from previous worker

    Returns:
        Array where entry [t,i] is the log probability of observations o_{t+1},...,o_T given h_t = i under the current model parameters.

    """
    log_transition = inference.model.log_transition
    log_prob = np.array(log_probability)
    first_beta, log_prob_carry = beta_carry

    beta = np.empty((log_probability.shape))
    beta_t = first_beta + log_transition + log_prob_carry
    beta[0] = logsumexp(beta_t, axis=1)

    for t in range(1, len(beta)):
        beta_t = beta[t - 1, None] + log_transition + log_prob[len(beta) - t]
        beta[t] = logsumexp(beta_t, axis=1)

    beta = np.flip(beta, axis=0)
    return beta
