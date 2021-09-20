"""Utility functions used across hmm classes, including for validation.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import random
from jax.scipy.special import logsumexp as lse
from scipy import stats
from scipy.special import logsumexp


def get_finite_observations_from_data(model, data):
    """Return finite observations as dataframe

    Arguments:
        model: HiddenMarkovModel object.
        data: dataframe of mixed data.

    Returns:
        dataframe of finite observations.
    """
    if len(model.finite_features) == 0:
        return None
    else:
        # Make sure that any integers are being cast as integers.
        float_to_int = {
            feature: "Int64"
            for feature in data[model.finite_features].select_dtypes("float")
        }
        data = data.astype(float_to_int, errors='ignore')
        return data.loc[:, model.finite_features]


def get_finite_observations_from_data_as_enum(model, data):
    """Return finite observations as series

    Arguments:
        model: DiscreteHMM object.
        data: dataframe of mixed data.

    Returns:
        Series of finite state indices.
    """
    if len(model.finite_features) == 0:
        return None
    else:
        finite_data = get_finite_observations_from_data(model, data)
        get_states = lambda x: (
            model.categorical_model.finite_values_dict_inverse[str(list(x))])
        states = np.array([get_states(x) for x in np.array(finite_data)])
        return pd.Series(states, index=data.index)


def get_gaussian_observations_from_data(model, data):
    """Return gaussian observations as dataframe

    Arguments:
        model: DiscreteHMM object.
        data: dataframe of mixed data.

    Returns:
        DataFrame with Gaussian observations.
    """
    if model.gaussian_mixture_model is None:
        return None
    else:
        gaussian_data = pd.DataFrame()
        continuous_values = model.continuous_values
        for col in continuous_values.columns:
            if continuous_values.loc['distribution', col] == 'gaussian':
                gaussian_data = gaussian_data.join(
                    data.loc[:, col], how='outer')
        return gaussian_data


def compute_marginal_probability_gaussian(partial_gaussian_observation, mean,
                                          covariance):
    """ Compute marginal probability of observation

    Arguments:
        partial_gaussian_observation: single row of a dataframe of gaussian observations where at least one entry is nan
        mean: vector of dimension equal to dimension of the gaussian space
        covariance: array with dimesion equal dimension of the gaussian space x dimension of the gaussian space

    Returns:
        marginal probability of observed portion of partial_gaussian_observation
    """
    covariance_inverse = np.linalg.solve(covariance, np.identity(
        len(covariance)))

    # Separate indices
    obs = np.array(partial_gaussian_observation.iloc[0])
    index_nan = np.argwhere(np.isnan(obs)).flatten()
    index_other = np.argwhere(~np.isnan(obs)).flatten()

    # Separate means
    mean_other = np.array([mean[i] for i in index_other])

    # Create block covariance matrices
    covariance_block_other = covariance[index_other[:, None], index_other]
    covariance_inverse_block_nan = covariance_inverse[index_nan[:, None],
                                                      index_nan]

    coeff = ((2 * np.pi)**(len(index_nan) - len(obs)) *
             (np.linalg.det(covariance_inverse_block_nan
                           ) / np.linalg.det(covariance)))**(1 / 2)

    error = obs[index_other] - mean_other
    marginal_prob = coeff * np.exp(
        (np.linalg.solve(covariance_block_other, error) @ error) * (-1 / 2))

    return marginal_prob


def compute_marginal_probability_gaussian_mixture(
        partial_gaussian_observation, means, covariances, component_weights):
    """ Compute marginal probability of observation in gaussian mixture

    Arguments:
        partial_gaussian_observation: single row of a dataframe of gaussian observations where at least one entry is nan
        means: list of vectors of dimension equal to dimension of the gaussian space
        covariances: list of arrays with dimesion equal dimension of the gaussian space x dimension of the gaussian space
        component_weights: list of weights of the gmm components

    Returns:
        marginal probability of observed portion of partial_gaussian_observation
    """
    marginal_prob_by_component = np.empty(len(component_weights))
    for i in range(len(component_weights)):
        mean = means[i]
        covariance_matrix = covariances[i]
        marginal_prob_by_component[i] = compute_marginal_probability_gaussian(
            partial_gaussian_observation, mean, covariance_matrix)

    marginal_prob = np.sum(
        np.array(component_weights * marginal_prob_by_component))

    return marginal_prob


def compute_mean_of_conditional_probability_gaussian(
        partial_gaussian_observation, mean, covariance):
    """ Compute mean of conditional probability of observation

    Arguments:
        partial_gaussian_observation: single row of a dataframe of gaussian observations where at least one entry is nan
        mean: vector of dimension equal to dimension of the gaussian space
        covariance: array with dimesion equal dimension of the gaussian space x dimension of the gaussian space

    Returns:
        mean for conditional probability of observed portion of partial_gaussian_observation
    """
    # Separate indices
    obs = np.array(partial_gaussian_observation.iloc[0])
    index_nan = np.argwhere(np.isnan(obs)).flatten()
    index_other = np.argwhere(~np.isnan(obs)).flatten()

    if len(index_other) > 0:

        # Separate means
        mean_nan = np.array([mean[i] for i in index_nan])
        mean_other = np.array([mean[i] for i in index_other])

        # Create block covariance matrices
        covariance_block_other = covariance[index_other[:, None], index_other]
        covariance_block_cross = covariance[index_nan[:, None], index_other]

        error = obs[index_other] - mean_other

        conditional_mean_nan = mean_nan - covariance_block_cross @ np.linalg.solve(
            covariance_block_other, error)

    else:
        conditional_mean_nan = mean

    return conditional_mean_nan


def compute_mean_of_conditional_probability_gaussian_mixture(
        partial_gaussian_observation, means, covariances, component_weights):
    """ Compute mean of conditional probability of observation

    Arguments:
        partial_gaussian_observation: single row of a dataframe of gaussian observations where at least one entry is nan
        means: vector of dimension equal to dimension of the gaussian space
        covariances: array with dimesion equal dimension of the gaussian space x dimension of the gaussian space
        component_weights: list of weights of the gmm components

    Returns:
        mean of conditional probability of observed portion of partial_gaussian_observation
    """
    obs = np.array(partial_gaussian_observation.iloc[0])
    index_nan = np.argwhere(np.isnan(obs)).flatten()

    conditional_means_by_component = np.empty((len(component_weights),
                                               len(index_nan)))
    for i in range(len(component_weights)):
        mean = means[i]
        covariance = covariances[i]
        conditional_means_by_component[
            i] = compute_mean_of_conditional_probability_gaussian(
                partial_gaussian_observation, mean, covariance)

    return np.sum(
        conditional_means_by_component * component_weights.reshape(-1, 1),
        axis=0)


def compute_covariance_of_conditional_probability_gaussian(
        partial_gaussian_observation, mean, covariance_matrix):
    """ Compute covariance of conditional probability of observation

    Arguments:
        partial_gaussian_observation: single row of a dataframe of gaussian observations where at least one entry is nan
        mean: vector of dimension equal to dimension of the gaussian space
        covariance_matrix: array with dimesion equal dimension of the gaussian space x dimension of the gaussian space

    Returns:
        covariance of conditional probability of observed portion of partial_gaussian_observation
    """
    obs = np.array(partial_gaussian_observation.iloc[0])
    covariance = np.array(covariance_matrix)
    covariance_inverse = np.linalg.solve(covariance, np.identity(
        len(covariance)))
    index_nan = np.argwhere(np.isnan(obs)).flatten()
    covariance_inverse_block_nan = covariance_inverse[index_nan[:, None],
                                                      index_nan]

    return np.linalg.solve(covariance_inverse_block_nan,
                           np.identity(len(index_nan)))


def compute_log_likelihood_with_inferred_pdf(
        full_gaussian_observation, partial_gaussian_observation, means,
        covariances, component_weights):
    """ Compute covariance of conditional probability of observation

    Arguments:
        full_gaussian_observation: single row of a dataframe of gaussian observations.
        partial_gaussian_observation: single row of a dataframe of gaussian observations where at least one entry is nan
        mean: vector of dimension equal to dimension of the gaussian space
        covariance_matrix: array with dimesion equal dimension of the gaussian space x dimension of the gaussian space
        component_weights: list of weights of the gmm components

    Returns:
        covariance of conditional probability of observed portion of partial_gaussian_observation
    """

    obs = np.array(partial_gaussian_observation.iloc[0])
    index_nan = np.argwhere(np.isnan(obs)).flatten()
    log_weights = np.log(component_weights)

    log_prob_by_component = np.empty((means.shape[0], means.shape[1]))
    for i in range(means.shape[0]):
        for j in range(means.shape[1]):
            mean = compute_mean_of_conditional_probability_gaussian(
                partial_gaussian_observation, means[i][j], covariances[i][j])
            covariance = compute_covariance_of_conditional_probability_gaussian(
                partial_gaussian_observation, means[i][j], covariances[i][j])
            log_prob_by_component[i][j] = stats.multivariate_normal.logpdf(
                [np.array(full_gaussian_observation)[0][k] for k in index_nan],
                mean,
                covariance,
                allow_singular=True)

    log_prob = np.empty(log_weights.shape[0])
    for i in range(log_weights.shape[0]):
        log_prob[i] = logsumexp(log_prob_by_component[i] + log_weights[i])

    return log_prob


def average_z_score(means, covariances, component_weights, actual_gaussian_data,
                    redacted_gaussian_data,
                    conditional_probability_by_hidden_state):
    """ Computes z score of imputed gaussian data averaged over observations.
        """
    redacted_index = redacted_gaussian_data[redacted_gaussian_data.isnull()
                                            .any(axis=1)].index
    z_score = np.empty(len(redacted_index))
    for i in range(len(redacted_index)):
        idx = redacted_index[i]
        partial_gaussian_observation = redacted_gaussian_data.loc[[idx]]
        redacted_values = np.array(redacted_gaussian_data.loc[idx][
            redacted_gaussian_data.loc[idx].isna()].index)
        actual_gaussian_observation = np.array(
            actual_gaussian_data.loc[[idx], redacted_values])[0]
        obs = np.array(partial_gaussian_observation.iloc[0])
        index_nan = np.argwhere(np.isnan(obs)).flatten()

        cond_prob = np.array(conditional_probability_by_hidden_state.loc[idx])
        new_means = np.empty((len(means), len(index_nan)))
        new_covariances = np.empty((len(means), len(index_nan), len(index_nan)))
        for j in range(len(means)):
            new_means[j] = cond_prob[
                j] * compute_mean_of_conditional_probability_gaussian_mixture(
                    partial_gaussian_observation, means[j], covariances[j],
                    component_weights[j])
            new_covariances_by_component = np.empty((len(covariances[1]),
                                                     len(index_nan),
                                                     len(index_nan)))
            for k in range(len(covariances[1])):
                new_covariances_by_component[k] = component_weights[j][
                    k] * compute_covariance_of_conditional_probability_gaussian(
                        partial_gaussian_observation, means[j][k],
                        covariances[j][k])
            new_covariances[j] = cond_prob[j] * np.sum(
                new_covariances_by_component, axis=0)

        new_mean = np.sum(new_means, axis=0)
        new_covariance = np.sum(new_covariances, axis=0)

        error = actual_gaussian_observation - new_mean
        z_score[i] = np.sqrt(np.linalg.solve(new_covariance, error) @ error)

    return np.mean(z_score)


def expected_proportional_accuracy(data, redacted_data):
    """ Computed expected accuracy of imputation done according to proportion.

    Arguments:
        data: dataframe with discrete data
        redacted_data: data with entries redacted

    Returns:
        Number between 0 and 1 giving expected rate of accuracy if values were imputed proportionally.
    """
    redacted_index = redacted_data[redacted_data.isnull().any(axis=1)].index
    df = data.copy()
    df['tuples'] = list(zip(*[data[c] for c in data.columns]))
    proportion = (df['tuples'].value_counts() / df.shape[0]).to_dict()

    return sum([proportion[tuple(x)] for x in df.loc[redacted_index, 'tuples']
               ]) / len(redacted_index)


def best_possible_accuracy_of_categorical_prediction(data, redacted_data):
    """ Computed expected accuracy of imputation done according to proportion.

    Arguments:
        data: dataframe with discrete data
        redacted_data: data with entries redacted

    Returns:
        Number between 0 and 1 giving expected rate of accuracy if values were imputed proportionally.
    """
    return 1 / expected_proportional_accuracy(data, redacted_data)


def check_jax_precision():
    """ Test that 64-bit precision is enabled.
    """
    x = random.uniform(random.PRNGKey(0), (1000,), dtype=jnp.float64)
    return x.dtype == 'float64'


@jax.jit
def jax_compute_forward_probabilities(log_initial_state, log_transition,
                                      log_probability):
    """ Compute forward probabilities faster with Jax.

    Arguments:
        log_initial_state: array of log initial state values
        log_transition: array of log transition matrix
        log_probability: array of log probability of hidden state

    Returns: Array where entry [t,i] is the log probability of observations
    o_0,...,o_t and h_t = i under the current model parameters.

    """
    if check_jax_precision() == False:
        raise ValueError(
            "To run forward-backward with Jax, you need to set the numerical "
            "precision to 64-bit, you can do this by either setting the "
            "environment variable `JAX_ENABLE_X64=True` or by setting the "
            "flag at startup with `config.update('jax_enable_x64', True)`.  "
            "You can find more details about this at "
            "https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#Double-(64bit)-precision"
        )
    log_prob = jnp.array(log_probability, jnp.float64)

    def forward_one_step(prev_alpha, index, log_transition, log_prob):
        alpha_t = prev_alpha + log_transition + log_prob[index]
        return lse(alpha_t, axis=1)

    def scan_fn(alpha_, index):
        return forward_one_step(alpha_, index, log_transition, log_prob), alpha_

    alpha = jnp.empty((log_probability.shape))
    init_prob = log_initial_state + log_prob[0]
    indices = np.array(range(1, len(alpha) + 1))

    prev_alpha, alpha = jax.lax.scan(scan_fn, init_prob, indices)

    return alpha


@jax.jit
def jax_compute_backward_probabilities(log_transition, log_probability):
    """ compute backard probabilities faster with Jax.

    Arguments:
        log_transition: array of log transition matrix
        log_probability: array of log probability of hidden state

    Returns: Array where entry [t,i] is the log probability of observations
    o_{t+1},...,o_T given h_t = i under the current model parameters.

    """
    log_prob = jnp.array(log_probability, jnp.float64)

    def backward_one_step(prev_beta, index, log_transition, log_prob):
        beta_t = prev_beta + log_transition + log_prob[index]
        return lse(beta_t, axis=1)

    def scan_backward_fn(beta_, index):
        return backward_one_step(beta_, index, log_transition, log_prob), beta_

    beta = jnp.empty((log_probability.shape))
    indices = np.array(range(len(beta) - 1, -1, -1))

    prev_beta, beta = jax.lax.scan(scan_backward_fn, beta[0], indices)
    beta = np.flip(beta, axis=0)
    return beta


def get_complete_data_chunks(data):
    """ Return dataframe with information on complete data chunks.

    Argument:
        data: dataframe with possible NaN entries

    Returns:
        Dataframe with 'start', 'end' and 'duration' values for each chunk
        of entries which are free of NaNs.
    """

    complete_data_chunks = pd.DataFrame(columns=['start', 'end', 'duration'])

    # Determine if no data is missing and drop any training NaN rows.
    complete_data = data[~data.isna().any(axis=1)]
    if complete_data.shape == data.shape:
        complete_data_chunks.loc[0, 'start'] = data.index[0]
        complete_data_chunks.loc[0, 'end'] = data.index[-1]
        complete_data_chunks.loc[0, 'duration'] = data.index[-1] - data.index[0]

        return complete_data_chunks

    else:
        data = data.loc[:complete_data.index[-1]]

    i = 0
    end = None
    while end != data.index[-1]:

        # Find first non-NaN entry and clip dataframe here from above.
        start = data[~data.isna().any(axis=1)].index[0]
        data = data.loc[start:]
        complete_data_chunks.loc[i, 'start'] = start

        # Find first NaN entry and clip dataframe here from below.
        nan_entries = data[data.isna().any(axis=1)].index
        if len(nan_entries) > 0:
            first_nan = nan_entries[0]
            data_chunk = data.loc[:first_nan].iloc[:-1]
            end = data_chunk.index[-1]
            complete_data_chunks.loc[i, 'end'] = end

            data = data.loc[first_nan:]
            i = i + 1

        else:
            end = data.index[-1]
            complete_data_chunks.loc[i, 'end'] = end

        complete_data_chunks[
            'duration'] = complete_data_chunks['end'] - complete_data_chunks['start']

    return complete_data_chunks
