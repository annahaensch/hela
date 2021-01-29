from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from dask.distributed import Client
from scipy.special import logsumexp

import hela.generation.hmm as gen
import hela.hmm as hmm
from hela.walk_forward_concordance import WalkForwardConcordance


@pytest.fixture(scope="module")
def random():
    return np.random.RandomState(0)


@pytest.fixture(scope="module")
def n_hidden_states(random):
    return random.choice(range(2, 6))


@pytest.fixture(scope="module")
def generative_model(random, n_hidden_states):
    model = gen.DiscreteHMMGenerativeModel(
        n_hidden_states=n_hidden_states,
        n_categorical_features=2,
        n_gaussian_features=1,
        n_gmm_components=1)

    hidden_states = model.generate_hidden_state_sequence(n_observations=800)
    dataset = model.generate_observations(hidden_states)

    training_parameters = gen.data_to_discrete_hmm_training_spec(
        hidden_states=hidden_states,
        n_hidden_states=model.n_hidden_states,
        data=dataset,
        categorical_features=list(model.categorical_values.columns),
        gaussian_features=list(model.gaussian_values.columns),
        n_gmm_components=model.n_gmm_components)

    dataset_redacted = dataset.copy()
    for _ in range(int(dataset.shape[0] / 10)):
        for i in range(len(dataset.columns) + 1):
            col_to_redact = np.random.choice(dataset.columns, i, replace=False)
            row_to_redact = dataset.index[np.random.choice(dataset.shape[0], 1)]
            dataset_redacted.loc[row_to_redact, col_to_redact] = np.nan

    return {
        "model": model,
        "dataset": dataset,
        "hidden_states": hidden_states,
        "training_parameters": training_parameters,
        "dataset_redacted": dataset_redacted
    }


@pytest.fixture(scope="module")
def factored_generative_model(random, n_hidden_states):
    model = gen.FactoredHMMGenerativeModel(
        ns_hidden_states=3 * [n_hidden_states], n_gaussian_features=1)

    return {"model": model}


@pytest.fixture(scope="module")
def distributed_learning_model(generative_model):

    dataset = generative_model["dataset"]
    training_parameters = generative_model["training_parameters"]
    workers = 2

    # Initializing Dask client
    client = Client(processes=True, n_workers=workers)
    # Split dataset for distributed learning
    split = len(dataset) // workers
    partitioned_data = [dataset[:split], dataset[split:]]

    model_config = hmm.DiscreteHMMConfiguration.from_spec(training_parameters)
    model = model_config.to_model()
    alg = hmm.distributed.distributed_init(partitioned_data, model, client)
    new_model = alg.run(model, 2)

    assert len(alg.model_results) == 2

    return new_model


def test_model_loads_from_spec(generative_model):

    training_parameters = generative_model["training_parameters"]

    model_config = hmm.DiscreteHMMConfiguration.from_spec(training_parameters)
    model = model_config.to_model()

    assert model.n_hidden_states == training_parameters['hidden_state']['count']


def test_model_learning_and_imputation(generative_model):

    training_parameters = generative_model["training_parameters"]
    dataset = generative_model["dataset"]
    dataset_redacted = generative_model["dataset_redacted"]
    hidden_states = generative_model["hidden_states"]

    model_config = hmm.DiscreteHMMConfiguration.from_spec(training_parameters)
    model = model_config.to_model()
    alg = hmm.LearningAlgorithm()
    new_model = alg.run(model, dataset, 2)

    assert len(alg.model_results) == 2

    # Check that log likelihood is increasing.
    alphas = []
    for s in alg.sufficient_statistics:
        s_inf = s.model.load_inference_interface()
        s_log_prob = s_inf.predict_hidden_state_log_probability(dataset)
        alpha = s_inf._compute_forward_probabilities(s_log_prob)
        alphas.append(logsumexp(alpha[-1]))
    assert alphas[0] < alphas[1]

    inf = new_model.load_inference_interface()

    predict_viterbi = inf.predict_hidden_states_viterbi(dataset)
    assert hidden_states[hidden_states ==
                         predict_viterbi].shape[0] / hidden_states.shape[0] > .1

    predict_gibbs = inf.predict_hidden_states_gibbs(
        data=dataset, n_iterations=5)
    assert hidden_states[hidden_states ==
                         predict_gibbs].shape[0] / hidden_states.shape[0] > .1

    dataset_imputed = inf.impute_missing_data(dataset_redacted, method='argmax')

    assert (dataset_redacted[(dataset_redacted.isna().any(axis=1))].shape[0] >
            0) & (dataset_imputed[(
                dataset_imputed.isna().any(axis=1))].shape[0] == 0)

    val = new_model.load_validation_interface(dataset)
    validation = val.validate_imputation(dataset_redacted, dataset_imputed)

    precision_recall = val.precision_recall_df_for_predicted_categorical_data(
        dataset_redacted, dataset_imputed)
    assert precision_recall['proportion'].sum() == 1

    val1 = validation['relative_accuracy_of_imputed_categorical_data']
    val2 = validation[
        'average_relative_log_likelihood_of_imputed_gaussian_data']
    val3 = validation['average_z_score_of_imputed_gaussian_data']

    assert val1 >= 1  #Accuracy should be at least as good as random guessing.
    assert val2 <= 0  #This metric returns


def test_distributed(distributed_learning_model, generative_model):

    training_parameters = generative_model["training_parameters"]
    dataset = generative_model["dataset"]
    dataset_redacted = generative_model["dataset_redacted"]

    inf = distributed_learning_model.load_inference_interface()
    prediction = inf.predict_hidden_states_viterbi(dataset)

    assert set(prediction.unique()).issubset(
        set([h for h in range(training_parameters['hidden_state']['count'])]))

    dataset_imputed = inf.impute_missing_data(dataset_redacted, method='argmax')

    assert (dataset_redacted[(dataset_redacted.isna().any(axis=1))].shape[0] >
            0) & (dataset_imputed[(
                dataset_imputed.isna().any(axis=1))].shape[0] == 0)

    val = distributed_learning_model.load_validation_interface(dataset)
    validation = val.validate_imputation(dataset_redacted, dataset_imputed)

    precision_recall = val.precision_recall_df_for_predicted_categorical_data(
        dataset_redacted, dataset_imputed)
    assert precision_recall['proportion'].sum() == 1

    val1 = validation['relative_accuracy_of_imputed_categorical_data']
    val2 = validation[
        'average_relative_log_likelihood_of_imputed_gaussian_data']
    val3 = validation['average_z_score_of_imputed_gaussian_data']

    assert val1 >= 1  #Accuracy should be at least as good as random guessing.
    assert val2 <= 0  #This metric returns
    # log p(actual value)-log p(imputed value) for conditonal
    # distribution. With imputation method 'argmax' the
    # imputed value should be at least as likely as the
    # actual value, so this should always be negative.
    assert val3 < 3.3  # This metric returns the average z score.
    # If this is larger than 3.3 then my imputed values,
    # are on average, worse than the 99.9% confidence
    # interval and something has gone wrong.


def test_forecasting(generative_model):

    training_parameters = generative_model["training_parameters"]
    dataset = generative_model["dataset"]

    model_config = hmm.DiscreteHMMConfiguration.from_spec(training_parameters)
    model = model_config.to_model()
    fi = model.load_forecasting_interface()

    horizon_timesteps = [7, 30]
    conditioning_date = dataset.index[int(dataset.shape[0] / 2)]

    forecast = fi.forecast_observation_at_horizons(dataset, horizon_timesteps,
                                                   conditioning_date)

    val = model.load_validation_interface(dataset)
    val.validate_forecast(forecast)

    assert forecast[~(forecast.isnull().any(axis=1))].shape[
        0] == len(horizon_timesteps) + 1

    steady_state = fi.steady_state_and_horizon(dataset)
    n_steps = steady_state['steady_state_horizon_timesteps'] + 1
    initial_prob = fi.hidden_state_probability_at_conditioning_date(
        dataset, dataset.index[-1])

    step1 = initial_prob @ np.linalg.matrix_power(
        np.exp(model.log_transition), n_steps)

    step2 = initial_prob @ np.linalg.matrix_power(
        np.exp(model.log_transition), n_steps + 1)

    assert np.max(np.abs(step1 - step2)) < 1e-05


def test_wfci(random, distributed_learning_model, generative_model):

    horizons = [pd.Timedelta(days=h) for h in [2, 7]]
    # validate on new, random datasets
    c = 0
    col = f"categorical_feature_{c}"
    n_assets = 4
    data = []
    for i in range(n_assets):
        hidden_states = generative_model[
            "model"].generate_hidden_state_sequence(n_observations=100)
        ds = generative_model["model"].generate_observations(hidden_states)
        data.append(ds)
    selected = [1]
    assets = [f"test_asset_{i}" for i in range(n_assets)]

    start = data[0].index[0]
    end = data[0].index[-1]
    time_resolution = pd.Timedelta(days=1)
    prediction_dates = pd.date_range(
        start.round(time_resolution),
        end.round(time_resolution),
        freq=time_resolution)

    # Initializing Dask client
    client = Client(processes=True, n_workers=n_assets)
    mapped = client.map(
        hmm.find_risk_at_horizons,
        data,
        assets,
        model=distributed_learning_model,
        label_column=col,
        selected_labels=selected,
        horizons=horizons,
        prediction_dates=prediction_dates,
        event_time_resolution=pd.Timedelta(days=1))
    risk_result = client.gather(mapped)
    client.shutdown()

    #calculate WFCI
    predictions = np.array([r[0] for r in risk_result])
    prediction_times = risk_result[0][1]
    validation_df = pd.concat([r[2] for r in risk_result], ignore_index=True)

    wfci = WalkForwardConcordance(
        predictions=predictions,
        asset_ids=assets,
        prediction_dates=prediction_times,
        validation_df=validation_df,
        predictive_horizons=horizons,
        resolution="1h")
    df = wfci.walk_forward_ci_df()[0]

    for c in df.columns[::2].to_list():  # ci_ and n_pairs columns
        assert len(df[~df[c].isna()]) > 0
        assert df[c].mean() > 0.4


def test_generative_model(generative_model, factored_generative_model):

    discrete_model = generative_model["model"]
    training_spec = discrete_model.generative_model_to_discrete_hmm_training_spec(
    )
    model_config = hmm.DiscreteHMMConfiguration.from_spec(training_spec)
    model = model_config.to_model()
    new_generative_model = gen.model_to_discrete_generative_spec(model)

    assert np.all(discrete_model.categorical_values ==
                  new_generative_model.categorical_values)
    assert np.all(discrete_model.means == new_generative_model.means)
    assert np.min(discrete_model.transition_matrix -
                  new_generative_model.transition_matrix) < 1e-08

    factored_model = factored_generative_model["model"]
    hidden_states = factored_model.generate_hidden_state_sequence(
        n_observations=1000)
    data = factored_model.generate_observations(hidden_states)

    assert data.columns == factored_model.gaussian_values.columns