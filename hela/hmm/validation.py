# Functions that help with validation of HMM predictions

import numpy as np
import pandas as pd


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
