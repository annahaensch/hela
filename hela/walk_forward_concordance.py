"""Compute walk-forward concordance index.

A model is trained on data before a *training* conditioning date, and
validated on a range of data between a *validation* conditioning date
VC and the censoring date CD (the last known date of data collection).
For a selection of timesteps t in the period [VC, CD-h], the concordance
index CI_h is computed for each horizon h in the predictive horizons
defined in the user specification. This method of validating predictions
on timeseries data is known as "walk-forward validation".
"""

import logging

import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored as ci_censored

logging.basicConfig(level=logging.INFO)


def general_horizon_to_str(td):
    """General tight string formatting of timedeltas.
    """
    sec = td.seconds
    hours = sec // 3600
    minutes = (sec - hours * 3600) // 60
    sec = int(sec - hours * 3600 - minutes * 60)
    # rounding at the second level
    return f"{td.days}d{hours}h{minutes}m{sec}s"


def general_str_to_horizon(string):
    """Reversing of general_horizon_to_str output.
    """
    d, r = string.split("d")
    h, r = r.split("h")
    m, r = r.split("m")
    s, r = r.split("s")
    return pd.Timedelta(
        days=int(d), hours=int(h), minutes=int(m), seconds=int(s))


def horizon_to_str(timedelta):
    """Returns string representation of default horizons.
    """
    fixed = horizon_str_map().get(timedelta)
    if fixed is None:
        return general_horizon_to_str(timedelta)
    else:
        return fixed


def str_to_horizon(string):
    str_horizon_map = {
        string: horizon
        for horizon, string in horizon_str_map().items()
    }
    fixed = str_horizon_map.get(string)
    if fixed is None:
        return general_str_to_horizon(string)
    else:
        return fixed


def horizon_str_map():
    return {
        pd.Timedelta(days=1): '1d',
        pd.Timedelta(days=2): '2d',
        pd.Timedelta(days=3): '3d',
        pd.Timedelta(days=7): "1wk",
        pd.Timedelta(days=14): "2wk",
        pd.Timedelta(days=21): "3wk",
        pd.Timedelta(days=30): "1mo",
        pd.Timedelta(days=91): "3mo",
        pd.Timedelta(days=182): "6mo",
        pd.Timedelta(days=365): "1yr",
        pd.Timedelta(days=730): "2yr"
    }


class WalkForwardConcordance(object):
    """Compute walk-forward concordance indices."""

    def __init__(self, predictions, asset_ids, prediction_dates, validation_df,
                 predictive_horizons, resolution):
        """Arguments:

            predictions (np.array): an array of predictions with shape
            (n_assets, n_timesteps, n_horizons, n_events).
            asset_ids (list): list of unique asset IDs, corresponding to the
            first dimension of the predictions matrix respectively.
            prediction_dates (list): a list of pd.Timestamps that correspond
            to the times at which the predictions were made.
            validation_df (pd.DataFrame): a dataframe where each row corresponds
            to a TTE episode. Each row must have the following non-null attributes:
            asset_id, start_date, event_date, event_id.
            predictive_horizons: horizons at which predictions were made (e.g. probability
            of event occurrence in the next 1 week).
            resolution (str): pd.Timedelta alias for the resolution at which TTEs will be
            calculated.
        """
        self.predictions = predictions

        assert len(asset_ids) == predictions.shape[0]
        self.asset_ids = asset_ids

        assert len(prediction_dates) == predictions.shape[
            1]  # n_timesteps check
        self.prediction_dates = prediction_dates

        # Rows of (asset_id, start_date, event_date, event_id), index=episode_id
        assert validation_df.asset_id.nunique() == predictions.shape[
            0]  # n_assets check
        self.validation_df = validation_df

        assert len(predictive_horizons) == predictions.shape[
            2]  # n_horizons check
        self.predictive_horizons = predictive_horizons

        self.resolution = pd.Timedelta(resolution)
        self.date_index = pd.date_range(
            min(self.prediction_dates),
            max(self.prediction_dates),
            freq=resolution)

        # Indices of the TTE matrix corresponding to dates of prediction
        # Prediction dates will be rounded up to the nearest discrete timestamp in the date index
        prediction_dates_np = [
            np.datetime64(date) for date in self.prediction_dates
        ]
        self.query_idxs = np.searchsorted(self.date_index.values,
                                          prediction_dates_np)
        self.str_horizons = [
            horizon_to_str(h) for h in self.predictive_horizons
        ]
        self.horizons_in_periods = [
            round(h / self.resolution) for h in self.predictive_horizons
        ]
        self.n_events = self.predictions.shape[-1]

    def construct_tte_matrix(self, event_id):
        """Encapsulate time-to-event data required to compute CI in array form.

        Loop through rows of validation_df in order to construct the following
        two arrays:

        - tte_matrix: each row encodes the time until the next event is to occur
        for a given asset, and is np.nan during non-TTE episodes.
        - event_id_matrix: encodes which entries of the TTE matrix should be compared
        when computing CI for a particular event ID.

        Arguments:
            event_id: event ID for which concordance index is currently being
            calculated.

        Returns:
            tte_array_matrix (np.array): The entry indexed by unit i and timestep j is equal to
            TTE for unit i at timestep j. If the unit is not currently active at timestep
            j (because the event has already occurred or the episode has not yet begun)
            the entry is np.nan.
            event_id_matrix (np.array): The entry indexed by unit i and timestep j is equal to
            1 if a TTE episode is active and the ID of the next event occurrence matches
            the event ID of the event currently under consideration (the *event_id* argument).
            Otherwise, the entry is 0.
        """
        dropped = [
        ]  # Episodes that fall through cracks of discretized timesteps
        tte_arrays = []
        event_id_arrays = []
        val_asset_ids = [
        ]  # To make sure validation assets are aligned with prediction matrix

        val_group = self.validation_df.groupby("asset_id")
        for asset, row_group in val_group:

            # Select episodes corresponding to event ID of interest
            row_group_event = row_group[row_group.event_id.isin([event_id,
                                                                 -1])].copy()
            row_group_event.sort_values("start_date", inplace=True)

            tte_array = np.full(len(self.date_index), np.nan)
            event_id_array = np.zeros(len(self.date_index))
            for _, index_row in row_group_event.iterrows():
                start_idx = np.searchsorted(self.date_index.values,
                                            np.datetime64(index_row.start_date))
                event_idx = np.searchsorted(self.date_index.values,
                                            np.datetime64(index_row.event_date))

                # Record when entire episode "falls through the cracks" of the discretized date range
                if start_idx == event_idx:
                    dropped.append(index_row.name)

                # TTE should be <= 1 at the index immediately preceding event date
                # TTE should equal event duration at index immediately after start date
                tte_array[start_idx:event_idx] = subtract_date_arrays(
                    index_row.event_date, self.date_index[start_idx:event_idx],
                    self.resolution)
                # Keep track of which episodes end in the event of interest
                if index_row.event_id == event_id:
                    event_id_array[start_idx:event_idx] = 1

            val_asset_ids.append(asset)
            tte_arrays.append(tte_array)
            event_id_arrays.append(event_id_array)

        if len(dropped) > 0:
            logging.warning(
                f"{len(dropped)} episodes did not span a single discretized timestep."
            )
            logging.warning(
                "These episodes will be excluded from the CI calcuation.")
            logging.warning(
                "Decrease the 'resolution' argument in the EP spec to address this issue."
            )

        # TTE matrix rows must be aligned with event prob prediction asset rows
        assert val_asset_ids == self.asset_ids, "Prediction asset IDs not aligned with validation asset IDs!"
        tte_matrix = np.array(tte_arrays)
        event_id_matrix = np.array(event_id_arrays)

        assert ~np.any(tte_matrix[~np.isnan(tte_matrix)] < 0)
        return tte_matrix, event_id_matrix

    def construct_event_indicator_matrix(self, predictions, tte_matrix,
                                         event_id_matrix):
        """Construct binary indicator of whether single event type occurred within horizon.

        Arguments:
            predictions: event probability predictions for a single event ID.
            tte_matrix: TTE for each asset and event in the validation set.
            event_id_matrix: binary matrix indicating which predictions are valid
            for the purposes of computing CI for the event ID in question.

        Returns:
            ei_matrix (np.array): An entry indexed by asset i, timestep j, and horizon h
            encodes whether at timestep j, an event for asset i occurs in the period [j, j+h].
            Note that unlike the TTE matrix, entries for asset i outside of valid TTE episodes
            for that asset are 0 (False) and not np.nan.
        """
        ei_matrix = np.zeros_like(predictions)
        tte_array_query = tte_matrix[:, self.query_idxs, None]
        event_id_query = event_id_matrix[:, self.query_idxs, None]

        # Condition 1: Event occurs within *horizon* timestemps of prediction time
        # Condition 2: The event that occurs matches the event ID we're considering
        with np.errstate(invalid='ignore', divide='ignore'):
            ei_matrix = (tte_array_query <=
                         self.horizons_in_periods) & (event_id_query == 1)

        return ei_matrix.astype(bool)

    def compute_walk_forward_ci(self, event_id):
        """Compute walk-forward concordance index for validation units.

        For each prediction time and predictive horizon, compute the concordance
        index and number of total pairwise comparisons.

        Arguments:
            event_id: event ID for which concordance index is currently being
            calculated.

        Returns:
            concordance index and total pairs compared for each timestep t and
            horizon h.
        """
        tte_matrix, event_id_matrix = self.construct_tte_matrix(event_id)

        predictions_event = self.predictions[..., event_id]
        ei_matrix = self.construct_event_indicator_matrix(
            predictions_event, tte_matrix, event_id_matrix)

        # All units for which an event has occurred by timestep t are excluded
        # from the concordance index calculation
        with np.errstate(invalid='ignore', divide='ignore'):
            valid_idxs_full = np.where(tte_matrix > 0, True, False)

        cis = {}
        pairs_compared = {}

        for hzn_ix, hzn in enumerate(self.str_horizons):
            logging.info("Computing CIs for horizon {} and event {}...".format(
                hzn, event_id))
            hzn_periods = self.horizons_in_periods[hzn_ix]

            cis[hzn] = []
            pairs_compared[hzn] = []

            # Compute CI for each prediction date
            for pred_idx, t in enumerate(self.query_idxs):
                # If t+horizon > max(event_date), return np.nan
                if t + hzn_periods > tte_matrix.shape[1]:
                    ci = np.nan
                    pairs_compared[hzn].append(0)
                    cis[hzn].append(ci)
                    continue

                # Pred_idx corresponds to the index of the prediction matrix ([0, 1, ...])
                # while t corresponds to the number of timesteps since the first element
                # of self.date_index. They represent the same points in time, however.
                valid_idxs = np.argwhere(valid_idxs_full[:, t] == 1).flatten()
                fp = predictions_event[valid_idxs, pred_idx, hzn_ix]
                tte = tte_matrix[valid_idxs, t]
                ei = ei_matrix[valid_idxs, pred_idx, hzn_ix]

                assert ~np.any(np.isnan(fp))
                assert ~np.any(np.isnan(tte))
                assert ~np.any(np.isnan(ei))

                if ~np.all(ei == 0) and len(ei) > 1:
                    ci, conc, disc, _, _ = ci_censored(ei, tte, fp)
                    pairs_compared[hzn].append(conc + disc)
                else:
                    # Zero or one events occurred (n_pairs=0). Set CI=NaN
                    ci = np.nan
                    pairs_compared[hzn].append(0)
                cis[hzn].append(ci)

        return cis, pairs_compared

    def walk_forward_ci_df(self):
        """Calculate walk-forward CI and package results into DataFrame format.

        Returns:
            DataFrame containing concordance index and number of pairs compared
            for each predictive horizon. Note that values in the range
            [max(event_date)-h, max(event_date)] are set to np.nan.
        """
        ci_dfs = {}
        for event_id in range(self.n_events):
            cis, pairs_compared = self.compute_walk_forward_ci(event_id)
            ci_df = pd.DataFrame([])

            for hzn_in_periods, str_hzn in zip(self.horizons_in_periods,
                                               self.str_horizons):
                ci = np.asarray(cis[str_hzn])
                n_pairs = np.asarray(pairs_compared[str_hzn])

                # Set all entries to NaN if horizon > max(event_date) - min(prediction_date)
                if min(self.prediction_dates) + str_to_horizon(str_hzn) >= max(
                        self.validation_df.event_date):
                    ci = np.full_like(self.date_index, np.nan)
                    n_pairs = np.full_like(self.date_index, np.nan)

                assert ci.shape[0] == n_pairs.shape[0] == len(self.query_idxs)
                ci_df["ci_{}".format(str_hzn)] = ci
                ci_df["n_pairs_{}".format(str_hzn)] = n_pairs

            ci_df.index = self.prediction_dates
            ci_dfs[event_id] = ci_df

        return ci_dfs


def convert_ep_to_asset(ep_ids, predictions):
    """Convert episode-level predictions to asset-level format.

    Currently, the TTE module is set up to make predictions at the
    episode- rather than the asset-level. This doesn't align well with
    how we think about prioritizing maintenance in the case of episodic TTE
    (where events reoccur over the course of an asset's lifetime). This function
    converts episode-level predictions (one nonzero entry per row) to asset-level
    predictions (multiple nonzero entries per row, each corresponding to a TTE
    episode).

    Arguments:
        ep_ids (list): list of episode IDs associated respectively with each
        row of predictions.
        predictions: array of shape (n_episodes, n_timesteps, n_horizons, n_events).

    Returns:
        asset_ids: ordered list of asset IDs, corresponding respectively to each row
        of the transformed predictions matrix.
        trasnformed_preds: transformed array of shape (n_assets, n_timesteps, n_horizons, n_events).
    """
    assert len(ep_ids) == predictions.shape[0]
    assert len(predictions.shape) == 4, "Incorrect input shape for predictions!"

    # Extract unique asset list from episode IDs.
    # This assumes that episode IDs follow the naming convention
    # "{asset_id}_{episode_number}"
    asset_ids = ["_".join(epid.split("_")[:-1]) for epid in ep_ids]
    asset_ids = sorted(list(set(asset_ids)))

    # Mapping between episode ID and row of prediction matrix
    ep_id_row_map = {ep: i for ep, i in zip(ep_ids, range(len(ep_ids)))}
    transformed_preds = np.zeros((len(asset_ids), *predictions.shape[1:]))

    for asset_row_idx, asset_id in enumerate(asset_ids):
        # Find all episodes associated with a given asset
        ep_assets = [ep for ep in ep_ids if asset_id in ep]

        # Loop through episodes associated with asset of interest
        for ep in ep_assets:
            episode_row = ep_id_row_map[ep]
            # Predictions will be nonzero only during TTE episodes
            # Find indices of nonzero preds in original array (one episode per row)
            nonzero_idxs = np.where(predictions[episode_row, ...] > 0)

            # Assign predictions to same indices of new asset-level row
            transformed_preds[asset_row_idx, nonzero_idxs[0], nonzero_idxs[
                1], nonzero_idxs[2]] = predictions[episode_row, nonzero_idxs[
                    0], nonzero_idxs[1], nonzero_idxs[2]]

    return asset_ids, transformed_preds


def subtract_date_arrays(scalar_date, date_array, resolution):
    """Subtract vector of dates from scalar date.

    Arguments:
        scalar_date: pd.Timestamp object.
        date_array: Index or array of timestamps (not just list).
        resolution: pd.Timedelta object.

    Returns:
        a vector of floats of length len(date_array) in time units of
        *resolution*.
    """
    return np.array((scalar_date - date_array) / resolution)
