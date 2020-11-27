"""Helper module for retrieving and plotting WireWall data."""
from pathlib import Path

import pandas as pd
from erddapy import ERDDAP

# use plotly for interactive plots
pd.options.plotting.backend = "plotly"


class WireWallMonitor:
    """A class to handle retrieval and plotting of WireWall data."""

    window_time_column = "time (UTC)"
    event_time_column = "event time (UTC)"
    series_column = "wireID (Dmnless)"
    datetime_fields = ["time (UTC)", "gpsTime (UTC)", "timestamp (UTC)"]

    def __init__(self, erddap_server, protocol="tabledap", response="csv"):
        """Initialise based on given ERDDAP instance."""
        self._erddap = ERDDAP(
            server=erddap_server,
            protocol=protocol,
            response=response,
        )

    def _add_event_columns(self, df):
        """Add a new columns which apply to events."""
        # calculate the event height with the baseline removed
        df["event height (cm)"] = df["elPTILE_6 (cm)"] - df["MEDelPTILE_2 (cm)"]

        df[self.event_time_column] = df[self.window_time_column].copy()
        time_delta = df["sampleNUM (Dmnless)"] - df["sampleNUM10 (Dmnless)"]

        # events occur at ~400Hz
        time_delta /= 400

        # we need this as an actual timedelta
        time_delta = time_delta.apply(pd.to_timedelta, unit="S")

        # check all events occur in the interval [0, 10] mins from the window start time
        assert time_delta.max() <= pd.to_timedelta(
            "10m"
        ), "Event time is after 10min window"
        assert time_delta.min() >= pd.to_timedelta(
            "0m"
        ), "Event time is before 10min window"

        df[self.event_time_column] += time_delta

    def _get_dataframe(self, dataset_id):
        """Retrieve a dataframe for the given dataset_id."""
        self._erddap.dataset_id = dataset_id

        df = self._erddap.to_pandas(parse_dates=self.datetime_fields)
        self._add_event_columns(df)

        self._erddap.dataset_id = None

        return df

    def _plot_window_variables(self, df, column_names):
        """Plot variables that are constant over a window timespan."""
        # since these variables are constant over a given window, for a given wire
        # we can remove any rows which are duplicated
        df = df.drop_duplicates(
            [self.window_time_column, self.series_column], keep="first"
        )
        df = df.pivot(index=self.window_time_column, columns=self.series_column)

        figs = [None] * len(column_names)

        # use a loop so we can update each fig
        for i, name in enumerate(column_names):
            fig = _plot_dataframe(name, df[name])
            fig.update_traces(mode="lines+markers", selector=dict(type="scatter"))
            figs[i] = fig

        return figs

    def _plot_event_variables(self, df, column_names):
        """Plot event variables."""
        # some windows don't have any events and so there may be rows without any
        # sample num value. We are only interested in actual events, so remove them
        df = df.dropna(subset=[self.event_time_column]).copy()
        df = df.pivot(index=self.event_time_column, columns=self.series_column)

        figs = [None] * len(column_names)

        # use a loop so we can update each fig
        for i, name in enumerate(column_names):
            figs[i] = _plot_dataframe(name, df[name])

        return figs

    def plot_variables(self, dataset_id, window_variables=None, event_variables=None):
        """Plot all the window and event variables for a given dataset.

        Args:
            dataset_id (str): the string identifier for the ERDDAP dataset.
            window_variables (list): a list of variable names (including units) which are
                constant over each window, to be plotted.
            event_variables (list): a list of variable names (including units) which are
                specific to each event, to be plotted.

        Returns: a list of figures generated, and calls .show() on all of them.
        """
        window_variables = window_variables or []
        event_variables = event_variables or []

        df = self._get_dataframe(dataset_id)
        window_figs = self._plot_window_variables(df, window_variables)
        event_figs = self._plot_event_variables(df, event_variables)

        figs = [*window_figs, *event_figs]

        for fig in figs:
            fig.show()

        return figs


def _plot_dataframe(name, df):
    """Plot the index on x-axis and columns on the y-axis."""
    fig = df.plot.scatter(x=df.index, y=df.columns)

    fig.update_layout(
        yaxis=dict(
            title=name,
        ),
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(step="all"),
                ],
            ),
            rangeslider=dict(visible=True),
            type="date",
        ),
    )

    return fig
