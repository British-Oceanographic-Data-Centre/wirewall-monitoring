"""Helper module for retrieving and plotting WireWall data."""
from pathlib import Path
from warnings import warn

import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
from erddapy import ERDDAP

_XAXIS_FORMAT = dict(
    rangeselector=dict(
        buttons=[
            dict(count=1, label="1d", step="day", stepmode="backward"),
            dict(step="all"),
        ],
    ),
    rangeslider=dict(visible=True),
    type="date",
)
_MARGIN_FORMAT = dict(t=50, b=50)


class WireWallMonitor:
    """A class to handle retrieval and plotting of WireWall data."""

    window_time_column = "time (UTC)"
    event_time_column = "event time (UTC)"
    series_column = "wireID (Dmnless)"
    datetime_fields = ["time (UTC)", "gpsTime (UTC)", "timestamp (UTC)"]

    def __init__(
        self, erddap_server, constraints=None, protocol="tabledap", response="csv"
    ):
        """Initialise based on given ERDDAP instance."""
        self._erddap = ERDDAP(
            server=erddap_server,
            protocol=protocol,
            response=response,
        )

        self._erddap.constraints = constraints or []

    def _add_event_columns(self, df):
        """Add a new columns which apply to events."""
        # calculate the event height with the baseline removed
        df["event depth preferred (cm)"] = df["elMEAN (cm)"] - df["MEDelMEAN (cm)"]
        df["event depth fallback (cm)"] = df["elPTILE_6 (cm)"] - df["MEDelPTILE_2 (cm)"]

        df[self.event_time_column] = df[self.window_time_column].copy()
        time_delta = df["sampleNUM (Dmnless)"] - df["sampleNUM10 (Dmnless)"]

        # events occur at ~400Hz
        time_delta /= 400

        # we need this as an actual timedelta
        time_delta = time_delta.apply(pd.to_timedelta, unit="S")

        # check all events occur in the interval [0, 10] mins from the window start time
        if time_delta.max() > pd.to_timedelta("10m"):
            warn(
                "Data has an event that occurs after the 10min sample window.",
                UserWarning,
            )

        if time_delta.min() < pd.to_timedelta("0m"):
            warn(
                "Data has an event that occurs before the 10min sample window.",
                UserWarning,
            )

        df[self.event_time_column] += time_delta

    def _get_dataframe(self, dataset_id):
        """Retrieve a dataframe for the given dataset_id."""
        self._erddap.dataset_id = dataset_id

        df = self._erddap.to_pandas(parse_dates=self.datetime_fields)
        df[self.series_column] = df[self.series_column].astype(str)
        self._add_event_columns(df)

        self._erddap.dataset_id = None

        return df

    def _plot_dataframe(self, df, x, y):
        """Plot the given columns of the dataframe."""
        fig = px.scatter(df, x=x, y=y, color=self.series_column)

        fig.update_layout(
            yaxis_title=y,
            xaxis=_XAXIS_FORMAT,
            margin=_MARGIN_FORMAT,
        )

        return fig

    def _plot_window_variables(self, df, column_names, column_names_secondary):
        """Plot variables that are constant over a window timespan."""
        # since these variables are constant over a given window, for a given wire
        # we can remove any rows which are duplicated
        df = df.drop_duplicates(
            [self.window_time_column, self.series_column], keep="first"
        )

        figs = [None] * len(column_names)

        # use a loop so we can update each fig
        for i, (name, name_secondary) in enumerate(
            zip(column_names, column_names_secondary)
        ):
            subfig1 = self._plot_dataframe(
                df,
                x=self.window_time_column,
                y=name,
            )

            yaxis_title = name

            if name_secondary is None:
                fig = subfig1
            else:
                fig = make_subplots()

                # get the units from the columns
                units = {s.split(" ")[1] for s in [name, name_secondary]}
                yaxis_title = "value " + " or ".join(units)

                subfig2 = self._plot_dataframe(
                    df,
                    x=self.window_time_column,
                    y=name_secondary,
                )

                # since this plot now has two series, rename them both
                subfig2.for_each_trace(
                    lambda trace: trace.update(
                        name=f"Wire {trace.name} {name_secondary}"
                    )
                )
                subfig1.for_each_trace(
                    lambda trace: trace.update(name=f"Wire {trace.name} {name}")
                )

                # distinguish the second trace from the first
                subfig2.update_traces(
                    marker_symbol="square",
                    line_dash="dot",
                )

                # combine the traces into one figure
                fig.add_traces(subfig1.data + subfig2.data)

            fig.update_traces(mode="lines+markers", selector=dict(type="scatter"))
            fig.update_layout(
                yaxis_title=yaxis_title,
                xaxis_title=self.window_time_column,
                xaxis=_XAXIS_FORMAT,
                margin=_MARGIN_FORMAT,
            )

            figs[i] = fig

        return figs

    def _plot_event_variables(self, df, column_names):
        """Plot event variables."""
        # some windows don't have any events and so there may be rows without any
        # sample num value. We are only interested in actual events, so remove them
        df = df.dropna(subset=[self.event_time_column]).copy()

        figs = [None] * len(column_names)

        # use a loop so we can update each fig
        for i, name in enumerate(column_names):
            figs[i] = self._plot_dataframe(df, x=self.event_time_column, y=name)

        return figs

    def plot_variables(
        self,
        dataset_id,
        window_variables=None,
        window_variables_secondary=None,
        event_variables=None,
    ):
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
        window_variables_secondary = window_variables_secondary or [None] * len(
            window_variables
        )
        event_variables = event_variables or []

        df = self._get_dataframe(dataset_id)
        window_figs = self._plot_window_variables(
            df, window_variables, window_variables_secondary
        )
        event_figs = self._plot_event_variables(df, event_variables)

        figs = [*window_figs, *event_figs]

        for fig in figs:
            fig.show()

        return figs
