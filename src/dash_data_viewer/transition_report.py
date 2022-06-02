from __future__ import annotations
from typing import TYPE_CHECKING
from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
from dataclasses import dataclass
from deprecation import deprecated

from dat_analysis import useful_functions as u
from dat_analysis.analysis_tools.square_wave import get_setpoint_indexes_from_times
from dat_analysis.plotting.plotly import OneD, TwoD
from dat_analysis.characters import DELTA, THETA, PM

from dash_data_viewer.cache import cache

if TYPE_CHECKING:
    from dash.development.base_component import Component
    from dat_analysis.dat_object.dat_hdf import DatHDF
    from dat_analysis.dat_object.attributes.square_entropy import Output
    from dat_analysis.analysis_tools.general_fitting import FitInfo
    from dat_analysis.dat_object.attributes.entropy import IntegrationInfo


@deprecated(deprecated_in='20220601', details='Going down a different route with dash now')
@dataclass
class TransitionData:
    x: np.ndarray
    data: np.ndarray
    avg_x: np.ndarray
    avg_data: np.ndarray
    avg_data_std: np.ndarray
    is2d: bool


class TransitionReport:
    """Class for generating dash components which summarize transition"""

    def __init__(self, dat: DatHDF):
        self.dat = dat
        self.p1d = OneD(dat=dat)
        self.p2d = TwoD(dat=dat)

        self.data: TransitionData = None
        self.fit: FitInfo = None

    def generate_data(self):
        if not self.data:
            dat = self.dat
            x = dat.Data.x
            data = dat.Data.i_sense

            if data.ndim == 2:
                is2d = True
                avg_data, avg_data_std, avg_x = dat.Transition.get_avg_data(check_exists=False,
                                                                            return_std=True,
                                                                            return_x=True)
            else:
                is2d = False
                avg_data, avg_data_std, avg_x = None, None, None

            self.data = TransitionData(
                x=x,
                data=data,
                avg_data=avg_data,
                avg_x=avg_x,
                avg_data_std=avg_data_std,
                is2d=is2d,
            )

    def generate_fit(self):
        if not self.fit:
            self.generate_data()
            x = self.data.avg_x if self.data.avg_x is not None else self.data.x
            data = self.data.avg_data if self.data.avg_data is not None else self.data.data
            self.fit = self.dat.Transition.get_fit(calculate_only=True, x=x, data=data)

    def figure_data2d(self) -> go.Figure:
        self.generate_data()
        if not self.data.is2d:
            raise ValueError(f'Dat{self.dat.datnum} is not 2D? (data.ndim = {self.data.data.ndim})')

        data = self.data
        y = self.dat.Data.y
        fig = self.p2d.plot(data=data.data, x=data.x, y=y,
                            title=f'Dat{self.dat.datnum}: 2D Transition Data')
        return fig

    def figure_avg_data_and_fit(self) -> go.Figure:
        """
        Returns a figure with avg data and fit to avg data
        Returns:

        """
        self.generate_data()
        self.generate_fit()
        p1d = self.p1d
        if self.data.avg_data is not None:
            x, data, data_err = self.data.avg_x, self.data.avg_data, self.data.avg_data_std
            title_text = 'Average'
        else:
            x, data, data_err = self.data.x, self.data.data, None
            title_text = '1D only'

        fig = p1d.plot(data=data, data_err=data_err, x=x,
                       ylabel='Current /nA',
                       title=f'Dat{self.dat.datnum}: {title_text} Transition data',
                       mode='markers+lines',
                       trace_kwargs=dict(
                           marker=dict(size=5, symbol='cross-thin'),
                           line=dict(width=2)
                       ))
        fig.add_trace(p1d.trace(data=self.fit.eval_fit(x=x), x=x, mode='lines'))
        return fig

    def text_fit_report(self) -> dcc.Markdown:
        self.generate_fit()
        param_info = {
            name: dict(val=param.value, std=param.stderr if param.stderr else 0, init=param.init_value)
            for name, param in self.fit.params.items()
        }
        strings = [f'{k}: {d["val"]:.3g}{PM}{d["std"]:.2g} (Init: {d["init"]:.1g})' for k, d in param_info.items()]
        text = '  \n'.join(strings)
        return dcc.Markdown(f'#### Dat{self.dat.datnum} Transition Fit Params \n' + text,
                            style={'white-space': 'pre', 'overflow-x': 'scroll'})

    def full_report(self) -> Component:
        self.generate_data()
        if self.data.is2d:
            layout = dbc.Card([
                dbc.CardHeader(html.H3('Transition Report:')),
                dbc.CardBody(
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(figure=self.figure_avg_data_and_fit())
                        ], width=4),
                        dbc.Col([
                            dcc.Graph(figure=self.figure_data2d())
                        ], width=4),
                        dbc.Col([
                            self.text_fit_report()
                        ], width=4),
                    ])
                )
            ])
        else:
            layout = dbc.Card([
                dbc.CardHeader(html.H3('Transition Report:')),
                dbc.CardBody(
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(figure=self.figure_avg_data_and_fit())
                        ], width=6),
                        dbc.Col([
                            self.text_fit_report()
                        ], width=6),
                    ])),
            ])
        return layout
