from __future__ import annotations
from typing import TYPE_CHECKING
from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np

from dat_analysis import useful_functions as u
from dat_analysis.analysis_tools.square_wave import get_setpoint_indexes_from_times
from dat_analysis.plotting.plotly import OneD, TwoD
from dat_analysis.characters import DELTA

from dash_data_viewer.cache import cache

if TYPE_CHECKING:
    from dash.development.base_component import Component
    from dat_analysis.dat_object.dat_hdf import DatHDF
    from dat_analysis.dat_object.attributes.square_entropy import Output
    from dat_analysis.analysis_tools.general_fitting import FitInfo
    from dat_analysis.dat_object.attributes.entropy import IntegrationInfo


class EntropyReport:
    """Class for generating dash components which summarize entropy"""

    def __init__(self, dat: DatHDF):
        self.dat = dat
        self.p1d = OneD(dat=dat)
        self.p2d = TwoD(dat=dat)

        self.out: Output = None
        self.fit: FitInfo = None
        self.int_info: IntegrationInfo = None

    def generate_out(self):
        if not self.out:
            try:
                out = self.dat.SquareEntropy.get_Outputs(name='default')
            except u.NotFoundInHdfError:
                start, _ = get_setpoint_indexes_from_times(self.dat, 0.005)
                pp = self.dat.SquareEntropy.get_ProcessParams(setpoint_start=start)
                out = self.dat.SquareEntropy.get_Outputs(name='default', calculate_only=False, process_params=pp)
            self.out = out

    def generate_fit(self):
        if not self.fit:
            self.generate_out()
            out = self.out

            self.fit = self.dat.Entropy.get_fit(calculate_only=True, x=out.x, data=out.average_entropy_signal)

    def generate_integration_info(self):
        if not self.int_info:
            try:
                self.int_info = self.dat.Entropy.get_integration_info('default')
            except u.NotFoundInHdfError:
                self.generate_out()
                out = self.out

                cold_fit = self.dat.SquareEntropy.get_fit(calculate_only=True, x=out.x, data=out.averaged,
                                                          transition_part='cold')
                hot_fit = self.dat.SquareEntropy.get_fit(calculate_only=True, x=out.x, data=out.averaged,
                                                         transition_part='hot')
                dT = hot_fit.best_values.theta - cold_fit.best_values.theta
                self.int_info = self.dat.Entropy.set_integration_info(dT=dT, amp=cold_fit.best_values.amp,
                                                                      name='default',
                                                                      overwrite=True)

    def figure_signal_and_fit(self) -> go.Figure:
        self.generate_out()
        self.generate_fit()
        p1d = self.p1d
        fig = p1d.plot(data=self.out.average_entropy_signal, x=self.out.x, ylabel='DeltaI /nA',
                       title=f'Dat{self.dat.datnum}: Average Entropy Signal (dS={self.fit.best_values.dS:.2f}kB)',
                       mode='markers+lines',
                       trace_kwargs=dict(
                           marker=dict(size=5, symbol='cross-thin'),
                           line=dict(width=2)
                       ))
        fig.add_trace(p1d.trace(data=self.fit.eval_fit(x=self.out.x), x=self.out.x, mode='lines'))
        return fig

    def figure_integrated(self) -> go.Figure:
        """
        Returns a figure with a single Integrated entropy trace and dashed lines for Ln2, Ln3, 0
        Returns:

        """
        self.generate_out()
        self.generate_integration_info()
        p1d = self.p1d
        integrated = self.int_info.integrate(self.out.average_entropy_signal)
        fig = p1d.figure(title=f'Dat{self.dat.datnum}: '
                               f'Integrated Entropy (dS={integrated[-1]:.2f}kB, '
                               f'dT={self.int_info.dT:.2f}mV)',
                         ylabel='Entropy /kB')
        fig.add_trace(p1d.trace(x=self.out.x, data=integrated, mode='lines'))
        p1d.add_line(fig, np.log(2), linetype='dash')
        p1d.add_line(fig, np.log(3), linetype='dash')
        p1d.add_line(fig, 0, linetype='solid')
        return fig

    def figure_square_wave(self, start: float = -np.inf, end: float = np.inf) -> go.Figure:
        self.generate_out()
        p1d = self.p1d

        x_range = [start, end]
        wavelen = self.dat.Logs.awg.wave_len
        measure_freq = self.dat.Logs.measure_freq
        x = self.dat.Data.x_array
        data = self.dat.Data.i_sense

        indexs = u.get_data_index(x, x_range)
        indexs = [ind - ind % wavelen for ind in indexs]
        z = np.mean(data[:, indexs[0]:indexs[1]], axis=0)
        z = np.reshape(z, (-1, wavelen))
        z = np.mean(z, axis=0)
        z = z - np.mean(z)
        lin_x = np.linspace(0, wavelen * 1 / measure_freq, wavelen)

        fig = p1d.figure(
            title=f'Dat{self.dat.datnum}: Square wave data averaged from  {x_range[0]}mV to {x_range[1]}mV',
            xlabel='Time through heating cycle /s',
            ylabel=f'{DELTA}I /nA'
        )
        fig.add_trace(p1d.trace(x=lin_x, data=z))
        return fig

    def full_report(self) -> Component:
        layout = dbc.Container([
            dbc.Row(html.H3('Entropy Report:')),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=self.figure_signal_and_fit())
                ], width=4),
                dbc.Col([
                    dcc.Graph(figure=self.figure_integrated())
                ], width=4),
                dbc.Col([
                    dcc.Graph(figure=self.figure_square_wave())
                ], width=4),
            ])
        ])
        return layout
