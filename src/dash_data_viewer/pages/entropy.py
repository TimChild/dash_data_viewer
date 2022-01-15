from __future__ import annotations
import dash
import copy
import numpy as np
import logging
import plotly.graph_objects as go
from dash_extensions.snippets import get_triggered
from dataclasses import dataclass, asdict
from dacite import from_dict
from dash import html, dcc, callback, Input, Output, MATCH, ALL, ALLSMALLER, State
import dash_bootstrap_components as dbc
from functools import lru_cache
from dash_data_viewer.layout_util import label_component
import uuid
import dash_data_viewer.components as c
from dash_data_viewer.layout_util import vertical_label, label_component

import dat_analysis
from dat_analysis import get_dat, get_dats
from dat_analysis.analysis_tools.general_fitting import calculate_fit, FitInfo
from dat_analysis.plotting.plotly import OneD, TwoD
from dat_analysis.analysis_tools.square_wave import get_setpoint_indexes_from_times, square_wave_time_array
from dat_analysis.dat_object.attributes.square_entropy import process_avg_parts
from dat_analysis.characters import DELTA, PM
from dat_analysis import useful_functions as U
from dat_analysis.dat_object.attributes.entropy import scaling, IntegrationInfo

# from dash_data_viewer.layout_util import label_component

from typing import TYPE_CHECKING, Optional, List, Union

if TYPE_CHECKING:
    from dash.development.base_component import Component
    from dat_analysis.dat_object.dat_hdf import DatHDF

FLOAT_REGEX = '^(-)?\d*(\.\d+)?$'


def sanitize_floats(*args):
    """
    Either returns None if None passed in or float not valid, otherwise converts string to float
    Args:
        *args ():

    Returns:

    """
    santized = []
    for arg in args:
        try:
            new = float(arg) if arg else None
        except ValueError:
            new = None
        santized.append(new)
    return santized


class EntropySignalAIO(html.Div):
    """
    Controls for getting entropy signal from dat (i.e. step_delay, centered)

    # Requires
    None

    # Provides
    Store(id=self.store_id) -- Filled Info class (e.g. including step_delay, centered)
    Store(id=self.square_store_id) -- Filled Info class from SquareSelectionAIO
    """

    @dataclass
    class Info:
        step_delay: str | float | int = 0.0
        centered: bool = False
        graph_type_2d: str = 'heatmap'

        def __post_init__(self):
            if isinstance(self.step_delay, str):
                try:
                    self.step_delay = float(self.step_delay)
                except ValueError:
                    self.step_delay = 0.0

    @classmethod
    def info_from_dict(cls, d: dict):
        if not d:
            info = cls.Info()
        else:
            info = from_dict(cls.Info, d)
        return info

    # Functions to create pattern-matching callbacks of the subcomponents
    class ids:
        @staticmethod
        def generic(aio_id, subcomponent, key: str):
            return {
                'component': 'EntropySignalAIO',
                'subcomponent': subcomponent,
                'key': key,
                'aio_id': aio_id,
            }

    def __init__(self, aio_id):
        self.store_id = self.ids.generic(aio_id, 'store', 'output')
        output_store = dcc.Store(id=self.store_id, storage_type='session')

        square_selection = SquareSelectionAIO(aio_id=aio_id)
        self.square_store_id = square_selection.store_id

        layout = dbc.Card([
            output_store,
            dbc.CardHeader('Entropy Signal'),
            dbc.CardBody([
                label_component(c.Input_(
                    id=self.ids.generic(aio_id, 'input', 'step_delay'),
                    type='text', pattern=FLOAT_REGEX,
                    persistence=True, persistence_type='local',
                ), 'Delay after step: '),
                dbc.Button(id=self.ids.generic(aio_id, 'button', 'centered'), children='Centered'),
                square_selection
            ]),
        ])
        super().__init__(children=layout)  # html.Div contains layout

    @staticmethod
    @callback(
        Output(ids.generic(MATCH, 'store', 'output'), 'data'),
        Output(ids.generic(MATCH, 'button', 'centered'), 'color'),
        Input(ids.generic(MATCH, 'input', 'step_delay'), 'value'),
        Input(ids.generic(MATCH, 'button', 'centered'), 'n_clicks'),
        State(ids.generic(MATCH, 'store', 'output'), 'data'),
    )
    def function(step_delay, center_clicks, state):
        state = EntropySignalAIO.info_from_dict(state)
        state.step_delay = step_delay if step_delay else 0
        triggered = get_triggered()
        if triggered.id and triggered.id['key'] == 'centered':
            state.centered = not state.centered  # Toggle button state
        return asdict(state), 'success' if state.centered else 'danger'


class SquareSelectionAIO(html.Div):
    """
    Controls to decide on selection of SquareWave

    # Requires
    None

    # Provides
    Store(id=self.store_id) -- Filled Info class
    """

    @dataclass
    class Info:
        peak: bool = True
        trough: bool = True
        width: float = 10
        range: bool = False
        range_start: Optional[float] = None
        range_stop: Optional[float] = None

    @classmethod
    def info_from_dict(cls, d: dict):
        if not d:
            info = cls.Info()
        else:
            info = from_dict(cls.Info, d)
        return info

    # Functions to create pattern-matching callbacks of the subcomponents
    class ids:
        @staticmethod
        def generic(aio_id, subcomponent, key: str):
            return {
                'component': 'SquareSelectionAIO',
                'subcomponent': subcomponent,
                'key': key,
                'aio_id': aio_id,
            }

        @staticmethod
        def input(aio_id, key: str):
            return {
                'component': 'SquareSelectionAIO',
                'subcomponent': 'input',
                'key': key,
                'aio_id': aio_id,
            }

    def __init__(self, aio_id):
        self.store_id = self.ids.generic(aio_id, 'store', 'output')
        output_store = dcc.Store(id=self.store_id, storage_type='session')

        layout = dbc.Card([
            output_store,
            dbc.CardHeader('Square Wave'),
            dbc.CardBody([
                c.MultiButtonAIO(button_texts=['Peak', 'Trough', 'Range'],
                                 button_values=['peak', 'trough', 'range'],
                                 aio_id=aio_id, allow_multiple=True, storage_type='local',
                                 ),
                dbc.Row([
                    dbc.Col(
                        vertical_label('Width',
                                       c.Input_(id=self.ids.input(aio_id, 'width'), type='number', value=10, min=0,
                                                persistence=True))),
                    dbc.Col(vertical_label('Range Start', c.Input_(id=self.ids.input(aio_id, 'start'), type='number',
                                                                   persistence=True))),
                    dbc.Col(vertical_label('Range Stop', c.Input_(id=self.ids.input(aio_id, 'stop'), type='number',
                                                                  persistence=True))),
                ]),
            ]),
        ])

        super().__init__(children=layout)  # html.Div contains layout

    @staticmethod
    @callback(
        Output(ids.generic(MATCH, 'store', 'output'), 'data'),
        Input(c.MultiButtonAIO.ids.store(MATCH), 'data'),
        Input(ids.input(MATCH, 'width'), 'value'),
        Input(ids.input(MATCH, 'start'), 'value'),
        Input(ids.input(MATCH, 'stop'), 'value'),
    )
    def function(buttons, width, start, stop):
        button_info = c.MultiButtonAIO.info_from_dict(buttons)
        buttons_selected = {}
        for k in ['peak', 'trough', 'range']:
            buttons_selected[k] = True if k in button_info.selected_values else False
        return asdict(SquareSelectionAIO.Info(**buttons_selected, width=width, range_start=start, range_stop=stop))


class IntegratedEntropyAIO(html.Div):
    """
    Controls for getting Integrated entropy

    # Requires
    None

    # Provides
    Store(id=self.store_id) -- Filled Info class (e.g. including step_delay, centered)
    Store(id=self.dt_store_id) -- Filled Info class from GetDtAIO()
    """

    @dataclass
    class Info:
        zero_pos: float | int | None = None
        end_pos: float | int | None = None

    @classmethod
    def info_from_dict(cls, d: dict):
        if not d:
            info = cls.Info()
        else:
            info = from_dict(cls.Info, d)
        return info

    # Functions to create pattern-matching callbacks of the subcomponents
    class ids:
        @staticmethod
        def generic(aio_id, subcomponent, key: str):
            return {
                'component': 'IntegratedEntropyAIO',
                'subcomponent': subcomponent,
                'key': key,
                'aio_id': aio_id,
            }

    def __init__(self, aio_id):
        self.store_id = self.ids.generic(aio_id, 'store', 'output')
        output_store = dcc.Store(id=self.store_id, storage_type='session')

        dtAIO = GetDtAIO(aio_id=aio_id)
        self.dt_store_id = dtAIO.store_id

        layout = dbc.Card([
            output_store,
            dbc.CardHeader('Integrated Entropy'),
            dbc.CardBody([
                dtAIO,
                label_component(
                    c.Input_(
                        id=self.ids.generic(aio_id, 'input', 'zero'),
                        type='text', pattern=FLOAT_REGEX,
                        persistence=True, persistence_type='local',
                    ),
                    'Zero Pos: '
                ),
                label_component(
                    c.Input_(
                        id=self.ids.generic(aio_id, 'input', 'end'),
                        type='text', pattern=FLOAT_REGEX,
                        persistence=True, persistence_type='local',
                    ),
                    'End Pos: '
                ),
            ]),
        ])
        super().__init__(children=layout)  # html.Div contains layout

    @staticmethod
    @callback(
        Output(ids.generic(MATCH, 'store', 'output'), 'data'),
        Input(ids.generic(MATCH, 'input', 'zero'), 'value'),
        Input(ids.generic(MATCH, 'input', 'end'), 'value'),
    )
    def function(zero, end):
        zero, end = sanitize_floats(zero, end)
        return asdict(IntegratedEntropyAIO.Info(zero_pos=zero, end_pos=end))


class GetDtAIO(html.Div):
    """
    Controls for how to get dT for data

    # Requires
    None

    # Provides
    Store(id=self.store_id) -- Filled Info class
    """

    @dataclass
    class Info:
        method: str = 'fixed'
        value: Optional[float] = None
        ref_channel_value: float | None = None
        ref_dt: float | None = None
        linear_term: float | None = None
        constant_term: float | None = None
        channel: Optional[str] = None

    @classmethod
    def info_from_dict(cls, d: dict):
        if not d:
            info = cls.Info()
        else:
            info = from_dict(cls.Info, d)
        return info

    # Functions to create pattern-matching callbacks of the subcomponents
    class ids:
        @staticmethod
        def generic(aio_id, subcomponent, key: str):
            return {
                'component': 'GetDtAIO',
                'subcomponent': subcomponent,
                'key': key,
                'aio_id': aio_id,
            }

        @staticmethod
        def input(aio_id, key: str):
            return {
                'component': 'GetDtAIO',
                'subcomponent': 'input',
                'key': key,
                'aio_id': aio_id,
            }

    def __init__(self, aio_id):
        self.store_id = self.ids.generic(aio_id, 'store', 'output')
        output_store = dcc.Store(id=self.store_id, storage_type='session')

        layout = dbc.Card([
            output_store,
            dbc.CardHeader('dT'),
            dbc.CardBody(
                dbc.Form([
                    c.MultiButtonAIO(button_texts=['Fixed', 'Dat', 'Equation'],
                                     button_values=['fixed', 'dat', 'eq'],
                                     aio_id=aio_id, allow_multiple=False, storage_type='local',
                                     ),
                    dbc.Label('Value', html_for=self.ids.input(aio_id, 'value')),
                    c.Input_(id=self.ids.input(aio_id, 'value'), type='text', pattern=FLOAT_REGEX, placeholder='1.0',
                             persistence=True),

                    dbc.Label('Ref Channel Value', html_for=self.ids.input(aio_id, 'ref_value')),
                    c.Input_(id=self.ids.input(aio_id, 'ref_value'), type='text', pattern=FLOAT_REGEX,
                             placeholder='-250',
                             persistence=True),

                    dbc.Label('Ref dT', html_for=self.ids.input(aio_id, 'ref_dt')),
                    c.Input_(id=self.ids.input(aio_id, 'ref_dt'), type='text', pattern=FLOAT_REGEX, placeholder='1.0',
                             persistence=True),

                    dbc.Label('Linear Term', html_for=self.ids.input(aio_id, 'linear')),
                    c.Input_(id=self.ids.input(aio_id, 'linear'), type='text', pattern=FLOAT_REGEX, placeholder='0.001',
                             persistence=True),

                    dbc.Label('Constant Term', html_for=self.ids.input(aio_id, 'constant')),
                    c.Input_(id=self.ids.input(aio_id, 'constant'), type='text', pattern=FLOAT_REGEX, placeholder='0.0',
                             persistence=True),

                    dbc.Label('Channel', html_for=self.ids.input(aio_id, 'channel')),
                    c.Input_(id=self.ids.input(aio_id, 'channel'), type='text', placeholder='ESC', persistence=True),
                ])
            ),
        ])

        super().__init__(children=layout)  # html.Div contains layout

    @staticmethod
    @callback(
        Output(ids.generic(MATCH, 'store', 'output'), 'data'),
        Input(c.MultiButtonAIO.ids.store(MATCH), 'data'),
        Input(ids.input(MATCH, 'value'), 'value'),
        Input(ids.input(MATCH, 'ref_value'), 'value'),
        Input(ids.input(MATCH, 'ref_dt'), 'value'),
        Input(ids.input(MATCH, 'linear'), 'value'),
        Input(ids.input(MATCH, 'constant'), 'value'),
        Input(ids.input(MATCH, 'channel'), 'value'),
    )
    def function(button, value, ref_value, ref_dt, linear, const, channel):
        button = c.MultiButtonAIO.info_from_dict(button)
        value, ref_value, ref_dt, linear, const = sanitize_floats(value, ref_value, ref_dt, linear, const)

        info = GetDtAIO.Info(
            method=button.selected_values,
            value=value,
            ref_channel_value=ref_value,
            ref_dt=ref_dt,
            linear_term=linear,
            constant_term=const,
            channel=channel,
        )
        return asdict(info)

    @staticmethod
    @callback(
        Output(ids.input(MATCH, 'value'), 'disabled'),
        Output(ids.input(MATCH, 'ref_value'), 'disabled'),
        Output(ids.input(MATCH, 'ref_dt'), 'disabled'),
        Output(ids.input(MATCH, 'linear'), 'disabled'),
        Output(ids.input(MATCH, 'constant'), 'disabled'),
        Output(ids.input(MATCH, 'channel'), 'disabled'),
        Input(c.MultiButtonAIO.ids.store(MATCH), 'data'),
    )
    def allowed(button):
        button = c.MultiButtonAIO.info_from_dict(button)
        disabled_state = [True]*6
        if button.selected_values == 'fixed':
            disabled_state[0] = False
        elif button.selected_values == 'eq':
            disabled_state = [False] * 6
            disabled_state[0] = True
        return disabled_state


class MainComponents(object):
    """Convenient holder for any components that will end up in the main area of the page"""
    signal_div = html.Div(id='entropy-div-signal')
    fit_div = html.Div(id='entropy-div-fit')
    integrated_div = html.Div(id='entropy-div-integrated')

    def layout(self):
        layout = html.Div(children=[
            dbc.Card(children=[
                dbc.CardHeader(html.H3('Entropy Signal')),
                dbc.CardBody(self.signal_div),
            ]
            ),
            dbc.Card(children=[
                dbc.CardHeader(html.H3('Fitting Data')),
                dbc.CardBody(self.fit_div),
            ]
            ),
            dbc.Card(children=[
                dbc.CardHeader(html.H3('Integrated Entropy')),
                dbc.CardBody(self.integrated_div),
            ]
            ),
        ]
        )
        return layout
    # graph_2d_transition = dcc.Graph(id='entropy-graph-2d-transition')
    # graph_average_transition = dcc.Graph(id='entropy-graph-average-transition')
    #
    # graph_2d_entropy = dcc.Graph(id='entropy-graph-2d-entropy')
    # graph_average_entropy = dcc.Graph(id='entropy-graph-average-entropy')
    #
    # graph_square_wave = dcc.Graph(id='entropy-graph-squarewave')
    # graph_integrated = dcc.Graph(id='entropy-graph-integrated')
    #
    # def layout(self):
    #     transition_graphs = dbc.Row([
    #         dbc.Col(self.graph_2d_transition, width=6), dbc.Col(self.graph_average_transition, width=6)
    #     ])
    #     entropy_graphs = dbc.Row([
    #         dbc.Row([dbc.Col(self.graph_2d_entropy, width=6), dbc.Col(self.graph_average_entropy, width=6)]),
    #         dbc.Row([dbc.Col(self.graph_integrated, width=6), dbc.Col(self.graph_square_wave, width=6)]),
    #     ])
    #     entropy_info = dbc.Row([
    #         html.Div(f'No info to show yet')
    #     ])
    #     layout_ = html.Div([
    #         transition_graphs,
    #         entropy_graphs,
    #         entropy_info,
    #     ])
    #     return layout_


class SidebarComponents(object):
    """Convenient holder for any components that will end up in the sidebar area of the page"""
    datnum = c.DatnumPickerAIO(aio_id='entropy-datnumPicker', allow_multiple=False)
    entropy_signal_controls = EntropySignalAIO(aio_id='entropy-signalAIO')
    fit_entropy_placeholder = dbc.Card([
        dbc.CardHeader('Fitting Info'),
        dbc.CardBody(),
    ])
    integrated_entropy_controls = IntegratedEntropyAIO(aio_id='entropy-integratedAIO')
    update = dbc.Button(id='entropy-update', children='Update', size='lg')

    def layout(self):
        layout_ = html.Div([
            self.update,
            self.datnum,
            self.entropy_signal_controls,
            self.fit_entropy_placeholder,
            self.integrated_entropy_controls,
        ])
        return layout_


# Initialize the components ONCE here.
main_components = MainComponents()
sidebar_components = SidebarComponents()


# TODO: Instead of updating the input value (which prevents the last value from being stored locally)
# TODO: Update another cell with a value based on input and rules (although this will depend on the dat, so not sure
# TODO: how to incoporate that properly).
@callback(
    Output(EntropySignalAIO.ids.generic(MATCH, 'input', 'step_delay'), 'value'),
    Input(EntropySignalAIO.ids.generic(MATCH, 'input', 'step_delay'), 'n_blur'),
    State(EntropySignalAIO.ids.generic(MATCH, 'input', 'step_delay'), 'value'),
    State(sidebar_components.datnum.dd_id, 'value'),
    State(c.ConfigAIO.store_id, 'data'),
    prevent_initial_callback=True,
)
def sanitize_step_delay(trigger, value, datnum, config):
    config = c.ConfigAIO.config_from_dict(config)
    if datnum and value:
        value = float(value)
        dat = get_dat(datnum, exp2hdf=config.experiment)
        awg = dat.SquareEntropy.square_awg
        full_setpoint_time = awg.info.wave_len / 4 * 1 / awg.info.measureFreq
        if value >= full_setpoint_time:
            value = round(full_setpoint_time - 0.0001, 4)
    return value


def check_dat_valid(dat: DatHDF) -> bool:
    try:
        awg = dat.Logs.awg
    except U.NotFoundInHdfError:
        return False
    return True


@callback(
    Output(main_components.signal_div.id, 'children'),
    Input(sidebar_components.update.id, 'n_clicks'),
    Input(sidebar_components.datnum.dd_id, 'value'),
    State(sidebar_components.entropy_signal_controls.store_id, 'data'),
    State(sidebar_components.entropy_signal_controls.square_store_id, 'data'),
    State(c.ConfigAIO.store_id, 'data'),
)
def update_signal_div(clicks, datnum, signal_info, square_info, config):
    config = c.ConfigAIO.config_from_dict(config)
    if datnum:
        signal_info = EntropySignalAIO.info_from_dict(signal_info)
        square_info = SquareSelectionAIO.info_from_dict(square_info)
        dat = get_dat(datnum, exp2hdf=config.experiment)
        if not check_dat_valid(dat):
            return html.Div(f'{dat} is Invalid')

        figs = []

        # 2D Transition
        transition_2d = get_transition_data2d(dat)
        p = TwoD(dat=dat)
        if signal_info.graph_type_2d == 'waterfall':
            p.MAX_POINTS = round(10000 / transition_2d.y.shape[0])
        fig = p.figure()
        fig.add_traces(p.trace(data=transition_2d.data, x=transition_2d.x, y=transition_2d.y,
                               trace_type=signal_info.graph_type_2d))
        fig.update_layout(title=f'Dat{dat.datnum}: Transition 2D')
        figs.append(fig)

        # Avg Transition
        if signal_info.centered:
            transition_1d = get_transition_data_avg(dat)
        else:
            transition_1d = Data1D(x=transition_2d.x, data=np.mean(transition_2d.data, axis=0),
                                   stderr=np.std(transition_2d.data, axis=0))
        p = OneD(dat=dat)
        fig = p.figure(ylabel='Current /nA')
        fig.add_trace(
            p.trace(data=transition_1d.data, data_err=transition_1d.stderr, x=transition_1d.x, mode='markers+lines'))
        fig.update_layout(title=f'Dat{dat.datnum}: Transition 1D')
        figs.append(fig)

        # 2D Entropy
        entropy_2d = get_entropy_data2d(dat, signal_info.step_delay)
        p = TwoD(dat=dat)
        if signal_info.graph_type_2d == 'waterfall':
            p.MAX_POINTS = round(10000 / transition_2d.y.shape[0])
        fig = p.figure()
        fig.add_traces(p.trace(data=entropy_2d.data, x=entropy_2d.x, y=entropy_2d.y,
                               trace_type=signal_info.graph_type_2d))
        fig.update_layout(title=f'Dat{dat.datnum}: Entropy 2D')
        figs.append(fig)

        # Avg Entropy
        entropy_1d = get_entropy_data_avg(dat, signal_info.step_delay, signal_info.centered)
        p = OneD(dat=dat)
        fig = p.figure(ylabel=f'{DELTA}Current /nA')
        fig.add_trace(p.trace(data=entropy_1d.data, data_err=entropy_1d.stderr, x=entropy_1d.x, mode='markers+lines'))
        fig.update_layout(title=f'Dat{dat.datnum}: Entropy 1D')
        figs.append(fig)

        # Square Wave Plots

        def avg_data_to_square_wave(data: Data1D, numpts: int, startx: float | None = None,
                                    endx: float | None = None) -> Data1D:
            """
            Takes 1D i_sense data which has not been separated into square wave parts, gets average of a single
            square wave cycle over the data from x=startx to x=endx.

            Args:
                data ():  Data1D (i.e. data and x array)
                numpts (): How many datapoints per full square wave
                startx (): Defaults to start of data
                endx ():  Defaults to end of data

            Returns:
                Data1D: Data and corresponding x-array
            """
            new_data = copy.copy(data)
            idx_start, idx_end = U.get_data_index(data.x, [startx, endx], is_sorted=True)

            if idx_start is not None:
                idx_start = int(
                    np.floor(idx_start / numpts) * numpts)  # So that it lines up with beginning of square wave
            if idx_end is not None:
                idx_end = int(np.ceil(idx_end / numpts) * numpts)  # So that it lines up with end of square wave

            new_data.data = new_data.data[idx_start: idx_end]
            new_data.x = new_data.x[idx_start: idx_end]

            new_data.data = np.reshape(new_data.data, (-1, num_pts))  # Line up all cycles on top of each other
            new_data.data = np.mean(new_data.data, axis=0)
            new_data.data = new_data.data - np.mean(new_data.data)
            return new_data

        def square_data_to_fig(plotter: OneD, square_data: Data1D) -> go.Figure():
            p = plotter
            masks = square_awg.get_single_wave_masks(num=0)  # list of Masks for each part of heating cycle
            fig = p.figure(xlabel='Time through Square Wave /s', ylabel=f'{DELTA}Current /nA')
            for mask, label in zip(masks, ['0_0', '+', '0_1', '-']):
                fig.add_trace(p.trace(data=square_data.data * mask, x=x, mode='lines', name=label))

            for sect_start in np.linspace(0, duration, 4, endpoint=False):
                p.add_line(fig, value=sect_start + signal_info.step_delay, mode='vertical', color='black',
                           linetype='dash')
            return fig

        square_awg = dat.SquareEntropy.square_awg
        num_pts = square_awg.info.wave_len
        duration = num_pts / square_awg.measure_freq
        x = square_wave_time_array(square_awg)  # Time array of single square wave

        data_avg = get_transition_data_avg(dat, centered=False)  # Centering does not keep alignment of square waves
        entropy_avg = get_entropy_data_avg(dat, signal_info.step_delay, signal_info.centered)
        data_avg.x = U.get_matching_x(entropy_avg.x, data_avg.data)  # Want x value to match the possibly centered data
        plotter = OneD(dat=dat)

        # Peak
        if square_info.peak:
            peak_x = entropy_avg.x[np.nanargmax(entropy_avg.data)]
            start_val, end_val = peak_x - square_info.width, peak_x + square_info.width  # Start and end in x
            peak_data = avg_data_to_square_wave(data_avg, numpts=num_pts, startx=start_val, endx=end_val)
            fig = square_data_to_fig(plotter, peak_data)
            fig.update_layout(title=f'Dat{dat.datnum}: Peak (x={peak_data.x[0]:.2f}->{peak_data.x[-1]:.2f}) averaged to one Square Wave')
            figs.append(fig)

        # Trough
        if square_info.trough:
            trough_x = data_avg.x[np.nanargmin(data_avg.data)]
            start_val, end_val = trough_x - square_info.width, trough_x + square_info.width
            trough_data = avg_data_to_square_wave(data_avg, numpts=num_pts, startx=start_val, endx=end_val)
            fig = square_data_to_fig(plotter, trough_data)
            fig.update_layout(
                title=f'Dat{dat.datnum}: Trough (x={trough_data.x[0]:.2f}->{trough_data.x[-1]:.2f}) averaged to one Square Wave')
            figs.append(fig)

        # Range
        if square_info.range:
            start_val, end_val = square_info.range_start, square_info.range_stop
            range_data = avg_data_to_square_wave(data_avg, numpts=num_pts, startx=start_val, endx=end_val)
            fig = square_data_to_fig(plotter, range_data)
            fig.update_layout(
                title=f'Dat{dat.datnum}: Range (x={range_data.x[0]:.2f}->{range_data.x[-1]:.2f}) averaged to one Square Wave')
            figs.append(fig)

        # General Info
        awg = dat.SquareEntropy.square_awg
        entropy_info = dbc.Col(
            [get_squarewave_info_md(dat)]
        )

        return dbc.Row([
                           dbc.Col([
                               c.GraphAIO(figure=fig)
                           ], width=12, lg=6) for fig in figs
                       ] + [entropy_info])

    return html.Div('No Dat Selected')


@callback(
    Output(main_components.fit_div.id, 'children'),
    Input(sidebar_components.update.id, 'n_clicks'),
    Input(sidebar_components.datnum.dd_id, 'value'),
    State(sidebar_components.entropy_signal_controls.store_id, 'data'),
    State(sidebar_components.entropy_signal_controls.square_store_id, 'data'),
    State(c.ConfigAIO.store_id, 'data'),
)
def update_fit_div(clicks, datnum, signal_info, square_info, config):
    config = c.ConfigAIO.config_from_dict(config)
    if datnum:
        signal_info = EntropySignalAIO.info_from_dict(signal_info)
        square_info = SquareSelectionAIO.info_from_dict(square_info)
        dat = get_dat(datnum, exp2hdf=config.experiment)
        if not check_dat_valid(dat):
            return html.Div(f'{dat} is Invalid')

        figs = []
        fit_info_mds = []

        # Transition Fits
        out = get_entropy_out_avg(dat, setpoint_start_time=signal_info.step_delay, centered=signal_info.centered)

        p = OneD(dat=dat)
        fig = p.figure(ylabel='Current /nA')
        for k, name, color in zip(['cold', 'hot'],
                                  ['Cold', 'Heated'],
                                  ['blue', 'red']):
            transition_data = get_transition_data_part(out, k)
            fit = get_transition_fit_from_data(data=transition_data)

            fig.add_traces([
                p.trace(data=transition_data.data, x=transition_data.x, mode='markers', name=name,
                        color=color),
                p.trace(data=fit.eval_fit(transition_data.x), x=transition_data.x, mode='lines', name=f'{name} fit',
                        color=color)
            ])
            transition_info_md = html.Div([
                f'Dat{dat.datnum}: {name} Transition Fit:',
                get_fit_info_md(fit)
            ])

            fit_info_mds.append(transition_info_md)
        fig.update_layout(title=f'Dat{dat.datnum}: Transition fits')
        figs.append(fig)

        # Avg Entropy
        entropy_1d = get_entropy_data_avg(dat, signal_info.step_delay, signal_info.centered)
        fit = get_entropy_fit_from_data(entropy_1d)
        p = OneD(dat=dat)
        fig = p.figure(ylabel=f'{DELTA}Current /nA')
        fig.add_traces([
            p.trace(data=entropy_1d.data, data_err=entropy_1d.stderr, x=entropy_1d.x, mode='markers', name='Data'),
            p.trace(data=fit.eval_fit(x=entropy_1d.x), x=entropy_1d.x, mode='lines', name='Fit')
        ])
        fig.update_layout(title=f'Dat{dat.datnum}: Entropy 1D')
        entropy_info_md = html.Div([
            f'Dat{dat.datnum}: Entropy Fit:',
            get_fit_info_md(fit)
        ])
        fit_info_mds.append(entropy_info_md)
        figs.append(fig)

        info_mds = [dbc.Col(md, width=6, lg=4) for md in fit_info_mds]

        return dbc.Row([
                           dbc.Col([
                               c.GraphAIO(figure=fig)
                           ], width=12, lg=6) for fig in figs
                       ] + info_mds)

    return html.Div('No Dat Selected')


@callback(
    Output(main_components.integrated_div.id, 'children'),
    Input(sidebar_components.update.id, 'n_clicks'),
    Input(sidebar_components.datnum.dd_id, 'value'),
    State(sidebar_components.entropy_signal_controls.store_id, 'data'),
    State(sidebar_components.entropy_signal_controls.square_store_id, 'data'),
    State(sidebar_components.integrated_entropy_controls.store_id, 'data'),
    State(sidebar_components.integrated_entropy_controls.dt_store_id, 'data'),
    State(c.ConfigAIO.store_id, 'data'),
)
def update_fit_div(clicks, datnum, signal_info, square_info, integrated_info, dt_info, config):
    config = c.ConfigAIO.config_from_dict(config)
    if datnum:
        signal_info = EntropySignalAIO.info_from_dict(signal_info)
        square_info = SquareSelectionAIO.info_from_dict(square_info)
        integrated_info = IntegratedEntropyAIO.info_from_dict(integrated_info)
        dt_info = GetDtAIO.info_from_dict(dt_info)
        dat = get_dat(datnum, exp2hdf=config.experiment)
        if not check_dat_valid(dat):
            return html.Div(f'{dat} is Invalid')

        # get dT
        dt = get_dT(dat, dt_info, setpoint_start=signal_info.step_delay, centered=signal_info.centered)

        # get Integration Info
        out = get_entropy_out_avg(dat, signal_info.step_delay, signal_info.centered)
        transition_data = get_transition_data_part(out, 'cold')
        transition_fit = get_transition_fit_from_data(transition_data)
        amp = transition_fit.best_values.amp
        dx = abs((transition_data.x[-1] - transition_data.x[0]) / transition_data.x.shape[-1])
        sf = scaling(dt, amp, dx)
        int_info = IntegrationInfo(dT=dt, amp=amp, dx=dx, sf=sf)

        # Other setup
        zero_pos = U.get_data_index(transition_data.x, integrated_info.zero_pos, is_sorted=True) \
            if integrated_info.zero_pos else 0
        end_pos = U.get_data_index(transition_data.x, integrated_info.end_pos, is_sorted=True) \
            if integrated_info.end_pos else -1

        figs = []
        # Integrated Avg
        entropy_avg = get_entropy_data_avg(dat,
                                           setpoint_start_time=signal_info.step_delay,
                                           centered=signal_info.centered)
        integrated = int_info.integrate(entropy_avg.data)
        integrated = integrated - integrated[zero_pos]
        int_err = int_info.integrate(entropy_avg.stderr) if entropy_avg.stderr else None
        p = OneD(dat=dat)
        fig = p.plot(data=integrated, data_err=int_err, x=entropy_avg.x, ylabel='Entropy /kB',
                     title=f'Dat{dat.datnum}: Integrated Entropy (Amp={amp:.3f}, dT={dt:.4f})', mode='lines')
        for pos in [integrated_info.zero_pos, integrated_info.end_pos]:
            if pos:
                p.add_line(fig, pos, mode='vertical', color='black', linetype='dash')
        figs.append(fig)

        #  Per Row integrated
        entropy_2d = get_entropy_data2d(dat, signal_info.step_delay)
        integrated = int_info.integrate(entropy_2d.data)
        integrated = integrated - integrated[:, zero_pos][:, None]
        p = TwoD(dat=dat)
        fig = p.plot(data=integrated, x=entropy_2d.x, ylabel='Entropy /kB',
                     title=f'Dat{dat.datnum}: Integrated Entropy (Amp={amp:.3f}, dT={dt:.4f})',
                     plot_type='waterfall')
        fig.update_layout(
            margin=go.layout.Margin(
                l=0,  # left margin
                r=0,  # right margin
                b=0,  # bottom margin
            )
        )
        # for pos in [integrated_info.zero_pos, integrated_info.end_pos]:
        #     if pos:
        #         p.add_line(fig, pos, mode='vertical', color='black', linetype='dash')
        figs.append(fig)

        # Per row value
        values = integrated[:, end_pos]
        p = OneD(dat=dat)
        fig = p.plot(data=values, x=entropy_2d.y, ylabel='Entropy /kB',
                     title=f'Dat{dat.datnum}: Integrated Entropy Values per Row (Amp={amp:.3f}, dT={dt:.4f})', mode='markers')
        figs.append(fig)

        return dbc.Row([
                           dbc.Col([
                               c.GraphAIO(figure=fig)
                           ], width=12, lg=6) for fig in figs
                       ])
    return html.Div(f'No dat selected')



def get_dT(dat, dt_info: GetDtAIO.Info, setpoint_start: float | None, centered: bool):
    """
    Get delta T by selected method (either fixed value, calculated from hot theta - cold theta, or from linear
    extrapolation

    Args:
        dat ():
        dt_info ():
        setpoint_start ():  Only used if calculating from dat (necessary for calculating fits)
        centered (): Only used if calculating from dat (necessary for calculating fits)

    Returns:

    """
    if dt_info.method == 'fixed':
        dt = dt_info.value if dt_info.value else 1.0
    elif dt_info.method == 'dat':
        out = get_entropy_out_avg(dat, setpoint_start_time=setpoint_start, centered=centered)
        transition_datas = [get_transition_data_part(out, k) for k in ['hot', 'cold']]
        fits = [get_transition_fit_from_data(data=data) for data in transition_datas]
        thetas = [fit.best_values.get('theta', None) if fit else None for fit in fits]
        dt = thetas[0] - thetas[1] if None not in thetas else 1.0
    elif dt_info.method == 'eq':
        if all([dt_info.ref_channel_value, dt_info.ref_dt, dt_info.channel]):
            ref_channel_val = dt_info.ref_channel_value
            ref_dt = dt_info.ref_dt
            channel = dt_info.channel
            linear = dt_info.linear_term if dt_info.linear_term else 0.0
            constant = dt_info.constant_term if dt_info.constant_term else 0.0

            try:
                new_channel_val = dat.Logs.dacs[channel]
            except KeyError as e:
                logging.error(f'{channel} not found in ({list(dat.Logs.dacs.keys())}')
                new_channel_val = None

            if new_channel_val:
                new_dt = ref_dt * (linear * ref_channel_val + constant) / (linear * new_channel_val + constant)
            else:
                dt = 1.0
        else:
            logging.error(f'Not enough info provided to calculate dT from equation {dt_info}')
            dt = 1.0
    else:
        logging.error(f'Invalid dt_info.method choice ({dt_info.method})')
        dt = 1.0
    return dt


@dataclass
class Data1D:
    x: np.ndarray
    data: np.ndarray
    stderr: Optional[np.ndarray]

    def __hash__(self):
        return hash(tuple([str(arr) for arr in [self.x, self.data, self.stderr]]))

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return hash(other) == hash(self)
        return False


@dataclass
class Data2D:
    x: np.ndarray
    y: np.ndarray
    data: np.ndarray

    def __hash__(self):
        return hash((str(arr) for arr in [self.x, self.data, self.y]))

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return hash(other) == hash(self)
        return False


@lru_cache(maxsize=128)
def get_transition_data2d(dat: DatHDF):
    data = np.atleast_2d(dat.Data.get_data('i_sense'))
    x = dat.Data.get_data('x')
    if 'y' in dat.Data.keys:
        y = dat.Data.get_data('y')
    else:
        y = np.arange(data.shape[0])
    return Data2D(x=x, y=y, data=data)


def get_entropy_data2d(dat: DatHDF, setpoint_start_time):
    out = get_entropy_out_no_average(dat, setpoint_start_time)
    if 'y' in dat.Data.keys:
        y = dat.Data.get_data('y')
    else:
        y = np.arange(out.entropy_signal.shape[0])
    return Data2D(x=out.x, y=y, data=out.entropy_signal)


def _get_setpoint_start_index(dat: DatHDF, setpoint_time: float | None):
    if setpoint_time is None:
        return 0
    awg = dat.SquareEntropy.square_awg
    full_setpoint_time = awg.info.wave_len * 1 / awg.info.measureFreq / 4
    if setpoint_time < full_setpoint_time:
        start_index, _ = get_setpoint_indexes_from_times(dat, start_time=setpoint_time, end_time=None)
    else:
        start_index = int(awg.info.wave_len / 4) - 1
    return start_index


@lru_cache(maxsize=128)
def get_entropy_out_no_average(dat: DatHDF, setpoint_start_time) -> dat_analysis.analysis_tools.square_wave.Output:
    start_index = _get_setpoint_start_index(dat, setpoint_start_time)
    i_sense_data = get_transition_data2d(dat)
    inputs = dat.SquareEntropy.get_Inputs(x_array=i_sense_data.x, i_sense=i_sense_data.data)
    process_params = dat.SquareEntropy.get_ProcessParams(setpoint_start=start_index)
    out = dat.SquareEntropy.get_row_only_output(calculate_only=True, inputs=inputs, process_params=process_params)
    return out


@lru_cache(maxsize=128)
def get_entropy_out_avg(dat: DatHDF, setpoint_start_time, centered) -> dat_analysis.analysis_tools.square_wave.Output:
    start_index = _get_setpoint_start_index(dat, setpoint_start_time)
    i_sense_data = get_transition_data2d(dat)
    if centered:
        centers = None  # Will use transition fits to cold part of I_sense to determine
    else:
        centers = [0] * i_sense_data.data.shape[0]
    inputs = dat.SquareEntropy.get_Inputs(x_array=i_sense_data.x, i_sense=i_sense_data.data, centers=centers)
    process_params = dat.SquareEntropy.get_ProcessParams(setpoint_start=start_index, transition_fit_func=None)
    out = dat.SquareEntropy.get_Outputs(calculate_only=True, inputs=inputs, process_params=process_params)
    return out


def get_entropy_data_avg(dat: DatHDF, setpoint_start_time, centered):
    out = get_entropy_out_avg(dat, setpoint_start_time, centered)
    return Data1D(x=out.x, data=out.average_entropy_signal,
                  stderr=None)  # TODO: Calculate the stderr of averaging (after centering if used)


@lru_cache(maxsize=128)
def get_transition_data_avg(dat: DatHDF, centered=True):
    data2d = get_transition_data2d(dat)
    if centered:
        fits = [dat.Transition.get_fit(calculate_only=True, data=data, x=data2d.x) for data in data2d.data]
        centers = [fit.best_values.mid for fit in fits]
    else:
        centers = [0] * data2d.data.shape[0]
    avg_data, avg_std, avg_x = dat.Transition.get_avg_data(x=data2d.x, data=data2d.data, centers=centers, return_x=True,
                                                           return_std=True, calculate_only=True)
    return Data1D(x=avg_x, data=avg_data, stderr=avg_std)


@lru_cache(maxsize=128)
def get_squarewave_info_md(dat: DatHDF):
    awg = dat.SquareEntropy.square_awg
    return dcc.Markdown(children=
                        f"""
                   Dat{dat.datnum}: Square Wave Info:
                      - Points per full square wave: {awg.info.wave_len} 
                      - Points per square wave step: {int(awg.info.wave_len / 4)} 
                      - Measurement frequency: {awg.info.measureFreq:.2f} Hz
                      - Number of DAC steps: {awg.info.num_steps} 
                      - Full heating cycles per DAC step: {awg.info.num_cycles} 
                      - Max AW output: {max(awg.get_single_wave(0)):.2f} mV 
                      - Min AW output: {min(awg.get_single_wave(0))} mV
                """
                        )


@lru_cache(maxsize=128)
def get_fit_info_md(fit: dat_analysis.analysis_tools.general_fitting.FitInfo):
    param_infos = []
    for p in fit.params.values():
        stderr = f'{PM}{p.stderr:.2g}' if p.stderr else ''
        info = f'{p.name}: {p.value:.4g}{stderr}'
        param_infos.append(info)
    param_info_str = '\n'.join([f'- {info}' for info in param_infos])
    return html.Div([
        dcc.Markdown(children=
                     f"""
                            - Success: {fit.success}
                            - Reduced Chi Square: {fit.reduced_chi_sq:.2g}
                            - Fit func: {fit.func_name}"""
                     ),
        dcc.Markdown(children=param_info_str)
    ])


@lru_cache(maxsize=128)
def get_transition_fit_from_data(data: Data1D) -> FitInfo:
    """
    Takes Output from SE and calculates transition fit to hot/cold part of data
    Args:
        data: Data1D to fit
        part: 'cold', 'hot', 0, 1, 2, 3

    Returns:
        FitInfo result
    """
    from dat_analysis.dat_object.attributes.transition import i_sense, get_param_estimates, _append_param_estimate_1d

    params = get_param_estimates(data.x, data.data)
    _append_param_estimate_1d(params, pars_to_add=None)  # Will be useful for fitting other than i_sense

    fit = calculate_fit(x=data.x, data=data.data, func=i_sense, params=params, generate_hash=True)
    return fit


def get_transition_data_part(out: dat_analysis.dat_object.attributes.square_entropy.Output, part: str) -> Data1D:
    from dat_analysis.dat_object.attributes.square_entropy import get_transition_part
    transition_data = Data1D(x=out.x,
                             data=get_transition_part(data=out.averaged, part=part),
                             stderr=None)
    return transition_data


@lru_cache(maxsize=128)
def get_entropy_fit_from_data(data: Data1D) -> FitInfo:
    """
    Takes Output from SE and calculates transition fit to hot/cold part of data
    Args:
        out: Output from Square Entropy

    Returns:
        FitInfo result
    """
    from dat_analysis.dat_object.attributes.entropy import entropy_nik_shape, get_param_estimates

    params = get_param_estimates(data.x, data.data)[0]

    fit = calculate_fit(x=data.x, data=data.data, func=entropy_nik_shape, params=params, generate_hash=True)
    return fit


def layout(add_config=False):
    """Must return the full layout for multipage to work"""
    sidebar_layout_ = sidebar_components.layout()
    main_layout_ = main_components.layout()

    if add_config:
        sidebar_layout_ = html.Div([
            c.ConfigAIO(experiment_options=['Nov21LD', 'Nov21Tim']),
            sidebar_layout_
        ])

    return html.Div([
        html.H1('Entropy'),
        dbc.Row([
            dbc.Col(
                dbc.Card(sidebar_layout_),
                sm=6, lg=3,
            ),
            dbc.Col(
                dbc.Card(main_layout_),
                sm=6, lg=9,
            )
        ]),
    ])


if __name__ == '__main__':
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = layout(add_config=True)
    app.run_server(debug=True, port=8052)
else:
    dash.register_page(__name__)
    pass
