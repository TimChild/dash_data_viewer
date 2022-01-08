from __future__ import annotations
import dash
import numpy as np
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

from dat_analysis import get_dat, get_dats


# from dash_data_viewer.layout_util import label_component

from typing import TYPE_CHECKING, Optional, List, Union

if TYPE_CHECKING:
    from dash.development.base_component import Component


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
        step_delay: float
        centered: bool

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
                    type='number',
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
        if not state:
            state = EntropySignalAIO.Info(step_delay=0, centered=False)
        else:
            state = from_dict(EntropySignalAIO.Info, state)
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
        peak: bool
        trough: bool
        width: float
        range: bool
        range_start: float
        range_stop: float

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
        buttons_selected = {}
        for k in ['peak', 'trough', 'range']:
            buttons_selected[k] = True if k in buttons else False
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
        zero_pos: float
        end_pos: float

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

        square_selection = SquareSelectionAIO(aio_id=aio_id)
        self.square_store_id = square_selection.store_id

        layout = dbc.Card([
            output_store,
            dbc.CardHeader('Integrated Entropy'),
            dbc.CardBody([
                dtAIO,
                label_component(
                    c.Input_(
                        id=self.ids.generic(aio_id, 'input', 'zero'),
                        type='number',
                        persistence=True, persistence_type='local',
                    ),
                    'Zero Pos: '
                ),
                label_component(
                    c.Input_(
                        id=self.ids.generic(aio_id, 'input', 'end'),
                        type='number',
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
        method: str
        value: Optional[float]
        equation: Optional[str]
        channel: Optional[str]

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
            dbc.CardBody([
                c.MultiButtonAIO(button_texts=['Fixed', 'Dat', 'Equation'],
                                 button_values=['fixed', 'dat', 'eq'],
                                 aio_id=aio_id, allow_multiple=False, storage_type='local',
                                 ),
                label_component(
                    c.Input_(id=self.ids.input(aio_id, 'value'), type='number', min=0, persistence=True),
                    'Value'
                ),
                label_component(
                    c.Input_(id=self.ids.input(aio_id, 'equation'), type='text', min=0, persistence=True,
                             placeholder='0.001*channel + 5.0'),
                    'Equation'
                ),
                label_component(
                    c.Input_(id=self.ids.input(aio_id, 'channel'), type='text', min=0, persistence=True,
                             placeholder='ESC'),
                    'Channel'
                ),
            ]),
        ])

        super().__init__(children=layout)  # html.Div contains layout

    @staticmethod
    @callback(
        Output(ids.generic(MATCH, 'store', 'output'), 'data'),
        Input(c.MultiButtonAIO.ids.store(MATCH), 'data'),
        Input(ids.input(MATCH, 'value'), 'value'),
        Input(ids.input(MATCH, 'equation'), 'value'),
        Input(ids.input(MATCH, 'channel'), 'value'),
    )
    def function(button, value, equation, channel):
        info = GetDtAIO.Info(
            method=button,
            value=value,
            equation=equation,
            channel=channel,
        )
        return asdict(info)



class MainComponents(object):
    """Convenient holder for any components that will end up in the main area of the page"""
    signal_div = html.Div(id='entropy-div-signal')
    fit_div = html.Div(id='entropy-div-fit')
    integrated_div = html.Div(id='entropy-div-integrated')

    def layout(self):
        layout = html.Div(
            dbc.Card(
                dbc.CardHeader(html.H3('Entropy Signal')),
                dbc.CardBody(self.signal_div),
            ),
            dbc.Card(
                dbc.CardHeader(html.H3('Entropy Signal')),
                dbc.CardBody(self.fit_div),
            ),
            dbc.Card(
                dbc.CardHeader(html.H3('Entropy Signal')),
                dbc.CardBody(self.integrated_div),
            ),
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
        dbc.CardHeader('Fit Entropy'),
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


@callback(
    Output(main_components.signal_div, 'children'),
    Input(sidebar_components.update, 'n_clicks'),
    Input(sidebar_components.datnum.dd_id, 'value'),
    State(sidebar_components.entropy_signal_controls.store_id, 'data'),
    State(sidebar_components.entropy_signal_controls.square_store_id, 'data'),
)
def update_signal_div(clicks, datnum, signal_info, square_info):
    if datnum:
        signal_info = from_dict(EntropySignalAIO.Info, signal_info)
        square_info = from_dict(EntropySignalAIO.Info, square_info)


    return html.Div('No Dat Selected')


@dataclass
class Data1D:
    x: np.ndarray
    data: np.ndarray
    stderr: Optional[np.ndarray]


@dataclass
class Data2D:
    x: np.ndarray
    y: np.ndarray
    data: np.ndarray


@lru_cache(maxsize=128)
def get_transition_data2d(datnum: int):
    dat = get_dat(datnum)
    data = dat.Data.get_data('i_sense')
    x = dat.Data.get_data('x')
    y = dat.Data.get_data('y')
    return Data2D(x=x, y=y, data=data)


@lru_cache(maxsize=128)
def get_transition_data_avg(datnum: int):
    dat = get_dat(datnum)
    data2d = get_transition_data2d(dat.datnum)
    fits = [dat.Transition.get_fit(calculate_only=True, data=data, x=data2d.x) for data in data2d.data]
    centers = [fit.best_values.mid for fit in fits]
    avg_data, avg_x, avg_std = dat.Transition.get_avg_data(x=data2d.x, data=data2d.data, centers=centers, return_x=True,
                                                           return_std=True)
    return Data1D(x=avg_x, data=avg_data, stderr=avg_std)



def layout():
    """Must return the full layout for multipage to work"""
    sidebar_layout_ = sidebar_components.layout()
    main_layout_ = main_components.layout()

    return html.Div([
        html.H1('Entropy'),
        dbc.Row([
            dbc.Col(
                dbc.Card(sidebar_layout_),
                width=3
            ),
            dbc.Col(
                dbc.Card(main_layout_)
            )
        ]),
    ])


if __name__ == '__main__':
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = layout()
    app.run_server(debug=True, port=8052)
else:
    # dash.register_page(__name__)
    pass
