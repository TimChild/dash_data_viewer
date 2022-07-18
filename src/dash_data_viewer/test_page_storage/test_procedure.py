# from __future__ import annotations
import dash
import h5py
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
from typing import TYPE_CHECKING, List, Union, Optional, Any, Tuple
import uuid
import numpy as np
import plotly.graph_objects as go

import dash_data_viewer.components as c
from dash_data_viewer.process_dash_extensions import ProcessInterface, NOT_SET
from dash_data_viewer.layout_util import label_component

from dat_analysis.analysis_tools.new_procedures import SeparateSquareProcess, Process, PlottableData, DataPlotter
from dat_analysis.hdf_file_handler import GlobalLock
from dat_analysis.useful_functions import data_to_json, data_from_json

if TYPE_CHECKING:
    from dash.development.base_component import Component


LOCK = GlobalLock('test_page_storage/test_file.txt.lock')
TEST_FILE = 'test_page_storage/test_file.txt'
TEST_FILE2 = 'test_page_storage/test_file2.txt'


def get_data(data_or_location: Union[str, list]) -> np.ndarray:
    """Either data will be stored as a dataset directly, or some location to the data (including dat_id, group_location)"""
    if isinstance(data_or_location, list):
        return np.array(data_or_location)
    elif isinstance(data_or_location, str):
        pass
    raise NotImplementedError(f"Type = {type(data_or_location)}")


class TestProcess(Process):
    def __base_data(self) -> [np.ndarray, np.ndarray]:
        return np.linspace(0, 10, 100), np.sin(np.linspace(0, 10, 100))

    def set_inputs(self, a, b, x=None, data=None):
        if x is None or data is None:
            x, data = self.__base_data()
        self.inputs = dict(a=a, b=b, x=x, data=data)

    def process(self):
        x, data = self.inputs['x'], self.inputs['data']
        data = (data+self.inputs['b'])*self.inputs['a']
        self.outputs = {
            'x': x,
            'data': data,
        }
        return PlottableData(
            data=data,
            x=x
        )

    def get_input_plotter(self) -> DataPlotter:
        x, data = self.inputs['x'], self.inputs['data']
        p = DataPlotter(PlottableData(data=data, x=x),
                        xlabel='x label', ylabel='y label', data_label='data label', title='input a and b')
        return p

    def get_output_plotter(self) -> DataPlotter:
        p = DataPlotter(self.process(), xlabel='x label', ylabel='y label', data_label='data label', title='output a and b')
        return p

    def save_progress(self, filepath, **kwargs):  #, group: h5py.Group, **kwargs):
        with LOCK:
            data_to_json(
                datas=[np.asanyarray(v) for v in list(self.inputs.values()) + list(self.outputs.values())],
                names=[f'in_{k}' for k in self.inputs.keys()] + [f'out_{k}' for k in self.outputs.keys()],
                filepath=filepath,
                )

    @classmethod
    def load_progress(cls, filepath, **kwargs) -> Any:  #group: h5py.Group) -> Process:
        with LOCK:
            data = data_from_json(filepath)
        inst = cls()
        inst.inputs = dict(a=float(data['in_a']), b=float(data['in_b']), x=data['in_x'], data=data['in_data'])
        inst.outputs = dict(x=data['out_x'], data=data['out_data'])
        return inst


class Test2Process(Process):
    def set_inputs(self, x, data, c):
        self.inputs = dict(x=x, data=data, c=c)

    def process(self) -> PlottableData:
        x, data, c = self.inputs['x'], self.inputs['data'], self.inputs['c']
        x = x*c
        self.outputs = {
            'x': x,
            'data': data,
        }
        return PlottableData(
            data=data,
            x=x
        )

    def get_input_plotter(self) -> DataPlotter:
        x, data, c = self.inputs['x'], self.inputs['data'], self.inputs['c']
        p = DataPlotter(PlottableData(data=data, x=x),
                        xlabel='x label', ylabel='y label', data_label='data label', title='input c')
        return p

    def get_output_plotter(self) -> DataPlotter:
        p = DataPlotter(self.process(), xlabel='x label', ylabel='y label', data_label='output c',
                        title='output c')
        return p

    def save_progress(self, filepath, **kwargs):  #, group: h5py.Group, **kwargs):
        with LOCK:
            data_to_json(
                datas=[np.asanyarray(v) for v in list(self.inputs.values()) + list(self.outputs.values())],
                names=[f'in_{k}' for k in self.inputs.keys()] + [f'out_{k}' for k in self.outputs.keys()],
                filepath=filepath,
            )

    @classmethod
    def load_progress(cls, filepath, **kwargs) -> Any:  #group: h5py.Group) -> Process:
        with LOCK:
            data = data_from_json(filepath)
        inst = cls()
        inst.inputs = dict(x=data['in_x'], data=data['in_data'], c=float(data['in_c']))
        inst.outputs = dict(x=data['out_x'], data=data['out_data'])
        return inst


class TestInterface2:
    @staticmethod
    def id(key) -> str:
        # TODO: "Test" should be changeable/settable somehow
        return f'Test-{key}'
    _NOT_SET = object()

    # Standard IDs
    sanitized_store_id = id('store-sanitized')
    output_store_id = id('store-output')

    # User Input IDs
    input_a_id = id('input-a')
    input_b_id = id('input-b')

    # Other Required Input IDs
    datpicker_id = _NOT_SET

    # Data input IDs
    data_input_id = _NOT_SET

    def make_user_inputs(self) -> html.Div:
        """Make user input components and return layout
        Note: keeping track of the IDs as well

        Also include sanitized_store
        """
        in_a = dbc.Input(id=self.input_a_id, type='number')
        in_b = dbc.Input(id=self.input_b_id, type='number')
        return html.Div([
            label_component(in_a, 'Input A'),
            label_component(in_b, 'Input B'),
        ])

    def get_stores(self) -> html.Div:
        """Get the stores that need to be placed somewhere on the page"""
        sanitized_store = dcc.Store(id=self.sanitized_store_id)
        output_store = dcc.Store(id=self.output_store_id)
        return html.Div([
            sanitized_store,
            output_store,
        ])

    def set_other_input_ids(self, datpicker_id: str):
        """Get the IDs for other required inputs"""
        self.datpicker_id = datpicker_id

    def setdata_input_ids(self, data_input_id: str):
        """Get the IDs for data inputs"""
        self.data_input_id = data_input_id

    def callback_sanitized_store(self):
        """Callback to update a store with inputs which are sanitized (including other required inputs, but excluding data)"""
        assert self.datpicker_id is not self._NOT_SET

        @callback(
            Output(self.sanitized_store_id, 'data'),
            Input(self.input_a_id, 'value'),
            Input(self.input_b_id, 'value'),
            # Input(self.datpicker_id, 'value'),
        )
        def update_sanitized_store(a, b):
            a = a if a else 1
            b = b if b else 0
            return dict(a=a, b=b)

    def display_sanitized_inputs(self) -> html.Div:
        """Make display for sanitized inputs and run callback (from sanitized store)"""
        md_id = self.id('md-sanitized')  # Should never need to be used by anything else, only the callback here
        output = html.Div(
            dcc.Markdown(
                id=md_id,
                style={'white-space': 'pre'}
            ))

        @callback(
            Output(md_id, 'children'),
            Input(self.sanitized_store_id, 'data')
        )
        def update_sanitized_display(s: dict):
            if s:
                return f'''
                Using:
                \ta = {s['a']:.4f} 
                \tb = {s['b']:.4f}
                '''
            return 'Nothing to show yet'

        return output

    def callback_output_store(self):
        """Update output store using sanitized inputs and data"""
        @callback(
            Output(self.output_store_id, 'data'),
            Input(self.sanitized_store_id, 'data'),
            Input(self.data_input_id, 'data'),
        )
        def update_output_store(inputs: dict, data: dict):
            # Only get the data from the data_store that we want here
            useful_x, useful_data = [get_data(data[key]) for key in ['x_a', 'data_a']]

            # Do the Processing
            process = TestProcess()
            process.set_inputs(a=inputs['a'], b=inputs['b'], x=useful_x, data=useful_data)
            out = process.process()

            # Rather than pass big datasets etc, save the Process and return the location to load it
            process.save_progress(TEST_FILE)
            return TEST_FILE


    def display_self(self) -> html.Div:
        """Make display for input/output and run callback
        TODO: How to account for general controls like setting x-min/max for viewing
        TODO: Don't want to always have to display Input and Output
        """
        in_graph_id = self.id('graph-in')
        out_graph_id = self.id('graph-out')
        output = html.Div([
            dbc.Card([
                dcc.Graph(id=in_graph_id)
            ]),
            dbc.Card([
                dcc.Graph(id=out_graph_id)
            ]),
        ]
        )

        @callback(
            Output(out_graph_id, 'figure'),
            Input(self.output_store_id, 'data'),
        )
        def update_output_graph(out_store):
            if out_store:
                process = TestProcess.load_progress(out_store)
                graph = process.get_output_plotter().plot_1d()
                return graph
            return go.Figure()

        @callback(
            Output(in_graph_id, 'figure'),
            Input(self.output_store_id, 'data'),
            # TODO: Below may let me separate processing step?
            # Input(self.sanitized_store_id, 'data'),
            # Input(self.data_input_id, 'data'),
        )
        def update_input_graph(out_store):
            if out_store:
                process = TestProcess.load_progress(out_store)
                fig = process.get_input_plotter().plot_1d()
                fig.add_hline(process.inputs['a']*process.inputs['b'])
                return fig
            return go.Figure()

        return output



class Test2Interface2(ProcessInterface):
    id_prefix = 'Test2'
    def __init__(self):
        ID = self.ID

        # User Input IDs
        self.input_c_id = ID('input-c')

        # Other Required Input IDs
        self.datpicker_id = NOT_SET

        # Data input IDs
        self.data_input_id = NOT_SET

        # Display IDs
        self.in_graph_id = ID('graph-in')
        self.out_graph_id = ID('graph-out')

    def set_other_input_ids(self, datpicker_id: str):
        """Get the IDs for other required inputs"""
        self.datpicker_id = datpicker_id

    def set_data_input_ids(self, data_input_id: str):
        """Get the IDs for data inputs"""
        self.data_input_id = data_input_id

    def get_user_inputs_layout(self) -> html.Div:
        """Make user input components and return layout
        Note: keeping track of the IDs as well

        Also include sanitized_store
        """
        in_c = dbc.Input(id=self.input_c_id, type='number')
        return html.Div([
            label_component(in_c, 'Input C'),
        ])

    def callback_sanitized_store(self):
        """Callback to update a store with inputs which are sanitized (including other required inputs, but excluding data)"""
        assert self.datpicker_id is not NOT_SET

        @callback(
            Output(self.sanitized_store_id, 'data'),
            Input(self.input_c_id, 'value'),
            # Input(self.datpicker_id, 'value'),
        )
        def update_sanitized_store(c):
            c = c if c else 10
            return dict(c=c)

    def info_for_sanitized_display(self) -> List[Union[str, dict]]:
        return [{'key':'c', 'name':'Input C', 'format':'.1f'}]

    def callback_output_store(self):
        """Update output store using sanitized inputs and data"""
        @callback(
            Output(self.output_store_id, 'data'),
            Input(self.sanitized_store_id, 'data'),
            Input(self.data_input_id, 'data'),
        )
        def update_output_store(inputs: dict, data: dict):
            # Get data from previous processing  # TODO: Need to make this clearer
            pre_process = TestProcess.load_progress(data)
            out = pre_process.process()
            useful_x, useful_data = out.x, out.data

            # Do the Processing
            process = Test2Process()
            process.set_inputs(c=inputs['c'], x=useful_x, data=useful_data)
            out = process.process()

            # Rather than pass big datasets etc, save the Process and return the location to load it
            process.save_progress(TEST_FILE2)
            return TEST_FILE2


    def get_input_display_layout(self) -> html.Div:
        """
        Make display layout for inputs
        Returns:

        """
        layout = html.Div([
            dbc.Card([
                dcc.Graph(id=self.in_graph_id)
            ]),
        ]
        )
        return layout


    def get_output_display_layout(self) -> html.Div:
        """Make display layout for output
        """
        output = html.Div([
            dbc.Card([
                dcc.Graph(id=self.out_graph_id)
            ]),
        ]
        )
        return output

    def callback_input_display(self):
        """Run callback for input display"""

        @callback(
            Output(self.in_graph_id, 'figure'),
            Input(self.output_store_id, 'data'),
            # TODO: Below may let me separate processing step?
            # Input(self.sanitized_store_id, 'data'),
            # Input(self.data_input_id, 'data'),
        )
        def update_input_graph(out_store):
            if out_store:
                process = Test2Process.load_progress(out_store)
                fig = process.get_input_plotter().plot_1d()
                fig.add_hline(process.inputs['c'])
                return fig
            return go.Figure()

    def callback_output_display(self):
        """
        Run callback for output display
        Returns:

        """
        @callback(
            Output(self.out_graph_id, 'figure'),
            Input(self.output_store_id, 'data'),
        )
        def update_output_graph(out_store):
            if out_store:
                process = Test2Process.load_progress(out_store)
                graph = process.get_output_plotter().plot_1d()
                return graph
            return go.Figure()



# Actually make the page

# Fake some data for testing
x, data = np.linspace(0, 10, 100), np.sin(np.linspace(0, 10, 100))
data_store_id = 'TESTING-data-store'
FAKE_DATA_STORE = dcc.Store(id=data_store_id, data={'x_a': x, 'data_a': data})

# Initialize Interfaces that help make dash page
TI = TestInterface2()
TI.setdata_input_ids(data_input_id=data_store_id)
TI.set_other_input_ids(datpicker_id='fake')
inputs = TI.make_user_inputs()
sanitized_inputs = TI.display_sanitized_inputs()
stores = TI.get_stores()
outputs = TI.display_self()
TI.callback_sanitized_store()
TI.callback_output_store()


# Use second ProcessInterface that builds on first
TI2 = Test2Interface2()
TI2.set_data_input_ids(data_input_id=TI.output_store_id)
TI2.set_other_input_ids(datpicker_id='fake')
inputs2 = TI2.get_user_inputs_layout()
sanitized_inputs2 = TI2.get_sanitized_inputs_layout()
stores2 = TI2.get_stores()
display_output = TI2.get_output_display_layout()
display_input = TI2.get_input_display_layout()
TI2.callback_sanitized_store()
TI2.callback_output_store()
TI2.callback_sanitized_inputs()
TI2.callback_output_display()
TI2.callback_input_display()

# Put everything together into a layout
layout = html.Div([
    FAKE_DATA_STORE,
    stores,
    stores2,
        dbc.Row([
            dbc.Col([
                dbc.Card([inputs, sanitized_inputs]),
                dbc.Card([inputs2, sanitized_inputs2]),
                ],
                width=3),
            dbc.Col([
                dbc.Card([outputs]),
                dbc.Card([display_input, display_output]),
            ], width=9),
        ])
    ]
    )


# def layout():
#
#     return html.Div([
#         dbc.Row([
#             dbc.Col([v for k, v in TestInterface().required_input_components().items()], width=3),
#             dbc.Col([v for k, v in TestInterface().all_outputs().items()], width=9),
#         ])
#     ]
#     )


if __name__ == '__main__':
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    # app.layout = layout()
    app.layout = layout
    app.run_server(debug=True, port=8051)
else:
    dash.register_page(__name__)
    pass

