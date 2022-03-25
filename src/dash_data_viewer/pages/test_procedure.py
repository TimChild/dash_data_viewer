from __future__ import annotations
import dash
import h5py
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
from typing import TYPE_CHECKING, List, Union, Optional, Any, Tuple
import uuid
import numpy as np
import plotly.graph_objects as go

import dash_data_viewer.components as c
from dash_data_viewer.process_dash_extensions import ProcessInterface, ProcessComponentInput, ProcessComponentOutput, ProcessComponentOutputGraph, DashEnabledProcess

from dat_analysis.analysis_tools.new_procedures import SeparateSquareProcess, Process, PlottableData, DataPlotter
from dat_analysis import get_dat

if TYPE_CHECKING:
    from dash.development.base_component import Component


class TestProcess(Process):

    def __base_data(self) -> [np.ndarray, np.ndarray]:
        return np.linspace(0, 10, 100), np.sin(np.linspace(0, 10, 100))

    def input_data(self, a, b):
        self._data_input = dict(a=a, b=b)

    def preprocess(self):
        pass

    def output_data(self) -> PlottableData:
        x, data = self.__base_data()
        data = (data+self._data_input['b'])*self._data_input['a']
        return PlottableData(
            data=data,
            x=x
        )

    def get_input_plotter(self) -> DataPlotter:
        x, data = self.__base_data()
        p = DataPlotter(PlottableData(data=data, x=x),
                        xlabel='x label', ylabel='y label', data_label='data label', title='title')
        return p

    def get_output_plotter(self) -> DataPlotter:
        p = DataPlotter(self.output_data(), xlabel='x label', ylabel='y label', data_label='data label', title='title')
        return p

    def save_progress(self, group: h5py.Group, **kwargs):
        print(f'Fake saving a = {self._data_input["a"]}, b = {self._data_input["b"]}')

    @classmethod
    def load_progress(cls, group: h5py.Group) -> Process:
        print(f'Fake Loading a = 2, b = 5')
        inst = cls()
        inst._data_input = dict(a=2, b=5)
        return inst



class TestInterface(ProcessInterface):
    # Collect all component ids used
    class ids:
        in_a = f'component=generic, subcomponent=generic, key=in_a'
        in_b = f'component=generic, subcomponent=generic, key=in_b'
        value_div = f'component=generic, subcomponent=generic, key=value_div'
        value_dict = f'component=generic, subcomponent=generic, key=value_dict'
        graph_out = f'component=generic, subcomponent=generic, key=graph_out'
        graph_in = f'component=generic, subcomponent=generic, key=graph_in'

    # Make the ids class a public class
    ids = ids

    def required_input_components(self) -> dict:
        in_a = dbc.Input(id=self.ids.in_a, type='number')
        in_b = dbc.Input(id=self.ids.in_b, type='number')
        value_div = html.Div(id=self.ids.value_div)
        value_dict = dcc.Store(id=self.ids.value_dict, storage_type='memory')
        d = {
            in_a.id: in_a,
            in_b.id: in_b,
            value_div.id: value_div,
            value_dict.id: value_dict,
        }
        return d

    def all_outputs(self) -> dict:
        in_graph = dcc.Graph(id=self.ids.graph_in)
        out_graph = dcc.Graph(id=self.ids.graph_out)
        return {
            in_graph.id: in_graph,
            out_graph.id: out_graph,
        }

    def main_outputs(self) -> List[ProcessComponentOutput]:
        return self.all_outputs()

    @staticmethod
    @callback(
        Output(ids.value_dict, 'data'),
        Input(ids.in_a, 'value'),
        Input(ids.in_b, 'value'),
    )
    def update_store(a, b) -> dict:
        a = a if a else 1
        b = b if b else 0
        return dict(a=a, b=b)

    @staticmethod
    @callback(
        Output(ids.value_div, 'children'),
        Input(ids.value_dict, 'data'),
    )
    def update_div(inputs: dict) -> str:
        return f'Using: {inputs}'


    @staticmethod
    @callback(
        Output(ids.graph_out, 'figure'),
        Input(ids.value_dict, 'data'),
    )
    def make_graph_out(inputs: dict) -> go.Figure:
        a = inputs['a']
        b = inputs['b']
        p = TestProcess()
        p.input_data(a, b)
        plotter = p.get_output_plotter()
        fig = plotter.plot_1d()
        return fig

    @staticmethod
    @callback(
        Output(ids.graph_in, 'figure'),
        Input(ids.value_dict, 'data'),
    )
    def make_graph_in(inputs: dict) -> go.Figure:
        a = inputs['a']
        b = inputs['b']
        p = TestProcess()
        p.input_data(a, b)
        plotter = p.get_input_plotter()
        fig = plotter.plot_1d()
        fig.add_hline(a*b)
        return fig


#
# class TestProcess2(Process):
#     def input_data(self, x, data, c):
#         self._data_input = dict(x=x, data=data, c=c)
#
#     def preprocess(self):
#         pass
#
#     def output_data(self) -> PlottableData:
#         x, data, c = self._data_input['x'], self._data_input['data'], self._data_input['c']
#         x = x*c
#         return PlottableData(
#             data=data,
#             x=x
#         )
#
#     def get_input_plotter(self) -> DataPlotter:
#         x, data, c = self._data_input['x'], self._data_input['data'], self._data_input['c']
#         p = DataPlotter(PlottableData(data=data, x=x),
#                         xlabel='x label', ylabel='y label', data_label='data label', title='title')
#         return p
#
#     def get_output_plotter(self) -> DataPlotter:
#         p = DataPlotter(self.output_data(), xlabel='x label', ylabel='y label', data_label='data label',
#                         title='title')
#         return p
#
#     def save_progress(self, group: h5py.Group, **kwargs):
#         raise NotImplementedError
#         print(f'Fake saving a = {self._data_input["a"]}, b = {self._data_input["b"]}')
#
#     @classmethod
#     def load_progress(cls, group: h5py.Group) -> Process:
#         raise NotImplementedError
#         print(f'Fake Loading a = 2, b = 5')
#         inst = cls()
#         inst._data_input = dict(a=2, b=5)
#         return inst
#
#
# class TestInterface2(TestInterface):
#     id_c_input = 'c_input_id'
#
#     def required_input_components(self) -> List[Union[ProcessComponentInput, List[ProcessComponentInput]]]:
#         existing_inputs = super().required_input_components()
#         input = dbc.Input(self.id_c_input, type='number')
#         input_w_label = html.Div([
#             dbc.Label('Input C'),
#             input
#         ])
#         existing_inputs.update({self.id_c_input: input_w_label})
#         return existing_inputs
#
#     def all_outputs(self) -> List[ProcessComponentOutput]:
#         in
#         pass
#
#     def main_outputs(self) -> List[ProcessComponentOutput]:
#         pass


def layout():
    print(TestInterface().all_outputs())
    return html.Div([
        dbc.Row([
            dbc.Col([v for k, v in TestInterface().required_input_components().items()], width=3),
            dbc.Col([v for k, v in TestInterface().all_outputs().items()], width=9),
        ])
    ]
    )


if __name__ == '__main__':
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = layout()
    app.run_server(debug=True, port=8051)
else:
    dash.register_page(__name__)
    pass

