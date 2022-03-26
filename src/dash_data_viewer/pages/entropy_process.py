from __future__ import annotations
import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
from typing import TYPE_CHECKING, List, Union, Optional, Any
import uuid

import dash_data_viewer.components as c
from dash_data_viewer.process_dash_extensions import ProcessInterface, ProcessComponentInput, ProcessComponentOutput, ProcessComponentOutputGraph, DashEnabledProcess

from dat_analysis.analysis_tools.new_procedures import SeparateSquareProcess, Process, PlottableData, DataPlotter
from dat_analysis import get_dat


if TYPE_CHECKING:
    from dash.development.base_component import Component


class SeparateSquareProcessDash(DashEnabledProcess, SeparateSquareProcess):
    def dash_full(self, *args, **kwargs) -> Component:
        input_plotter = self.get_input_plotter()
        output_plotter = self.get_output_plotter()

        input_fig = input_plotter.plot_1d()
        output_figs = [output_plotter.plot_heatmap(), output_plotter.plot_waterfall()]
        return html.Div([
            self._dash_children_full(),
            dbc.Row([
                dbc.Col([ProcessComponentOutputGraph(input_fig)], width=6),
                *[dbc.Col([ProcessComponentOutputGraph(fig)], width=6) for fig in output_figs],
            ]),
        ])
        pass

    def mpl_full(self, *args, **kwargs):
        return None

    def pdf_full(self, *args, **kwargs):
        return None


# class SeparateSquareProcessInput(ProcessComponentInput):
#     def __init__(self, aio_id=None):
#         if aio_id is None:
#             aio_id = str(uuid.uuid4())
#         store = dcc.Store(id=self.ids.generic(aio_id, 'store'), storage_type='memory')
#
#         super().__init__(children=[store, ])  # html.Div contains layout
#
#     @staticmethod
#     @callback(
#     )
#     def function():
#         pass


# For making dash interface
class SeparateSquareProcessInterface(ProcessInterface):
    def user_inputs(self, *args, **kwargs) -> html.Div:
        pass

    def other_inputs(self, *args, **kwargs):
        pass

    def required_input_components(self) -> List[Component]:
        return [c.DatnumPickerAIO(aio_id='entropy-process-datpicker', allow_multiple=False)]

    def all_outputs(self) -> List[Component]:
        return [html.Div(id='entropy-process-separated_out')]

    def main_outputs(self) -> List[Component]:
        return []

    # @staticmethod
    # @callback(
    #     Output('entropy-process-separated_out', 'children'),
    #     Input(c.DatnumPickerAIO.ids.dropdown('entropy-process-datpicker'), 'value'),
    # )
    # def do_something(datnum):
    #     if datnum:
    #         dat = get_dat(2164, exp2hdf='febmar21tim')
    #
    #         square_process = SeparateSquareProcessDash()
    #         square_process.input_data(dat.Data.i_sense, dat.Data.x, dat.Logs.measure_freq, round(dat.AWG.info.wave_len/4), 0.01, dat.Data.y)
    #         square_process.preprocess()
    #         square_process.output_data()
    #
    #         layout = square_process.dash_full()
    #         print(layout)
    #
    #         return layout


class MainComponents(object):
    """Convenient holder for any components that will end up in the main area of the page"""


class SidebarComponents(object):
    """Convenient holder for any components that will end up in the sidebar area of the page"""


# Initialize the components ONCE here.
main_components = MainComponents()
sidebar_components = SidebarComponents()


square_interface = SeparateSquareProcessInterface()


def main_layout() -> Component:
    global main_components
    m = main_components
    layout_ = html.Div([
        *square_interface.all_outputs()
    ])
    return layout_


def sidebar_layout() -> Component:
    global sidebar_components
    s = sidebar_components
    layout_ = html.Div([
        *square_interface.required_input_components()
    ])
    return layout_


def layout():
    """Must return the full layout for multipage to work"""

    sidebar_layout_ = sidebar_layout()
    main_layout_ = main_layout()

    return html.Div([
        html.H1('Entropy Process'),
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
    app.run_server(debug=True, port=8051)
else:
    # dash.register_page(__name__)
    pass
