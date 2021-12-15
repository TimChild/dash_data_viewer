from __future__ import annotations
import dash
import dash_extensions.snippets
from dash import html, dcc, callback, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
from dash_data_viewer.cache import cache

from dat_analysis.plotting.plotly import OneD, TwoD
from dat_analysis.dat_object.make_dat import DatHandler, get_newest_datnum, get_dat, get_dats
import dat_analysis.useful_functions as u

import logging
import numpy as np
import plotly.graph_objects as go

from typing import TYPE_CHECKING, Tuple

from dash_data_viewer.entropy_report import EntropyReport
from dash_data_viewer.transition_report import TransitionReport
from dash_data_viewer.layout_util import label_component
# from dash_data_viewer.layout_util import label_component
from dash_data_viewer.cache import cache

if TYPE_CHECKING:
    from dash.development.base_component import Component
    from dat_analysis.dat_object.dat_hdf import DatHDF
    from dat_analysis.dat_object.attributes.square_entropy import Output


class MainComponents(object):
    """Convenient holder for any components that will end up in the main area of the page"""
    div = html.Div(id='noise-div-text')
    graph = dcc.Graph(id='noise-graph-graph1')
    graph2 = dcc.Graph(id='noise-graph-graph2')


class SidebarComponents(object):
    """Convenient holder for any components that will end up in the sidebar area of the page"""
    button = dbc.Button('Click Me', id='button-click')
    datnum = dbc.Input(id='noise-input-datnum', persistence='local', debounce=True, placeholder="Datnum", type='number')


# Initialize the components ONCE here.
main_components = MainComponents()
sidebar_components = SidebarComponents()

# @cache.memoize()
def get_data(datnum) -> Tuple[np.darray, np.ndarray]:
    print(f'getting data for dat{datnum}')
    data = None
    x = None
    if datnum:
        dat = get_dat(datnum)
        x = dat.Data.get_data('x')
        if 'Experiment Copy/current' in dat.Data.keys:
            data = dat.Data.get_data('Experiment Copy/current')
        elif 'Experiment Copy/cscurrent' in dat.Data.keys:
            data = dat.Data.get_data('Experiment Copy/cscurrent')
        elif 'Experiment Copy/current_2d' in dat.Data.keys:
            data = dat.Data.get_data('Experiment Copy/current_2d')[0]
        elif 'Experiment Copy/cscurrent_2d' in dat.Data.keys:
            data = dat.Data.get_data('Experiment Copy/cscurrent_2d')[0]
        else:
            print(f'No data found. valid keys are {dat.Data.keys}')
    return x, data



@callback(Output(main_components.graph.id, 'figure'), Input(sidebar_components.datnum.id, 'value'))
def update_graph(datnum) -> go.Figure:
    x, data = get_data(datnum)
    if x is not None and data is not None:
        p1d = OneD(dat=None)
        fig = p1d.figure(title=f'Test')
        fig.add_trace(p1d.trace(x=x, data=data))
        
        return fig
    return go.Figure()

@callback(Output(main_components.graph2.id, 'figure'), Input(sidebar_components.datnum.id, 'value'))
def update_graph(datnum) -> go.Figure:
    x, data = get_data(datnum)
    if x is not None and data is not None:
        p1d = OneD(dat=None)
        fig = p1d.figure(title=f'Test 2')
        fig.add_trace(p1d.trace(x=x, data=data))
        
        return fig
    return go.Figure()

def main_layout() -> Component:
    global main_components
    m = main_components
    layout_ = html.Div([
        m.graph,
        m.graph2,
    ])
    return layout_


def sidebar_layout() -> Component:
    global sidebar_components
    s = sidebar_components
    layout_ = html.Div([
        s.button,
        label_component(s.datnum, 'Datnum:')
    ])
    return layout_


def layout():
    """Must return the full layout for multipage to work"""

    sidebar_layout_ = sidebar_layout()
    main_layout_ = main_layout()

    return html.Div([
        html.H1('Noise Analysis'),
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
    cache.init_app(app.server)
    app.run_server(debug=True, port=8051)
else:
    dash.register_page(__name__)
