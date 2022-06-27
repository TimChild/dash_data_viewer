from __future__ import annotations
import dash
import json
import os
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
from typing import TYPE_CHECKING, List, Union, Optional, Any, Tuple

import numpy as np
import plotly.graph_objects as go

import dash_data_viewer.components as c
from dash_data_viewer.layout_util import label_component

from dat_analysis.analysis_tools.general_fitting import calculate_fit, FitInfo
from dat_analysis.useful_functions import data_to_json, data_from_json, get_data_index, mean_data
from dat_analysis.plotting.plotly.dat_plotting import OneD
from dat_analysis.new_dat.dat_hdf import get_dat_from_exp_filepath
from dat_analysis.new_dat.new_dat_util import get_local_config

import logging
logger = logging.getLogger(__name__)

config = get_local_config()


def get_dat(data_path):
    dat = None
    if data_path and os.path.exists(data_path):
        try:
            dat = get_dat_from_exp_filepath(experiment_data_path=data_path, overwrite=False)
        except Exception as e:
            logger.warning(f'Failed to load dat at {data_path}. Raised {e}')
    else:
        logger.info(f'No file at {data_path}')

    return dat




"""
Plan:
Overall aim
- Select Host, User, Experiment (/possibly deeper) name
- Select Dat num (from available)
    - Or range of dats etc
- See some Logs info automatically (FastDAC settings, Temperatures, etc)
- See data plotted
    - 1D with x_axis
    - 2D as heatmap with x and y axis
        - A line cut slider



"""

global_persistence = 'session'
persistence_on = True

dat_selector = html.Div([
    label_component(c.Input_(id='inp-host-name', placeholder='e.g. qdev-xld',
                             persistence=persistence_on, persistence_type=global_persistence), 'Host Name'),
    label_component(c.Input_(id='inp-user-name', placeholder='e.g. Tim',
                             persistence=persistence_on, persistence_type=global_persistence), 'User Name'),
    label_component(c.Input_(id='inp-experiment-name', placeholder='e.g. 202206_TestCondKondoQPC',
                             persistence=persistence_on, persistence_type=global_persistence), 'User Name'),
    label_component(c.Input_(id='inp-datnum', placeholder='e.g. 1', inputmode='numeric',
                             persistence=persistence_on, persistence_type=global_persistence), 'Datnum'),
    label_component(dbc.RadioButton(id='tog-raw'), 'Raw'),
    dcc.Store('store-data-path')
])

@callback(
    Output('store-data-path', 'data'),
    Input('inp-host-name', 'value'),
    Input('inp-user-name', 'value'),
    Input('inp-experiment-name', 'value'),
    Input('inp-datnum', 'value'),
    Input('tog-raw', 'value'),
)
def generate_data_path(host, user, experiment, datnum, raw):
    host = host if host else ''
    user = user if user else ''
    experiment = experiment if experiment else ''
    datnum = datnum if datnum else 0
    base_path = config['loading']['path_to_measurement_data']
    datfile = f'dat{datnum}_RAW.h5' if raw else f'dat{datnum}.h5'
    data_path = os.path.join(base_path, host, user, experiment, datfile)
    return data_path


data_options = label_component(
    dcc.Dropdown(id='dd-data-names', value='', options=[]),
    'Data')


@callback(
    Output('dd-data-names', 'options'),
    Output('dd-data-names', 'value'),
    Input('store-data-path', 'data'),
    State('dd-data-names', 'value'),
)
def update_data_options(data_path, current_value):
    dat = get_dat(data_path)
    options = []
    value = None
    if dat:
        options = dat.Data.data_keys
        if current_value and current_value in options:
            value = current_value
        elif options:
            value = options[0]
    return options, value


graphs = html.Div([
    dcc.Graph(id='graph-1', figure=go.Figure())
])



@callback(
    Output('graph-1', 'figure'),
    Input('store-data-path', 'data'),
    Input('dd-data-names', 'value'),
)
def update_graph(data_path, data_key) -> go.Figure():
    dat = get_dat(data_path)
    fig = go.Figure()
    if dat:
        data = dat.Data.get_data(data_key)
        x = dat.Data.x
        y = dat.Data.y

        if data is not None and data.ndim > 0:
            x = x if x is not None else np.linspace(0, data.shape[-1], data.shape[-1])

            if data.ndim == 1:
                fig.add_trace(go.Scatter(x=x, y=data))
            elif data.ndim == 2:
                y = y if y is not None else np.linspace(0, data.shape[-2], data.shape[-2])
                fig.add_trace(go.Heatmap(x=x, y=y, z=data))
            else:
                pass

    return fig



logs_info = html.Div([
    html.H3('Logs'),
    html.Div(id='div-logs-info', children='Not yet updated')
])

@callback(
    Output('div-logs-info', 'children'),
    Input('store-data-path', 'data'),
)
def update_logs_area(data_path):
    dat = get_dat(data_path)
    entries = []
    if dat:
        logs = dat.Logs
        entries.append(dcc.Markdown(f'''Available logs are: {dat.Logs.logs_keys}'''))

        dacs = logs.dacs
        if dacs is not None and isinstance(dacs, dict):
            entries.append(dcc.Markdown(json.dumps(dacs, indent=2)))

        temperatures = logs.temperatures
        if temperatures is not None and isinstance(temperatures, dict):
            entries.append(dcc.Markdown(json.dumps(temperatures, indent=2)))

    return html.Div([entry for entry in entries])



sidebar = dbc.Container([
    dat_selector,
    data_options
],
)

main = dbc.Container([
   graphs,
   logs_info
])





layout = dbc.Container([
    dbc.Row([
        dbc.Col([sidebar], width=3),
        dbc.Col([main], width=9)
    ])
], fluid=True)

if __name__ == '__main__':
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    # app.layout = layout()
    app.layout = layout
    app.run_server(debug=True, port=8051)
else:
    dash.register_page(__name__)
    pass