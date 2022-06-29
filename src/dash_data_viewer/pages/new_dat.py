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
from dat_analysis.new_dat.new_dat_util import get_local_config, NpEncoder
from dat_analysis.hdf_file_handler import GlobalLock
import tempfile

import logging
logger = logging.getLogger(__name__)

config = get_local_config()
global_lock = GlobalLock(os.path.join(tempfile.gettempdir(), 'dash_lock.lock'))


def get_dat(data_path):
    dat = None
    with global_lock:
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

global_persistence = 'local'
persistence_on = True


ddir = config['loading']['path_to_measurement_data']
host_options = [k for k in os.listdir(ddir) if os.path.isdir(os.path.join(ddir, k))]

dat_selector = html.Div([
    label_component(dcc.Dropdown(id='dd-host-name', options=host_options, persistence=True, persistence_type=global_persistence), 'Host Name'),
    label_component(dcc.Dropdown(id='dd-user-name'), 'User Name'),
    label_component(dcc.Dropdown(id='dd-experiment-name'), 'Experiment Name'),
    label_component(c.Input_(id='inp-datnum', placeholder='0', persistence=True, persistence_type=global_persistence), 'Datnum'),
    label_component(dbc.RadioButton(id='tog-raw', persistence=True, persistence_type=global_persistence), 'Raw'),
    dcc.Store('store-data-path'),
    dcc.Store('store-selections', storage_type=global_persistence),
])


def _default_selections_dict():
    return {
        'host': None,
        'user': None,
        'experiment': None,
        'datnum': None,  # Not using yet
        'data': None,  # Not using yet
    }


def ensure_selections_dict(d):
    """If not already a proper selections dict, will replace with default one"""
    if not d or not isinstance(d, dict) or _default_selections_dict().keys() != d.keys():
        d = _default_selections_dict()
    return d


@callback(
    Output('store-selections', 'data'),
    State('store-selections', 'data'),
    Input('dd-host-name', 'value'),
    Input('dd-user-name', 'value'),
    Input('dd-experiment-name', 'value'),
    Input('inp-datnum', 'value'),
    Input('dd-data-names', 'value'),

)
def store_selections(old_selections, new_host, new_user, new_exp, new_datnum, new_dataname):
    old_selections = ensure_selections_dict(old_selections)
    print(f'store_selections: {old_selections}, {new_host, new_user, new_exp, new_datnum, new_dataname}')
    updated = False
    for k, v in {'host': new_host, 'user': new_user, 'experiment': new_exp, 'datnum': new_datnum, 'data': new_dataname}.items():
        if v != old_selections[k]:
            updated = True
            old_selections[k] = v
    if updated:
        return old_selections
    else:
        return dash.no_update


@callback(
    Output('dd-user-name', 'options'),
    Output('dd-user-name', 'value'),
    Input('dd-host-name', 'value'),
    State('store-selections', 'data'),
)
def update_user_options(host_name, current_selections):
    current_selections = ensure_selections_dict(current_selections)
    # new_user = dash.no_update
    new_user = None
    new_options = []
    print(f'update_user: {host_name}, {current_selections}')
    if host_name:
        new_options = os.listdir(os.path.join(ddir, host_name))
        if current_selections['user'] in new_options:
            new_user = current_selections['user']
        else:
            new_user = None
    return new_options, new_user


@callback(
    Output('dd-experiment-name', 'options'),
    Output('dd-experiment-name', 'value'),
    State('store-selections', 'data'),
    State('dd-host-name', 'value'),
    Input('dd-user-name', 'value'),
)
def update_experiment_options(current_selections, host_name, user_name):
    current_selections = ensure_selections_dict(current_selections)
    # host = current_selections['host'] if current_selections['host'] else ''
    host = host_name
    # user = selections['user'] if selections['user'] else ''
    user = user_name

    print(f'update_experiment: {host_name, user_name}, {current_selections}')
    new_exp = None
    new_options = []
    if host and user:
        new_options = os.listdir(os.path.join(ddir, host, user))
        if current_selections['experiment'] in new_options:
            new_exp = current_selections['experiment']
        else:
            new_exp = None
    return new_options, new_exp


@callback(
    Output('store-data-path', 'data'),
    Input('dd-host-name', 'value'),
    Input('dd-user-name', 'value'),
    Input('dd-experiment-name', 'value'),
    Input('inp-datnum', 'value'),
    Input('tog-raw', 'value'),
)
def generate_dat_path(host, user, experiment, datnum, raw):
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
def update_data_options(dat_path, current_value):
    dat = get_dat(dat_path)
    options = []
    value = None
    if dat:
        options = dat.Data.data_keys
        if current_value and current_value in options:
            value = current_value
        elif options:
            value = options[0]
    return options, value


g1 = c.GraphAIO(aio_id='graph-1', figure=None)
graphs = html.Div([
    g1,
])


@callback(
    Output(g1.graph_id, 'figure'),
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

        fig.update_layout(
            title=f'Dat{dat.datnum}: {data_key}',
        )
        fig.update_xaxes(title_text=dat.Logs.x_label)

        if data is not None and data.ndim > 0:
            x = x if x is not None else np.linspace(0, data.shape[-1], data.shape[-1])

            if data.ndim == 1:
                fig.add_trace(go.Scatter(x=x, y=data))
            elif data.ndim == 2:
                y = y if y is not None else np.linspace(0, data.shape[-2], data.shape[-2])
                fig.update_yaxes(title_text=dat.Logs.y_label)
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
        if temperatures is not None:
            try:
                temp_dict = temperatures.asdict()
                entries.append(dcc.Markdown(json.dumps(temp_dict, indent=2, cls=NpEncoder)))
            except TypeError as e:
                logger.error(f'TypeError when loading temperatures ({temperatures})\n'
                             f'Error: {e}')

    return html.Div([entry for entry in entries])


all_graphs = html.Div(id='div-all-graphs')

@callback(
    Output('div-all-graphs', 'children'),
    Input('store-data-path', 'data'),
    Input('dd-data-names', 'value'),
)
def generate_all_data_graphs(dat_path, avoid_selected):
    dat = get_dat(dat_path)
    figs = []
    if dat:
        x = dat.Data.x
        y = dat.Data.y
        for k in dat.Data.data_keys:
            if k != avoid_selected:
                fig = go.Figure()
                data = dat.Data.get_data(k)
                if data is not None and data.ndim > 0:
                    x = x if x is not None else np.linspace(0, data.shape[-1], data.shape[-1])
                    fig.update_layout(
                        title=f'Dat{dat.datnum}: {k}',
                    )
                    fig.update_xaxes(title_text=dat.Logs.x_label)
                    if data.ndim == 1:
                        fig.add_trace(go.Scatter(x=x, y=data))
                        figs.append(fig)
                    elif data.ndim == 2:
                        y = y if y is not None else np.linspace(0, data.shape[-2], data.shape[-2])
                        fig.add_trace(go.Heatmap(x=x, y=y, z=data))
                        fig.update_yaxes(title_text=dat.Logs.y_label)
                        figs.append(fig)
                    else:
                        pass
    dash_figs = [c.GraphAIO(figure=fig) for fig in figs]
    if not dash_figs:
        dash_figs = html.Div('No other data to display')
    return dash_figs


sidebar = dbc.Container([
    dat_selector,
    data_options
],
)

main = dbc.Container([
   graphs,
   logs_info,
    all_graphs,
])


layout = dbc.Container([
    dbc.Row([
        dbc.Col([sidebar], width=4),
        dbc.Col([main], width=8)
    ])
], fluid=True)

if __name__ == '__main__':
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    # app.layout = layout()
    app.layout = layout
    app.run_server(debug=True, port=8051, dev_tools_hot_reload=False, use_reloader=False)
else:
    dash.register_page(__name__)
    pass
