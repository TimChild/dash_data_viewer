from __future__ import annotations
import dash
import json
import os
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc

import numpy as np
import plotly.graph_objects as go

import dash_data_viewer.components as c
from dash_data_viewer.layout_util import label_component
from dash_data_viewer.new_dat_util import get_dat, ExperimentFileSelector

from dat_analysis.new_dat.new_dat_util import get_local_config, NpEncoder
from dat_analysis.hdf_file_handler import GlobalLock
import tempfile



import logging
logger = logging.getLogger(__name__)

config = get_local_config()

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


dat_selector = ExperimentFileSelector()
data_path_store = dcc.Store(id='store-data-path', storage_type='session')


datnum = c.Input_(id='inp-datnum', type='number', value=0, persistence_type=global_persistence, persistence=persistence_on)
raw_tog = dcc.RadioItems(id='tog-raw', persistence=persistence_on, persistence_type=global_persistence)


@callback(
    Output('store-data-path', 'data'),
    Input(dat_selector.store_id, 'data'),
    Input(datnum.id, 'value'),
    Input(raw_tog.id, 'value'),
)
def generate_dat_path(filepath, datnum, raw):
    datnum = datnum if datnum else 0
    data_path = None
    if filepath and os.path.exists(filepath):
        if os.path.isdir(filepath):
            datfile = f'dat{datnum}_RAW.h5' if raw else f'dat{datnum}.h5'
            data_path = os.path.join(filepath, datfile)
        else:
            data_path = filepath
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
    datnum,
    raw_tog,
    data_options,
    data_path_store,
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
