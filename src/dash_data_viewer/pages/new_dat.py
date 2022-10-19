import dash
import json
import os
import pandas as pd
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc

import numpy as np
import plotly.graph_objects as go

import dash_data_viewer.components as c
from dash_data_viewer.layout_util import label_component
from dash_data_viewer.new_dat_util import get_dat_from_exp_path

from dat_analysis.dat.dat_util import get_local_config, NpEncoder
import dat_analysis.useful_functions as U
import tempfile

import logging
logger = logging.getLogger(__name__)

config = get_local_config()

"""
Plan:
Overall aim
- See some Logs info automatically (FastDAC settings, Temperatures, etc)
- See data plotted
    - 1D with x_axis
    - 2D as heatmap with x and y axis
        - A line cut slider



"""


global_persistence = 'session'
persistence_on = True


def dacs_to_component(dac_dict: dict):
    table_rows = []
    for k, v in dac_dict.items():
        table_rows.append(
            html.Tr([html.Th(k), html.Td(f'{v:.2f}')])
        )
    table_header = [html.Thead([html.Th('DAC'), html.Th('Value /mV')])]
    table_body = [html.Tbody(table_rows)]
    return dbc.Table(table_header+table_body, bordered=True, size='sm')


def temps_to_component(temp_dict: dict):
    table_rows = []
    keys = {
        'fiftyk': '50K',
        'fourk': '4K',
        'magnet': 'Magnet',
        'still': 'Still',
        'mc': 'MC',
    }
    for k, v in temp_dict.items():
        val = f'{v:.1f} K' if v > 1 else f'{v*1000:.0f} mK'
        table_rows.append(
            html.Tr([html.Th(keys[k]), html.Td(val)])
        )
    table_header = [html.Thead([html.Th('Plate'), html.Th('Temperature')])]
    table_body = [html.Tbody(table_rows)]
    return dbc.Table(table_header+table_body, bordered=True, size='sm')


dat_selector = c.DatSelectorAIO()

data_options = label_component(
    dcc.Dropdown(id='dd-data-names', value='', options=[]),
    'Data')


@callback(
    Output('dd-data-names', 'options'),
    Output('dd-data-names', 'value'),
    # Input('store-data-path', 'data'),
    Input(dat_selector.store_id, 'data'),
    State('dd-data-names', 'value'),
)
def update_data_options(dat_path, current_value):
    dat = get_dat_from_exp_path(dat_path)
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


def fig_from_dat(dat, data_key):
    fig = go.Figure()
    data = dat.Data.get_data(data_key)
    x = dat.Data.x
    y = dat.Data.y

    fig.update_layout(
        title=f'Dat{dat.datnum}: {data_key}',
    )
    if data is not None and data.ndim > 0:
        if x is not None and x.shape[0] == data.shape[-1]:
            fig.update_xaxes(title_text=dat.Logs.x_label)
        else:
            x = np.linspace(0, data.shape[-1], data.shape[-1])
            fig.update_xaxes(title_text='data index')

        if data.ndim == 1:
            data, x = U.resample_data(data, x=x, max_num_pnts=500, resample_method='bin')
            fig.add_trace(go.Scatter(x=x, y=data))
        elif data.ndim == 2:
            if y is not None and y.shape[0] == data.shape[-2]:
                fig.update_yaxes(title_text=dat.Logs.y_label)
            else:
                y = np.linspace(0, data.shape[-2], data.shape[-2])
                fig.update_yaxes(title_text='data index')
            data, x, y = U.resample_data(data, x=x, y=y, max_num_pnts=500, resample_method='bin')
            fig.update_yaxes(title_text=dat.Logs.y_label)
            fig.add_trace(go.Heatmap(x=x, y=y, z=data))
        else:
            pass
    return fig


@callback(
    Output(g1.update_figure_store_id, 'data'),
    # Input('store-data-path', 'data'),
    Input(dat_selector.store_id, 'data'),
    Input('dd-data-names', 'value'),
)
def update_graph(data_path, data_key) -> go.Figure():
    dat = get_dat_from_exp_path(data_path)
    if dat:
        return fig_from_dat(dat, data_key)
    else:
        return c.blank_figure()


logs_info = html.Div([
    html.H3('Logs'),
    html.Div(id='div-logs-info', children='Not yet updated')
])


@callback(
    Output('div-logs-info', 'children'),
    # Input('store-data-path', 'data'),
    Input(dat_selector.store_id, 'data'),
)
def update_logs_area(data_path):
    dat = get_dat_from_exp_path(data_path)
    entries = []
    if dat:
        logs = dat.Logs
        entries.append(dcc.Markdown(f'''Available logs are: {dat.Logs.logs_keys}'''))

        dacs = logs.dacs
        if dacs is not None and isinstance(dacs, dict):
            try:
                entries.append(dacs_to_component(dacs))
                # entries.append(dcc.Markdown(json.dumps(dacs, indent=2, cls=NpEncoder)))
            except Exception as e:
                logger.error(f'Error loading DACs ({dacs}): {e}')
                entries.append(html.Div('Error loading dacs'))

        temperatures = logs.temperatures
        if temperatures is not None:
            try:
                temp_dict = temperatures.asdict()
                # entries.append(dcc.Markdown(json.dumps(temp_dict, indent=2, cls=NpEncoder)))
                entries.append(temps_to_component(temp_dict))
            except TypeError as e:
                logger.error(f'TypeError when loading temperatures ({temperatures})\n'
                             f'Error: {e}')
                entries.append(html.Div('Error loading temperatures'))

    return html.Div([entry for entry in entries])


all_graphs = html.Div(id='div-all-graphs')


@callback(
    Output('div-all-graphs', 'children'),
    # Input('store-data-path', 'data'),
    Input(dat_selector.store_id, 'data'),
    Input('dd-data-names', 'value'),
)
def generate_all_data_graphs(dat_path, avoid_selected):
    dat = get_dat_from_exp_path(dat_path)
    figs = []
    if dat:
        for k in dat.Data.data_keys:
            if k not in avoid_selected:
                figs.append(fig_from_dat(dat, k))

    dash_figs = [c.GraphAIO(figure=fig) for fig in figs]
    if not dash_figs:
        dash_figs = html.Div('No other data to display')
    return dash_figs


sidebar = dbc.Container([
    dat_selector,
    # datnum,
    # raw_tog,
    data_options,
    # data_path_store,
],
)

main = dbc.Container([
    graphs,
    logs_info,
    all_graphs,
])


layout = dbc.Container([
    dbc.Row([
        dbc.Col([sidebar], width=6, lg=4),
        dbc.Col([main], width=6, lg=8)
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
