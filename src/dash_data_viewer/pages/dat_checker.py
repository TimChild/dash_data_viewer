import dash
import json
import os
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc

import numpy as np
import plotly.graph_objects as go

import dash_data_viewer.components as c
from dash_data_viewer.layout_util import label_component
from dash_data_viewer.new_dat_util import get_dat

from dat_analysis.new_dat.new_dat_util import get_local_config, NpEncoder
from dat_analysis.new_dat.dat_hdf import DatHDF
from dat_analysis.hdf_file_handler import GlobalLock
import dat_analysis.useful_functions as U
import tempfile

import logging

logger = logging.getLogger(__name__)

config = get_local_config()

"""
Plan:
Overall aim
- Ensure that things are loaded in HDF file correctly, and if not display a nice message that helps fix
- Check all things that might exist for a given file, or that should exist

- Maybe do a version that does not copy data, so that it is very quick to check many metadatas which are the most likely
to have problems



"""

global_persistence = 'local'
persistence_on = True

dat_selector = c.DatSelectorAIO()

logs_info = html.Div([
    html.H3('Logs'),
    c.CollapseAIO(content=html.Div(id='dc-div-logs-info', children='Not yet updated'), button_text='Logs', start_open=True),
    html.Hr(),
])


class CheckMessage:
    def message(self, request, info, *args, dump_json=False, **kwargs):
        if dump_json:
            info = json.dumps(info, indent=2, cls=NpEncoder)
        message = html.Div([
            html.H5(request),
            dcc.Markdown(info, *args, **kwargs, style={'white-space': 'pre'}),
            html.Hr(),
        ])
        return message


class ErrorMessage:
    def __init__(self, call_args=None, call_kwargs=None):
        self.call_args = call_args
        self.call_kwargs = call_kwargs

    def message(self, description, exception, *args, **kwargs):
        message = html.Div([
            html.H4(description),
            html.P('\n'.join(['Exception Raised:', f'{exception}'])),
            dcc.Markdown(f'Call args: \n {self.call_args} \nCall kwargs: \n {self.call_kwargs}',
                         style={'white-space': 'pre'}),
            html.Hr(),
        ],
            style={'color':'red'},
        )
        return message


@callback(
    Output('dc-div-logs-info', 'children'),
    Input(dat_selector.store_id, 'data'),
)
def update_logs_area(data_path):
    entries = []

    error = ErrorMessage(call_kwargs={'data_path': data_path})
    check = CheckMessage()

    try:
        dat = get_dat(data_path)
    except Exception as e:
        dat = None
        entries.append(error.message('While initializing dat', e))
    if dat is None:
        return html.Div([entry for entry in entries])

    try:
        logs = dat.Logs
        entries.append(check.message('dat.Logs', f'''Available logs are: {dat.Logs.logs_keys}'''))

    except Exception as e:
        logs = None
        entries.append(error.message(f'While opening dat.Logs', e))

    try:
        dacs = logs.dacs
        if dacs is not None and isinstance(dacs, dict):
            entries.append(check.message('dat.Logs.dacs', dacs, dump_json=True))
    except Exception as e:
        entries.append(error.message(f'While opening dat.Logs.dacs ', e))

    try:
        temperatures = logs.temperatures
        temp_dict = temperatures.asdict()
        entries.append(check.message('dat.Logs.temperatures', temp_dict, dump_json=True))
    except Exception as e:
        entries.append(error.message(f'While opening dat.Logs.temperatures ', e))

    try:
        comments = logs.comments
        entries.append(check.message('dat.Logs.comments', comments))
    except Exception as e:
        entries.append(error.message(f'While opening dat.Logs.comments', e))

    return html.Div([entry for entry in entries])


sidebar = dbc.Container([
    dat_selector,
],
)

main = dbc.Container([
    logs_info,
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
