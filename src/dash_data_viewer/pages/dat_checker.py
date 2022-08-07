import dash
import numbers
import json
import uuid
import os
from dash import html, dcc, callback, Input, Output, State, MATCH, ALL
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
    c.CollapseAIO(content=html.Div(id='dc-div-logs-info', children='Not yet updated'), button_text='Logs',
                  start_open=True),
    html.Hr(),
])


class Messages:
    class ids:
        _id_counter = 0

        @classmethod
        def generate_id(cls, which: str, num=None):
            num = num if num else cls._id_num()
            return {
                'component': f'{which}-message',
                'key': num,
            }

        @classmethod
        def _id_num(cls):
            id_ = cls._id_counter
            cls._id_counter += 1
            return id_

    ids = ids

    def __init__(self, call_kwargs):
        self.call_kwargs = call_kwargs
        self._request = None

    def setup(self, request):
        self._request = request

    def success_message(self, returned, dump_json=False):
        # TODO: Add a collapse all successful callback
        if dump_json:
            returned = json.dumps(returned, indent=2, cls=NpEncoder)
        message = html.Div([
            html.H5(f'Success: {self._request}'),
            dcc.Markdown(str(returned), style={'white-space': 'pre'}),
            html.Hr(),
        ],
            id=self.ids.generate_id('success'),
            style={'color': 'green'},
        )
        return message

    def warning_message(self, returned, expected):
        message = html.Div([
            html.H5(f'Warning: {self._request}'),
            dcc.Markdown(f'Found:\n{returned}\n\nExpected:\n{expected}'),
            html.Hr(),
        ],
            id=self.ids.generate_id('warning'),
            style={'color': 'orange'}
        )
        return message

    def error_message(self, exception, additional_info=None):
        additional_info = html.P(f'{additional_info}') if additional_info else html.Div()
        message = html.Div([
            html.H5(f'Error: {self._request}'),
            html.P('\n'.join(['Exception Raised:', f'{exception}'])),
            additional_info,
            dcc.Markdown(f'Call kwargs: \n {self.call_kwargs}',
                         style={'white-space': 'pre'}),
            html.Hr(),
        ],
            id=self.ids.generate_id('error'),
            style={'color': 'red'},
        )
        return message

    @classmethod
    def collapse_all_button(cls, which: str):
        button = dbc.Button(id=f'dc-button-{which}-visible', children=which)

        @callback(
            Output(cls.ids.generate_id(which, ALL), 'hidden'),
            Input(button, 'n_clicks'),
            State(cls.ids.generate_id(which, ALL), 'hidden'),
        )
        def toggle_visible(clicks, current_states):
            if clicks:
                return [bool(clicks % 2)]*len(current_states)
            return [False]*len(current_states)

        return button


@callback(
    Output('dc-div-logs-info', 'children'),
    Input(dat_selector.store_id, 'data'),
)
def update_logs_area(data_path):
    entries = []

    message = Messages(call_kwargs={'data_path': data_path})

    message.setup(request='test_request_responses')
    entries.append(message.success_message('success returned'))
    entries.append(message.warning_message('warning returned', 'warning expected'))
    entries.append(message.error_message('error returned'))

    message.setup(request=f'get_dat({data_path})')
    try:
        dat = get_dat(data_path)
        message.success_message(dat)
    except Exception as e:
        dat = None
        entries.append(message.error_message(e))
    if dat is None:
        return html.Div([entry for entry in entries])

    message.setup(request='dat.Logs')
    try:
        logs = dat.Logs
        entries.append(message.success_message(dat.Logs.logs_keys))

    except Exception as e:
        logs = None
        entries.append(message.error_message(e))

    message.setup(request='dat.Logs.comments')
    try:
        comments = logs.comments
        entries.append(message.success_message(','.join(comments)))
    except Exception as e:
        entries.append(message.error_message(e))

    message.setup(request='dat.Logs.dacs')
    try:
        dacs = logs.dacs
        if dacs is not None and isinstance(dacs, dict):
            entries.append(message.success_message(dat.Logs.dacs, dump_json=True))
    except Exception as e:
        entries.append(message.error_message(e))

    message.setup(request='dat.Logs.temperatures')
    try:
        temperatures = logs.temperatures
        temp_dict = temperatures.asdict()
        entries.append(message.success_message(temp_dict, dump_json=True))
    except Exception as e:
        entries.append(message.error_message(e))

    message.setup(request='dat.Logs.x_label')
    try:
        entries.append(message.success_message(logs.x_label))
    except Exception as e:
        entries.append(message.error_message(e))

    message.setup(request='dat.Logs.y_label')
    try:
        entries.append(message.success_message(logs.y_label))
    except Exception as e:
        entries.append(message.error_message(e))

    message.setup(request='dat.Logs.measure_freq')
    try:
        measure_freq = logs.measure_freq
        # If FastDACs present and no measure_freq
        if not isinstance(measure_freq, numbers.Number) and any([f'FastDAC{i}' in logs.logs_keys for i in range(1, 5)]):
            entries.append(message.warning_message(measure_freq, 'Numeric measure frequency if FastDAC was used'))
        else:
            entries.append(message.success_message(measure_freq))
    except Exception as e:
        entries.append(message.error_message(e))

    message.setup(request='dat.Logs.get_fastdac(X)')
    try:
        fd_messages = []
        for i in range(1, 5):
            fd = logs.get_fastdac(i)
            if fd is not None:
                fd_messages.append(f'FastDAC{i}:\n{json.dumps(fd.asdict(), indent=2, cls=NpEncoder)}')
            else:
                fd_messages.append(f'FastDAC{i} not found')
        entries.append(message.success_message('\n\n'.join(fd_messages)))
    except Exception as e:
        entries.append(message.error_message(e))


    return html.Div([entry for entry in entries])


sidebar = dbc.Container([
    dat_selector,
    Messages.collapse_all_button('success'),
    Messages.collapse_all_button('warning'),
    Messages.collapse_all_button('error'),

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
