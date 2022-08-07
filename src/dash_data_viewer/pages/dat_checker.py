import dash
import numbers
import json
from dash import html, callback, Input, Output
import dash_bootstrap_components as dbc

import dash_data_viewer.components as c
from dash_data_viewer.new_dat_util import get_dat

from dat_analysis.new_dat.new_dat_util import get_local_config, NpEncoder

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

UNIQUE_PAGE_ID = 'dat-checker'

dat_selector = c.DatSelectorAIO()

logs_info = html.Div([
    html.H3('Logs'),
    c.CollapseAIO(content=html.Div(id='dc-div-logs-info', children='Not yet updated'), button_text='Logs',
                  start_open=True),
    html.Hr(),
])


@callback(
    Output('dc-div-logs-info', 'children'),
    Input(dat_selector.store_id, 'data'),
)
def update_logs_area(data_path):
    entries = []

    message = c.MessagesAIO(call_kwargs={'data_path': data_path}, unique_id=UNIQUE_PAGE_ID)

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
    dbc.Card([
        dbc.CardHeader('Toggle Visibility'),
        dbc.CardBody([
            c.MessagesAIO.collapse_all_button(unique_id=UNIQUE_PAGE_ID, which='success'),
            c.MessagesAIO.collapse_all_button(unique_id=UNIQUE_PAGE_ID, which='warning'),
            c.MessagesAIO.collapse_all_button(unique_id=UNIQUE_PAGE_ID, which='error'),
        ])
    ]),
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
