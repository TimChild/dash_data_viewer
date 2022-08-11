import dash
import numbers
import os
import json
from dash import html, callback, Input, Output, ctx
import dash_bootstrap_components as dbc

import dash_data_viewer.components as c
from dash_data_viewer.new_dat_util import get_dat_from_exp_path

from dat_analysis.dat.dat_util import get_local_config, NpEncoder

import logging

logger = logging.getLogger(__name__)

config = get_local_config()
ddir = config['loading']['path_to_measurement_data']

"""
Plan:
Overall aim
- Ensure that things are loaded in HDF file correctly, and if not display a nice message that helps fix
- Check all things that might exist for a given file, or that should exist

- Maybe do a version that does not copy data, so that it is very quick to check many metadatas which are the most likely
to have problems






#### For possible use later
  'transform': 'scale(0.5)',
  '-ms - transform': 'scale(.5)',  # *IE9 * /
  '-webkit-transform': 'scale(.5)',  # / *Safari and Chrome * /
  '-o - transform': 'scale(.5)',  # / *Opera * /
  '-moz - transform': 'scale(.5)',  # / *Firefox * /

"""

global_persistence = 'local'
persistence_on = True


class DatList:
    def __init__(self, host, user, experiment, datnums, expected_type=None):
        self.host = host
        self.user = user
        self.experiment = experiment
        self.datnums = datnums
        self.expected_type = expected_type


RANGE_OF_DATS = [
    DatList(host='qdev-xld', user='Tim', experiment='202207_InstabilityTest_Bale_Dendi', datnums=[1, 2, 3],
            expected_type='first few scans'),
    DatList(host='qdev-xld', user='Tim', experiment='202207_InstabilityTest_Bale_Dendi', datnums=[80, 148, 162],
            expected_type='noise'),
    DatList(host='qdev-xld', user='Tim', experiment='202207_InstabilityTest_Bale_Dendi', datnums=[25, 26, 31, 319],
            expected_type='ohmic check'),
    DatList(host='qdev-xld', user='Tim', experiment='202207_InstabilityTest_Bale_Dendi', datnums=[33, 34, 48, 179, 181],
            expected_type='pinch off'),
    DatList(host='qdev-xld', user='Tim', experiment='202207_InstabilityTest_Bale_Dendi', datnums=[251, 252, 261],
            expected_type='dot tune'),
    # DatList(host='qdev-xld', user='Owen', experiment='GaAs/non_local_entropy_febmar21', datnums=[717, 718, 719],
    #         expected_type=''),
    # DatList(host='qdev-xld', user='Owen', experiment='GaAs/non_local_entropy_febmar21', datnums=[748, 754, 1558],
    #         expected_type='noise'),
    # DatList(host='qdev-xld', user='Owen', experiment='GaAs/non_local_entropy_febmar21', datnums=[1690, 1691, 1714, 1715],
    #         expected_type='noise on off transition'),
    # DatList(host='qdev-xld', user='Owen', experiment='GaAs/non_local_entropy_febmar21', datnums=[738, 739, 751, 816, 818],
    #         expected_type='pinch off'),
    # DatList(host='qdev-xld', user='Owen', experiment='GaAs/non_local_entropy_febmar21', datnums=[758, 759, 766],
    #         expected_type='dot tune'),
    # DatList(host='qdev-xld', user='Owen', experiment='GaAs/non_local_entropy_febmar21', datnums=[796, 797, 807, 813],
    #         expected_type='transition'),
    # DatList(host='qdev-xld', user='Owen', experiment='GaAs/non_local_entropy_febmar21', datnums=[1100, 1101, 1106, 1114],
    #         expected_type='entropy'),
    # DatList(host='qdev-xld', user='Owen', experiment='GaAs/non_local_entropy_febmar21', datnums=[],
    #         expected_type=''),
]

UNIQUE_PAGE_ID = 'dat-checker'

dat_selector = c.DatSelectorAIO()
overwrite_dat_button = dbc.Button('Fully reset dat', id='dc-but-reset', color='danger')
specific_dat_collapse = c.CollapseAIO(content=html.Div([dat_selector, overwrite_dat_button]), button_text='Check Specific Dat', start_open=True)


def generate_dat_check_div(data_path, overwrite=False):
    entries = []

    message = c.MessagesAIO(call_kwargs={'data_path': data_path}, unique_id=UNIQUE_PAGE_ID)

    message.setup(request='test_request_responses')
    entries.append(message.success_message('success returned'))
    entries.append(message.warning_message('warning returned', 'warning expected'))
    entries.append(message.error_message('error returned'))

    message.setup(request=f'get_dat({data_path})')
    try:
        dat = get_dat_from_exp_path(data_path, overwrite=overwrite)
        if dat is not None:
            entries.append(message.success_message(dat))
        else:
            entries.append(message.warning_message(dat, f'Valid dat from {data_path}'))
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

    message.setup(request='dat.Logs.sweeplogs_string')
    try:
        sweeplogs_string = logs.sweeplogs_string
        entries.append(message.success_message(sweeplogs_string))
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


@callback(
    Output('dc-div-logs-info', 'children'),
    Input(specific_dat_collapse.collapse, 'is_open'),
    Input(dat_selector.store_id, 'data'),
    Input(overwrite_dat_button, 'n_clicks'),
)
def update_logs_area(specific_dat, data_path, overwrite_clicks):
    if specific_dat:
        overwrite = True if ctx.triggered_id == 'dc-but-reset' else False
        return generate_dat_check_div(data_path, overwrite=overwrite)
    else:
        dat_results = []
        for datlist in RANGE_OF_DATS:
            header = html.H3(
                f'Host: {datlist.host}\nUser: {datlist.user}\nExperiment: {datlist.experiment}\nType: {datlist.expected_type}',
                style={'white-space': 'pre-wrap'})
            dirpath = os.path.join(ddir, datlist.host, datlist.user, datlist.experiment)
            body = []
            for datnum in datlist.datnums:
                path = os.path.join(dirpath, f'dat{datnum}.h5')
                dat_entry = dbc.Card([
                    dbc.CardHeader(
                        html.H4(f'Dat{datnum}:'),
                    ),
                    dbc.CardBody(
                        generate_dat_check_div(path)
                    ),
                ])
                body.append(dat_entry)
            dat_results.append(
                dbc.Container([
                    dbc.Card([
                        dbc.CardHeader([
                            header
                        ]),
                        dbc.CardBody([
                            dbc.Container([
                                dbc.Container(b, style={
                                    'display': 'flex',
                                    'transform': 'rotateX(180deg)'
                                }) for b in body],
                                style={
                                    'display': 'flex',
                                    'overflow-x': 'scroll',
                                    'transform': 'rotateX(180deg)',
                                }),
                        ]),
                    ]),
                ])
            )
        layout = dbc.Container(
            # html.Div(
            html.Div(
                [
                    dbc.Container(result, style={
                        'display': 'flex',
                        # 'transform': 'scale(0.5)',
                    }
                                  ) for result in dat_results],
                style={
                    'transform': 'rotateX(180deg)',
                    'display': 'flex',
                }
            ),
            style={'transform': 'rotateX(180deg)',
                   'overflow-x': 'scroll',
                   'display': 'flex'}
            # ),
        )

        # 'transform': 'rotateX(180deg)',
        return layout


logs_info = html.Div([
    html.H3('Logs'),
    c.CollapseAIO(content=html.Div(id='dc-div-logs-info', children='Not yet updated'), button_text='Logs',
                  start_open=True),
    html.Hr(),
])

sidebar = dbc.Container([
    specific_dat_collapse,

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
