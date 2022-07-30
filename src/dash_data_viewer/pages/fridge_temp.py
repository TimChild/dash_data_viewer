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
from dat_analysis.analysis_tools.transition import CenteredAveragingProcess, TransitionFitProcess
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
- Select multiple dats at varying fridge temp
- Fit transition to them all (with some options for centering etc
- Plot avg thetas
- Fit linear region to convert from theta to fridge temp

"""

global_persistence = 'local'

dat_selector = c.DatSelectorAIO(multi_select=True)

data_options = label_component(
    dd_opts := dcc.Dropdown(id='ft-dd-data-names', value='', options=[]), 'Data'
)


@callback(
    Output('ft-dd-data-names', 'options'),
    Output('ft-dd-data-names', 'value'),
    # Input('store-data-path', 'data'),
    Input(dat_selector.store_id, 'data'),
    State('ft-dd-data-names', 'value'),
)
def update_data_options(dat_paths, current_value):
    options = None
    value = None
    for path in dat_paths:
        dat = get_dat(path)
        if dat:
            if options is None:
                options = dat.Data.data_keys
            else:
                options = list(set(options).intersection(set(dat.Data.data_keys)))  # Only options valid for all dats

    if options is None:  # If options never got set at all
        options = []
    if current_value and current_value in options:
        value = current_value
    elif options:
        value = options[0]
    return options, value


g1 = c.GraphAIO(aio_id='theta-graph', figure=None)
theta_graphs = html.Div([
    g1,
])




@callback(
    Output(g1.update_figure_store_id, 'data'),
    # Input('store-data-path', 'data'),
    Input(dat_selector.store_id, 'data'),
    Input('ft-dd-data-names', 'value'),
)
def update_thetas_graph(data_paths, data_key) -> go.Figure():
    fig = go.Figure()
    for path in data_paths:
        dat = get_dat(path)
        if dat:
            averaging = CenteredAveragingProcess()
            averaging.set_inputs(
                x=dat.Data.x,
                datas=dat.Data.get_data(data_key),
                center_by_fitting=True,
                fit_start_x=None, fit_end_x=None,
                initial_params=None,
                override_centers_for_averaging=None,
            )
            averaging.process(),  # TODO: Save this result to dat

            avg_data = averaging.outputs['averaged']
            avg_x = averaging.outputs['x']

            fit_process = TransitionFitProcess()
            fit_process.set_inputs(
                x=avg_x,
                transition_data=avg_data,
            )
            fit_process.process()
            fit = fit_process.outputs['fits']
            if fit and fit.success:
                temp = dat.Logs.temperatures['mc']*1000
                fig.add_trace(go.Scatter(x=[temp], y=[fit.outputs['fits'][0].best_values.get('theta', np.nan)], name=f'Dat{dat.datnum}'))
    return fig


dat_graphs = html.Div(id='ft-div-dat-graphs')
per_dat_graphs = html.Div([
    per_dat_collapse := c.CollapseAIO(content=dat_graphs, button_text='Per Dat Graphs', start_open=True)
])


def check_exists(dat: DatHDF, group_path: str):
    if dat:
        with dat as f:
            # if 'dash' in f.keys() and 'fridge_temp' in f['dash'].keys() and 'centering' in f['dash']['fridge_temp'].keys():
            if group := f.get(group_path, None):
                if group is not None:
                    return True
    return False


def get_averaging(dat: DatHDF, data_key: str):
    save_name = f'centering_{data_key.replace("/", ".")}'

    averaging = None
    with dat as f:  # Read only first
        if check_exists(dat, f'dash/fridge_temp/{save_name}'):
            averaging = CenteredAveragingProcess.from_hdf(f['dash']['fridge_temp'], name=save_name)

    if not averaging:
        dat.mode = 'r+'
        with dat as f:  # Now with write mode
            if check_exists(dat, 'dash/fridge_temp/centering'):  # In case this call was queued behind another call
                averaging = CenteredAveragingProcess.from_hdf(f['dash']['fridge_temp'], name=save_name)
            else:
                averaging = CenteredAveragingProcess()
                averaging.set_inputs(
                    x=dat.Data.x,
                    datas=dat.Data.get_data(data_key),
                    center_by_fitting=True,
                    fit_start_x=None, fit_end_x=None,
                    initial_params=None,
                    override_centers_for_averaging=None,
                )
                averaging.process(),  # TODO: Save this result to dat
                ft_group = f.require_group('dash/fridge_temp')
                averaging.save_to_hdf(ft_group, name='centering')
    return averaging


@callback(
    Output(dat_graphs, 'children'),
    Input(dat_selector.store_id, 'data'),
    Input(dd_opts, 'value'),
    State(per_dat_collapse.collapse, 'is_open'),
)
def update_per_dat_graphs(data_paths, data_key, open):
    def data_with_centers_fig(x, y, data_2d, centers):
        fig = go.Figure()
        fig.add_trace(go.Heatmap(x=x, y=y, z=data_2d))
        fig.add_trace(go.Scatter(x=y, y=centers, mode='markers', marker=dict(color='white')))
        return fig

    def data_with_stdev_fig(x, data, stdev):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=data, yerr=stdev))
        return fig

    if not open:
        return []
    all_entries = []
    for path in data_paths:
        dat = get_dat(path)
        if dat:
            figs = []
            averaging = get_averaging(dat, data_key=data_key)

            figs.append(data_with_centers_fig(x=averaging.inputs['x'], y=dat.Data.y, data_2d=averaging.inputs['datas'], centers=averaging.outputs['centers']))
            figs.append(data_with_stdev_fig(x=averaging.outputs['x'], data=averaging.outputs['averaged'], stdev=averaging.outputs['std_errs']))

            entry = html.Div([
                html.H3(f'Dat{dat.datnum}'),
                *[c.GraphAIO(figure=fig) for fig in figs],
            ])

            all_entries.append(entry)

    layout = []
    for entry in all_entries:
        layout.append(entry)
        layout.append(html.Hr())
    return html.Div(children=layout)





sidebar = dbc.Container([
    dat_selector,
    data_options,
],
)

main = dbc.Container([
    per_dat_graphs,
    theta_graphs,
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
