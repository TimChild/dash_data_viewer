import threading

import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc

import numpy as np
import plotly.graph_objects as go

import dash_data_viewer.components as c
from dash_data_viewer.layout_util import label_component
from dash_data_viewer.new_dat_util import get_dat_from_exp_path

from dat_analysis.dat.dat_util import get_local_config
from dat_analysis.analysis_tools.transition import CenteredAveragingProcess, TransitionFitProcess
from dat_analysis.dat.dat_hdf import DatHDF
import dat_analysis.useful_functions as U

import logging

from new_dat_util import check_exists

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
U.set_default_logging()


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
    logger.debug('start')
    print(f'{threading.get_ident()}: starting update_data_options for {dat_paths}')
    options = None
    value = None
    for path in dat_paths:
        dat = get_dat_from_exp_path(path)
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
    logger.debug('finish')
    print(f'{threading.get_ident()}: finishing update_data_options for {dat_paths}')
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
    logger.debug('start')
    print(f'{threading.get_ident()}: starting update_thetas_graph for {data_paths}')
    fig = go.Figure()
    for path in data_paths:
        logger.debug(f'Adding dat {path} to thetas')
        dat = get_dat_from_exp_path(path)
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
            fits = fit_process.outputs['fits']
            if fits and isinstance(fits, list) and len(fits) > 0:
                fit = fits[0]
                if fit.success:
                    temp = dat.Logs.temperatures['mc']*1000
                    fig.add_trace(go.Scatter(x=[temp], y=[fit.best_values.get('theta', np.nan)], name=f'Dat{dat.datnum}'))
    logger.debug(f'finish')
    print(f'{threading.get_ident()}: finishing update_thetas_graph for {data_paths}')
    return fig


dat_graphs = html.Div(id='ft-div-dat-graphs')
per_dat_graphs = html.Div([
    per_dat_collapse := c.CollapseAIO(content=dat_graphs, button_text='Per Dat Graphs', start_open=True)
])


def get_averaging(dat: DatHDF, data_key: str):
    save_name = f'centering_{data_key.replace("/", ".")}'

    averaging = None
    with dat.hdf_read as f:  # Read only first
        if check_exists(dat, f'dash/fridge_temp/{save_name}'):
            averaging = CenteredAveragingProcess.from_hdf(f['dash']['fridge_temp'], name=save_name)

    if not averaging:
        with dat.hdf_write as f:  # Now with write mode
            if check_exists(dat, f'dash/fridge_temp/{save_name}'):  # In case this call was queued behind another call
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
                averaging.process()
                ft_group = f.require_group('dash/fridge_temp')
                averaging.save_to_hdf(ft_group, name=save_name)
        dat.mode = 'r'
    return averaging


@callback(
    Output(dat_graphs, 'children'),
    Input(dat_selector.store_id, 'data'),
    Input(dd_opts, 'value'),
    State(per_dat_collapse.collapse, 'is_open'),
)
def update_per_dat_graphs(data_paths, data_key, open):
    logger.debug(f'start')
    print(f'{threading.get_ident()}: starting update_per_dat_graphs for {data_paths}')

    def data_with_centers_fig(x, y, data_2d, centers):
        fig = go.Figure()
        data_2d, x, y = U.resample_data(data_2d, x=x, y=y, max_num_pnts=500)
        fig.add_trace(go.Heatmap(x=x, y=y, z=data_2d))
        fig.add_trace(go.Scatter(x=centers, y=y, mode='markers', marker=dict(color='white')))
        return fig

    def data_with_stdev_fig(x, data, stdev):
        fig = go.Figure()
        data, x = U.resample_data(data, x=x, max_num_pnts=500)
        stdev = U.resample_data(stdev, max_num_pnts=500, resample_method='downsample')
        fig.add_trace(go.Scatter(x=x, y=data, error_y=dict(type='data', array=stdev, visible=True)))
        return fig

    if not open:
        return []
    all_entries = []
    for path in data_paths:
        dat = get_dat_from_exp_path(path)
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
    logger.debug(f'finish')
    print(f'{threading.get_ident()}: finishing update_per_dat_graphs for {data_paths}')
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
