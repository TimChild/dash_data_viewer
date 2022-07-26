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
    dcc.Dropdown(id='dd-data-names', value='', options=[]), 'Data')


@callback(
    Output('dd-data-names', 'options'),
    Output('dd-data-names', 'value'),
    # Input('store-data-path', 'data'),
    Input(dat_selector.store_id, 'data'),
    State('dd-data-names', 'value'),
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


g1 = c.GraphAIO(aio_id='graph-1', figure=None)
graphs = html.Div([
    g1,
])


@callback(
    Output(g1.update_figure_store_id, 'data'),
    # Input('store-data-path', 'data'),
    Input(dat_selector.store_id, 'data'),
    Input('dd-data-names', 'value'),
)
def update_graph(data_paths, data_key) -> go.Figure():
    # fig = go.Figure()
    # fig.add_annotation(text=', '.join(data_paths), xref='paper', yref='paper', x=0.5, y=0.5, showarrow=False)
    # return fig

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

            fit = TransitionFitProcess()
            fit.set_inputs(
                x=avg_x,
                transition_data=avg_data,
            )
            fit.process()

            temp = dat.Logs.temperatures['mc']*1000

            fig.add_trace(go.Scatter(x=[temp], y=[fit.outputs['fits'][0].best_values.get('theta', np.nan)], name=f'Dat{dat.datnum}'))
    return fig

    # dat = get_dat(data_path)
    # if dat:
    #     fig = go.Figure()
    #     data = dat.Data.get_data(data_key)
    #     x = dat.Data.x
    #     y = dat.Data.y
    #
    #     fig.update_layout(
    #         title=f'Dat{dat.datnum}: {data_key}',
    #     )
    #     fig.update_xaxes(title_text=dat.Logs.x_label)
    #
    #     if data is not None and data.ndim > 0:
    #         x = x if x is not None else np.linspace(0, data.shape[-1], data.shape[-1])
    #
    #         if data.ndim == 1:
    #             data, x = U.resample_data(data, x=x, max_num_pnts=500, resample_method='bin')
    #             fig.add_trace(go.Scatter(x=x, y=data))
    #         elif data.ndim == 2:
    #             y = y if y is not None else np.linspace(0, data.shape[-2], data.shape[-2])
    #             data, x, y= U.resample_data(data, x=x, y=y, max_num_pnts=500, resample_method='bin')
    #             fig.update_yaxes(title_text=dat.Logs.y_label)
    #             fig.add_trace(go.Heatmap(x=x, y=y, z=data))
    #         else:
    #             pass
    #
    #     return fig
    # else:
    #     return c.blank_figure()


sidebar = dbc.Container([
    dat_selector,
    data_options,
],
)

main = dbc.Container([
    graphs,
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
