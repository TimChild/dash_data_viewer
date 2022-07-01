from __future__ import annotations
import dash
from dash import MATCH, ALL, ALLSMALLER
import json
import os
import uuid
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
from typing import TYPE_CHECKING, List, Union, Optional, Any, Tuple

import numpy as np
import plotly.graph_objects as go

import dash_data_viewer.components as c
from dash_data_viewer.new_dat_util import DatSelector
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


class Test(html.Div):
    """
    """

    # Functions to create pattern-matching callbacks of the subcomponents
    class ids:
        @staticmethod
        def file_dropdown(aio_id, level: int):
            return {
                'component': 'DatSelector',
                'subcomponent': 'FileDropdown',
                'level': level,
                'aio_id': aio_id,
            }

        @staticmethod
        def test_out(aio_id, level):
            return {
                'component': 'DatSelector',
                'subcomponent': 'DivOut',
                'level': level,
                'aio_id': aio_id,
            }

    # Make the ids class a public class
    ids = ids

    def __init__(self, aio_id=None):
        if aio_id is None:
            aio_id = str(uuid.uuid4())

        layout = html.Div([
            dcc.Dropdown(id=self.ids.file_dropdown(aio_id, 1), options=['a', 'b']),
            dcc.Dropdown(id=self.ids.file_dropdown(aio_id, 2), options=['c', 'd']),
            dcc.Dropdown(id=self.ids.file_dropdown(aio_id, 3), options=['c', 'd']),
            html.Div(id=self.ids.test_out(aio_id, 2)),
        ])

        super().__init__(children=[layout])  # html.Div contains layout

    @staticmethod
    @callback(
        Output(ids.test_out(MATCH, MATCH), 'children'),
        Input(ids.file_dropdown(MATCH, ALLSMALLER), 'value'),
    )
    def update_options(values):
        print(values)
        return values



selector = DatSelector()

@callback(
    Output('div-path', 'children'),
    Input(selector.store_id, 'data')
)
def update_path(path):
    return path

layout = dbc.Container([selector,
                        html.Div(id='div-path')])
# layout = html.Div(Test())

if __name__ == '__main__':
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    # app.layout = layout()
    app.layout = layout
    app.run_server(debug=True, port=8049, dev_tools_hot_reload=False, use_reloader=False)
else:
    dash.register_page(__name__)
    pass
