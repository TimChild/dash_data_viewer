from __future__ import annotations

import uuid
from dash import html, dcc, MATCH, ALL, ALLSMALLER, ctx
from .layout_util import label_component
import dash
import json
import os
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
from typing import TYPE_CHECKING, List, Union, Optional, Any, Tuple

import numpy as np
import plotly.graph_objects as go

import dash_data_viewer.components as c
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

ddir = config['loading']['path_to_measurement_data']


class DatSelector(html.Div):
    """
    Select a dat from measurement directory with dropdown menus for each level of folder
    heirarchy

    # Requires
    No required components
    Does require dat_analysis local_config to be set up with a measurement directory and save directory

    # Provides
    Full path to selected Dat

    """

    # Functions to create pattern-matching callbacks of the subcomponents
    class ids:
        @staticmethod
        def generic(aio_id, key: str):
            return {
                'component': 'DatSelector',
                'subcomponent': f'generic',
                'key': key,
                'aio_id': aio_id,
            }

        @staticmethod
        def file_dropdown(aio_id, level: int):
            return {
                'component': 'DatSelector',
                'subcomponent': 'FileDropdown',
                'level': level,
                'aio_id': aio_id,
            }

        @staticmethod
        def store(aio_id):
            return {
                'component': 'DatSelector',
                'subcomponent': 'Store',
                'aio_id': aio_id,
            }

    # Make the ids class a public class
    ids = ids

    def __init__(self, aio_id=None):
        if aio_id is None:
            aio_id = str(uuid.uuid4())
        self.store_id = self.ids.store(aio_id)

        layout = self.layout(aio_id)
        super().__init__(children=[layout])  # html.Div contains layout

    def layout(self, aio_id):
        host_options = [k for k in os.listdir(ddir) if os.path.isdir(os.path.join(ddir, k))]
        layout = html.Div([
            dcc.Store(id=self.ids.generic(aio_id, 'aio_id'), data=aio_id),
            dcc.Store(id=self.ids.generic(aio_id, 'selections'), storage_type='session'),
            html.H5('Folder/File Path:'),
            label_component(dcc.Dropdown(id=self.ids.generic(aio_id, 'host'), options=host_options, persistence=True,
                                         persistence_type='session'), 'Host Name'),
            label_component(dcc.Dropdown(id=self.ids.generic(aio_id, 'user'), persistence=True,
                                         persistence_type='session'), 'User Name'),
            label_component(html.Div(id=self.ids.generic(aio_id, 'div-experiment-selections')), 'File Path'),
            dcc.Store(self.ids.store(aio_id), storage_type='session'),
        ])
        return layout

    @staticmethod
    @callback(
        Output(ids.generic(MATCH, 'user'), 'options'),
        Input(ids.generic(MATCH, 'host'), 'value'),
    )
    def create_dropdowns_for_user(host):
        opts = []
        if host:
            opts = os.listdir(os.path.join(ddir, host))
        return opts

    @staticmethod
    @callback(
        Output(ids.generic(MATCH, 'div-experiment-selections'), 'children'),
        Input(ids.file_dropdown(MATCH, ALL), 'value'),
        Input(ids.generic(MATCH, 'host'), 'value'),
        Input(ids.generic(MATCH, 'user'), 'value'),
        State(ids.generic(MATCH, 'aio_id'), 'data'),
        State(ids.generic(MATCH, 'div-experiment-selections'), 'children'),
        State(ids.generic(MATCH, 'selections'), 'data'),
    )
    def add_dropdown_for_experiment(values, host, user, aio_id, existing, stored_values):
        """Always aim to have an empty dropdown available (i.e. for next depth of folders)"""
        values = [v if v else '' for v in values]

        # If host or user is trigger, or no experiment dropdowns already
        if any([v == ctx.triggered_id.get('key', None) if ctx.triggered_id else False for v in ['host', 'user']]) or not existing:
            if host and user:
                opts = os.listdir(os.path.join(ddir, host, user))
                if not existing and stored_values:  # Page reload
                    stored_values = [s if s else '' for s in stored_values]
                    if os.path.exists(os.path.join(ddir, host, user, *stored_values)):
                        dds = []
                        p = os.path.join(ddir, host, user)
                        for v in stored_values:
                            opts = os.listdir(p)
                            dds.append(dcc.Dropdown(id=DatSelector.ids.file_dropdown(aio_id, 0), options=opts, value=v))
                            p = os.path.join(p, v)
                        return dds
                    else:
                        return [dcc.Dropdown(id=DatSelector.ids.file_dropdown(aio_id, 0), options=opts)]
                elif not existing or not os.path.exists(os.path.join(host, user, *values)):  # Make first set of options
                    return [dcc.Dropdown(id=DatSelector.ids.file_dropdown(aio_id, 0), options=opts)]
            else:  # Don't know what options to show yet
                return []
        # If there are existing dropdowns, decide if some need to be removed or added
        elif values:
            for i, v in enumerate(values):
                if not v and i < len(values) - 1:
                    new = existing[:i + 1]  # Remove dropdowns after first empty
                    return new

            last_val = values[-1]
            if last_val:  # If last dropdown is filled, add another (unless it isn't a directory)
                depth = len(existing)
                if os.path.isdir(os.path.join(ddir, host, user, *values)):
                    opts = os.listdir(os.path.join(ddir, host, user, *values))
                    existing.append(dcc.Dropdown(id=DatSelector.ids.file_dropdown(aio_id, depth), options=opts))
                    return existing
        return dash.no_update

    @staticmethod
    @callback(
        Output(ids.store(MATCH), 'data'),
        Input(ids.generic(MATCH, 'host'), 'value'),
        Input(ids.generic(MATCH, 'user'), 'value'),
        Input(ids.file_dropdown(MATCH, ALL), 'value'),
    )
    def update_full_path(host, user, exp_path):
        host = host if host else ''
        user = user if user else ''
        exp_path = [p if p else '' for p in exp_path]
        return os.path.join(ddir, host, user, *exp_path)

    @staticmethod
    @callback(
        Output(ids.generic(MATCH, 'selections'), 'data'),
        Input(ids.file_dropdown(MATCH, ALL), 'value'),
    )
    def persistent_selections(values):
        return values

