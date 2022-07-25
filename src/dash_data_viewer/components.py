import dash
from dash import Input, Output, State, dcc, html, callback, MATCH, ALL, ALLSMALLER, dash, ctx
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from dacite import from_dict
from dash_extensions.snippets import get_triggered
from dash_extensions import Download
from typing import Optional, List, Union, Any
import uuid
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import logging
import tempfile
import filelock
import os
import time
from deprecation import deprecated

from dat_analysis import useful_functions as u, get_local_config
from layout_util import label_component, vertical_label


tempdir = os.path.join(tempfile.gettempdir(), 'dash_viewer/')
os.makedirs(tempdir, exist_ok=True)
global_lock = filelock.FileLock(os.path.join(tempdir, 'components_lock.lock'))

config = get_local_config()
ddir = config['loading']['path_to_measurement_data']


def blank_figure() -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text='Blank Figure', xref='paper', yref='paper', x=0.5, y=0.5, showarrow=False)
    return fig


def Input_(id=None, type='text', value=None, autocomplete=False, inputmode='text', max=None, maxlength=None, min=None,
           minlength=None, step=None, size='md', valid=False, required=False, placeholder='', name='', debounce=True,
           persistence=False, persistence_type='session', **kwargs) -> dbc.Input:
    """
    Wrapper around dbc.Input with some different default values

    Args:
        id (): ID in page
        type (): The type of control to render. (a value equal to: "text", 'number', 'password', 'email', 'range', 'search', 'tel', 'url', 'hidden'; optional):
        value (): initial value of input
        autocomplete (): Enable browser autocomplete
        inputmode (): (a value equal to: "verbatim", "latin", "latin-name", "latin-prose", "full-width-latin", "kana", "katakana", "numeric", "tel", "email", "url"; optional): Provides a hint to the browser as to the type of data that might be entered by the user while editing the element or its contents.
        max (): Max value (or date) allowed
        maxlength (): Max string length allowed
        min (): Min value (or date) allowed
        minlength (): Min string length allowed
        step (): Step size for value type
        size (): 'sm', 'md', or 'lg'. Defaults to 'md'
        valid (): Apply valid style (adds a tick)
        required (): Entry required for form submission
        placeholder (): Placeholder text (not an actual value)
        name (): Name of component submitted with form data
        debounce (): Update only on focus loss or enter key
        persistence (): Whether the value should persist
        persistence_type (): (a value equal to: 'local', 'session', 'memory'; default 'local'). memory: only kept in memory, reset on page refresh. local: window.localStorage, data is kept after the browser quit (Note: Shared across multiple browser tabs/windows). session: window.sessionStorage, data is cleared once the browser quit.
        **kwargs (): Any other kwargs accepted by dbc.Input

    Returns:

    """
    return dbc.Input(id=id, type=type, value=value, autocomplete=autocomplete, inputmode=inputmode, max=max,
                     maxlength=maxlength, min=min, minlength=minlength, step=step, size=size, valid=valid,
                     required=required, placeholder=placeholder, name=name, debounce=debounce, persistence=persistence,
                     persistence_type=persistence_type, **kwargs)


class TemplateAIO(html.Div):
    """
    DESCRIPTION

    # Requires
    What components require outside callbacks in order to work (if any)

    # Provides
    What component outputs are intended to be used by other callbacks (if any)

    """

    # Functions to create pattern-matching callbacks of the subcomponents
    class ids:
        @staticmethod
        def generic(aio_id, key: str):
            return {
                'component': 'TemplateAIO',
                'subcomponent': f'generic',
                'key': key,
                'aio_id': aio_id,
            }

    # Make the ids class a public class
    ids = ids

    def __init__(self, aio_id=None):
        if aio_id is None:
            aio_id = str(uuid.uuid4())

        super().__init__(children=[])  # html.Div contains layout

    # UNCOMMENT -- This callback is run when module is imported regardless of use
    # @staticmethod
    # @callback(
    # )
    # def function():
    #     pass


class CollapseAIO(html.Div):
    """
    A collapsable div that handles the Callbacks for collapsing/expanding

    # Requires
    None - self contained (just put content in)

    # Provides
    None - self contained.

    """

    # Functions to create pattern-matching callbacks of the subcomponents
    class ids:
        @staticmethod
        def generic(aio_id, key: str):
            return {
                'component': 'CollapseAIO',
                'subcomponent': f'generic',
                'key': key,
                'aio_id': aio_id,
            }

    # Make the ids class a public class
    ids = ids

    def __init__(self, aio_id=None, content: html.Div = None, button_text: str = 'Expand', start_open=False):
        if aio_id is None:
            aio_id = str(uuid.uuid4())

        layout = [
            dbc.Button(button_text, id=self.ids.generic(aio_id, 'expand-button')),
            dbc.Collapse(children=content, id=self.ids.generic(aio_id, 'collapse'), is_open=start_open),
        ]
        super().__init__(children=layout)  # html.Div contains layout

    @staticmethod
    @callback(
        Output(ids.generic(MATCH, 'collapse'), 'is_open'),
        Input(ids.generic(MATCH, 'expand-button'), 'n_clicks'),
        State(ids.generic(MATCH, 'collapse'), 'is_open'),
    )
    def toggle_collapse(clicks, is_open: bool):
        if clicks:
            return not is_open
        return is_open


class ExperimentFileSelectorAIO(html.Div):
    """
    Select a dat from measurement directory with dropdown menus for each level of folder
    heirarchy

    # Requires
    No required components
    Does require dat_analysis local_config to be set up with a measurement directory and save directory

    # Provides
    Full path to selected Dat or other experiment file or folder

    """

    # Functions to create pattern-matching callbacks of the subcomponents
    class ids:
        @staticmethod
        def generic(aio_id, key: str):
            return {
                'component': 'FileSelector',
                'subcomponent': f'generic',
                'key': key,
                'aio_id': aio_id,
            }

        @staticmethod
        def file_dropdown(aio_id, level: int):
            return {
                'component': 'FileSelector',
                'subcomponent': 'FileDropdown',
                'level': level,
                'aio_id': aio_id,
            }

        @staticmethod
        def store(aio_id):
            return {
                'component': 'FileSelector',
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
                opts = sorted(os.listdir(os.path.join(ddir, host, user)))
                if not existing and stored_values:  # Page reload
                    stored_values = [s if s else '' for s in stored_values]
                    if os.path.exists(os.path.join(ddir, host, user, *stored_values)):
                        dds = []
                        p = os.path.join(ddir, host, user)
                        for v in stored_values:
                            opts = sorted(os.listdir(p))
                            dds.append(dcc.Dropdown(id=ExperimentFileSelectorAIO.ids.file_dropdown(aio_id, 0), options=opts, value=v))
                            p = os.path.join(p, v)
                        return dds
                    else:
                        return [dcc.Dropdown(id=ExperimentFileSelectorAIO.ids.file_dropdown(aio_id, 0), options=opts)]
                elif not existing or not os.path.exists(os.path.join(host, user, *values)):  # Make first set of options
                    return [dcc.Dropdown(id=ExperimentFileSelectorAIO.ids.file_dropdown(aio_id, 0), options=opts)]
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
                    opts = sorted(os.listdir(os.path.join(ddir, host, user, *values)))
                    existing.append(dcc.Dropdown(id=ExperimentFileSelectorAIO.ids.file_dropdown(aio_id, depth), options=opts))
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


class DatSelectorAIO(html.Div):

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
        def store(aio_id):
            return {
                'component': 'DatSelector',
                'subcomponent': 'Store',
                'aio_id': aio_id,
            }

    def __init__(self, aio_id=None):
        if aio_id is None:
            aio_id = str(uuid.uuid4())
        self.store_id = self.ids.store(aio_id)
        store = dcc.Store(id=self.store_id)

        datnum_input = Input_(id=self.ids.generic(aio_id, key='inp-datnum'), type='number', inputmode='numeric', min=0,
                              step=1, persistence=True, persistence_type='session')

        raw_tog = dbc.RadioButton(id=self.ids.generic(aio_id, key='tog-raw'), persistence=True, persistence_type='session')

        layout = html.Div([
            store,
            ExperimentFileSelectorAIO(aio_id=aio_id),
            label_component(datnum_input, 'Datnum:'),
            label_component(raw_tog, 'RAW File:'),
        ])
        super().__init__(children=[layout])  # html.Div contains layout

    @staticmethod
    @callback(
        Output(ids.store(MATCH), 'data'),
        Input(ExperimentFileSelectorAIO.ids.store(MATCH), 'data'),
        Input(ids.generic(MATCH, key='inp-datnum'), 'value'),
        Input(ids.generic(MATCH, key='tog-raw'), 'value'),
        State(ids.store(MATCH), 'data'),
    )
    def update_selection(filepath, datnum, raw, current_path):
        datnum = datnum if datnum else 0
        data_path = None
        if filepath and os.path.exists(filepath):
            if os.path.isdir(filepath):
                datfile = f'dat{datnum}_RAW.h5' if raw else f'dat{datnum}.h5'
                data_path = os.path.join(filepath, datfile)
            else:
                data_path = filepath
        if data_path == current_path:
            return dash.no_update
        return data_path


@deprecated(details='2022-07-05 -- Use improved DatSelectorAIO which works with measurement-data layout instead')
class DatnumPickerAIO(html.Div):
    """
    A group of buttons with custom text where one or multiple dats can be selected at a time.
    Selected dats are accessed through (picker.dd_id, 'value')

    Examples:
        picker = DatnumPickerAIO(aio_id='testpage-datpicker')  # Fixed ID required for persistence to work

        @callback(Input(picker.dd_id, 'value'))  # Selected Datnums are accessible from the dd.value
        def foo(datnums: list):
            return datnums

    """

    # Functions to create pattern-matching callbacks of the subcomponents
    class ids:
        @staticmethod
        def input(aio_id, key):
            return {
                'component': 'DatnumPickerAIO',
                'subcomponent': f'input',
                'key': key,
                'aio_id': aio_id,
            }

        @staticmethod
        def button(aio_id, type):
            return {
                'component': 'DatnumPickerAIO',
                'subcomponent': f'button',
                'key': type,
                'aio_id': aio_id,
            }

        @staticmethod
        def dropdown(aio_id):
            return {
                'component': 'DatnumPickerAIO',
                'subcomponent': f'dropdown',
                'aio_id': aio_id,
            }

        @staticmethod
        def options_store(aio_id):
            return {
                'component': 'DatnumPickerAIO',
                'subcomponent': f'opts',
                'aio_id': aio_id,
            }

    # Make the ids class a public class
    ids = ids

    def __init__(self, aio_id, allow_multiple=True):
        self.dd_id = self.ids.dropdown(aio_id)  # For easy access to this component ('value' contains selected dats)

        opts_store = dcc.Store(id=self.ids.options_store(aio_id),
                               data=dict(multi=allow_multiple)
                               )

        input_infos = {
            'start': dict(component=None, label='Start', placeholder='Start'),
            'stop': dict(component=None, label='Stop', placeholder='Stop'),
            'step': dict(component=None, label='Step', placeholder='Step'),
        }

        button_infos = {
            'add': dict(component=None, text='Add'),
            'remove': dict(component=None, text='Remove'),
        }

        for key, info in input_infos.items():
            info['component'] = dbc.Input(id=self.ids.input(aio_id, key),
                                          type='number',
                                          placeholder=info['placeholder'],
                                          debounce=True,
                                          persistence=True,
                                          persistence_type='local')

        for key, info in button_infos.items():
            info['component'] = dbc.Button(id=self.ids.button(aio_id, key), children=info['text'])

        dd_datnums = dcc.Dropdown(id=self.ids.dropdown(aio_id), multi=allow_multiple,
                                  clearable=True,
                                  placeholder='Add options first',
                                  searchable=True,
                                  persistence=True,
                                  persistence_type='local',
                                  # style=,
                                  )

        layout = dbc.Card(
            [
                dbc.CardHeader(dbc.Col(html.H3('Dat Selector'))),
                dbc.CardBody([
                    dbc.Row([
                        *[vertical_label(info['label'], info['component']) for info in input_infos.values()],
                        dbc.Col([info['component'] for info in button_infos.values()])
                    ]),
                    dbc.Row([
                        dbc.Col(dd_datnums)
                    ])
                ])
            ])

        super().__init__(children=[opts_store, layout])  # html.Div contains layout

    @staticmethod
    @callback(
        Output(ids.dropdown(MATCH), 'options'),
        Output(ids.dropdown(MATCH), 'value'),
        Input(ids.button(MATCH, 'add'), 'n_clicks'),
        Input(ids.button(MATCH, 'remove'), 'n_clicks'),
        State(ids.input(MATCH, 'start'), 'value'),
        State(ids.input(MATCH, 'stop'), 'value'),
        State(ids.input(MATCH, 'step'), 'value'),
        State(ids.dropdown(MATCH), 'options'),
        State(ids.dropdown(MATCH), 'value'),
        State(ids.options_store(MATCH), 'data'),
    )
    def update_datnums(add_clicks: Optional[int], remove_clicks: Optional[int],
                       start: Optional[int], stop: Optional[int], step: Optional[int],
                       prev_options: Optional[List[dict]],
                       current_datnums: Optional[List[int]],
                       options: dict):
        """
        Update the list of datnums in the selectable dropdown thing and what is currently selected
        Args:
            add_clicks ():
            remove_clicks ():
            start ():
            stop ():
            step ():
            prev_options ():
            current_datnums ():
            options (): Additional options that are stored on creation of AIO

        Returns:

        """
        logging.info(f'Current datnums: {current_datnums}')
        prev_options = prev_options if prev_options else []
        if options['multi'] is True:
            current_datnums = current_datnums if current_datnums else []

        triggered = get_triggered()
        if add_clicks and start:
            step = step if step else 1
            stop = stop if stop and stop > start else start
            vals = range(start, stop + 1, step)
            prev_opts_keys = [opt['value'] for opt in prev_options]
            if triggered.id['key'] == 'add':
                for v in vals:
                    if v not in prev_opts_keys:
                        prev_options.append({'label': v, 'value': v})
                    if options['multi'] is True:
                        if v not in current_datnums:
                            current_datnums.append(v)
            elif triggered.id['key'] == 'remove':
                prev_options = [p for p in prev_options if p['value'] not in vals]
                if options['multi'] is True:
                    current_datnums = [d for d in current_datnums if d not in vals]
                else:
                    if current_datnums in vals:
                        current_datnums = None
            else:
                logging.warning(f'Unexpected trigger. trig.id = {triggered.id}')

            prev_options = [opts for opts in sorted(prev_options, key=lambda item: item['value'])]
            if options['multi'] is True:
                current_datnums = list(sorted(current_datnums))
            else:
                if current_datnums is None and prev_options:
                    current_datnums = prev_options[0]['value']
        return prev_options, current_datnums  # Return each in [] because of use of ALL


class MultiButtonAIO(html.Div):
    """
    A group of buttons with custom text where one(multiple) can be selected at a time and returns either the text or a specific
    value

    # Requires
    None

    # Provides
    dcc.Store(id=self.store_id) -- Filled Info class

    Examples:
    # Example init
    ex = TemplateAIO()

    # Example input required (if relevant)
    # Example output provided (if relevant)

    """

    @dataclass
    class Info:
        selected_values: Union[List[Any], Any]
        selected_numbers: List[int]  # Button numbers (so I know which to color)

    @classmethod
    def info_from_dict(cls, d: dict) -> Info:
        if not d:
            return cls.Info([], [])
        else:
            return from_dict(cls.Info, d)

    @dataclass
    class Config:
        allow_multiple: bool

    # Functions to create pattern-matching callbacks of the subcomponents
    class ids:
        @staticmethod
        def button(aio_id, num, value):
            return {
                'component': 'MultiButtonAIO',
                'subcomponent': f'button',
                'number': num,
                'aio_id': aio_id,
                'value': value,
            }

        @staticmethod
        def store(aio_id):
            return {
                'component': 'MultiButtonAIO',
                'subcomponent': f'store',
                'aio_id': aio_id,
            }

        @staticmethod
        def config(aio_id):
            return {
                'component': 'MultiButtonAIO',
                'subcomponent': f'config_store',
                'aio_id': aio_id,
            }

    # Make the ids class a public class
    ids = ids

    def __init__(self, button_texts: list[str], button_values: list | None = None, button_props: dict = None,
                 aio_id=None, allow_multiple=False, storage_type='memory'):
        if button_props is None:
            button_props = {}
        if aio_id is None:
            aio_id = str(uuid.uuid4())

        config = self.Config(allow_multiple=allow_multiple)

        # # Store other optional params here so that the callback can see them
        config_store = dcc.Store(id=self.ids.config(aio_id), storage_type='session',
                                 data=asdict(config))

        if button_values is None:
            button_values = button_texts

        self.store_id = self.ids.store(aio_id)

        buttons = [dbc.Button(
            id=self.ids.button(aio_id, num, value), children=text, color='danger', **button_props
        ) for num, (text, value) in
            enumerate(zip(button_texts, button_values))
        ]
        if not allow_multiple:
            buttons[0].color = 'success'

        button_layout = dbc.ButtonGroup(buttons)

        super().__init__(children=[
            button_layout,
            dcc.Store(id=self.ids.store(aio_id), storage_type=storage_type),
            config_store,
        ])

    @staticmethod
    @callback(
        Output(ids.store(MATCH), 'data'),
        Output(ids.button(MATCH, ALL, ALL), 'color'),
        Input(ids.button(MATCH, ALL, ALL), 'n_clicks'),
        State(ids.button(MATCH, ALL, ALL), 'color'),
        State(ids.store(MATCH), 'data'),
        State(ids.config(MATCH), 'data'),
    )
    def update_button_output(all_clicks, button_colors, current_data, config):
        config = from_dict(MultiButtonAIO.Config, config)
        if not current_data or not isinstance(current_data, dict):
            current_data = MultiButtonAIO.Info(selected_values=[], selected_numbers=[])
        else:
            current_data = from_dict(MultiButtonAIO.Info, current_data)

        triggered = get_triggered()
        if triggered.id:
            selected = triggered.id.get('value')
            num = triggered.id.get('number')
            if config.allow_multiple is False:
                current_data.selected_values = selected
                current_data.selected_numbers = [num]
            else:  # Allowing multiple selection
                if selected in current_data.selected_values:  # Remove if already selected
                    current_data.selected_values.remove(selected)
                    current_data.selected_numbers.remove(num)
                else:  # Add otherwise
                    current_data.selected_values.append(selected)
                    current_data.selected_numbers.append(num)
        colors = ['danger' if i not in current_data.selected_numbers else 'success' for i, _ in
                  enumerate(button_colors)]
        return asdict(current_data), colors


class GraphAIO(html.Div):
    """
    Displays a graph an adds some options for download of the data or figure

    # Requires
    self.graph_id, 'figure' -- should be updated to update the figure in the graph

    # Provides
    None
    """

    # Functions to create pattern-matching callbacks of the subcomponents
    class ids:
        @staticmethod
        def generic(aio_id, key: str):
            return {
                'component': 'GraphAIO',
                'subcomponent': f'generic',
                'key': key,
                'aio_id': aio_id,
            }

        @staticmethod
        def input(aio_id, key: str):
            return {
                'component': 'GraphAIO',
                'subcomponent': f'input',
                'key': key,
                'aio_id': aio_id,
            }

        @staticmethod
        def button(aio_id, key: str):
            return {
                'component': 'GraphAIO',
                'subcomponent': f'button',
                'key': key,
                'aio_id': aio_id,
            }

        @staticmethod
        def graph(aio_id):
            return {
                'component': 'GraphAIO',
                'subcomponent': f'figure',
                'aio_id': aio_id,
            }

        @staticmethod
        def update_figure_store(aio_id):
            """Updating this store with a fig.to_dict() will be passed on to the graph figure without requiring a
            second callback
            I.e. The callback to update the actual figure is defined in the AIO, so can't be duplicated elsewhere
            """
            return {
                'component': 'GraphAIO',
                'subcomponent': f'update_figure',
                'aio_id': aio_id,
            }

    # Make the ids class a public class
    ids = ids

    def __init__(self, aio_id=None, figure=None, **graph_kwargs):
        figure = figure if figure else go.Figure()

        if aio_id is None:
            aio_id = str(uuid.uuid4())

        self.update_figure_store_id = self.ids.update_figure_store(aio_id)
        update_fig_store = dcc.Store(id=self.update_figure_store_id, data=figure)
        self.graph_id = self.ids.graph(aio_id)
        fig = dcc.Graph(id=self.graph_id, figure=figure, **graph_kwargs)

        download_buttons = dbc.ButtonGroup([
            dbc.Button(children=name, id=self.ids.button(aio_id, name)) for name in ['HTML', 'Jpeg', 'SVG', 'Data']
        ])

        options_layout = dbc.Form([
            dbc.Label(children='Download Name', html_for=self.ids.input(aio_id, 'downloadName')),
            Input_(id=self.ids.input(aio_id, 'downloadName')),
            dbc.Label(children='Download'),
            download_buttons,
            html.Hr(),
            vertical_label('Waterfall', dbc.RadioButton(id=self.ids.generic(aio_id, 'tog-waterfall'))),
        ])

        options_button = dbc.Button(id=self.ids.button(aio_id, 'optsPopover'), children='Options', size='sm',
                                    color='light')
        options_popover = dbc.Popover(children=dbc.PopoverBody(options_layout), target=options_button.id,
                                      trigger='click')

        full_layout = html.Div([
            update_fig_store,
            dcc.Download(id=self.ids.generic(aio_id, 'download')),
            options_popover,
            fig,
            html.Div(children=options_button, style={'position': 'absolute', 'top': 0, 'left': 0}),
        ], style={'position': 'relative'})

        super().__init__(children=full_layout)  # html.Div contains layout

    @staticmethod
    @callback(
        Output(ids.graph(MATCH), 'figure'),
        Input(ids.update_figure_store(MATCH), 'data'),
        Input(ids.generic(MATCH, 'tog-waterfall'), 'value'),
        State(ids.graph(MATCH), 'figure'),
    )
    def update_figure(update_fig, waterfall, existing_fig):
        fig = go.Figure(existing_fig)
        if ctx.triggered:
            if ctx.triggered_id.get('subcomponent', None) == 'update_figure':
                fig = update_fig

        fig = fig_waterfall(fig, waterfall)

        if fig:
            return fig
        else:
            return blank_figure()

    @staticmethod
    @callback(
        Output(ids.generic(MATCH, 'download'), 'data'),
        Input(ids.button(MATCH, ALL), 'n_clicks'),
        State(ids.input(MATCH, 'downloadName'), 'value'),
        State(ids.graph(MATCH), 'figure'),
        prevent_initial_callback=True
    )
    def download(selected, download_name, figure):
        triggered = get_triggered()
        if triggered.id and triggered.id['subcomponent'] == 'button' and figure:
            selected = triggered.id['key'].lower()
            fig = go.Figure(figure)
            if not download_name:
                download_name = fig.layout.title.text if fig.layout.title.text else 'fig'
            download_name = download_name.split('.')[0]  # To remove any extensions in name
            if selected == 'html':
                return dict(content=fig.to_html(), filename=f'{download_name}.html', type='text/html')
            elif selected == 'jpeg':
                filepath = os.path.join(tempdir, 'jpgdownload.jpg')
                with global_lock:  # TODO: Send a file object directly rather than actually writing to disk first
                    time.sleep(0.1)  # Here so that any previous one has time to be sent before being overwritten
                    fig.write_image(filepath, format='jpg')
                    return dcc.send_file(filepath, f'{download_name}.jpg', type='image/jpg')
            elif selected == 'svg':
                filepath = os.path.join(tempdir, 'svgdownload.svg')
                with global_lock:  # TODO: Send a file object directly rather than actually writing to disk first
                    time.sleep(0.1)  # Here so that any previous one has time to be sent before being overwritten
                    fig.write_image(filepath, format='svg')
                    return dcc.send_file(filepath, f'{download_name}.svg', type='image/svg+xml')
            elif selected == 'data':
                filepath = os.path.join(tempdir, 'datadownload.json')
                with global_lock:  # TODO: Send a file object directly rather than actually writing to disk first
                    time.sleep(0.1)  # Here so that any previous one has time to be sent before being overwritten
                    u.fig_to_data_json(fig, filepath)
                    return dcc.send_file(filepath, f'{download_name}.json', type='application/json')
        return dash.no_update


def fig_waterfall(fig: go.Figure, waterfall_state: bool):
    if fig:
        fig = go.Figure(fig)
    if fig and fig.data:
        if len(fig.data) == 1 and isinstance(fig.data[0], go.Heatmap) and waterfall_state:  # Convert from heatmap to waterfall
            hm = fig.data[0]
            fig.data = ()
            x = hm.x
            for r, y in zip(hm.z, hm.y):
                fig.add_trace(go.Scatter(x=x, y=r, name=y))
            fig.update_layout(legend=dict(title=fig.layout.yaxis.title.text), yaxis_title=fig.layout.coloraxis.colorbar.title.text)
        elif len(fig.data) > 1 and all([isinstance(d, go.Scatter) for d in fig.data]) and not waterfall_state:  # Convert from waterfall to heatmap
            rows = fig.data
            fig.data = ()
            x = rows[0].x
            y = []
            for i, r in enumerate(rows):
                try:
                    name = r.name
                    v = float(name)
                except Exception as e:
                    v = i
                y.append(v)
            z = np.array([r.y for r in rows])
            fig.add_trace(go.Heatmap(x=x, y=y, z=z))
            fig.update_layout(yaxis_title=fig.layout.legend.title.text, legend=None, coloraxis=dict(colorbar=dict(title=fig.layout.yaxis.title.text)))
        else:
            pass
    return fig




