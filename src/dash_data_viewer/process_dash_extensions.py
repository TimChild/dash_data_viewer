"""
Provide the dash functionality to dat_analysis Process
"""
from __future__ import annotations
import abc
from typing import TYPE_CHECKING, List, Union, Type, TypeVar, Optional
from dash import html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import dash
import uuid

from dat_analysis.analysis_tools.new_procedures import Process, SeparateSquareProcess
from dash_data_viewer.layout_util import label_component
import dash_data_viewer.components as c

from dash_data_viewer.components import CollapseAIO

from dat_analysis import get_dat
from dat_analysis.hdf_file_handler import HDFFileHandler

if TYPE_CHECKING:
    from dash.development.base_component import Component

NOT_SET = object()  # To be able to check if things still need to be set

T = TypeVar('T', bound=Process)
def load(location_dict: dict, process_class: Type[T]) -> T:
    dat_id = location_dict.get('dat_id', None)
    save_path = location_dict.get('save_path', None)
    if dat_id and save_path:
        dat = get_dat(id=dat_id)
        with HDFFileHandler(dat.hdf.hdf_path, 'r') as f:
            group = f.get(save_path)
            process = process_class.load_progress(group)
        return process
    raise ValueError(f'dat_id or save_path not found in location_dict ({location_dict})')


class ProcessInterface(abc.ABC):
    """
    Things necessary to put a process into a dash page or report with easy human friendly input. I.e. for building a
    new dash page

    human friendly input, and id of file to load from (or data passed in)
    """

    @property
    @abc.abstractmethod
    def id_prefix(self) -> str:
        """
        String to prepend to all IDs generated in this Interface
        Note: Can just set a class attribute to override this
        """
        pass

    def ID(self, key):
        return f'{self.id_prefix}-{key}'

    # Set some IDs which will usually be used
    @property
    def sanitized_store_id(self) -> str:
        return self.ID('store-sanitized')

    @property
    def output_store_id(self) -> str:
        return self.ID('store-output')

    @property
    def sanitized_inputs_id(self) -> str:
        return self.ID('md-sanitized')

    # Common methods that usually won't need to be overridden
    def get_stores(self) -> html.Div:
        """
        Put the stores into a div to be placed somewhere on the page
        """
        return html.Div([
            dcc.Store(id=self.sanitized_store_id),
            dcc.Store(id=self.output_store_id),
        ])

    def get_sanitized_inputs_layout(self) -> html.Div:
        """Make display layout for sanitized inputs

        Note: Can be overridden for more control over layout (override callback_sanitized_inputs as well)
        """
        output = html.Div(
            dcc.Markdown(
                id=self.sanitized_inputs_id,
                style={'white-space': 'pre'}
            ))
        return output

    def _sanitized_info_dicts(self) -> List[dict]:
        info_to_display = self.info_for_sanitized_display()
        full_info = []
        for entry in info_to_display:
            if isinstance(entry, str):
                entry = dict(key=entry)
            if 'key' not in entry:
                raise KeyError(f'{entry} has no "key". Must provide "key" that matches the sanitized input store '
                               f'in self.info_for_sanitized_display')
            entry.setdefault('name', entry['key'])
            entry.setdefault('format', 'g')
            full_info.append(entry)
        return full_info

    def callback_sanitized_inputs(self):
        """Run callback for sanitized inputs display"""
        info_to_display = self._sanitized_info_dicts()

        @callback(
            Output(self.sanitized_inputs_id, 'children'),
            Input(self.sanitized_store_id, 'data')
        )
        def update_sanitized_display(s: dict):
            if s:
                md = 'Using:\n'
                for d in info_to_display:
                    md += f"\t{d['name']} = {s[d['key']]:{d['format']}}\n"
                return md
            return 'Nothing to show yet'

    # Required methods
    @abc.abstractmethod
    def set_other_input_ids(self, *args):
        """
        Get the IDs of any other existing input components that will be used by this process (needed for callbacks)
        Args:
            *args ():

        Returns:

        """
        pass

    @abc.abstractmethod
    def set_data_input_ids(self, *args):
        """
        Get the IDs of any data/process stores that are used in this Process (needed for callbacks)
        Args:
            *args ():

        Returns:

        """
        pass

    @abc.abstractmethod
    def get_user_inputs_layout(self) -> html.Div:
        """
        Make user input components and return layout
        """
        pass

    @abc.abstractmethod
    def callback_sanitized_store(self):
        """
        Callback to update the sanitized input store with information from user inputs and other inputs after setting
        defaults and checking the inputs are OK (i.e. may use info from data_in to check bounds or make defaults etc)

        Note: This is a good place to check all your existing inputs have been set
            (i.e. assert self.existing_id is not NOT_SET)
        Returns:

        """

    @abc.abstractmethod
    def info_for_sanitized_display(self) -> List[Union[str, dict]]:
        """
        Generate list of which sanitized inputs should be displayed.
        Return just keys to use default formatting for numbers only.
        Or return a dict with 'key' and optionally other keys:
            {'key': <key>, 'display_name': <name>, 'format': '.2f'}

        Note: For ultimate control, override get_sanitized_inputs_layout and callback_sanitized_inputs
            (in which case this method is not used)

        Examples:
            return ['a', 'b']  # Use default formatting to display inputs 'a' and 'b'

            return [{'key': 'a', 'name': 'Input A', 'format': '.3g'}, 'b']  # Display input 'a' using 'Input A' as the
                name, with '.3g' formatting for the value. Use default formatting to display input 'b'
        """
        pass

    @abc.abstractmethod
    def callback_output_store(self):
        """
        Callback to update output store using sanitized inputs and data
        Returns:

        """
        pass

    @abc.abstractmethod
    def get_input_display_layout(self):
        """
        Make display layout for inputs of Process (e.g. input data, with initial fit values)
        Returns:

        """
        pass

    @abc.abstractmethod
    def get_output_display_layout(self):
        """
        Make display layout for output of Process (e.g. data with fit)
        Returns:

        """
        pass

    @abc.abstractmethod
    def callback_input_display(self):
        """
        Callback to update the input displays (e.g. input data with initial fit)
        Returns:

        """

    @abc.abstractmethod
    def callback_output_display(self):
        """
        Callback to update the output displays (e.g. input data with final fit)
        Returns:

        """
        pass


class TemplateProcessInterface(ProcessInterface):

    id_prefix = 'TEMPLATE'

    def __init__(self):
        # Set up the non-standard IDs which will be used for various inputs/outputs
        ID = self.ID  # Just to not have to write self.ID every time

        # User Input IDs
        self.input_a_id = ID('input-a')
        self.input_b_id = ID('input-b')

        # Other Required Input IDs
        self.dat_id = NOT_SET
        self.existing_input_id = NOT_SET

        # Data input IDs
        self.data_input_id = NOT_SET  # May just take data directly from some store/saved location
        self.previous_process_id = NOT_SET  # Or may be easier to continue on from the output of another process

        # Display IDs
        self.in_graph_id = ID('graph-in')  # Display input to this Process (plus useful info, e.g. initial fit)
        self.out_graph_id = ID('graph-out')  # Display output of this Process

    def set_other_input_ids(self, dat_id, existing_c_id):
        self.dat_id = dat_id
        self.existing_input_id = existing_c_id

    def set_data_input_ids(self, previous_process_id):
        self.previous_process_id = previous_process_id

    def get_user_inputs_layout(self) -> html.Div:
        in_a = dbc.Input(id=self.input_a_id, type='number')
        in_b = dbc.Input(id=self.input_b_id, type='number')
        return html.Div([
            label_component(in_a, 'Input A'),
            label_component(in_b, 'Input B'),
        ])

    def callback_sanitized_store(self):
        assert self.existing_input_id is not NOT_SET

        @callback(
            Output(self.sanitized_store_id, 'data'),
            Input(self.input_a_id, 'value'),
            Input(self.input_b_id, 'value'),
            Input(self.existing_input_id, 'value'),
        )
        def update_sanitized_store(a, b, c) -> dict:
            a = a if a else 0
            b = b if b else 10
            c = c if a < c < b else (a + b) / 2
            return dict(a=a, b=b, c=c)

    def info_for_sanitized_display(self) -> List[Union[str, dict]]:
        return [
            {'key': 'a', 'name': 'Input A', 'format': '.1f'},  # For more control to display 'a'
            'b',  # Use defaults to display 'b'
            'c',  # Use defaults to display 'c'
        ]

    def callback_output_store(self):
        """Update output store using sanitized inputs and data"""
        assert self.dat_id is not NOT_SET

        @callback(
            Output(self.output_store_id, 'data'),
            Input(self.dat_id, 'value'),
            Input(self.sanitized_store_id, 'data'),
            Input(self.data_input_id, 'data'),
        )
        def update_output_store(dat_id, inputs: dict, data_path: dict):
            # # Get data from previous processing
            # dat = get_dat(dat_id)
            # previous_process = load(data_path, PreviousProcess)
            # out = previous_process.outputs
            # useful_x, useful_data = out['x'], out['data']
            #
            # # Do the Processing
            # process = ThisProcess()
            # process.set_inputs(a=inputs['a'], b=inputs['b'], c=inputs['c'], x=useful_x, data=useful_data)
            # out = process.process()
            #
            # # Rather than pass big datasets etc, save the Process and return the location to load it
            # with HDFFileHandler(dat.hdf.hdf_path, 'r+') as f:
            #     process_group = f.require_group('/Process')
            #     save_group = process.save_progress(process_group, name=None)  # Save at top level with default name
            #     save_path = save_group.name
            # return {'dat_id': dat.dat_id, 'save_path': save_path}
            pass

    def get_input_display_layout(self):
        layout = html.Div([
            c.GraphAIO(aio_id=self.in_graph_id)
        ])
        return layout

    def get_output_display_layout(self):
        layout = html.Div([
            c.GraphAIO(aio_id=self.out_graph_id)
        ])
        return layout

    def callback_input_display(self):
        @callback(
            Output(c.GraphAIO.ids.graph(self.in_graph_id), 'figure'),
            Input(self.output_store_id, 'data'),
        )
        def update_input_graph(out_store):
            # if out_store:
            #     process = load(out_store, ThisProcess)
            #     fig = process.get_input_plotter().plot_1d()
            #     return fig
            return go.Figure()

    def callback_output_display(self):
        @callback(
            Output(c.GraphAIO.ids.graph(self.out_graph_id), 'figure'),
            Input(self.output_store_id, 'data'),
        )
        def update_output_graph(out_store):
            # if out_store:
            #     process = load(out_store, ThisProcess)
            #     fig = process.get_output_plotter().plot_1d()
            #     return fig
            return go.Figure()


def standard_input_layout(process_name: str, user_inputs, sanitized_inputs,
                          start_open: bool=True) -> dbc.Card:
    layout = dbc.Card([
        dbc.CardHeader(html.H5(process_name)),
        dbc.CardBody([user_inputs, sanitized_inputs])
    ])
    layout = c.CollapseAIO(content=layout, button_text=process_name, start_open=start_open)
    return layout

def standard_output_layout(process_name: str, output_display, input_display = None,
                           start_open: bool = True) -> dbc.Card:
    input_display = input_display if input_display else html.Div()
    layout = dbc.Card([
        dbc.CardHeader(html.H5(process_name)),
        dbc.CardBody([output_display, input_display])
    ])
    layout = c.CollapseAIO(content=layout, button_text=process_name, start_open=start_open)
    return layout


