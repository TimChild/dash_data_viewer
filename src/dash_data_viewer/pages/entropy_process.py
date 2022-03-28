from __future__ import annotations
import dash
import h5py
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
from typing import TYPE_CHECKING, List, Union, Optional, Any, Tuple
import uuid
import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass

import dash_data_viewer.components as c
from dash_data_viewer.process_dash_extensions import ProcessInterface, NOT_SET, load
from dash_data_viewer.layout_util import label_component

from dat_analysis.analysis_tools.new_procedures import SeparateSquareProcess, Process, PlottableData, DataPlotter
from dat_analysis import get_dat
from dat_analysis.hdf_file_handler import GlobalLock, HDFFileHandler
from dat_analysis.useful_functions import data_to_json, data_from_json, get_data_index
from dat_analysis.dat_object.attributes.square_entropy import get_transition_parts
from dat_analysis.plotting.plotly.dat_plotting import OneD

if TYPE_CHECKING:
    from dash.development.base_component import Component


# For now making the Processes here, but should be moved to dat_analysis package
@dataclass
class SeparateSquareProcess(Process):
    def set_inputs(self, x: np.ndarray, i_sense: np.ndarray,
                   measure_freq: float,
                   samples_per_setpoint: int,

                   setpoint_average_delay: Optional[float] = 0,
                   ):
        self.inputs = dict(
            x=x,
            i_sense=i_sense,
            measure_freq=measure_freq,
            samples_per_setpoint=samples_per_setpoint,
            setpoint_average_delay=setpoint_average_delay,
        )

    def _preprocess(self):
        i_sense = np.atleast_2d(self.inputs['i_sense'])

        data_by_setpoint = i_sense.reshape((i_sense.shape[0], -1, 4, self.inputs['samples_per_setpoint']))

        delay_index = round(self.inputs['setpoint_average_delay'] * self.inputs['measure_freq'])
        if delay_index > self.inputs['samples_per_setpoint']:
            setpoint_duration = self.inputs['samples_per_setpoint'] / self.inputs['measure_freq']
            raise ValueError(f'setpoint_average_delay ({self.inputs["setpoint_average_delay"]} s) is longer than '
                             f'setpoint duration ({setpoint_duration:.5f} s)')

        setpoint_duration = self.inputs['samples_per_setpoint'] / self.inputs['measure_freq']

        self._preprocessed = {
            'data_by_setpoint': data_by_setpoint,
            'delay_index': delay_index,
            'setpoint_duration': setpoint_duration,
        }

    def process(self):
        self._preprocess()
        separated = np.mean(
            self._preprocessed['data_by_setpoint'][:, :, :, self._preprocessed['delay_index']:], axis=-1)

        x = self.inputs['x']
        x = np.linspace(x[0], x[-1], separated.shape[-2])
        self.outputs = {
            'x': x,
            'separated': separated,
        }
        return self.outputs


    def get_input_plotter(self,
                          xlabel: str = 'Sweepgate /mV', data_label: str = 'Current /nA',
                          title: str = 'Data Averaged to Single Square Wave',
                          start_x: Optional[float] = None, end_x: Optional[float] = None,  # To only average between
                          y: Optional[np.ndarray] = None,
                          start_y: Optional[float] = None, end_y: Optional[float] = None,  # To only average between
                          ) -> DataPlotter:
        self._preprocess()
        by_setpoint = self._preprocessed['data_by_setpoint']
        x = self.inputs['x']

        if start_y or end_y:
            if y is None:
                raise ValueError(f'Need to pass in y_array to use start_y and/or end_y')
            indexes = get_data_index(y, [start_y, end_y])
            s_ = np.s_[indexes[0], indexes[1]]
            by_setpoint = by_setpoint[s_]  # slice of rows, all_dac steps, 4 parts, all datapoints

        if start_x or end_x:
            indexes = get_data_index(x, [start_x, end_x])
            s_ = np.s_[indexes[0], indexes[1]]
            by_setpoint = by_setpoint[:, s_]  # All rows, slice of dac steps, 4 parts, all datapoints

        averaged = np.nanmean(by_setpoint, axis=0)  # Average rows together
        averaged = np.moveaxis(averaged, 1, 0)  # 4 parts, num steps, samples
        averaged = np.nanmean(averaged, axis=-2)  # 4 parts, samples  (average steps together)
        averaged = averaged.flatten()  # single 1D array with all 4 setpoints sequential

        duration = self._preprocessed['setpoint_duration']
        time_x = np.linspace(0, 4 * duration, averaged.shape[-1])

        data = PlottableData(
            data=averaged,
            x=time_x,
        )

        plotter = DataPlotter(
            data=data,
            xlabel=xlabel,
            data_label=data_label,
            title=title,
        )
        return plotter

    def get_output_plotter(self,
                           y: Optional[np.ndarray] = None,
                           xlabel: str = 'Sweepgate /mV', data_label: str = 'Current* /nA',
                           ylabel: str = 'Repeats',
                           part: Union[str, int] = 'cold',  # e.g. hot, cold, vp, vm, or 0, 1, 2, 3
                           title: str = 'Separated into Square Wave Parts',
                           x_spacing: float = 0,
                           y_spacing: float = 0.3,
                           ) -> DataPlotter:
        if not self.processed:
            self.process()
        separated = self.outputs['separated']  # rows, dac steps, 4 parts
        y = y if y is not None else np.arange(separated.shape[0])

        data_part = self.data_part_out(part)

        data = PlottableData(
            data=data_part,
            x=self.outputs['x'],
            y=y,
        )
        plotter = DataPlotter(
            data=data,
            xlabel=xlabel,
            ylabel=ylabel,
            data_label=data_label,
            title=title,
            xspacing=x_spacing,
            yspacing=y_spacing,
        )
        return plotter

    def data_part_out(self,
                      part: Union[str, int] = 'cold',  # e.g. hot, cold, vp, vm, or 0, 1, 2, 3
                      ) -> np.ndarray:
        if not self.processed:
            self.process()
        separated = self.outputs['separated']
        part = get_transition_parts(part)  # Convert to Tuple (e.g. (1,3) for 'hot')
        data_part = np.nanmean(np.take(separated, part, axis=2), axis=2)
        return data_part




class SeparateProcessInterface(ProcessInterface):
    id_prefix = 'Separate'

    def __init__(self):
        # Set up the non-standard IDs which will be used for various inputs/outputs
        ID = self.ID  # Just to not have to write self.ID every time

        # User Input IDs
        self.input_delay_id = ID('input-delay')

        # User Inputs for viewing only
        # TODO: Fill with things like start-x, end-x etc

        # Other Required Input IDs
        self.dat_id = NOT_SET

        # Data input IDs
        self.data_input_id = NOT_SET  # Probably going to be taking data from HDF file

        # Display IDs
        self.in_graph_id = ID('graph-in')  # Display input to this Process (plus useful info, e.g. initial fit)
        self.out_graph_id = ID('graph-out')  # Display output of this Process

    def set_other_input_ids(self, dat_id):
        self.dat_id = dat_id  # For things like measure_freq etc
        # TODO: later, make it possible to manually set measure_freq etc (useful if dropping data in)

    def set_data_input_ids(self, previous_process_id=None):
        # self.data_input_id = previous_process_id
        pass

    def get_user_inputs_layout(self) -> html.Div:
        in_delay = dbc.Input(id=self.input_delay_id, type='number', debounce=True)
        return html.Div([
            label_component(in_delay, 'Settling Delay'),
        ])

    def callback_sanitized_store(self):
        assert self.dat_id is not NOT_SET

        @callback(
            Output(self.sanitized_store_id, 'data'),
            Input(self.input_delay_id, 'value'),
            Input(self.dat_id, 'value'),
        )
        def update_sanitized_store(delay, dat_id) -> dict:
            delay = delay if delay else 0
            # FOR TESTING
            dat_id = {
                'datnum':2164,
                'experiment_name': 'febmar21tim',
                'datname': 'base',
            }

            if dat_id:
                dat = get_dat(id=dat_id)
                measure_freq = dat.Logs.measure_freq
                samples_per_setpoint = round(dat.AWG.info.wave_len/4)
                # data = dat.Data.i_sense
                # x = dat.Data.x

                delay_index = round(delay * measure_freq)
                if delay_index > samples_per_setpoint:
                    delay_index = samples_per_setpoint
                    delay = delay_index / measure_freq

                return dict(
                    delay=delay,
                    measure_freq=measure_freq,
                    samples_per_setpoint=samples_per_setpoint,
                    valid=True,
                )
            return dict(
                delay = delay,
                measure_freq=0, samples_per_setpoint=0,
                valid=False
            )

    def info_for_sanitized_display(self) -> List[Union[str, dict]]:
        return [
            {'key': 'delay', 'name': 'Settle Delay', 'format': 'f'},
            {'key': 'measure_freq', 'name': 'Measure Frequency', 'format': 'g'},
            {'key': 'samples_per_setpoint', 'name': 'Setpoint Samples', 'format': 'd'},
        ]

    def callback_output_store(self):
        """Update output store using sanitized inputs and data"""
        assert self.dat_id is not NOT_SET
        print(f'running callback')

        @callback(
            Output(self.output_store_id, 'data'),
            Input(self.dat_id, 'value'),
            Input(self.sanitized_store_id, 'data'),
            # Input(self.data_input_id, 'data'),  # Taking data directly from datHDF (First Process Only)
        )
        def update_output_store(dat_id, inputs: dict): #, data_path: dict):
            # FOR TESTING
            dat_id = {
                'datnum':2164,
                'experiment_name': 'febmar21tim',
                'datname': 'base',
            }
            if dat_id and inputs and inputs.get('valid', False):
                # Get data from previous processing
                dat = get_dat(dat_id)

                # Get Initial data directly from Dat (First Process Only)
                i_sense = dat.Data.i_sense
                x = dat.Data.x

                # Do the Processing
                process = SeparateSquareProcess()
                process.set_inputs(x=x, i_sense=i_sense,
                                   measure_freq=inputs['measure_freq'],
                                   samples_per_setpoint=inputs['samples_per_setpoint'],
                                   setpoint_average_delay=inputs['delay'],
                                   )

                out = process.process()

                # Rather than pass big datasets etc, save the Process and return the location to load it
                with HDFFileHandler(dat.hdf.hdf_path, 'r+') as f:
                    process_group = f.require_group('/Process')  # First Process, so make sure Process Group exists

                    save_group = process.save_progress(process_group, name=None)  # Save at top level with default name
                    save_path = save_group.name

                return {'dat_id': dat.dat_id, 'save_path': save_path}
            return dash.no_update

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
            if out_store:
                process: SeparateSquareProcess = load(out_store, SeparateSquareProcess)
                fig = process.get_input_plotter().plot_1d()

                delay = process.inputs['setpoint_average_delay']
                sp_duration = process.inputs['samples_per_setpoint']/process.inputs['measure_freq']
                for i in range(4):
                    fig.add_vline(sp_duration*i, line=dict(color='black', dash='solid'))
                    fig.add_vline(delay+sp_duration*i, line=dict(color='green', dash='dash'))

                return fig
            return go.Figure()

    def callback_output_display(self):
        @callback(
            Output(c.GraphAIO.ids.graph(self.out_graph_id), 'figure'),
            Input(self.output_store_id, 'data'),
        )
        def update_output_graph(out_store):
            if out_store:
                process: SeparateSquareProcess = load(out_store, SeparateSquareProcess)
                p = OneD(dat=None)
                x = process.outputs['x']
                fig = p.figure(xlabel='Sweepgate /mV', ylabel='Current /nA', title='Naively Averaged Separated Data')
                for i, label in enumerate(['0_0', '+', '0_1', '-']):
                    data = process.data_part_out(i)
                    data = np.nanmean(data, axis=0)
                    fig.add_trace(p.trace(data=data, x=x, name=label, mode='lines'))
                return fig
            return go.Figure()



# Actually make the page
dat_picker = c.DatnumPickerAIO(aio_id='entropy-process', allow_multiple=False)
stores = []

# Initialize Interfaces that help make dash page
separate_interface = SeparateProcessInterface()
separate_interface.set_data_input_ids(None)
separate_interface.set_other_input_ids(dat_id=dat_picker.dd_id)

stores.append(separate_interface.get_stores())
separate_inputs = separate_interface.get_user_inputs_layout()
separate_sanitized = separate_interface.get_sanitized_inputs_layout()
separate_input_display = separate_interface.get_input_display_layout()
separate_output_display = separate_interface.get_output_display_layout()

separate_interface.callback_output_store()
separate_interface.callback_sanitized_store()

separate_interface.callback_sanitized_inputs()
separate_interface.callback_input_display()
separate_interface.callback_output_display()




# Put everything together into a layout
layout = html.Div([
    *stores,
    dbc.Row([
        dbc.Col([
            dat_picker,
            dbc.Card([separate_inputs, separate_sanitized]),
        ],
            width=3),
        dbc.Col([
            dbc.Card([separate_input_display, separate_output_display]),
        ], width=9),
    ])
]
)


if __name__ == '__main__':
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    # app.layout = layout()
    app.layout = layout
    app.run_server(debug=True, port=8051)
else:
    dash.register_page(__name__)
    pass

