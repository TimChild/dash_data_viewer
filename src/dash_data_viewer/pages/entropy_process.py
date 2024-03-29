import logging

import dash
import os
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
from typing import TYPE_CHECKING, List, Union
import numpy as np
import plotly.graph_objects as go

import dash_data_viewer.components as c
from dash_data_viewer.process_dash_extensions import ProcessInterface, NOT_SET, load, standard_input_layout, standard_output_layout
from dash_data_viewer.layout_util import label_component
from dash_data_viewer.dash_hdf import DashHDF, HdfId
from dash_data_viewer.new_dat_util import get_dat_from_exp_path

from dat_analysis.analysis_tools.entropy import EntropySignalProcess
from dat_analysis.analysis_tools.new_procedures import PlottableData, DataPlotter
from dat_analysis.analysis_tools.square_wave import SeparateSquareProcess
from dat_analysis.analysis_tools.transition import CenteredAveragingProcess
from dat_analysis.plotting.plotly.dat_plotting import OneD


if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

DELTA = '\u0394'
SIGMA = '\u03c3'


# For now making the Processes here, but should be moved to dat_analysis package


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
        self.dashdata_id = NOT_SET

        # Data input IDs
        self.data_input_id = NOT_SET

        # Display IDs
        self.in_graph_id = ID('graph-in')  # Display input to this Process (plus useful info, e.g. initial fit)
        self.out_graph_id = ID('graph-out')  # Display output of this Process

    def set_other_input_ids(self, dashdata_id):
        self.dashdata_id = dashdata_id  # For things like measure_freq etc
        # TODO: later, make it possible to manually set measure_freq etc (useful if dropping data in)

    def set_data_input_ids(self, previous_process_id=None):
        logger.warning(f'Nothing happens here')
        # self.data_input_id = previous_process_id
        pass

    def get_user_inputs_layout(self) -> html.Div:
        in_delay = dbc.Input(id=self.input_delay_id, type='number', debounce=True)
        return html.Div([
            label_component(in_delay, 'Settling Delay'),
        ])

    def callback_sanitized_store(self):
        assert self.dashdata_id is not NOT_SET

        @callback(
            Output(self.sanitized_store_id, 'data'),
            Input(self.input_delay_id, 'value'),
            Input(self.dashdata_id, 'data'),
        )
        def update_sanitized_store(delay, dashd_id) -> dict:
            delay = delay if delay else 0

            if dashd_id:
                dashd = DashHDF(dashd_id)
                measure_freq = dashd.get_info('measure_freq')
                samples_per_setpoint = dashd.get_info('samples_per_setpoint')

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
        assert self.dashdata_id is not NOT_SET

        @callback(
            Output(self.output_store_id, 'data'),
            Input(self.dashdata_id, 'data'),
            Input(self.sanitized_store_id, 'data'),
        )
        def update_output_store(dashd_id, inputs: dict):
            if dashd_id and inputs and inputs.get('valid', False):
                # Get data from previous processing
                dashd = DashHDF(dashd_id)
                i_sense = dashd.get_data('i_sense', subgroup=None)
                x = dashd.get_data('x', subgroup=None)

                # Do the Processing
                process = SeparateSquareProcess()
                process.set_inputs(x=x, i_sense=i_sense,
                                   measure_freq=inputs['measure_freq'],
                                   samples_per_setpoint=inputs['samples_per_setpoint'],
                                   setpoint_average_delay=inputs['delay'],
                                   )

                process.process()

                # Rather than pass big datasets etc, save the Process and return the location to load it
                dashd.mode = 'r+'
                with dashd as f:
                    process_group = f.require_group('/Process')  # First Process, so make sure Process Group exists

                    save_group = process.save_progress(process_group, name=None)  # Save at top level with default name
                    save_path = save_group.name

                logger.debug('Updating Separate Output Store')
                return {'dashd_id': dashd.id, 'save_path': save_path}
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
            Output(c.GraphAIO.ids.update_figure_store(self.in_graph_id), 'data'),
            Input(self.output_store_id, 'data'),
        )
        def update_input_graph(out_store):
            if out_store:
                process: SeparateSquareProcess = load(out_store, SeparateSquareProcess)
                p = process.get_input_plotter()
                fig = p.fig_1d()
                fig.add_trace(p.trace_1d())

                delay = process.inputs['setpoint_average_delay']
                sp_duration = process.inputs['samples_per_setpoint']/process.inputs['measure_freq']
                for i in range(4):
                    fig.add_vline(sp_duration*i, line=dict(color='black', dash='solid'))
                    fig.add_vline(delay+sp_duration*i, line=dict(color='green', dash='dash'))

                return fig
            return go.Figure()

    def callback_output_display(self):
        @callback(
            Output(c.GraphAIO.ids.update_figure_store(self.out_graph_id), 'data'),
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


class EntropySignalInterface(ProcessInterface):
    id_prefix = 'entropy-signal'

    def __init__(self):
        # Set up the non-standard IDs which will be used for various inputs/outputs
        ID = self.ID  # Just to not have to write self.ID every time

        # User Input IDs

        # Other Required Input IDs
        self.dashd_id = NOT_SET

        # Data input IDs
        self.previous_process_id = NOT_SET  # Or may be easier to continue on from the output of another process

        # Display IDs
        self.in_graph_id = ID('graph-in')  # Display input to this Process (plus useful info, e.g. initial fit)
        self.out_graph2d_id = ID('graph2d-out')  # Display output of this Process
        self.out_graph_avg_id = ID('graph-avg-out')  # Display output of this Process

    def set_other_input_ids(self, dashd_id):
        self.dashd_id = dashd_id

    def set_data_input_ids(self, previous_process_id):
        self.previous_process_id = previous_process_id

    def get_user_inputs_layout(self) -> html.Div:
        return html.Div([
        ])

    def callback_sanitized_store(self):
        # @callback(
        #     Output(self.sanitized_store_id, 'data'),
        # )
        # def update_sanitized_store() -> dict:
        #     return {}
        pass

    def info_for_sanitized_display(self) -> List[Union[str, dict]]:
        # return [
        #     {'key': 'a', 'name': 'Input A', 'format': '.1f'},  # For more control to display 'a'
        #     'b',  # Use defaults to display 'b'
        #     'c',  # Use defaults to display 'c'
        # ]
        return []

    def callback_output_store(self):
        """Update output store using sanitized inputs and data"""
        assert self.dashd_id is not NOT_SET

        @callback(
            Output(self.output_store_id, 'data'),
            Input(self.dashd_id, 'data'),
            Input(self.sanitized_store_id, 'data'),
            Input(self.previous_process_id, 'data'),
        )
        def update_output_store(dashd_id, inputs: dict, previous_process_location: dict):
            if dashd_id and previous_process_location:
                # Get data from previous processing
                previous: SeparateSquareProcess = load(previous_process_location, SeparateSquareProcess)
                data_dict = previous.outputs
                x = data_dict['x']
                separated = data_dict['separated']

                # Do the Processing
                process = EntropySignalProcess()
                process.set_inputs(x=x, separated_data=separated)
                process.process()

                # Rather than pass big datasets etc, save the Process and return the location to load it
                dashd = DashHDF(dashd_id)
                with dashd as f:
                    process_group = f.require_group('/Process')  # First Process, so make sure Process Group exists
                    save_group = process.save_progress(process_group, name=None)  # Save at top level with default name
                    save_path = save_group.name

                logger.debug('updating entropy signal store')
                return {'dashd_id': dashd.hdf_id.asdict(), 'save_path': save_path}
            return dash.no_update

    def get_input_display_layout(self):
        layout = html.Div([
        ])
        return layout

    def get_output_display_layout(self):
        layout = html.Div([
            c.GraphAIO(aio_id=self.out_graph2d_id),
            c.GraphAIO(aio_id=self.out_graph_avg_id),
        ])
        return layout

    def callback_input_display(self):
        # @callback(
        #     Output(c.GraphAIO.ids.update_figure_store(self.in_graph_id), 'data'),
        #     Input(self.output_store_id, 'data'),
        # )
        # def update_input_graph(out_store):
        #     # if out_store:
        #     #     process = load(out_store, ThisProcess)
        #     #     fig = process.get_input_plotter().plot_1d()
        #     #     return fig
        #     return go.Figure()
        pass

    def callback_output_display(self):
        @callback(
            Output(c.GraphAIO.ids.update_figure_store(self.out_graph2d_id), 'data'),
            Output(c.GraphAIO.ids.update_figure_store(self.out_graph_avg_id), 'data'),
            Input(self.output_store_id, 'data'),
        )
        def update_output_graph(out_store):
            if out_store:
                process = load(out_store, EntropySignalProcess)
                data = process.outputs['entropy']
                x = process.outputs['x']
                plotter = process.get_output_plotter()

                fig2d = plotter.fig_heatmap()
                fig2d.add_trace(plotter.trace_heatmap())

                fig_avg = plotter.fig_1d()
                fig_avg.add_trace(plotter.trace_1d(avg=True))
                fig_avg.update_layout(title='Naively Averaged Entropy Signal')

                return fig2d, fig_avg
            return go.Figure(), go.Figure()


class CenteredEntropyAveragingInterface(ProcessInterface):

    id_prefix = 'entropy-averaging'

    def __init__(self):
        # Set up the non-standard IDs which will be used for various inputs/outputs
        ID = self.ID  # Just to not have to write self.ID every time

        # User Input IDs
        self.toggle_centering_id = c.MultiButtonAIO.ids.store(aio_id=self.ID('centering'))  # TODO: Just use a regular button and make callback for it to change color etc
        self.input_fit_start_id = ID('input-fit-start')
        self.input_fit_end_id = ID('input-fit-end')
        # TODO: Add options for initial params

        # Other Required Input IDs
        self.dashd_id = NOT_SET

        # Data input IDs
        self.separated_data_id = NOT_SET
        self.entropy_signal_id = NOT_SET

        # Display IDs
        self.in_graph_id = ID('graph-in')  # Display input to this Process (plus useful info, e.g. initial fit)
        self.out_graph_isense_id = ID('graph-out-isense')  # Display output of this Process
        self.out_graph_entropy_id = ID('graph-out-entropy')  # Display output of this Process

    def set_other_input_ids(self, dashd_id: str):
        self.dashd_id = dashd_id

    def set_data_input_ids(self, separated_data_id, entropy_signal_id):
        self.separated_data_id = separated_data_id
        self.entropy_signal_id = entropy_signal_id

    def get_user_inputs_layout(self) -> html.Div:
        tog_center = c.MultiButtonAIO(button_texts=['Centered Averaging'], allow_multiple=False, aio_id=self.ID('centering'))
        in_start = dbc.Input(id=self.input_fit_start_id, type='number', debounce=True)
        in_end = dbc.Input(id=self.input_fit_end_id, type='number', debounce=True)
        return html.Div([
            tog_center,
            label_component(in_start, 'Fit Start'),
            label_component(in_end, 'Fit End'),
        ])

    def callback_sanitized_store(self):
        assert self.dashd_id
        assert self.separated_data_id
        assert self.entropy_signal_id

        @callback(
            Output(self.sanitized_store_id, 'data'),
            Input(self.toggle_centering_id, 'data'),
            Input(self.input_fit_start_id, 'value'),
            Input(self.input_fit_end_id, 'value'),
            Input(self.separated_data_id, 'data'),
            # State(self.entropy_signal_id, 'data'),  # TODO: Possibly want to check this exists as well?
        )
        def update_sanitized_store(centered, fit_start, fit_end, prev_process_location) -> dict:
            if prev_process_location:
                prev_process: SeparateSquareProcess = load(prev_process_location, SeparateSquareProcess)
                x = prev_process.outputs['x']
                if not fit_start or not np.nanmin(x) < fit_start < np.nanmax(x):
                    fit_start = np.nanmin(x)
                if not fit_end or not np.nanmin(x) < fit_end < np.nanmax(x):
                    fit_end = np.nanmax(x)
                logger.debug(f'Value of centered is {centered}')
                centered = True if centered['selected_values'] else False
                return dict(
                    centered = centered,
                    fit_start = fit_start,
                    fit_end = fit_end,
                    valid=True
                )
            return dict(
                centered=False,
                fit_start=0,
                fit_end=0,
                valid=False,
            )

    def info_for_sanitized_display(self) -> List[Union[str, dict]]:
        return [
            {'key': 'centered', 'name': 'Do centering', 'format': 'g'},
            {'key': 'fit_start', 'name': 'Fit From x', 'format': 'f'},
            {'key': 'fit_end', 'name': 'Fit To x', 'format': 'f'},
        ]

    def callback_output_store(self):
        """Update output store using sanitized inputs and data"""
        assert self.dashd_id

        @callback(
            Output(self.output_store_id, 'data'),
            Input(self.dashd_id, 'data'),
            Input(self.sanitized_store_id, 'data'),
            Input(self.separated_data_id, 'data'),
            Input(self.entropy_signal_id, 'data'),
        )
        def update_output_store(dashd_id: dict, inputs: dict, separated_path, entropy_path):
            if dashd_id and inputs['valid']:

                # Get data from previous processing
                separated_process = load(separated_path, SeparateSquareProcess)
                out = separated_process.outputs
                x, separated = out['x'], out['separated']

                cold = np.mean(np.take(np.atleast_3d(separated), (0, 2), axis=2), axis=2)
                hot = np.mean(np.take(np.atleast_3d(separated), (1, 3), axis=2), axis=2)

                entropy_process = load(entropy_path, EntropySignalProcess)
                entropy = entropy_process.outputs['entropy']

                # Do the Processing
                process = CenteredAveragingProcess()
                process.set_inputs(x=x, datas=[cold, hot, entropy],
                                   center_by_fitting=inputs['centered'],
                                   fit_start_x=inputs['fit_start'],
                                   fit_end_x=inputs['fit_end'],
                                   initial_params=None,
                                   )
                out = process.process()

                # Rather than pass big datasets etc, save the Process and return the location to load it
                dashd = DashHDF(dashd_id)
                with dashd as f:
                    process_group = f.require_group('/Process')
                    save_group = process.save_progress(process_group, name=None)  # Save at top level with default name
                    save_path = save_group.name
                logger.debug(f'Updating Center and average store')
                return {'dashd_id': dashd.id, 'save_path': save_path}
            return dash.no_update

    def get_input_display_layout(self):
        layout = html.Div([
            c.GraphAIO(aio_id=self.in_graph_id)
        ])
        return layout

    def get_output_display_layout(self):
        layout = html.Div([
            c.GraphAIO(aio_id=self.out_graph_isense_id),
            c.GraphAIO(aio_id=self.out_graph_entropy_id),
        ])
        return layout

    def callback_input_display(self):
        @callback(
            Output(c.GraphAIO.ids.update_figure_store(self.in_graph_id), 'data'),
            Input(self.output_store_id, 'data'),
        )
        def update_input_graph(out_store):
            if out_store:
                process: CenteredAveragingProcess = load(out_store, CenteredAveragingProcess)
                cold_data, _, _ = process.inputs['datas']
                y = np.arange(cold_data.shape[0])
                data = PlottableData(cold_data, x=process.inputs['x'], y=y)

                p = DataPlotter(data, xlabel='Sweepgate /mV', ylabel='Row', data_label='Current /nA',
                                title='Unheated part of Data with Center Values')
                fig = p.fig_heatmap()
                fig.add_trace(p.trace_heatmap())
                fig.add_trace(go.Scatter(x=process.outputs['centers'], y=y, mode='markers')) #, marker=dict(size=5, symbol='cross-thin', line=dict(color='white'))))
                return fig
            return go.Figure()

    def callback_output_display(self):
        @callback(
            Output(c.GraphAIO.ids.update_figure_store(self.out_graph_isense_id), 'data'),
            Input(self.output_store_id, 'data'),
        )
        def update_i_sense_graph(out_store):
            if out_store:
                process: CenteredAveragingProcess = load(out_store, CenteredAveragingProcess)

                x_label = 'Sweepgate /mV'
                title = f'Averaged Transition data (with {SIGMA} of average)'
                if process.inputs['center_by_fitting']:
                    x_label = 'Centered ' + x_label
                    title = 'Centered ' + title

                p = DataPlotter(None, xlabel=x_label, data_label='Current /nA',
                                title=title)
                fig = p.fig_1d()

                datas = process.outputs['averaged']
                errs = process.outputs['std_errs']
                x = process.outputs['x']

                for data, err in zip(datas[:2], errs[:2]):
                    plottable = PlottableData(data, x=x,
                                         data_err=err)
                    p.data = plottable
                    fig.add_trace(p.trace_1d())

                return fig
            return go.Figure()

        @callback(
            Output(c.GraphAIO.ids.update_figure_store(self.out_graph_entropy_id), 'data'),
            Input(self.output_store_id, 'data'),
        )
        def update_entropy_graph(out_store):
            if out_store:
                process: CenteredAveragingProcess = load(out_store, CenteredAveragingProcess)

                x_label = 'Sweepgate /mV'
                title = f'Averaged Entropy Signal (with {SIGMA} of average)'
                if process.inputs['center_by_fitting']:
                    x_label = 'Centered ' + x_label
                    title = 'Centered ' + title

                p = DataPlotter(None, xlabel=x_label, data_label=f'{DELTA}Current /nA',
                                title=title)
                fig = p.fig_1d()

                data = process.outputs['averaged'][-1]
                err = process.outputs['std_errs'][-1]
                x = process.outputs['x']

                plottable = PlottableData(data, x=x,
                                          data_err=err)
                p.data = plottable
                fig.add_trace(p.trace_1d())

                return fig
            return go.Figure()


# Actually make the page
dat_selector = c.DatSelectorAIO()

# # Store to hold the DashHDF ID where any external info (i.e. from Dat or File) is stored
# # This is to avoid relying on any specific original source of data but keep data local and only send pointer to store
external_data_path_store = dcc.Store(id='entropy-store-from-external')


@callback(
    Output(external_data_path_store, 'data'),
    Input(dat_selector.store_id, 'data'),
)
def get_external_data_and_info(dat_filepath):
    """For now this only gets data from Dat, but can later take data from other sources and everything else should
    still work
    Note: This is the ONLY place that external data should load from.
    """

    if dat_filepath and os.path.exists(dat_filepath) and os.path.isfile(dat_filepath):
        data_path = dat_filepath
    else:
        data_path = None

    if data_path:
        dat = get_dat_from_exp_path(data_path)
        hdf_id = HdfId(page='entropy-process', number=dat.datnum)
        dashd = DashHDF(hdf_id)  # TODO: Maybe want to use 'w' mode to overwrite previous?
        subgroup = None  # In case I later want to save this data in e.g. a "main" group

        # Any Data that will likely be used by a later process (if it is unlikely to be used, set it in a relevant callback)
        dashd.save_data(dat.Data.get_data('standard/i_sense', None), 'i_sense', subgroup=subgroup)
        dashd.save_data(dat.Data.x, 'x', subgroup=subgroup)

        # Any Info that will likely be used by a later process (if it is unlikely to be used, set it in a relevant callback)
        dashd.save_info(dat.Logs.measure_freq, 'measure_freq', subgroup=subgroup)
        fd_logs = dat.Logs.get_fastdac(1)
        dashd.save_info(fd_logs.AWG['waveLen']/4, 'samples_per_setpoint', subgroup=subgroup)  # fd_logs.AWG['info']['wave_len']/4
        return dashd.id
    else:
        return None

# dat_picker = c.DatnumPickerAIO(aio_id='entropy-process', allow_multiple=False)
stores = [external_data_path_store]

# Initialize Interfaces that help make dash page
# Separating into parts of heating wave
separate_interface = SeparateProcessInterface()
separate_interface.set_data_input_ids(None)
separate_interface.set_other_input_ids(dashdata_id=external_data_path_store.id)

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


# Turning into Entropy signal
signal_interface = EntropySignalInterface()
signal_interface.set_data_input_ids(previous_process_id=separate_interface.output_store_id)
signal_interface.set_other_input_ids(dashd_id=external_data_path_store.id)

stores.append(signal_interface.get_stores())
signal_inputs = signal_interface.get_user_inputs_layout()
signal_sanitized = signal_interface.get_sanitized_inputs_layout()
signal_input_display = signal_interface.get_input_display_layout()
signal_output_display = signal_interface.get_output_display_layout()

signal_interface.callback_output_store()
signal_interface.callback_sanitized_store()

signal_interface.callback_sanitized_inputs()
signal_interface.callback_input_display()
signal_interface.callback_output_display()


# Centering and Averaging
centering_interface = CenteredEntropyAveragingInterface()
centering_interface.set_data_input_ids(
    separated_data_id=separate_interface.output_store_id,
    entropy_signal_id=signal_interface.output_store_id,
)
centering_interface.set_other_input_ids(dashd_id=external_data_path_store.id)

stores.append(centering_interface.get_stores())
centering_inputs = centering_interface.get_user_inputs_layout()
centering_sanitized = centering_interface.get_sanitized_inputs_layout()
centering_input_display = centering_interface.get_input_display_layout()
centering_output_display = centering_interface.get_output_display_layout()

centering_interface.callback_output_store()
centering_interface.callback_sanitized_store()

centering_interface.callback_sanitized_inputs()
centering_interface.callback_input_display()
centering_interface.callback_output_display()


# Put everything together into a layout
layout = html.Div([
    *stores,
    dbc.Row([
        dbc.Col([
            dat_selector,
            standard_input_layout('Separate Heating Parts', separate_inputs, separate_sanitized),
            standard_input_layout('Make Entropy Signal', signal_inputs, signal_sanitized),
            standard_input_layout('Centering and Averaging', centering_inputs, centering_sanitized),
        ],
            width=6, md=5, lg=4),
        dbc.Col([
            standard_output_layout('Separate Heating Parts', separate_output_display, separate_input_display),
            standard_output_layout('Entropy Signal', signal_output_display, signal_input_display),
            standard_output_layout('Centering and Averaging', centering_output_display, centering_input_display),
        ], width=6, md=7, lg=8),
    ])
]
)


if __name__ == '__main__':
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    # app.layout = layout()
    app.layout = layout
    app.run_server(debug=True, port=8051, dev_tools_hot_reload=False, use_reloader=False)
else:
    dash.register_page(__name__)
    pass

