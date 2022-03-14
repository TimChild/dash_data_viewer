from __future__ import annotations
import dash
import dash_extensions.snippets
from dash import html, dcc, callback, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
from dash_data_viewer.cache import cache
from dataclasses import dataclass

from dat_analysis.plotting.plotly import OneD, TwoD
from dat_analysis.dat_object.make_dat import DatHandler, get_newest_datnum, get_dat, get_dats
import dat_analysis.useful_functions as u

import logging
import numpy as np
import plotly.graph_objects as go
from scipy import signal
from scipy.fft import rfft, rfftfreq
from scipy.integrate import cumulative_trapezoid

from dash_data_viewer.entropy_report import EntropyReport
from dash_data_viewer.transition_report import TransitionReport
from dash_data_viewer.layout_util import label_component
from dash_data_viewer.components import DatnumPickerAIO
from dash_data_viewer.cache import cache

from typing import TYPE_CHECKING, Optional, List, Union, Tuple

logging.basicConfig(level=logging.INFO)

if TYPE_CHECKING:
    from dash.development.base_component import Component
    from dat_analysis.dat_object.dat_hdf import DatHDF
    from dat_analysis.dat_object.attributes.square_entropy import Output


class MainComponents(object):
    """Convenient holder for any components that will end up in the main area of the page"""
    div = html.Div(id='noise-div-text')
    graph1 = dcc.Graph(id='noise-graph-graph1')
    graph2 = dcc.Graph(id='noise-graph-graph2')
    noise_info_text = html.Div(id='noise-div-dataInfoText')


class SidebarComponents(object):
    """Convenient holder for any components that will end up in the sidebar area of the page"""
    button = dbc.Button('Click Me', id='button-click')
    datnum = dbc.Input(id='noise-input-datnum', persistence='local', debounce=True, placeholder="Datnum", type='number')
    datnum_final = dbc.Input(id='noise-input-datnum-final', persistence='local', debounce=True, placeholder="Datnum_final", type='number')
    datnum_choice = dbc.Input(id='noise-input-datnum-choice', persistence='local', debounce=True, placeholder="Datnum_choicel", type='number')
    specdropdown = dcc.Dropdown(id = 'psd-type', options = [
                                                            {'label': 'Spectrum', 'value':0},
                                                            {'label': 'Density', 'value': 1}],
                                                            value = 0)
    lindbdropdown = dcc.Dropdown(id = 'lin-db', options = [
                                                            {'label': 'Linear', 'value':0},
                                                            {'label': 'dB', 'value': 1}],
                                                            value = 1)   
    psdtype_checklist = dcc.Checklist(id = 'pdgm-welch', options = [
                                                            {'label': 'Periodogram', 'value':'PDGM'},
                                                            {'label': 'Welch', 'value': 'WELCH'}],
                                                            value = ['PDGM', 'WELCH'],
                                                            labelStyle={'display': 'inline-block'})                                                                                                           
    dat_selector = DatnumPickerAIO(aio_id='transition-datpicker', allow_multiple=False)
    integrate_check = dcc.Checklist(id = 'integrated', options = [
                                                            {'label': 'Integrate', 'value':'INT'}],
                                                            value = ['INT'],
                                                            labelStyle={'display': 'inline-block'})
    freq_selector = DatnumPickerAIO(aio_id='freqpicker', allow_multiple=True, button_title='Frequency Choice') 
    freq_width = dcc.Input(id="freq_width", type="number", placeholder=0, min=0, debounce=True)    
    width_check = dcc.Checklist(id = 'width_freq', options = [
                                                            {'label': '', 'value':'WIDTH'}],
                                                            value = [],
                                                            labelStyle={'display': 'inline-block'})                                             

# Initialize the components ONCE here.
main_components = MainComponents()
sidebar_components = SidebarComponents()


###############################################################################
# START: noise functions, leave here for now...
###############################################################################
# If you want to compare the magnitude of your spectra, you should use scaling='spectrum'. 
# If you compare integrals over a bandwidth, you should use scaling='density'.
def psd(x: np.ndarray, delf: float, type_psd: list, n: float = None) -> np.ndarray:
    """Returns 2d array of PSD computed with specified method

    Args:
        x (np.ndarray): Values in time domain
        delf (float): Sampling Rate
        type (list): [x, y] x=0 psd, x=1 psd density && y=0 standard psd, y=1 welch's method
        n (float): Number of Segments with Welch's method

    Returns:
        np.ndarray: 2D array, frequency values and computed PSD values
    """ 

    if type_psd[0] == 0:
        scale = 'spectrum'
    else:
        scale = 'density'

    if type_psd[1] == 0:
        f, P_xx = signal.periodogram(x, delf, scaling=scale)
    else:
        num_points = len(x)/n
        f, P_xx = signal.welch(x, delf, nperseg=num_points, scaling=scale)

    return [f, P_xx]


def integrate(x, y):
    '''
    Takes an array of y values and the x spacing between each point
    (assumed constant) and returns an array of the cumulative 
    integration (area)
    '''

    return cumulative_trapezoid(y, x, initial=0)


def dBScale(x: np.array) -> np.ndarray:
    """Converts array into decibels

    Args:
        x (np.array): 

    Returns:
        np.ndarray: 
    """
    return 10 * np.log10(x)

###############################################################################
# END: noise functions
###############################################################################

@dataclass
class NoiseData:
    x: np.ndarray
    y: Optional[np.ndarray]
    data: np.ndarray
    sample_freq: Optional[float]
    pdgm: Optional[np.ndarray]
    welch: Optional[np.ndarray]
    welch_window: Optional[int]
    select_freq: Optional[np.ndarray]
    width_freq: Optional[float]



# # @cache.memoize()
# def get_data(datnum) -> Tuple[np.darray, np.ndarray]:
#     print(f'getting data for dat{datnum}')
#     data = None
#     x = None
#     if datnum:
#         dat = get_dat(datnum)
#         x = dat.Data.get_data('x')
#         if 'Experiment Copy/current' in dat.Data.keys:
#             data = dat.Data.get_data('Experiment Copy/current')
#         elif 'Experiment Copy/cscurrent' in dat.Data.keys:
#             data = dat.Data.get_data('Experiment Copy/cscurrent')
#         elif 'Experiment Copy/current_2d' in dat.Data.keys:
#             data = dat.Data.get_data('Experiment Copy/current_2d')[0]
#         elif 'Experiment Copy/cscurrent_2d' in dat.Data.keys:
#             data = dat.Data.get_data('Experiment Copy/cscurrent_2d')[0]
#         else:
#             print(f'No data found. valid keys are {dat.Data.keys}')
#     return x, data

# @cache.memoize()
def get_noise_datas(datnums: int) -> Optional[Union[NoiseData, List[NoiseData]]]:
    tdatas = None
    if datnums:
        was_int = isinstance(datnums, int)
        if isinstance(datnums, int):
            datnums = [datnums]
        elif not isinstance(datnums, list):
            return None
        dats = get_dats(datnums)
        ndatas = []
        for dat in dats:
            x = dat.Data.get_data('x')
            data = dat.Data.get_data('i_sense')
            if 'y' in dat.Data.keys:
                y = dat.Data.get_data('y')
            else:
                y = None
            ndata = NoiseData(x=x, y=y, data=data)
            ndatas.append(ndata)
        if was_int is True:
            return ndatas[0]
    return ndatas


@callback(
    Output(main_components.graph1.id, 'figure'),
    Input(sidebar_components.datnum.id, 'value'))
def update_graph(datnum) -> go.Figure:
    x, data = get_data(datnum)
    if x is not None and data is not None:
        p1d = OneD(dat=None)
        fig = p1d.figure(title=f'Test')
        fig.add_trace(p1d.trace(x=x, data=data))
        
        return fig
    return go.Figure()


# Updating graphs
@callback(Output(main_components.graph2.id, 'figure'), 
          Input(sidebar_components.dat_selector.dd_id, 'value'),
          Input(sidebar_components.freq_selector.dd_id, 'value'),
          Input(sidebar_components.specdropdown.id, 'value'),
          Input(sidebar_components.lindbdropdown.id, 'value'),
          Input(sidebar_components.psdtype_checklist.id, 'value'))
def update_graph(datnums: list, freqs: list, specdropdown, lineardb, ps_check) -> go.Figure:
    print(f'Hitting graphing callback')
    print( f'Dats = {datnums}, Datnumtype = {type(datnums)}, Frequencies = {freqs}, Freqtype = {type(freqs)}')

    # Creating correct y label
    if lineardb == 0:
        if specdropdown == 0:
            y_label = 'Power Spectrum V^2'
        else:
            y_label = 'Power Spectrum V^2/Hz' 
    else:
        if specdropdown == 0:
            y_label = 'Power Spectrum dB V^2'
        else:
            y_label = 'Power Spectrum dB V^2/Hz'


    p1d = OneD(dat=None)
    p1d.MAX_POINTS = 10000

    if lineardb == 1:
        fig = p1d.figure(title=f'Power Spectrum', ylabel=y_label, xlabel='Frequency Hz')
    else:
        fig = p1d.figure(title=f'Power Spectrum', ylabel=y_label, xlabel='Frequency Hz')
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
    # If none of the dats are good then return empty figure
    print(f'Here are our datvals = {datnums}')
    if datnums == None:
        return go.Figure()
    else:
        print(f'Getting dat {datnums}')

        x, data = get_dat(datnums)
        dat = get_dat(datnums)
        freq = dat.Logs.measure_freq

        if 'PDGM' in ps_check:
            psd_pdgm = psd(data, freq, [specdropdown, 0])
            psd_pdgm_pxx = psd_pdgm[1]
            freq_pdgm = psd_pdgm[0]
            if lineardb == 1:   
                psd_pdgm_pxx = dBScale(psd_pdgm_pxx)
            fig.add_trace(p1d.trace(x=freq_pdgm, data=psd_pdgm_pxx, mode='lines', name=f'Dat{dat_val}: Periodogram'))

        if 'WELCH' in ps_check:
            psd_welch = psd(data, freq, [specdropdown, 1], n=20)
            psd_welch_pxx = psd_welch[1]
            freq_welch = psd_welch[0]
            if lineardb == 1:   
                psd_welch_pxx = dBScale(psd_welch_pxx) 
            fig.add_trace(p1d.trace(x=freq_welch, data=psd_welch_pxx, mode='lines', name=f'Dat{dat_val}: Welch'))
        
    return fig


# Updating table of information
@callback(
    Output(main_components.noise_info_text.id, 'children'),
    Input(sidebar_components.dat_selector.dd_id, 'value'),
    Input(sidebar_components.freq_selector.dd_id, 'value'),
)
def update_data_shapes(datnums: list, freqs: list):
    """Returns Data Info"""
    print( f'Dats = {datnums}, Datnumtype = {type(datnums)}, Frequencies = {freqs}, Freqtype = {type(freqs)}')


    if datnums is not None:
        print('works')
        md = dcc.Markdown(f'''
        Dat: {datnums}  
        ''')
        # f_values = [
        #     {'freq': f} for f in freqs
        # ]
        f_values = [
            {'freq': freqs}
        ]
        print(f'f_values = {f_values}')
        # table = dash_table.DataTable(data=f_values, columns=[{'freq': k} for k in ['freq']])
        table = dash_table.DataTable(data=f_values[0], columns=['freq'])

        return dbc.Card([
            dbc.CardHeader(dbc.Col(html.H3(f'Dat{datnums} Data Summary:'))),
            dbc.CardBody(table),
            ])
    return html.Div()


    ########## CHECK THIS OUT ##########


    # @dataclass
    # class ParInfo:
    #     key: str
    #     name: str
    #     units: str

    # if dat_id is not None:
    #     dat = get_dat_from_id(dat_id)
    #     infos: list[ParInfo] = []
    #     for k in dat.Data.keys:
    #         data = dat.Data.get_data(k)
    #         shape = data.shape
    #         min_ = np.nanmin(data)
    #         max_ = np.nanmax(data)
    #         infos.append(ParInfo(
    #             name=k,
    #             shape=shape,
    #             min=min_,
    #             max=max_,
    #         ))

    #     data_info = [
    #         {'Name': info.name, 'Shape': f'{info.shape}', 'Min': f'{info.min:.4f}', 'Max': f'{info.max:.4f}'}
    #         for info in infos
    #     ]
    #     table = dash_table.DataTable(data=data_info, columns=[{'name': k, 'id': k} for k in ['Name', 'Shape', 'Min', 'Max']])

    #     return dbc.Card([
    #         dbc.CardHeader(dbc.Col(html.H3(f'Dat{dat.datnum} Data Summary:'))),
    #         dbc.CardBody(table),
    #     ])
    # return html.Div()


    

def main_layout() -> Component:
    global main_components
    m = main_components
    layout_ = html.Div([
        m.graph1,
        m.graph2,
        m.noise_info_text
    ])
    return layout_


def sidebar_layout() -> Component:
    global sidebar_components
    s = sidebar_components
    layout_ = html.Div([
        s.dat_selector,
        label_component(s.specdropdown, 'PSD type:'),
        label_component(s.lindbdropdown, 'Scaling:'),
        label_component(s.psdtype_checklist, 'PS Check:'),
        label_component(s.integrate_check, 'Integrate:'),
        s.freq_selector,
        label_component(s.freq_width, 'Frequency Width:'),
        label_component(s.width_check, 'Show Width:'),
    ])
    return layout_


def layout():
    """Must return the full layout for multipage to work"""

    sidebar_layout_ = sidebar_layout()
    main_layout_ = main_layout()

    return html.Div([
        html.H1('Noise Analysis'),
        dbc.Row([
            dbc.Col(
                dbc.Card(sidebar_layout_),
                width=3
            ),
            dbc.Col(
                dbc.Card(main_layout_)
            )
        ]),
    ])


if __name__ == '__main__':
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = layout()
    cache.init_app(app.server)
    app.run_server(debug=True, port=8051)
else:
    dash.register_page(__name__)
