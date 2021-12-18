from __future__ import annotations
import dash
import dash_extensions.snippets
from dash import html, dcc, callback, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
from dash_data_viewer.cache import cache

from dat_analysis.plotting.plotly import OneD, TwoD
from dat_analysis.dat_object.make_dat import DatHandler, get_newest_datnum, get_dat, get_dats
import dat_analysis.useful_functions as u

import logging
import numpy as np
import plotly.graph_objects as go

from typing import TYPE_CHECKING, Tuple

from dash_data_viewer.entropy_report import EntropyReport
from dash_data_viewer.transition_report import TransitionReport
from dash_data_viewer.layout_util import label_component
# from dash_data_viewer.layout_util import label_component
from dash_data_viewer.cache import cache

if TYPE_CHECKING:
    from dash.development.base_component import Component
    from dat_analysis.dat_object.dat_hdf import DatHDF
    from dat_analysis.dat_object.attributes.square_entropy import Output

from scipy import signal
from scipy.fft import rfft, rfftfreq
from scipy.integrate import cumulative_trapezoid


class MainComponents(object):
    """Convenient holder for any components that will end up in the main area of the page"""
    div = html.Div(id='noise-div-text')
    graph = dcc.Graph(id='noise-graph-graph1')
    graph2 = dcc.Graph(id='noise-graph-graph2')


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
    # x = np.arange(0, len(y), dt)

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

# Initialize the components ONCE here.
main_components = MainComponents()
sidebar_components = SidebarComponents()

# @cache.memoize()
def get_data(datnum) -> Tuple[np.darray, np.ndarray]:
    print(f'getting data for dat{datnum}')
    data = None
    x = None
    if datnum:
        dat = get_dat(datnum)
        x = dat.Data.get_data('x')
        if 'Experiment Copy/current' in dat.Data.keys:
            data = dat.Data.get_data('Experiment Copy/current')
        elif 'Experiment Copy/cscurrent' in dat.Data.keys:
            data = dat.Data.get_data('Experiment Copy/cscurrent')
        elif 'Experiment Copy/current_2d' in dat.Data.keys:
            data = dat.Data.get_data('Experiment Copy/current_2d')[0]
        elif 'Experiment Copy/cscurrent_2d' in dat.Data.keys:
            data = dat.Data.get_data('Experiment Copy/cscurrent_2d')[0]
        else:
            print(f'No data found. valid keys are {dat.Data.keys}')
    return x, data



@callback(Output(main_components.graph.id, 'figure'), Input(sidebar_components.datnum.id, 'value'))
def update_graph(datnum) -> go.Figure:
    x, data = get_data(datnum)
    if x is not None and data is not None:
        p1d = OneD(dat=None)
        fig = p1d.figure(title=f'Test')
        fig.add_trace(p1d.trace(x=x, data=data))
        
        return fig
    return go.Figure()

@callback(Output(main_components.graph2.id, 'figure'), 
          Input(sidebar_components.datnum.id, 'value'), 
          Input(sidebar_components.datnum_final.id, 'value'),
          Input(sidebar_components.datnum_choice.id, 'value'),
          Input(sidebar_components.specdropdown.id, 'value'),
          Input(sidebar_components.lindbdropdown.id, 'value'),
          Input(sidebar_components.psdtype_checklist.id, 'value'))
def update_graph(datnum, datnum_final, datnum_choice, specdropdown, lineardb, ps_check) -> go.Figure:

    dat_vals = []
    if datnum_final == None:
        datnum_final = datnum + 1
        datnum_choice = 1

    for dat in np.arange(datnum, datnum_final, datnum_choice):
        x, data = get_data(dat)
        if  x is not None and data is not None:
            dat_vals.append(dat)


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
    if len(dat_vals) == 0:
        return go.Figure()
    else:
        for dat_val in dat_vals:
            print(f'Getting dat {dat_val}')

            x, data = get_data(dat_val)
            dat = get_dat(dat_val)
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



def main_layout() -> Component:
    global main_components
    m = main_components
    layout_ = html.Div([
        m.graph,
        m.graph2,
    ])
    return layout_


def sidebar_layout() -> Component:
    global sidebar_components
    s = sidebar_components
    layout_ = html.Div([
        s.button,
        label_component(s.datnum, 'Datnum:'),
        label_component(s.datnum_final, 'Datnum Final:'),
        label_component(s.datnum_choice, 'For ith Dat:'),
        label_component(s.specdropdown, 'PSD type:'),
        label_component(s.lindbdropdown, 'Scaling:'),
       label_component(s.psdtype_checklist, 'PS Check:')
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
