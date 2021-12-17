from __future__ import annotations
import dash
import lmfit as lm
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
from dataclasses import dataclass

from dash_data_viewer.layout_util import label_component
from dash_data_viewer.cache import cache
from dash_data_viewer.transition_report import TransitionReport

from dat_analysis.dat_object.make_dat import get_dat, get_dats
from dat_analysis.plotting.plotly import OneD, TwoD
from dat_analysis.analysis_tools.general_fitting import FitInfo
from dat_analysis.useful_functions import mean_data

# from dash_data_viewer.layout_util import label_component

from typing import TYPE_CHECKING, Tuple, Optional, List

if TYPE_CHECKING:
    from dash.development.base_component import Component


class MainComponents(object):
    """Convenient holder for any components that will end up in the main area of the page"""
    graph_2d_transition = dcc.Graph(id='transition-graph-2d-transition')
    graph_average_transition = dcc.Graph(id='transition-graph-average-transition')

    div_row_fit_graphs = html.Div(id='transition-div-perRowFitGraphs')


class SidebarComponents(object):
    """Convenient holder for any components that will end up in the sidebar area of the page"""
    input_datnum = dbc.Input(id='transition-input-datnum', type='number', placeholder='Datnum', debounce=True,
                             persistence=True, persistence_type='local')
    toggle_centering = dbc.Checklist(id='transition-toggle-centering', switch=True,
                                     persistence=True, persistence_type='local',
                                     options=[
                                         {'label': "", 'value': True}
                                     ],
                                     value=[True])
    dropdown_2dtype = dcc.Dropdown(id='transition-dropdown-2dtype',
                                   options=[
                                       {'label': 'Heatmap', 'value': 'heatmap'},
                                       {'label': 'Waterfall', 'value': 'waterfall'},
                                   ],
                                   value='heatmap')


# Initialize the components ONCE here.
main_components = MainComponents()
sidebar_components = SidebarComponents()


@dataclass
class TransitionData:
    x: np.ndarray
    y: Optional[np.ndarray]
    data: np.ndarray


@dataclass
class RowFitInfo:
    params: List[lm.Parameters]
    centers: List[float]


# Functions and Callbacks specific to this page

# @cache.memoize()
def get_transition_data(datnum: int) -> Optional[TransitionData]:
    tdata = None
    if datnum:
        dat = get_dat(datnum)
        if dat:
            x = dat.Data.get_data('x')
            data = dat.Data.get_data('i_sense')
            if 'y' in dat.Data.keys:
                y = dat.Data.get_data('y')
            else:
                y = None
            tdata = TransitionData(x=x, y=y, data=data)
    return tdata


# @cache.memoize()
def get_transition_row_fits(datnum: int) -> Optional[RowFitInfo]:
    tdata = get_transition_data(datnum)
    row_fits = None
    if tdata:
        dat = get_dat(datnum)
        if dat:
            if tdata.data.ndim == 1:
                row_fits = [dat.Transition.get_fit(which='row', row=0, x=tdata.x, data=tdata.data, calculate_only=True)]
            elif tdata.data.ndim == 2:
                row_fits = [
                    dat.Transition.get_fit(which='row', row=i, x=tdata.x, data=tdata.data[i], calculate_only=True) for i
                    in range(tdata.data.shape[0])]
    if row_fits:
        row_fit_info = RowFitInfo(
            params=[f.params for f in row_fits],
            centers=[f.best_values.mid for f in row_fits]
        )
    else:
        row_fit_info = None
    return row_fit_info


@callback(
    Output(main_components.graph_2d_transition.id, 'figure'),
    Input(sidebar_components.input_datnum.id, 'value'),
    Input(sidebar_components.dropdown_2dtype.id, 'value'),
)
def graph_transition_data(datnum: int, graph_type: str) -> go.Figure:
    """
    Either 1D or 2D transition data just as it was recorded (after resampling)
    Returns:

    """
    tdata = get_transition_data(datnum)
    if tdata:
        dat = get_dat(datnum)
        if dat:
            if tdata.y is not None and tdata.data.ndim == 2:
                p = TwoD(dat=dat)
                fig = p.figure()
                fig.add_traces(p.trace(data=tdata.data, x=tdata.x, y=tdata.y, trace_type=graph_type))
            elif tdata.data.ndim == 1:
                p = OneD(dat=dat)
                fig = p.figure(ylabel='Current /nA')
                fig.add_trace(p.trace(data=tdata.data, x=tdata.x, mode='markers+lines'))
            else:
                fig = go.Figure()

            fig.update_layout(title=f'Dat{dat.datnum}: Averaged Transition Data')
            return fig
    return go.Figure()


@callback(
    Output(main_components.graph_average_transition.id, 'figure'),
    Input(sidebar_components.input_datnum.id, 'value'),
    Input(sidebar_components.toggle_centering.id, 'value'),
)
def average_transition_data(datnum: int, centered: int) -> go.Figure:
    tdata = get_transition_data(datnum)
    if tdata:
        dat = get_dat(datnum)
        if dat:
            if tdata.data.ndim == 1:
                return graph_transition_data(datnum, 'not used')
            elif tdata.y is not None and tdata.data.ndim == 2:
                if centered:
                    row_fits = get_transition_row_fits(datnum)
                    centers = row_fits.centers
                else:
                    centers = [0] * tdata.data.shape[0]
                avg_data, avg_x, avg_std = mean_data(x=tdata.x, data=tdata.data, centers=centers, return_x=True,
                                                     return_std=True)

                p = OneD(dat=dat)
                fig = p.figure(ylabel='Current /nA')
                fig.add_trace(p.trace(data=avg_data, x=avg_x, mode='markers+lines'))
            else:
                fig = go.Figure()

            fig.update_layout(title=f'Dat{dat.datnum}: Transition Data')
            return fig
    return go.Figure()


@callback(
    Output(main_components.div_row_fit_graphs.id, 'children'),
    Input(sidebar_components.input_datnum.id, 'value'),
)
def plot_per_row_fit_params(datnum: int) -> Component:
    @dataclass
    class ParInfo:
        key: str
        name: str
        units: str

    if datnum:
        dat = get_dat(datnum)
        row_fits = get_transition_row_fits(datnum)

        p1d = OneD(dat=dat)
        graphs = []
        for par_info in [
            ParInfo(key='mid', name='center', units='mV'),
        ]:
            if par_info.key in row_fits.params[0].keys():
                fig = p1d.figure(title=f'Dat{dat.datnum}: {par_info.name} values',
                                 xlabel="Row num",
                                 ylabel=f'{par_info.name} /{par_info.units}')
                key_params = [params[par_info.key] for params in row_fits.params]
                fig.add_trace(p1d.trace(
                    x=dat.Data.y,
                    data=[p.value for p in key_params],
                    # data_err=[p.stderr for p in key_params],
                    mode='markers+lines',
                ))
                graphs.append(fig)
            else:
                pass
        return html.Div([dcc.Graph(figure=fig) for fig in graphs])
    return html.Div('No data to show')


def main_layout() -> Component:
    global main_components
    m = main_components
    transition_graphs = dbc.Row([
        dbc.Col(m.graph_2d_transition, width=6), dbc.Col(m.graph_average_transition, width=6)
    ])
    transition_info = dbc.Row([
        html.Div(f'No info to show yet')
    ])
    layout_ = html.Div([
        transition_graphs,
        transition_info,
        m.div_row_fit_graphs,
    ])
    return layout_


def sidebar_layout() -> Component:
    global sidebar_components
    s = sidebar_components
    layout_ = html.Div([
        label_component(s.input_datnum, 'Datnum:'),
        label_component(s.dropdown_2dtype, 'Graph Type: '),
        label_component(s.toggle_centering, 'Center first: ')
    ])
    return layout_


def layout():
    """Must return the full layout for multipage to work"""
    sidebar_layout_ = sidebar_layout()
    main_layout_ = main_layout()

    return html.Div([
        html.H1('Transition'),
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
    app.run_server(debug=True, port=8053)
else:
    dash.register_page(__name__)
    pass
