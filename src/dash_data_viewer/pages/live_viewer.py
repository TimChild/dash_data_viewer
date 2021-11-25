from __future__ import annotations
import dash
from dash import html, dcc, callback, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
from dash_data_viewer.cache import cache

from dat_analysis.plotting.plotly import OneD, TwoD
from dat_analysis.dat_object.make_dat import DatHandler, get_newest_datnum
import dat_analysis.useful_functions as u
from dat_analysis.analysis_tools.square_wave import get_setpoint_indexes_from_times
from dat_analysis.dat_object.attributes.entropy import IntegrationInfo

import logging
import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass
import json

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dash.development.base_component import Component
    from dat_analysis.dat_object.dat_hdf import DatHDF
    from dat_analysis.dat_object.attributes.square_entropy import Output

DH = DatHandler()  # This is a singleton, so any thread in the same process should get the same DatHandler() I think!


class MainComponents(object):
    """Convenient holder for any components that will end up in the main area of the page"""
    graph1 = dcc.Graph(id='live-graph-1')
    analysed_out = html.Div(id='live-div-anlaysed-out')  # Can be filled will multiple components however is useful
    dat_info_text = html.Div(id='live-div-dat-info-text')


class SidebarComponents(object):
    """Convenient holder for any components that will end up in the sidebar area of the page"""
    update = dcc.Interval(id='live-interval-update', interval=10000)
    newest_datnum = dcc.Store(id='live-store-newest-datnum')
    dat_id = dcc.Store(id='live-store-datid')
    data = dcc.Store(id='live-store-data')
    button_live = dbc.Button(id='live-button-live', children='Toggle Live Update', color='success')
    datnum = dbc.Input(id='live-input-datnum', type='number', placeholder='Datnum', debounce=True, persistence=True,
                       persistence_type='local')

# Initialize the components ONCE here.
main_components = MainComponents()
sidebar_components = SidebarComponents()


def initialize_dat_from_datnum(datnum: int) -> str:
    """This gets the dat the usual way. I.e. It initialized them if they aren't already open, or if they don't exist etc
    This should be used as sparingly as possible to avoid multiple threads all trying to write the same dat
    Returns:
        dat_id which can be used to get dat using get_dat_from_id
    """
    if datnum:
        global DH
        dat = DH.get_dat(datnum)
        return DH.get_open_dat_id(dat)
    raise ValueError(f'datnum = None is not valid, avoid calling this unless datnum exists')


def get_dat_from_id(id: str) -> DatHDF:
    """Getting a dat from here does no initialization, it just returns an already opened dat (i.e. should be safer to
    call from lots of callbacks simultaneously)"""
    if id and id in DH.open_dats:
        return DH.open_dats[id]
    raise FileNotFoundError(f'dat with id {id} not in DH.open_dats. Should it be?')


@cache.memoize()
def get_entropy_out(dat: DatHDF) -> Output:
    try:
        out = dat.SquareEntropy.get_Outputs(name='default')
    except u.NotFoundInHdfError:
        start, _ = get_setpoint_indexes_from_times(dat, 0.005)
        pp = dat.SquareEntropy.get_ProcessParams(setpoint_start=start)
        out = dat.SquareEntropy.get_Outputs(name='default', calculate_only=False, process_params=pp)
    return out


@callback(
    Output(sidebar_components.newest_datnum.id, 'data'),
    Output(sidebar_components.dat_id.id, 'data'),
    Output(sidebar_components.button_live.id, 'color'),
    Input(sidebar_components.update.id, 'n_intervals'),
    Input(sidebar_components.button_live.id, 'n_clicks'),
    Input(sidebar_components.datnum.id, 'value'),
    State(sidebar_components.newest_datnum.id, 'data'),
)
def update_newest_datnum(n_intervals, n_clicks, selected_datnum, latest_datnum: int):
    """Updates the newest datnum and dat_id if there has been a change"""
    n_clicks = n_clicks if n_clicks else 0
    live_active = False if n_clicks%2 else True

    new_datnum = dash.no_update
    dat_id = dash.no_update
    color = 'success' if live_active else 'danger'

    if live_active:
        new_datnum = get_newest_datnum()
        if new_datnum != latest_datnum:
            dat_id = initialize_dat_from_datnum(new_datnum)
    else:
        if selected_datnum:
            new_datnum = selected_datnum
            if selected_datnum != latest_datnum:
                dat_id = initialize_dat_from_datnum(selected_datnum)
    return new_datnum, dat_id, color


@callback(
    Output(main_components.graph1.id, 'figure'),
    Input(sidebar_components.dat_id.id, 'data')
)
def update_graph1(dat_id: str):
    if dat_id is not None:
        dat = get_dat_from_id(dat_id)
        data = dat.Data.i_sense
        if data.ndim == 1:
            p1d = OneD(dat=dat)
            fig = p1d.plot(data)
        elif data.ndim == 2:
            p2d = TwoD(dat=dat)
            fig = p2d.plot(data)
        else:
            logging.warning(f'Dat{dat.datnum} found, but data.shape {data.shape} not right')
            fig = go.Figure()
        fig.update_layout(title=f'Dat{dat.datnum}: Time completed = {dat.Logs.time_completed}')
        return fig
    return go.Figure()


@callback(
    Output(main_components.analysed_out.id, 'children'),
    Input(sidebar_components.dat_id.id, 'data')
)
def update_analysed_out(dat_id: str) -> Component:
    if dat_id:
        dat = get_dat_from_id(dat_id)
        p1d = OneD(dat=dat)
        if 'square entropy' in [v.strip() for v in dat.Logs.comments.split(',')]:
            entropy_out = get_entropy_out(dat)

            # Entropy signal and fit
            fig = p1d.plot(data=entropy_out.average_entropy_signal, x=entropy_out.x, ylabel='DeltaI /nA',
                           trace_name='Averaged Data', title=f'Dat{dat.datnum}: Average Entropy Signal',
                           mode='markers+lines', trace_kwargs=dict(
                    marker=dict(size=5, symbol='cross-thin'), line=dict(width=2)
                ))
            fit = dat.Entropy.get_fit(calculate_only=True, x=entropy_out.x, data=entropy_out.average_entropy_signal)
            fig.add_trace(p1d.trace(data=fit.eval_fit(x=entropy_out.x), x=entropy_out.x, mode='lines',
                                    name=f'Fit: dS={fit.best_values.dS:.2f}kB'))

            # Integrated entropy
            try:
                int_info = dat.Entropy.get_integration_info('default')
            except u.NotFoundInHdfError:
                cold_fit = dat.SquareEntropy.get_fit(calculate_only=True, x=entropy_out.x, data=entropy_out.averaged,
                                                     transition_part='cold')
                hot_fit = dat.SquareEntropy.get_fit(calculate_only=True, x=entropy_out.x, data=entropy_out.averaged,
                                                     transition_part='hot')
                dT = hot_fit.best_values.theta - cold_fit.best_values.theta
                int_info = dat.Entropy.set_integration_info(dT=dT, amp=cold_fit.best_values.amp, name='default',
                                                            overwrite=True)
            integrated = int_info.integrate(entropy_out.average_entropy_signal)
            fig2 = p1d.figure(title=f'Dat{dat.datnum}: Integrated Entropy (dS={integrated[-1]:.2f}kB, dT={int_info.dT:.2f}mV)',
                              ylabel='Entropy /kB')
            fig2.add_trace(p1d.trace(x=entropy_out.x, data=integrated, mode='lines'))
            p1d.add_line(fig2, np.log(2), linetype='dash')
            p1d.add_line(fig2, np.log(3), linetype='dash')
            p1d.add_line(fig2, 0, linetype='solid')

            layout = dbc.Row([
                dbc.Col(
                    dcc.Graph(figure=fig)
                ),
                dbc.Col(
                    dcc.Graph(figure=fig2)
                )
            ])

            return layout
    return html.Div(f'No additional analysis to show')


@callback(
    Output(main_components.dat_info_text.id, 'children'),
    Input(sidebar_components.dat_id.id, 'data'),
)
def update_dat_info(dat_id: str) -> Component:
    """Returns general information about dat"""
    if dat_id is not None:
        dat = get_dat_from_id(dat_id)
        try:
            mag = dat.Logs.mags.magy
            field = mag.field
        except u.NotFoundInHdfError:
            field = 0

        md = dcc.Markdown(f'''
        ### Dat{dat.datnum}:
        Comments: {dat.Logs.comments}  
        Time elapsed: {dat.Logs.time_elapsed}/s  
        Time completed: {dat.Logs.time_completed}  
        Sweeprate: {dat.Logs.sweeprate:.1f}mV/s  
        Measure Freq: {dat.Logs.measure_freq}Hz  
        Mag Field (Y): {field}mT
        ''')
        dac_values = [
            {'DAC': k, 'Value': v} for k, v in dat.Logs.dacs.items()
        ]
        table = dash_table.DataTable(data=dac_values, columns=[{'name': k, 'id': k} for k in ['DAC', 'Value']])

        return dbc.Container([
            dbc.Row([
                dbc.Col(md), dbc.Col(table, width=4)
            ])
        ])
    return html.Div()


def label_component(component, label: str) -> Component:
    return dbc.Row(
        [
            dbc.Label(label, html_for=label, width=4),
            dbc.Col(
                component,
                width=8
            ),
        ],
        # className="mb-3",
    )


def main_layout() -> Component:
    global main_components
    m = main_components
    layout_ = html.Div([
        m.graph1,
        m.dat_info_text,
        m.analysed_out,
    ])
    return layout_


def sidebar_layout() -> Component:
    global sidebar_components
    s = sidebar_components
    layout_ = html.Div([
        s.update,
        s.newest_datnum,  # Invisible
        s.dat_id,  # Invisible
        s.button_live,
        label_component(s.datnum, 'Datnum:'),
    ])
    return layout_


def layout():
    """Must return the full layout for multipage to work"""

    sidebar_layout_ = sidebar_layout()
    main_layout_ = main_layout()

    return html.Div([
        html.H1('Live Viewer'),
        dbc.Row([
            dbc.Col(
                dbc.Card(sidebar_layout_),
                width=3
            ),
            dbc.Col(
                dbc.Card(main_layout_),
                width=9
            )
        ]),
    ])


if __name__ == '__main__':
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    cache.init_app(app.server)
    app.layout = layout()
    app.run_server(debug=True, port=8051)
else:
    dash.register_page(__name__)
