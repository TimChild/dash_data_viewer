from __future__ import annotations
import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc

from dash_data_viewer.layout_util import label_component

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dash.development.base_component import Component


class MainComponents(object):
    """Convenient holder for any components that will end up in the main area of the page"""
    graph_2d_transition = dcc.Graph(id='entropy-graph-2d-transition')
    graph_average_transition = dcc.Graph(id='entropy-graph-average-transition')

    graph_2d_entropy = dcc.Graph(id='entropy-graph-2d-entropy')
    graph_average_entropy = dcc.Graph(id='entropy-graph-average-entropy')

    graph_square_wave = dcc.Graph(id='entropy-graph-squarewave')
    graph_integrated = dcc.Graph(id='entropy-graph-integrated')


class SidebarComponents(object):
    """Convenient holder for any components that will end up in the sidebar area of the page"""
    input_datnum = dbc.Input(id='entropy-input-datnum', type='number', placeholder='Datnum', debounce=True,
                             persistence=True, persistence_type='local')
    toggle_centering = dbc.Checklist(id='entropy-toggle-centering', switch=True,
                                     persistence=True, persistence_type='local')
    dropdown_squarewave_mode = dcc.Dropdown(
        id='entropy-dropdown-squarewavemode')  # I.e. Average all, near peak, near dip, selected range?
    slider_squarewave_range = dcc.RangeSlider(
        id='entropy-slider-squarewaverange')  # If using range mode, then this selects range

    input_squarewave_start = dbc.Input(id='entropy-input-squarewavestart', type='number', value=0.0, debounce=True,
                                       persistence=True, persistence_type='local')

    dropdown_integration_mode = dcc.Dropdown(id='entropy-dropdown-integration-mode')  # I.e. Calculate dT from hot-cold, use fixed values, extrapolate from linear

    input_integration_zero = dbc.Input(id='entropy-input-integration-zero')  # Where should zero be defined for integrated entropy



# Initialize the components ONCE here.
main_components = MainComponents()
sidebar_components = SidebarComponents()


def main_layout() -> Component:
    global main_components
    m = main_components
    transition_graphs = dbc.Row([
        dbc.Col(m.graph_2d_transition, width=6), dbc.Col(m.graph_average_transition, width=6)
    ])
    entropy_graphs = dbc.Row([
        dbc.Row([dbc.Col(m.graph_2d_entropy, width=6), dbc.Col(m.graph_average_entropy, width=6)]),
        dbc.Row([dbc.Col(m.graph_integrated, width=6), dbc.Col(m.graph_square_wave, width=6)]),
    ])
    entropy_info = dbc.Row([
        html.Div(f'No info to show yet')
    ])
    layout_ = html.Div([
        transition_graphs,
        entropy_graphs,
        entropy_info,
    ])
    return layout_


def sidebar_layout() -> Component:
    global sidebar_components
    s = sidebar_components
    square_processing = dbc.Row()
    layout_ = html.Div([
        label_component(s.input_datnum, 'Datnum:'),

    ])
    return layout_


def layout():
    """Must return the full layout for multipage to work"""
    sidebar_layout_ = sidebar_layout()
    main_layout_ = main_layout()

    return html.Div([
        html.H1('Entropy'),
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
    app.run_server(debug=True, port=8052)
else:
    dash.register_page(__name__)
