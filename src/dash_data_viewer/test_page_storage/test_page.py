from __future__ import annotations
import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dash.development.base_component import Component


class MainComponents(object):
    """Convenient holder for any components that will end up in the main area of the page"""
    div = html.Div(id='div-text')


class SidebarComponents(object):
    """Convenient holder for any components that will end up in the sidebar area of the page"""
    button = dbc.Button('Click Me', id='button-click')


# Initialize the components ONCE here.
main_components = MainComponents()
sidebar_components = SidebarComponents()


@callback(Output(main_components.div.id, 'children'), Input(sidebar_components.button.id, 'n_clicks'))
def click(n_clicks):
    return n_clicks


def main_layout() -> Component:
    global main_components
    m = main_components
    layout_ = html.Div([
        m.div,
    ])
    return layout_


def sidebar_layout() -> Component:
    global sidebar_components
    s = sidebar_components
    layout_ = html.Div([
        s.button,
    ])
    return layout_


def layout():
    """Must return the full layout for multipage to work"""

    sidebar_layout_ = sidebar_layout()
    main_layout_ = main_layout()

    return html.Div([
        html.H1('Test Page'),
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
    app.run_server(debug=True, port=8051)
else:
    dash.register_page(__name__)
