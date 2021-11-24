import dash
from dash import html, dcc, callback, Input, Output
from flask_caching import Cache
import dash_bootstrap_components as dbc
import dash_extensions.snippets

from dash_extensions.enrich import ServersideOutput

# from dash_extensions.enrich import DashProxy, ServersideOutputTransform
# app = DashProxy(__name__,
#                 transforms=[ServersideOutputTransform()],
#                 external_stylesheets=[dbc.themes.BOOTSTRAP])


@callback(Output('div-text', 'children'), Input('button-click', 'n_clicks'))
def click(n_clicks):
    return n_clicks


def layout():
    button = dbc.Button('Click Me', id='button-click')
    div = html.Div(id='div-text')

    return html.Div([
        button,
        html.H1('Test Page'),
        div,
    ])


if __name__ == '__main__':
    import dash_bootstrap_components as dbc

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = layout()
    app.run_server(debug=True, port=8051)
else:
    dash.register_page(__name__)
