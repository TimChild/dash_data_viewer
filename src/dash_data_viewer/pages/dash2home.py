from dash import html, dcc, callback, Input, Output


import dash
dash.register_page(__name__, path='/')


def layout():
    return html.H1('Home Page')