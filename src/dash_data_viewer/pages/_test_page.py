from dash import html, dash
import dash_bootstrap_components as dbc


box = html.Div(style={'background-color': 'red', 'width': '200px', 'height': '200px', 'border': '5px dotted black'})
boxes = [box]*10


def row_col():
    return dbc.Row([
        dbc.Col(box) for box in boxes
    ], style={'overflow-x': 'scroll'})


def flex_container():
    return dbc.Container([
        dbc.Container(box, style={'display': 'flex'}) for box in boxes
    ], fluid=True, style={'overflow-x': 'scroll', 'display': 'flex'})

def test():
    return dbc.Container([
        dbc.Row([
            dbc.Col(box) for box in boxes
            ],  style={'display': 'flex'})
    ], fluid=True, style={'overflow-x': 'scroll', 'display': 'flex'})


layout = dbc.Container(children=flex_container(),
                       fluid=True)

if __name__ == '__main__':
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    # app.layout = layout()
    app.layout = layout
    app.run_server(debug=True, port=8051, dev_tools_hot_reload=True, use_reloader=True)
