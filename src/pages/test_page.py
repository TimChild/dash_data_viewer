import dash
import dash_html_components as html
# import dash.html as html
import dash_core_components as dcc
# import dash.dcc as dcc
import dash_bootstrap_components as dbc

import dash_labs as dl

from dash_extensions.enrich import DashProxy

from src.multipage_util import PageInfo, MyFlexibleCallbacks

from dataclasses import dataclass
import numpy as np
import plotly.graph_objects as go


@dataclass
class Info:
    x: np.ndarray
    y: np.ndarray
    data: np.ndarray
    x_label: str
    y_label: str


def get_info():
    x = np.linspace(0, 100, 100)
    y = np.linspace(0, 10, 20)
    data = np.random.random((x.shape[0], y.shape[0]))
    return Info(x, y, data=data, x_label='X /mV', y_label='Y /mV')


def get_figure():
    info = get_info()
    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=info.x, y=info.y, z=info.data))
    return fig


def make_app(app: dash.Dash = None) -> dash.Dash:
    if app is None:
        app = DashProxy(transforms=[], plugins=[MyFlexibleCallbacks()])

    tpl = dl.templates.DbcSidebarTabs(app, ['First', 'Second'], title='Test Page', sidebar_columns=2)
    graph = dcc.Graph(id='graph1', figure=get_figure())
    button = dbc.Button('Click Me')

    tpl.add_component(graph, 'First', component_property='figure')
    tpl.add_component(button, 'sidebar', component_property='n_clicks')

    @app.callback(dl.Output(graph, 'figure'), dl.Input(button, 'n_clicks'))
    def get_1d_figure(clicks):
        if clicks:
            x = np.linspace(0, 100, 200)
            y = np.sin(x*clicks)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y))
            return fig
        return go.Figure()

    datnum_input = dbc.Input('datnum', debounce=True, type='number', placeholder='Datnum', persistence=True, persistence_type='local')
    wavename_input = dbc.Input('wavenames', debounce=True, type='text', placeholder='wavename')
    # graph2 = dcc.Graph('graph2')
    graph2 = html.H2('graph2')

    @app.callback(dl.Output(graph2, 'children'), dl.Input(datnum_input), dl.Input(wavename_input))
    def get_data(datnum: int, wavename: str):
        if datnum:
            return f'Datnum:{datnum}, wavename:{wavename}'
        return f'Nothing selected'

    tpl.add_component(datnum_input, 'sidebar', 'Datnum')
    tpl.add_component(wavename_input, 'sidebar', 'Wave name')
    tpl.add_component(graph2, 'Second', 'Second Graph')

    app.layout = html.Div(
        tpl.children
    )
    return app


page_info = PageInfo(page_name='Test Page', app_function=make_app)


if __name__ == '__main__':
    from src.multipage_util import run_app
    run_app(make_app(), debug=True, debug_port=8059)