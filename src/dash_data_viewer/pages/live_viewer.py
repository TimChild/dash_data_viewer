from __future__ import annotations
import dash
import dash_html_components as html
# import dash.html as html
import dash_core_components as dcc
# import dash.dcc as dcc
import dash_labs as dl
from dash_extensions.enrich import DashProxy, ServersideOutput, ServersideOutputTransform

from dash_data_viewer.multipage_util import PageInfo, MyFlexibleCallbacks

#######
from typing import TYPE_CHECKING
import plotly.graph_objects as go
from dat_analysis.dat_object.make_dat import get_dat, DatHandler
from dat_analysis.plotting.plotly import OneD, TwoD

if TYPE_CHECKING:
    from dat_analysis.dat_object.dat_hdf import DatHDF


last_datnum = 400


def full_id(dat) -> str:
    if dat:
        for k, v in DatHandler().open_dats.items():
            if dat == v:
                return k
    return None


def get_last_dat(interval) -> str:
    global last_datnum
    dh = DatHandler()
    dat = None
    for i in range(last_datnum, 9000):
        try:
            dat = get_dat(i)
            if dat:
                dh.remove(dat)
        except ValueError:
            dh.clear_dats()
            dat = get_dat(i-1)
            last_datnum = i-1
            break
    return full_id(dat)


def plot_i_sense(dat_id: str):
    dat = DatHandler().open_dats[dat_id]
    data = dat.Data.i_sense
    if data.ndim == 1:
        p1d = OneD(dat=dat)
        fig = p1d.plot(data)
    elif data.ndim == 2:
        p2d = TwoD(dat=dat)
        fig = p2d.plot(data)
    else:
        fig = go.Figure()
    fig.update_layout(title=f'Dat{dat.datnum}: Time completed = {dat.Logs.time_completed}')
    return fig


def test_interval(n_intervals):
    return n_intervals


def make_app(app: dash.Dash = None) -> dash.Dash:
    if app is None:
        app = DashProxy(transforms=[ServersideOutputTransform()], plugins=[MyFlexibleCallbacks()])

    tpl = dl.templates.DbcSidebarTabs(app, ['First', 'Second'], title='TEMPLATE')

    dat_store = dcc.Store('dat-store')
    tpl.add_component(dat_store, 'sidebar')  # Doesn't show up, just has to be in layout somewhere

    timer = dcc.Interval('interval-update-dat', interval=10000)  # To trigger self updating every 1s
    tpl.add_component(timer, 'sidebar')  # Doesn't show up

    graph = dcc.Graph('graph-main')
    tpl.add_component(graph, 'First')

    app.callback(dl.Output(dat_store, 'data'), dl.Input(timer, 'n_intervals'))(get_last_dat)
    # app.callback(dl.Output(dat_store, 'data'), dl.Input(timer, 'n_intervals'))(get_last_dat)
    app.callback(dl.Output(graph, 'figure'), dl.Input(dat_store, 'data'))(plot_i_sense)

    app.layout = html.Div(
        tpl.children
    )

    return app


page_info = PageInfo(page_name='Live Viewer', app_function=make_app)


if __name__ == '__main__':
    from dash_data_viewer.multipage_util import run_app
    run_app(make_app(), debug=True, debug_port=8050)
