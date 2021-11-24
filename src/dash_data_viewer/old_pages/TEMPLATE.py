import dash
import dash_html_components as html
# import dash.html as html
# import dash.dcc as dcc

import dash_labs as dl

from dash_extensions.enrich import DashProxy

from src.dash_data_viewer.multipage_util import PageInfo, MyFlexibleCallbacks


def make_app(app: dash.Dash = None) -> dash.Dash:
    if app is None:
        app = DashProxy(transforms=[], plugins=[MyFlexibleCallbacks()])

    app.layout = html.Div(
        dl.templates.DbcSidebarTabs(app, ['First', 'Second'], title='TEMPLATE').children
    )
    return app


page_info = PageInfo(page_name='TEMPLATE', app_function=make_app)


if __name__ == '__main__':
    from src.dash_data_viewer.multipage_util import run_app
    run_app(make_app(), debug=True, debug_port=8050)