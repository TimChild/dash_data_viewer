from __future__ import annotations
from typing import Optional, Any

import dash_labs as dl
import dash_bootstrap_components as dbc
import dash_html_components as html

from dash_extensions.enrich import DashProxy
from src.multipage_util import MyFlexibleCallbacks, PageInfo

from dataclasses import dataclass

@dataclass
class _template:
    tpl: dl.templates.dbc.BaseDbcTemplate | Any
    loc_output: str
    loc_input: str


def make_app(app: Optional[DashProxy] = None):
    if app is None:
        app = DashProxy(transforms=[], plugins=[MyFlexibleCallbacks()])

    templates = dl.templates
    tpls = [
        _template(templates.DbcSidebar(app, title='DbcSidebar Title', sidebar_columns=4), 'main', 'sidebar'),
        _template(templates.DbcSidebarTabs(app, title='DbcSidebarTabs Title', tab_locations=['main', 'main_2'], sidebar_columns=3), 'main', 'sidebar'),
        _template(templates.DbcCard(app, title='DbcCard Title with 2 columns', columns=2), 'top', 'bottom'),
        _template(templates.HtmlCard(app, title='HtmlCard Title', width=None), 'top', 'bottom'),
        _template(templates.DbcRow(app, title='DbcRow Title height', left_cols=5), 'right', 'left')
    ]

    for i, temp in enumerate(tpls):
        temp.tpl.add_component(component=dbc.Button('Example button input'), location=temp.loc_input, label='Button Input Label')
        temp.tpl.add_component(component=html.H3('Example H3 output'), location=temp.loc_output, label='H3 Output label')

    app.layout = dbc.Container(
        children=[
            dbc.Row(dbc.Col([html.Div(children=temp.tpl.children), html.Hr()])) for temp in tpls
        ],
        # children=[
        #     html.Div(children=[*list(tpl.children), html.Hr()]) for tpl in tpls
        # ],
        fluid=True)

    return app


# Multipage app will look for this in order to add to multipage
page_info = PageInfo(
    page_name='Dash Labs Templates',  # The name which will show up in the NavBar
    app_function=make_app,  # This function should take a DashProxy instance
)

if __name__ == '__main__':
    page_app = make_app()
    page_app.run_server(debug=True, port=8060)
