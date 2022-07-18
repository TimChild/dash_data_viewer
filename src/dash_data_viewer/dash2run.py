import logging
logging.basicConfig(level=logging.INFO, force=True, format=f'%(levelname)s:%(module)s:%(lineno)d:%(funcName)s:%(message)s')

import dash
import dash_bootstrap_components as dbc
from dash_data_viewer.cache import cache
import argparse

app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.NavbarSimple([
        dbc.NavItem(dbc.NavLink(page['name'], href=page['path']))
        for page in dash.page_registry.values()  # page_registry added to dash in pages_plugin
        if page['module'] != 'pages.not_found_404'
                     ]),
    dash.page_container
], fluid=True)


if __name__ == '__main__':
    from dat_analysis.useful_functions import set_default_logging
    set_default_logging()

    cache.init_app(app.server)

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', action=argparse.BooleanOptionalAction)
    parser.set_defaults(remote=False)
    args = parser.parse_args()
    print(args.r)

    if args.r:
        app.run_server(debug=False, port=9050, threaded=True, host='0.0.0.0')
    else:
        app.run_server(debug=True, port=8052, threaded=True, dev_tools_hot_reload=False, use_reloader=False)