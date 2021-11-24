import dash
import dash_bootstrap_components as dbc
import pages_plugin

from dash_extensions.enrich import DashProxy, ServersideOutputTransform

app = dash.Dash(__name__, plugins=[pages_plugin], external_stylesheets=[dbc.themes.BOOTSTRAP])
# app = DashProxy(__name__,
#                 plugins=[pages_plugin],
#                 transforms=[ServersideOutputTransform()],
#                 external_stylesheets=[dbc.themes.BOOTSTRAP])


app.layout = dbc.Container([
    dbc.NavbarSimple([
        dbc.NavItem(dbc.NavLink(page['name'], href=page['path']))
        for page in dash.page_registry.values()  # page_registry added to dash in pages_plugin
        if page['module'] != 'pages.not_found_404'
    ]),
    pages_plugin.page_container,
], fluid=True)


if __name__ == '__main__':
    app.run_server(debug=True, port=8050)