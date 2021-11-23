from dash import Dash
import dash_bootstrap_components as dbc
# import dash.html as html
import dash_html_components as html
# import dash.dcc as dcc
import dash_core_components as dcc
import dash.dependencies as dd

from dash_labs.plugins import FlexibleCallbacks
from dash_labs._callback import _callback as dx_callback

from dash_extensions.enrich import DashProxy, PrefixIdTransform

from dataclasses import dataclass
import urllib.parse
from typing import Callable, Optional, List
from types import MethodType
from functools import partial


CONTENT_ID = 'content'
URL_ID = 'url'


class MyFlexibleCallbacks(FlexibleCallbacks):
    """
    My override so that Dash labs callbacks can work with dash-extensions multipage
    (only change Dash.callback to type(app).callback)
    """
    def plug(self, app: Dash):
        # Instead of wrapping Dash.callback, wrap type(app) (e.g. in case it is a DashProxy from dash_extensions
        _wrapped_callback = type(app).callback  # type(app) so that I don't wrap a method which has 'self' as first arg
        app._wrapped_callback = _wrapped_callback
        app.callback = MethodType(
            partial(dx_callback, _wrapped_callback=_wrapped_callback), app
        )


def register_dash_proxy_callbacks_to_app(proxy_app: DashProxy, new_app: Dash):
    """Register all callbacks in a DashProxy to a Dash app"""
    callbacks, clientside_callbacks = list(proxy_app._resolve_callbacks())
    new_app = super() if new_app is None else new_app
    for callback in callbacks + clientside_callbacks:
        # Replacing this line from dash_extensions. Otherwise scalar returns end up wrapped in a list
        # outputs = callback[Output][0] if len(callback[Output]) == 1 else callback[Output]
        outputs = callback[dd.Output]
        new_app.callback(outputs, callback[dd.Input], callback[dd.State], **callback["kwargs"])(callback["f"])


def run_app(app: Dash, debug: bool = True, debug_port: int = 8050, real_port: int = None, threaded=True):
    """Handles running dash app with reasonable settings for debug or real"""
    if isinstance(app, DashProxy):  # turn into a dash.Dash app (otherwise single outputs get turned into lists when they shouldn't)
        real_app = Dash(__name__, plugins=[MyFlexibleCallbacks()])
        real_app.index_string = app.index_string  # Copy across any inline CSS style added by dash_labs plugins
        real_app.config.external_stylesheets = app.config.external_stylesheets  # Copy across external stylesheets
        register_dash_proxy_callbacks_to_app(app, real_app)  # Copy across callbacks
        real_app.layout = app._layout_value()  # Copy across layout
        app = real_app

    # Resizes things better based on actual device width rather than just pixels (good for mobile)
    meta_tags = [
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
    app.config.meta_tags = meta_tags
    if debug is False:
        assert isinstance(real_port, int)
        app.run_server(debug=False, port=real_port, host='0.0.0.0', threaded=threaded)
    else:
        app.run_server(debug=True, port=debug_port, threaded=threaded)


@dataclass
class PageInfo:
    """
    Mostly new pages are made as if they are a single page, then this class just ties things together so that a
    multipage app can more predictably interact with each page. I.e. it can expect to find a page name for example
    """
    page_name: str  # The name of the page which will show up in a Nav bar
    app_function: Callable  # Function which returns the complete app (and can be passed in an app to fill)
    page_id: Optional[str] = None  # Prefix to add to all interactive layout/callback components

    def __post_init__(self):
        self.page_id = self.page_id if self.page_id else self.page_name


@dataclass
class _ProcessedPage:
    """Used when generating a multipage app"""
    name: str
    layout: list


def make_multipage_app(module_pages: list,
                       app_name: str) -> Dash:
    """
    Make a multipage app from a list of module_pages (which each need to implement a PageInfo, see template page)

    Args:
        module_pages (): list of modules which are app pages
        app_name (): Name of multipage app to display in Navbar

    Returns:

    """
    multipage_app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
    pages = _modules_to_pages(module_pages, multipage_app)
    navbar = _make_navbar(pages, app_name)
    _page_select_callbacks(multipage_app, pages)
    multipage_app.layout = _multipage_layout(navbar)
    multipage_app.validation_layout = html.Div([multipage_app.layout, *[p.layout for p in pages]])
    return multipage_app


def _modules_to_pages(modules: list, app: Dash) -> List[_ProcessedPage]:
    pages = []
    for page in modules:
        page_info = getattr(page, 'page_info', None)
        if page_info is None or not isinstance(page_info, PageInfo):
            raise RuntimeError(f'Need to implement `page_info` for {page}')
        # TODO: Need to add more transforms if used in other pages, or maybe add that to PageInfo so as not to
        # TODO: load more transforms than necessary for each page
        page_app = page_info.app_function(DashProxy(__name__,
                                                    transforms=[PrefixIdTransform(page_info.page_id)],
                                                    plugins=[MyFlexibleCallbacks()])
                                          )
        pages.append(
            _ProcessedPage(
                name=page_info.page_name,
                layout=page_app._layout_value(),
            )
        )
        # Callbacks must be registered AFTER page_app._layout_value() to work with dl.Plugins.FlexibleCallbacks() where
        # component is made inside of app.callback(...), otherwise prefix is added twice!
        register_dash_proxy_callbacks_to_app(page_app, app)
    return pages


def _page_select_callbacks(app: Dash, pages: List[_ProcessedPage]):
    page_dict = {urllib.parse.quote_plus(p.name): p for p in pages}

    @app.callback(dd.Output(CONTENT_ID, 'children'), dd.Input(URL_ID, 'pathname'))
    def page_select(pathname):
        if pathname:
            pathname = pathname[1:]  # get rid of the first leading "/"
        if pathname in page_dict:
            return page_dict[pathname].layout
        else:
            return pages[0].layout


def _multipage_layout(navbar: dbc.NavbarSimple) -> dbc.Container:
    return dbc.Container(
        [
            dcc.Location(id=URL_ID),
            navbar,
            html.Div(id=CONTENT_ID),
        ], fluid=True
    )


def _make_navbar(pages: List[_ProcessedPage], app_name: str) -> dbc.NavbarSimple:
    navbar = dbc.NavbarSimple(
        [
            *[dbc.NavItem(dbc.NavLink(p.name, href=urllib.parse.quote_plus(p.name))) for p in pages]
        ], brand=app_name,
    )
    return navbar

