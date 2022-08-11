# from __future__ import annotations
from typing import TYPE_CHECKING
import dash_bootstrap_components as dbc
from dash import html


if TYPE_CHECKING:
    pass
from dash.development.base_component import Component


def label_component(component, label: str) -> dbc.Row:
    return dbc.Row(
        [
            dbc.Col(dbc.Label(html.H5(label), html_for=label)),
            dbc.Col(
                component,
            ),
        ],
        # className="mb-3",
    )


def vertical_label(label: str, component: Component, **kwargs) -> dbc.Col:
    """
    Add a label above a component wrapped in a column and two rows
    Args:
        label ():
        component ():

    Returns:

    """
    return dbc.Col([
        dbc.Row(dbc.Col(html.H5(label), width=12)),
        dbc.Row(dbc.Col(component, width=12)),
    ],
        **kwargs
    )