from __future__ import annotations
from typing import TYPE_CHECKING
import dash_bootstrap_components as dbc
from dash import html


if TYPE_CHECKING:
    from dash.development.base_component import Component


def label_component(component, label: str) -> Component:
    return dbc.Row(
        [
            dbc.Label(html.H4(label), html_for=label, width=4),
            dbc.Col(
                component,
                width=8
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
        dbc.Row(dbc.Col(html.H4(label), width=12)),
        dbc.Row(dbc.Col(component, width=12)),
    ],
        **kwargs
    )