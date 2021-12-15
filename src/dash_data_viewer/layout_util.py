from __future__ import annotations
from typing import TYPE_CHECKING
import dash_bootstrap_components as dbc



if TYPE_CHECKING:
    from dash.development.base_component import Component

def label_component(component, label: str) -> Component:
    return dbc.Row(
        [
            dbc.Label(label, html_for=label, width=4),
            dbc.Col(
                component,
                width=8
            ),
        ],
        # className="mb-3",
    )