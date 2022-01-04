from dash import Input, Output, State, dcc, html, callback, MATCH, ALL, ALLSMALLER, dash
from dash_extensions.snippets import get_triggered
from typing import Optional, List
import uuid
import dash_bootstrap_components as dbc
import logging

from .layout_util import vertical_label


class DatnumPickerAIO(html.Div):
    """
    A group of buttons with custom text where one can be selected at a time and returns either the text or a specific
    value

    Examples:
        picker = DatnumPickerAIO()  # Requires no arguments to instantiate (but can take aio_id)

        @callback(Input(picker.dd_id, 'value'))  # Selected Datnums are accessible from the dd.value
        def foo(datnums: list):
            return datnums

        app.layout = picker  # picker is a self contained layout

    """

    # Functions to create pattern-matching callbacks of the subcomponents
    class ids:
        @staticmethod
        def input(aio_id, key):
            return {
                'component': 'DatnumPickerAIO',
                'subcomponent': f'input',
                'key': key,
                'aio_id': aio_id,
            }

        @staticmethod
        def button(aio_id, type):
            return {
                'component': 'DatnumPickerAIO',
                'subcomponent': f'button',
                'key': type,
                'aio_id': aio_id,
            }

        @staticmethod
        def dropdown(aio_id: str):
            return {
                'component': 'DatnumPickerAIO',
                'subcomponent': f'dropdown',
                'aio_id': aio_id,
            }

    # Make the ids class a public class
    ids = ids

    def __init__(self, aio_id=None, allow_multiple=True):
        if aio_id is None:
            aio_id = str(uuid.uuid4())

        multi = 'true' if allow_multiple else 'false'

        self.dd_id = self.ids.dropdown(aio_id)  # For easy access to this component ('value' contains selected dats)

        input_infos = {
            'start': dict(component=None, label='Start', placeholder='Start'),
            'stop': dict(component=None, label='Stop', placeholder='Stop'),
            'step': dict(component=None, label='Step', placeholder='Step'),
        }

        button_infos = {
            'add': dict(component=None, text='Add'),
            'remove': dict(component=None, text='Remove'),
        }

        for key, info in input_infos.items():
            info['component'] = dbc.Input(id=self.ids.input(aio_id, key),
                                          type='number',
                                          placeholder=info['placeholder'],
                                          debounce=True,
                                          persistence=True,
                                          persistence_type='local')

        for key, info in button_infos.items():
            info['component'] = dbc.Button(id=self.ids.button(aio_id, key), children=info['text'])

        dd_datnums = dcc.Dropdown(id=self.ids.dropdown(aio_id), multi=allow_multiple,
                                  clearable=True,
                                  placeholder='Add options first',
                                  searchable=True,
                                  persistence=True,
                                  persistence_type='local',
                                  # style=,
                                  )

        layout = dbc.Card(
            [
                dbc.CardHeader(dbc.Col(html.H3('Dat Selector'))),
                dbc.CardBody([
                    dbc.Row([
                        *[vertical_label(info['label'], info['component']) for info in input_infos.values()],
                        dbc.Col([info['component'] for info in button_infos.values()])
                    ]),
                    dbc.Row([
                        dbc.Col(dd_datnums)
                    ])
                ])
            ])

        super().__init__(children=[layout])  # html.Div contains layout

    @staticmethod
    @callback(
        Output(ids.dropdown(MATCH), 'options'),
        Output(ids.dropdown(MATCH), 'value'),
        Input(ids.button(MATCH, 'add'), 'n_clicks'),
        Input(ids.button(MATCH, 'remove'), 'n_clicks'),
        State(ids.input(MATCH, 'start'), 'value'),
        State(ids.input(MATCH, 'stop'), 'value'),
        State(ids.input(MATCH, 'step'), 'value'),
        State(ids.dropdown(MATCH), 'options'),
        State(ids.dropdown(MATCH), 'value'),
    )
    def update_datnums(add_clicks: Optional[int], remove_clicks: Optional[int],
                       start: Optional[int], stop: Optional[int], step: Optional[int],
                       prev_options: Optional[List[dict]],
                       current_datnums: Optional[List[int]]) -> \
            tuple[list[dict], list[int]]:
        """
        Update the list of datnums in the selectable dropdown thing and what is currently selected
        Args:
            add_clicks ():
            remove_clicks ():
            start ():
            stop ():
            step ():
            prev_options ():
            current_datnums ():

        Returns:

        """
        prev_options = prev_options if prev_options else []
        current_datnums = current_datnums if current_datnums else []
        logging.info(f'Current datnums: {current_datnums}')

        triggered = get_triggered()
        if add_clicks and start:
            step = step if step else 1
            stop = stop if stop and stop > start else start
            vals = range(start, stop + 1, step)
            prev_opts_keys = [opt['value'] for opt in prev_options]
            for v in vals:
                if triggered.id['key'] == 'add':
                    if v not in prev_opts_keys:
                        prev_options.append({'label': v, 'value': v})
                    if v not in current_datnums:
                        current_datnums.append(v)
                elif triggered.id['key'] == 'remove':
                    prev_options = [p for p in prev_options if p['value'] not in vals]
                    current_datnums = [d for d in current_datnums if d not in vals]
                else:
                    logging.warning(f'Unexpected trigger. trig.id = {triggered.id}')

            prev_options = [opts for opts in sorted(prev_options, key=lambda item: item['value'])]
            current_datnums = list(sorted(current_datnums))
        return prev_options, current_datnums



