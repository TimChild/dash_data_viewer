from dash import Input, Output, State, dcc, html, callback, MATCH, ALL, ALLSMALLER, dash
from dash_extensions.snippets import get_triggered
from typing import Optional, List
import uuid
import dash_bootstrap_components as dbc
import logging

from .layout_util import vertical_label


class TemplateAIO(html.Div):
    """
    DESCRIPTION

    Examples:
        # Example init
        ex = TemplateAIO()

        # Example use of any main components
        @callback(
            Input(ex.store_id, 'data'),
            Output(ex.val_id, 'value'),
        )
        def foo(data: list[str]) -> str:  # Showing type of arguments
    """

    # Functions to create pattern-matching callbacks of the subcomponents
    class ids:
        @staticmethod
        def input(aio_id):
            return {
                'component': 'TemplateAIO',
                'subcomponent': f'input',
                'aio_id': aio_id,
            }

    # Make the ids class a public class
    ids = ids

    def __init__(self, aio_id=None):
        if aio_id is None:
            aio_id = str(uuid.uuid4())

        super().__init__(children=[])  # html.Div contains layout

    # UNCOMMENT -- This callback is run when module is imported regardless of use
    # @staticmethod
    # @callback(
    # )
    # def function():
    #     pass


class DatnumPickerAIO(html.Div):
    """
    A group of buttons with custom text where one can be selected at a time and returns either the text or a specific
    value

    Examples:
        picker = DatnumPickerAIO(aio_id='testpage-datpicker')  # Fixed ID required for persistence to work

        @callback(Input(picker.dd_id, 'value'))  # Selected Datnums are accessible from the dd.value
        def foo(datnums: list):
            return datnums

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
        def dropdown(aio_id):
            return {
                'component': 'DatnumPickerAIO',
                'subcomponent': f'dropdown',
                'aio_id': aio_id,
            }

        @staticmethod
        def options_store(aio_id):
            return {
                'component': 'DatnumPickerAIO',
                'subcomponent': f'opts',
                'aio_id': aio_id,
            }

    # Make the ids class a public class
    ids = ids

    def __init__(self, aio_id, allow_multiple=True):
        self.dd_id = self.ids.dropdown(aio_id)   # For easy access to this component ('value' contains selected dats)

        opts_store = dcc.Store(id=self.ids.options_store(aio_id),
                               data=dict(multi=allow_multiple)
                               )

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

        super().__init__(children=[opts_store, layout])  # html.Div contains layout

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
        State(ids.options_store(MATCH), 'data')
    )
    def update_datnums(add_clicks: Optional[int], remove_clicks: Optional[int],
                       start: Optional[int], stop: Optional[int], step: Optional[int],
                       prev_options: Optional[List[dict]],
                       current_datnums: Optional[List[int]],
                       options: dict):
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
            options (): Additional options that are stored on creation of AIO

        Returns:

        """
        logging.info(f'Current datnums: {current_datnums}')
        prev_options = prev_options if prev_options else []
        if options['multi'] is True:
            current_datnums = current_datnums if current_datnums else []

        triggered = get_triggered()
        if add_clicks and start:
            step = step if step else 1
            stop = stop if stop and stop > start else start
            vals = range(start, stop + 1, step)
            prev_opts_keys = [opt['value'] for opt in prev_options]
            if triggered.id['key'] == 'add':
                for v in vals:
                    if v not in prev_opts_keys:
                        prev_options.append({'label': v, 'value': v})
                    if options['multi'] is True:
                        if v not in current_datnums:
                            current_datnums.append(v)
            elif triggered.id['key'] == 'remove':
                prev_options = [p for p in prev_options if p['value'] not in vals]
                if options['multi'] is True:
                    current_datnums = [d for d in current_datnums if d not in vals]
                else:
                    if current_datnums in vals:
                        current_datnums = None
            else:
                logging.warning(f'Unexpected trigger. trig.id = {triggered.id}')

            prev_options = [opts for opts in sorted(prev_options, key=lambda item: item['value'])]
            if options['multi'] is True:
                current_datnums = list(sorted(current_datnums))
            else:
                if current_datnums is None and prev_options:
                    current_datnums = prev_options[0]['value']
        return prev_options, current_datnums  # Return each in [] because of use of ALL


class TestAIO(html.Div):
    """
    DESCRIPTION

    Examples:
        # Example init
        ex = TemplateAIO()

        # Example use of any main components
        @callback(
            Input(ex.store_id, 'data'),
            Output(ex.val_id, 'value'),
        )
        def foo(data: list[str]) -> str:  # Showing type of arguments
    """

    # Functions to create pattern-matching callbacks of the subcomponents
    class ids:
        @staticmethod
        def input(aio_id, key):
            return {
                'component': 'TemplateAIO',
                'subcomponent': f'input',
                'key': key,
                'aio_id': aio_id,
            }

        @staticmethod
        def div(aio_id, key):
            return {
                'component': 'TemplateAIO',
                'subcomponent': f'div',
                'key': key,
                'aio_id': aio_id,
            }

    # Make the ids class a public class
    ids = ids

    def __init__(self, aio_id=None):
        if aio_id is None:
            aio_id = str(uuid.uuid4())

        input = dbc.Input(id=self.ids.input(aio_id, True))
        div = html.Div(id=self.ids.div(aio_id, 0), children='Nothing Updated')
        super().__init__(children=[input, div])  # html.Div contains layout

    @staticmethod
    @callback(
        Output(ids.div(MATCH, ALL), 'children'),
        Input(ids.input(MATCH, ALL), 'value'),
    )
    def function(value):
        logging.info(f'value = {value}')
        logging.info(get_triggered().id)
        return [value]
