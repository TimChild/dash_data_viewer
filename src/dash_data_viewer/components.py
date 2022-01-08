from dash import Input, Output, State, dcc, html, callback, MATCH, ALL, ALLSMALLER, dash
from dataclasses import dataclass, asdict
from dacite import from_dict
from dash_extensions.snippets import get_triggered
from typing import Optional, List, Union, Any
import uuid
import dash_bootstrap_components as dbc
import logging

from .layout_util import vertical_label


class TemplateAIO(html.Div):
    """
    DESCRIPTION

    # Requires
    What components require outside callbacks in order to work (if any)

    # Provides
    What component outputs are intended to be used by other callbacks (if any)

    Examples:
        # Example init
        ex = TemplateAIO()

        # Example input required (if relevant)
        # Example output provided (if relevant)

    """

    # Functions to create pattern-matching callbacks of the subcomponents
    class ids:
        @staticmethod
        def generic(aio_id, key: str):
            return {
                'component': 'StockSummaryAIO',
                'subcomponent': f'generic',
                'key': key,
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


def Input_(id=None, type='text', value=None, autocomplete=False, inputmode='text', max=None, maxlength=None, min=None,
           minlength=None, step=None, size='md', valid=False, required=False, placeholder='', name='', debounce=True,
           persistence=False, persistence_type='local', **kwargs) -> dbc.Input:
    """

    Args:
        id (): ID in page
        type (): The type of control to render. (a value equal to: "text", 'number', 'password', 'email', 'range', 'search', 'tel', 'url', 'hidden'; optional):
        value (): initial value of input
        autocomplete (): Enable browser autocomplete
        inputmode (): (a value equal to: "verbatim", "latin", "latin-name", "latin-prose", "full-width-latin", "kana", "katakana", "numeric", "tel", "email", "url"; optional): Provides a hint to the browser as to the type of data that might be entered by the user while editing the element or its contents.
        max (): Max value (or date) allowed
        maxlength (): Max string length allowed
        min (): Min value (or date) allowed
        minlength (): Min string length allowed
        step (): Step size for value type
        size (): 'sm', 'md', or 'lg'. Defaults to 'md'
        valid (): Apply valid style (adds a tick)
        required (): Entry required for form submission
        placeholder (): Placeholder text (not an actual value)
        name (): Name of component submitted with form data
        debounce (): Update only on focus loss or enter key
        persistence (): Whether the value should persist
        persistence_type (): (a value equal to: 'local', 'session', 'memory'; default 'local'). memory: only kept in memory, reset on page refresh. local: window.localStorage, data is kept after the browser quit. session: window.sessionStorage, data is cleared once the browser quit.
        **kwargs (): Any other kwargs accepted by dbc.Input

    Returns:

    """
    return dbc.Input(id=id, type=type, value=value, autocomplete=autocomplete, inputmode=inputmode, max=max,
                     maxlength=maxlength, min=min, minlength=minlength, step=step, size=size, valid=valid,
                     required=required, placeholder=placeholder, name=name, debounce=debounce, persistence=persistence,
                     persistence_type=persistence_type, **kwargs)


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
        self.dd_id = self.ids.dropdown(aio_id)  # For easy access to this component ('value' contains selected dats)

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


class MultiButtonAIO(html.Div):
    """
    A group of buttons with custom text where one(multiple) can be selected at a time and returns either the text or a specific
    value

    # Requires
    None

    # Provides
    dcc.Store(id=self.store_id) -- Filled Info class

    Examples:
    # Example init
    ex = TemplateAIO()

    # Example input required (if relevant)
    # Example output provided (if relevant)

    """

    @dataclass
    class Info:
        selected_values: Union[List[Any], Any]
        selected_numbers: List[int]  # Button numbers (so I know which to color)

    @dataclass
    class Config:
        allow_multiple: bool

    # Functions to create pattern-matching callbacks of the subcomponents
    class ids:
        @staticmethod
        def button(aio_id, num, value):
            return {
                'component': 'MultiButtonAIO',
                'subcomponent': f'button',
                'number': num,
                'aio_id': aio_id,
                'value': value,
            }

        @staticmethod
        def store(aio_id):
            return {
                'component': 'MultiButtonAIO',
                'subcomponent': f'store',
                'aio_id': aio_id,
            }

        @staticmethod
        def config(aio_id):
            return {
                'component': 'MultiButtonAIO',
                'subcomponent': f'config_store',
                'aio_id': aio_id,
            }

    # Make the ids class a public class
    ids = ids

    def __init__(self, button_texts: list[str], button_values: list | None = None, button_props: dict = None,
                 aio_id=None, allow_multiple=False, storage_type='memory'):
        if button_props is None:
            button_props = {}
        if aio_id is None:
            aio_id = str(uuid.uuid4())

        config = self.Config(allow_multiple=allow_multiple)

        # # Store other optional params here so that the callback can see them
        config_store = dcc.Store(id=self.ids.config(aio_id), storage_type='session',
                                 data=asdict(config))

        if button_values is None:
            button_values = button_texts

        self.store_id = self.ids.store(aio_id)

        buttons = [dbc.Button(
            id=self.ids.button(aio_id, num, value), children=text, color='danger', **button_props
        ) for num, (text, value) in
            enumerate(zip(button_texts, button_values))
        ]
        if not allow_multiple:
            buttons[0].color = 'success'

        button_layout = dbc.ButtonGroup(buttons)

        super().__init__(children=[
            button_layout,
            dcc.Store(id=self.ids.store(aio_id), storage_type=storage_type),
            config_store,
        ])

    @staticmethod
    @callback(
        Output(ids.store(MATCH), 'data'),
        Output(ids.button(MATCH, ALL, ALL), 'color'),
        Input(ids.button(MATCH, ALL, ALL), 'n_clicks'),
        State(ids.button(MATCH, ALL, ALL), 'color'),
        State(ids.store(MATCH), 'data'),
        State(ids.config(MATCH), 'data'),
    )
    def update_button_output(all_clicks, button_colors, current_data, config):
        config = from_dict(MultiButtonAIO.Config, config)
        if not current_data or not isinstance(current_data, dict):
            current_data = MultiButtonAIO.Info(selected_values=[], selected_numbers=[])
        else:
            current_data = from_dict(MultiButtonAIO.Info, current_data)

        triggered = get_triggered()
        if triggered.id:
            selected = triggered.id.get('value')
            num = triggered.id.get('number')
            if config.allow_multiple is False:
                current_data.selected_values = selected
                current_data.selected_numbers = [num]
            else:  # Allowing multiple selection
                if selected in current_data.selected_values:  # Remove if already selected
                    current_data.selected_values.remove(selected)
                    current_data.selected_numbers.remove(num)
                else:  # Add otherwise
                    current_data.selected_values.append(selected)
                    current_data.selected_numbers.append(num)
        colors = ['danger' if i not in current_data.selected_numbers else 'success' for i, _ in
                  enumerate(button_colors)]
        return asdict(current_data), colors


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
