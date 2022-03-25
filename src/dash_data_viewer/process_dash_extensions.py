"""
Provide the dash functionality to dat_analysis Process
"""
from __future__ import annotations
import abc
from typing import TYPE_CHECKING, List, Union
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import dash
import uuid

from dat_analysis.analysis_tools.new_procedures import Process, SeparateSquareProcess

from dash_data_viewer.components import CollapseAIO

if TYPE_CHECKING:
    from dash.development.base_component import Component


class DashEnabledProcess(Process, abc.ABC):
    @abc.abstractmethod
    def dash_full(self, *args, **kwargs) -> Component:
        """
        Combine multiple figures/text/etc to make an output component
        (can assume 3/4 page wide, any height)
        Args:

        Returns:

        """
        input_plotter = self.get_input_plotter()
        output_plotter = self.get_output_plotter()

        input_fig = ProcessComponentOutputGraph(fig=input_plotter.plot_1d())
        output_fig = ProcessComponentOutputGraph(fig=output_plotter.plot_1d())

        return html.Div([
            input_fig.layout(),
            output_fig.layout()
        ])

        pass

    def _dash_children_full(self) -> html.Div:
        collapses = []
        for child in self.child_processes:
            collapses.append(CollapseAIO(aio_id=None, content=child.dash_full(), button_text=child.default_name,
                                         start_open=False))
        return html.Div(collapses)

    @abc.abstractmethod
    def mpl_full(self, *args, **kwargs):  # -> plt.Figure
        raise NotImplementedError

    @abc.abstractmethod
    def pdf_full(self, *args, **kwargs):  # -> pdf
        raise NotImplementedError


class ProcessInterface(abc.ABC):
    """
    Things necessary to put a process into a dash page or report with easy human friendly input. I.e. for building a
    new dash page

    human friendly input, and id of file to load from (or data passed in)
    """

    def __init__(self):
        self.store_id = ''
        self.sub_store_ids = []

    @abc.abstractmethod
    def required_input_components(self) -> List[Union[ProcessComponentInput, List[ProcessComponentInput]]]:
        """
        Give the list of components that need to be placed in order for the Process to be carried out
        Should all update stores or possibly the dat file (TODO: how to get lower steps of data?)
        Returns:

        """
        return []

    @abc.abstractmethod
    def all_outputs(self) -> List[ProcessComponentOutput]:
        """List of components that display the process in detail"""
        return []

    @abc.abstractmethod
    def main_outputs(self) -> List[ProcessComponentOutput]:
        """List of components that display the main features of the process"""
        return [self.all_outputs()[0]]


class ProcessComponent(abc.ABC):
    """
    A dash component with callbacks etc to interface with user
    """
    # # Functions to create pattern-matching callbacks of the subcomponents
    # class ids:
    #     @staticmethod
    #     def generic(aio_id, key: str):
    #         return {
    #             'component': 'TemplateAIO',
    #             'subcomponent': f'generic',
    #             'key': key,
    #             'aio_id': aio_id,
    #         }
    #
    # # Make the ids class a public class
    # ids = ids
    #
    # def __init__(self, aio_id=None):
    #     if aio_id is None:
    #         aio_id = str(uuid.uuid4())
    #
    #     super().__init__(children=[])  # html.Div contains layout
    #
    # # UNCOMMENT -- This callback is run when module is imported regardless of use
    # # @staticmethod
    # # @callback(
    # # )
    # # def function():
    # #     pass



class ProcessComponentInput(ProcessComponent, abc.ABC):
    """
    Component mostly for user input
    """

    def __init__(self, title, num_buttons, etc):
        super().__init__()


class ProcessComponentOutput(ProcessComponent, abc.ABC):
    """
    Component mostly for output to graph/table/file etc
    """


class ProcessComponentOutputGraph(ProcessComponentOutput):
    """
    Component for displaying a figure (with additional options)
    """

    def __init__(self, fig: go.Figure):
        super().__init__()
        self.fig = fig

    def layout(self):
        return dcc.Graph(figure=self.fig)

    def run_callbacks(self, **kwargs):
        pass



