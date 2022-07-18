# from __future__ import annotations
# import dash
# import lmfit as lm
# from dash import html, dcc, callback, Input, Output, State
# from dash.dependencies import Component
# import dash_bootstrap_components as dbc
# import plotly.graph_objects as go
# import numpy as np
# from dataclasses import dataclass
# import dash_extensions.snippets
# import logging
#
# from dash_data_viewer.layout_util import label_component
# from dash_data_viewer.components import DatnumPickerAIO
#
# from dat_analysis.dat_object.make_dat import get_dat, get_dats
# from dat_analysis.plotting.plotly import OneD, TwoD
# from dat_analysis.useful_functions import mean_data
#
# # from dash_data_viewer.layout_util import label_component
#
# from typing import TYPE_CHECKING, Optional, List, Union
#
# logging.basicConfig(level=logging.INFO)
#
#
# class MainComponents(object):
#     """Convenient holder for any components that will end up in the main area of the page"""
#     graph_2d_transition = dcc.Graph(id='transition-graph-2d-transition')
#     graph_average_transition = dcc.Graph(id='transition-graph-average-transition')
#
#     div_row_fit_graphs = html.Div(id='transition-div-perRowFitGraphs')
#
#     test_selected_datnums = html.Div(id='test-selected-datnums')
#
#
# class SidebarComponents(object):
#     """Convenient holder for any components that will end up in the sidebar area of the page"""
#     input_datnum = dbc.Input(id='transition-input-datnum', type='number', placeholder='Datnum', debounce=True,
#                              persistence=True, persistence_type='local')
#     # dd_datnums = dcc.Dropdown(id='transition-dd-datnums', multi=True,
#     #                           clearable=True,
#     #                           placeholder='Add options first',
#     #                           searchable=True,
#     #                           persistence=True,
#     #                           persistence_type='local',
#     #                           # style=,
#     #                           )
#     toggle_centering = dbc.Checklist(id='transition-toggle-centering', switch=True,
#                                      persistence=True, persistence_type='local',
#                                      options=[
#                                          {'label': "", 'value': True}
#                                      ],
#                                      value=[True])
#     dropdown_2dtype = dcc.Dropdown(id='transition-dropdown-2dtype',
#                                    options=[
#                                        {'label': 'Heatmap', 'value': 'heatmap'},
#                                        {'label': 'Waterfall', 'value': 'waterfall'},
#                                    ],
#                                    value='heatmap',
#                                    persistence=True,
#                                    persistence_type='local',
#                                    optionHeight=20,
#                                    )
#     dat_selector = DatnumPickerAIO(aio_id='transition-datpicker', allow_multiple=False)
#     # inp_datnums_start = dbc.Input(id='transition-inp-datstart', type='number', placeholder='Start', debounce=True,
#     #                               persistence=True, persistence_type='local')
#     # inp_datnums_stop = dbc.Input(id='transition-inp-datStop', type='number', placeholder='Stop', debounce=True,
#     #                              persistence=True, persistence_type='local')
#     # inp_datnums_step = dbc.Input(id='transition-inp-datStep', type='number', placeholder='Step', debounce=True,
#     #                              persistence=True, persistence_type='local')
#     #
#     # but_datnums_add = dbc.Button(id='transition-but-datadd', children='Add')
#     # but_datnums_remove = dbc.Button(id='transition-but-datremove', children='Remove', color='danger')
#
#
# # Initialize the components ONCE here.
# main_components = MainComponents()
# sidebar_components = SidebarComponents()
#
#
# @dataclass
# class TransitionData:
#     x: np.ndarray
#     y: Optional[np.ndarray]
#     data: np.ndarray
#
#
# @dataclass
# class RowFitInfo:
#     params: List[lm.Parameters]
#     centers: List[float]
#
#
# # Functions and Callbacks specific to this page
#
# # @cache.memoize()
# def get_transition_datas(datnums: int) -> Optional[Union[TransitionData, List[TransitionData]]]:
#     tdatas = None
#     if datnums:
#         was_int = isinstance(datnums, int)
#         if isinstance(datnums, int):
#             datnums = [datnums]
#         elif not isinstance(datnums, list):
#             return None
#         dats = get_dats(datnums)
#         tdatas = []
#         for dat in dats:
#             x = dat.Data.get_data('x')
#             data = dat.Data.get_data('i_sense')
#             if 'y' in dat.Data.keys:
#                 y = dat.Data.get_data('y')
#             else:
#                 y = None
#             tdata = TransitionData(x=x, y=y, data=data)
#             tdatas.append(tdata)
#         if was_int is True:
#             return tdatas[0]
#     return tdatas
#
#
# # @cache.memoize()
# def get_transition_row_fits(datnum: int) -> Optional[RowFitInfo]:
#     tdata = get_transition_datas(datnum)
#     row_fits = None
#     if tdata:
#         dat = get_dat(datnum)
#         if dat:
#             if tdata.data.ndim == 1:
#                 row_fits = [dat.Transition.get_fit(which='row', row=0, x=tdata.x, data=tdata.data, calculate_only=True)]
#             elif tdata.data.ndim == 2:
#                 row_fits = [
#                     dat.Transition.get_fit(which='row', row=i, x=tdata.x, data=tdata.data[i], calculate_only=True) for i
#                     in range(tdata.data.shape[0])]
#     if row_fits:
#         row_fit_info = RowFitInfo(
#             params=[f.params for f in row_fits],
#             centers=[f.best_values.mid for f in row_fits]
#         )
#     else:
#         row_fit_info = None
#     return row_fit_info
#
#
# # @callback(
# #     Output(sidebar_components.dd_datnums.id, 'options'),
# #     Output(sidebar_components.dd_datnums.id, 'value'),
# #     Input(sidebar_components.but_datnums_add.id, 'n_clicks'),
# #     Input(sidebar_components.but_datnums_remove.id, 'n_clicks'),
# #     State(sidebar_components.inp_datnums_start.id, 'value'),
# #     State(sidebar_components.inp_datnums_stop.id, 'value'),
# #     State(sidebar_components.inp_datnums_step.id, 'value'),
# #     State(sidebar_components.dd_datnums.id, 'options'),
# #     State(sidebar_components.dd_datnums.id, 'value'),
# # )
# # def update_datnums(add_clicks: Optional[int], remove_clicks: Optional[int],
# #                    start: Optional[int], stop: Optional[int], step: Optional[int],
# #                    prev_options: Optional[List[dict]],
# #                    current_datnums: Optional[List[int]]) -> \
# #         tuple[list[dict], list[int]]:
# #     """
# #     Update the list of datnums in the selectable dropdown thing and what is currently selected
# #     Args:
# #         add_clicks ():
# #         remove_clicks ():
# #         start ():
# #         stop ():
# #         step ():
# #         prev_options ():
# #         current_datnums ():
# #
# #     Returns:
# #
# #     """
# #     prev_options = prev_options if prev_options else []
# #     current_datnums = current_datnums if current_datnums else []
# #
# #     triggered = dash_extensions.snippets.get_triggered()
# #     if add_clicks and start:
# #         step = step if step else 1
# #         stop = stop if stop and stop > start else start
# #         vals = range(start, stop + 1, step)
# #         prev_opts_keys = [opt['value'] for opt in prev_options]
# #         for v in vals:
# #             if triggered.id == sidebar_components.but_datnums_add.id:
# #                 if v not in prev_opts_keys:
# #                     prev_options.append({'label': v, 'value': v})
# #                 if v not in current_datnums:
# #                     current_datnums.append(v)
# #             elif triggered.id == sidebar_components.but_datnums_remove.id:
# #                 prev_options = [p for p in prev_options if p['value'] not in vals]
# #                 current_datnums = [d for d in current_datnums if d not in vals]
# #             else:
# #                 logging.info(
# #                     f'trig.id = {triggered.id}, but.id = {sidebar_components.but_datnums_add.id}, trig==but: {triggered.id == sidebar_components.but_datnums_add.id}')
# #                 logging.warning(f'Unexpected trigger')
# #
# #         prev_options = [opts for opts in sorted(prev_options, key=lambda item: item['value'])]
# #         current_datnums = list(sorted(current_datnums))
# #     return prev_options, current_datnums
#
#
# @callback(
#     Output(main_components.graph_2d_transition.id, 'figure'),
#     Input(sidebar_components.input_datnum.id, 'value'),
#     Input(sidebar_components.dropdown_2dtype.id, 'value'),
# )
# def graph_transition_data(datnum: int, graph_type: str) -> go.Figure:
#     """
#     Either 1D or 2D transition data just as it was recorded (after resampling)
#     Returns:
#
#     """
#     tdata = get_transition_datas(datnum)
#     if tdata:
#         dat = get_dat(datnum)
#         if dat:
#             if tdata.y is not None and tdata.data.ndim == 2:
#                 p = TwoD(dat=dat)
#                 if graph_type == 'waterfall':
#                     p.MAX_POINTS = round(10000/tdata.y.shape[0])
#                 fig = p.figure()
#                 fig.add_traces(p.trace(data=tdata.data, x=tdata.x, y=tdata.y, trace_type=graph_type))
#             elif tdata.data.ndim == 1:
#                 p = OneD(dat=dat)
#                 fig = p.figure(ylabel='Current /nA')
#                 fig.add_trace(p.trace(data=tdata.data, x=tdata.x, mode='markers+lines'))
#             else:
#                 fig = go.Figure()
#
#             fig.update_layout(title=f'Dat{dat.datnum}: Averaged Transition Data')
#             return fig
#     return go.Figure()
#
#
# @callback(
#     Output(main_components.graph_average_transition.id, 'figure'),
#     Input(sidebar_components.input_datnum.id, 'value'),
#     Input(sidebar_components.toggle_centering.id, 'value'),
# )
# def average_transition_data(datnum: int, centered: int) -> go.Figure:
#     tdata = get_transition_datas(datnum)
#     if tdata:
#         dat = get_dat(datnum)
#         if dat:
#             if tdata.data.ndim == 1:
#                 return graph_transition_data(datnum, 'not used')
#             elif tdata.y is not None and tdata.data.ndim == 2:
#                 if centered:
#                     row_fits = get_transition_row_fits(datnum)
#                     centers = row_fits.centers
#                 else:
#                     centers = [0] * tdata.data.shape[0]
#                 avg_data, avg_x, avg_std = mean_data(x=tdata.x, data=tdata.data, centers=centers, return_x=True,
#                                                      return_std=True)
#
#                 p = OneD(dat=dat)
#                 fig = p.figure(ylabel='Current /nA')
#                 fig.add_trace(p.trace(data=avg_data, x=avg_x, mode='markers+lines'))
#             else:
#                 fig = go.Figure()
#
#             fig.update_layout(title=f'Dat{dat.datnum}: Transition Data')
#             return fig
#     return go.Figure()
#
#
# @callback(
#     Output(main_components.div_row_fit_graphs.id, 'children'),
#     Input(sidebar_components.input_datnum.id, 'value'),
# )
# def plot_per_row_fit_params(datnum: int) -> Component:
#     @dataclass
#     class ParInfo:
#         key: str
#         name: str
#         units: str
#
#     if datnum:
#         dat = get_dat(datnum)
#         row_fits = get_transition_row_fits(datnum)
#
#         p1d = OneD(dat=dat)
#         graphs = []
#         for par_info in [
#             ParInfo(key='mid', name='center', units='mV'),
#         ]:
#             if par_info.key in row_fits.params[0].keys():
#                 fig = p1d.figure(title=f'Dat{dat.datnum}: {par_info.name} values',
#                                  xlabel="Row num",
#                                  ylabel=f'{par_info.name} /{par_info.units}')
#                 key_params = [params[par_info.key] for params in row_fits.params]
#                 fig.add_trace(p1d.trace(
#                     x=dat.Data.y,
#                     data=[p.value for p in key_params],
#                     data_err=[p.stderr if p.stderr else 0 for p in key_params],
#                     mode='markers+lines',
#                 ))
#                 graphs.append(fig)
#             else:
#                 pass
#         return html.Div([dcc.Graph(figure=fig) for fig in graphs])
#     return html.Div('No data to show')
#
#
# @callback(
#     Output(main_components.test_selected_datnums.id, 'children'),
#     Input(sidebar_components.dat_selector.dd_id, 'value'),
#     # Input(sidebar_components.dd_datnums.id, 'value'),
# )
# def show_datnums(datnums):
#     return f'{datnums}'
#
#
# def main_layout() -> Component:
#     global main_components
#     m = main_components
#     transition_graphs = dbc.Row([
#         dbc.Col(m.graph_2d_transition, width=6), dbc.Col(m.graph_average_transition, width=6)
#     ])
#     transition_info = dbc.Row([
#         html.Div(f'No info to show yet')
#     ])
#     layout_ = html.Div([
#         transition_graphs,
#         transition_info,
#         m.div_row_fit_graphs,
#         m.test_selected_datnums,
#     ])
#     return layout_
#
#
# def sidebar_layout() -> Component:
#     global sidebar_components
#     s = sidebar_components
#
#     # dat_manager = dbc.Card([
#     #     dbc.CardHeader(dbc.Col(html.H3('Dat Manager'))),
#     #     dbc.CardBody([
#     #         dbc.Row([
#     #             vertical_label('Start', s.inp_datnums_start),
#     #             vertical_label('Stop', s.inp_datnums_stop),
#     #             vertical_label('Step', s.inp_datnums_step),
#     #             dbc.Col([s.but_datnums_add, s.but_datnums_remove])
#     #
#     #         ]),
#     #         dbc.Row([
#     #             dbc.Col(s.dd_datnums)
#     #         ])
#     #     ]),
#     # ])
#
#     layout_ = html.Div([
#         label_component(s.input_datnum, 'Datnum:'),
#         label_component(s.dropdown_2dtype, 'Graph Type: '),
#         label_component(s.toggle_centering, 'Center first: '),
#         s.dat_selector,
#         # dat_manager,
#     ])
#     return layout_
#
#
# def layout():
#     """Must return the full layout for multipage to work"""
#     sidebar_layout_ = sidebar_layout()
#     main_layout_ = main_layout()
#
#     return html.Div([
#         html.H1('Transition'),
#         dbc.Row([
#             dbc.Col(
#                 dbc.Card(sidebar_layout_),
#                 width=3
#             ),
#             dbc.Col(
#                 dbc.Card(main_layout_)
#             )
#         ]),
#     ])
#
#
# if __name__ == '__main__':
#     app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
#     app.layout = layout()
#     app.run_server(debug=True, port=8053, threaded=True)
# else:
#     dash.register_page(__name__)
#     pass
