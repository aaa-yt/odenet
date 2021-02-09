import sys
import json
from datetime import datetime
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np
import seaborn as sns

sys.path.append("../")
from config import Config

class App1D1D:
    def __init__(self, config: Config, app):
        self.config = config

        self.app = app
        self.layout = html.Div(children=[
            html.Div(children=[
                html.H2(children="Training data", style={"textAlign": "center"}),
                dcc.Graph(
                    id="training-data-graph"
                )
            ], id="training-data", style={"width": "48%", "margin": "10px"}),
            html.Div(children=[
                html.H2(children="Validation data", style={"textAlign": "center"}),
                dcc.Graph(
                    id="validation-data-graph"
                )
            ], id="validation-data", style={"width": "48%", "margin": "10px"})
        ], id="data", style={"display": "flex", "justify-content": "center"})
        
        @self.app.callback(
            Output("training-data-graph", "figure"),
            Output("validation-data-graph", "figure"),
            Input("interval-component", "n_intervals")
        )
        def update_dataset(n):
            with open(self.config.resource.result_visualize_data_path, "rt") as f:
                dataset = json.load(f)
            x_train = np.array(dataset.get("Train").get("Input")).reshape(-1)
            y_train = np.array(dataset.get("Train").get("Output")).reshape(-1)
            x_train, y_train = zip(*sorted(zip(x_train, y_train)))
            x_val = np.array(dataset.get("Validation").get("Input")).reshape(-1)
            y_val = np.array(dataset.get("Validation").get("Output")).reshape(-1)
            x_val, y_val = zip(*sorted(zip(x_val, y_val)))
            with open(self.config.resource.result_visualize_data_predict_path, "rt") as f:
                dataset = json.load(f)
            x_train_pred = np.array(dataset.get("Train").get("Input")).reshape(-1)
            y_train_pred = np.array(dataset.get("Train").get("Output")).reshape(-1)
            x_train_pred, y_train_pred = zip(*sorted(zip(x_train_pred, y_train_pred)))
            x_val_pred = np.array(dataset.get("Validation").get("Input")).reshape(-1)
            y_val_pred = np.array(dataset.get("Validation").get("Output")).reshape(-1)
            x_val_pred, y_val_pred = zip(*sorted(zip(x_val_pred, y_val_pred)))

            fig_data_train = go.Figure(data=[
                go.Scatter(x=x_train, y=y_train, name="original", mode="lines", marker={"color": "#0000ff"}),
                go.Scatter(x=x_train_pred, y=y_train_pred, name="predict", mode="lines", marker={"color": "#ff0000"})
            ])
            fig_data_train.update_xaxes(title="x")
            fig_data_train.update_yaxes(title="F(x)")

            fig_data_val = go.Figure(data=[
                go.Scatter(x=x_val, y=y_val, name="original", mode="lines", marker={"color": "#0000ff"}),
                go.Scatter(x=x_val_pred, y=y_val_pred, name="predict", mode="lines", marker={"color": "#ff0000"})
            ])
            fig_data_val.update_xaxes(title="x")
            fig_data_val.update_yaxes(title="F(x)")
            return fig_data_train, fig_data_val


class App2D1D:
    def __init__(self, config: Config, app):
        self.config = config

        self.app = app
        self.layout = html.Div(children=[
            html.Div(children=[
                html.H2(children="Training data", style={"textAlign": "center"}),
                html.Div(children=[
                    html.Div(children=[
                        html.H3(children="Correct data", style={"textAlign": "center"}),
                        dcc.Graph(
                            id="training-data-correct-graph"
                        )
                    ], id="training-data-correct", style={"width": "48%", "margin": "10px"}),
                    html.Div(children=[
                        html.H3(children="Predict data", style={"textAlign": "center"}),
                        dcc.Graph(
                            id="training-data-predict-graph"
                        )
                    ], id="training-data-predict", style={"width": "48%", "margin": "10px"})
                ], id="training-data", style={"display": "flex", "justify-content": "center"})
            ], id="training"),
            html.Div(children=[
                html.H2(children="Validation data", style={"textAlign": "center"}),
                html.Div(children=[
                    html.Div(children=[
                        html.H3(children="Correct data", style={"textAlign": "center"}),
                        dcc.Graph(
                            id="validation-data-correct-graph"
                        )
                    ], id="validation-data-correct", style={"width": "48%", "margin": "10px"}),
                    html.Div(children=[
                        html.H3(children="Predict data", style={"textAlign": "center"}),
                        dcc.Graph(
                            id="validation-data-predict-graph"
                        )
                    ], id="validation-data-predict", style={"width": "48%", "margin": "10px"})
                ], id="validation-data", style={"display": "flex", "justify-content": "center"})
            ], id="validation")
        ])

        @self.app.callback(
            Output("training-data-correct-graph", "figure"),
            Output("validation-data-correct-graph", "figure"),
            Input("interval-component", "n_intervals")
        )
        def update_dataset_correct(n):
            with open(self.config.resource.result_visualize_data_path, "rt") as f:
                dataset = json.load(f)
            x_train = np.array(dataset.get("Train").get("Input"))
            y_train = np.array(dataset.get("Train").get("Output")).reshape(-1)
            y_train = np.where(y_train>0.5, 1, 0)
            x_val = np.array(dataset.get("Validation").get("Input"))
            y_val = np.array(dataset.get("Validation").get("Output")).reshape(-1)
            y_val = np.where(y_val>0.5, 1, 0)
            
            fig_data_train = go.Figure(data=[
                go.Scatter(x=x_train[np.where(y_train==1)][:, 0], y=x_train[np.where(y_train==1)][:, 1], name="y=1", mode="markers", marker={"color": "#ff0000"}),
                go.Scatter(x=x_train[np.where(y_train==0)][:, 0], y=x_train[np.where(y_train==0)][:, 1], name="y=0", mode="markers", marker={"color": "#0000ff"})
            ], layout=go.Layout(yaxis=dict(scaleanchor='x')))
            fig_data_train.update_xaxes(title="x1")
            fig_data_train.update_yaxes(title="x2")
            fig_data_val = go.Figure(data=[
                go.Scatter(x=x_val[np.where(y_val==1)][:, 0], y=x_val[np.where(y_val==1)][:, 1], name="y=1", mode="markers", marker={"color": "#ff0000"}),
                go.Scatter(x=x_val[np.where(y_val==0)][:, 0], y=x_val[np.where(y_val==0)][:, 1], name="y=0", mode="markers", marker={"color": "#0000ff"})
            ], layout=go.Layout(yaxis=dict(scaleanchor='x')))
            fig_data_val.update_xaxes(title="x1")
            fig_data_val.update_yaxes(title="x2")
            return fig_data_train, fig_data_val
        

        @self.app.callback(
            Output("training-data-predict-graph", "figure"),
            Output("validation-data-predict-graph", "figure"),
            Input("interval-component", "n_intervals")
        )
        def update_dataset_predict(n):
            with open(self.config.resource.result_visualize_data_predict_path, "rt") as f:
                dataset = json.load(f)
            x_train = np.array(dataset.get("Train").get("Input"))
            y_train = np.array(dataset.get("Train").get("Output")).reshape(-1)
            y_train = np.where(y_train>0.5, 1, 0)
            x_val = np.array(dataset.get("Validation").get("Input"))
            y_val = np.array(dataset.get("Validation").get("Output")).reshape(-1)
            y_val = np.where(y_val>0.5, 1, 0)

            fig_data_train = go.Figure(data=[
                go.Scatter(x=x_train[np.where(y_train==1)][:, 0], y=x_train[np.where(y_train==1)][:, 1], name="y=1", mode="markers", marker={"color": "#ff0000"}),
                go.Scatter(x=x_train[np.where(y_train==0)][:, 0], y=x_train[np.where(y_train==0)][:, 1], name="y=0", mode="markers", marker={"color": "#0000ff"})
            ], layout=go.Layout(yaxis=dict(scaleanchor='x')))
            fig_data_train.update_xaxes(title="x1")
            fig_data_train.update_yaxes(title="x2")
            fig_data_val = go.Figure(data=[
                go.Scatter(x=x_val[np.where(y_val==1)][:, 0], y=x_val[np.where(y_val==1)][:, 1], name="y=1", mode="markers", marker={"color": "#ff0000"}),
                go.Scatter(x=x_val[np.where(y_val==0)][:, 0], y=x_val[np.where(y_val==0)][:, 1], name="y=0", mode="markers", marker={"color": "#0000ff"})
            ], layout=go.Layout(yaxis=dict(scaleanchor='x')))
            fig_data_val.update_xaxes(title="x1")
            fig_data_val.update_yaxes(title="x2")
            return fig_data_train, fig_data_val

class App2D2D:
    def __init__(self, config: Config, app):
        self.config = config

        self.app = app
        self.layout = html.Div(children=[
            html.Div(children=[
                html.H2(children="Training data", style={"textAlign": "center"}),
                html.Div(children=[
                    html.Div(children=[
                        html.H3(children="Correct data", style={"textAlign": "center"}),
                        dcc.Graph(
                            id="training-data-correct-graph"
                        )
                    ], id="training-data-correct", style={"width": "48%", "margin": "10px"}),
                    html.Div(children=[
                        html.H3(children="Predict data", style={"textAlign": "center"}),
                        dcc.Graph(
                            id="training-data-predict-graph"
                        )
                    ], id="training-data-predict", style={"width": "48%", "margin": "10px"})
                ], id="training-data", style={"display": "flex", "justify-content": "center"})
            ], id="training"),
            html.Div(children=[
                html.H2(children="Validation data", style={"textAlign": "center"}),
                html.Div(children=[
                    html.Div(children=[
                        html.H3(children="Correct data", style={"textAlign": "center"}),
                        dcc.Graph(
                            id="validation-data-correct-graph"
                        )
                    ], id="validation-data-correct", style={"width": "48%", "margin": "10px"}),
                    html.Div(children=[
                        html.H3(children="Predict data", style={"textAlign": "center"}),
                        dcc.Graph(
                            id="validation-data-predict-graph"
                        )
                    ], id="validation-data-predict", style={"width": "48%", "margin": "10px"})
                ], id="validation-data", style={"display": "flex", "justify-content": "center"})
            ], id="validation")
        ])

        @self.app.callback(
            Output("training-data-correct-graph", "figure"),
            Output("validation-data-correct-graph", "figure"),
            Input("interval-component", "n_intervals")
        )
        def update_dataset_correct(n):
            with open(self.config.resource.result_visualize_data_path, "rt") as f:
                dataset = json.load(f)
            x_train = np.array(dataset.get("Train").get("Input"))
            y_train = np.array(dataset.get("Train").get("Output"))
            y_train = np.argmax(y_train, 1)
            x_val = np.array(dataset.get("Validation").get("Input"))
            y_val = np.array(dataset.get("Validation").get("Output"))
            y_val = np.argmax(y_val, 1)

            fig_data_train = go.Figure()
            fig_data_val = go.Figure()
            colors_data = self.get_colorpalette("hls", self.config.model.output_dimension)
            for i in range(self.config.model.output_dimension):
                fig_data_train.add_trace(go.Scatter(x=x_train[np.where(y_train==i)][:, 0], y=x_train[np.where(y_train==i)][:, 1], name="y={}".format(i), mode="markers", marker={"color": colors_data[i]}))
                fig_data_val.add_trace(go.Scatter(x=x_val[np.where(y_val==i)][:, 0], y=x_val[np.where(y_val==i)][:, 1], name="y={}".format(i), mode="markers", marker={"color": colors_data[i]}),)
            fig_data_train.update_layout(yaxis=dict(scaleanchor='x'))
            fig_data_train.update_xaxes(title="x1")
            fig_data_train.update_yaxes(title="x2")
            fig_data_val.update_layout(yaxis=dict(scaleanchor='x'))
            fig_data_val.update_xaxes(title="x1")
            fig_data_val.update_yaxes(title="x2")

            return fig_data_train, fig_data_val
        
        @self.app.callback(
            Output("training-data-predict-graph", "figure"),
            Output("validation-data-predict-graph", "figure"),
            Input("interval-component", "n_intervals")
        )
        def update_dataset_predict(n):
            with open(self.config.resource.result_visualize_data_predict_path, "rt") as f:
                dataset = json.load(f)
            x_train = np.array(dataset.get("Train").get("Input"))
            y_train = np.array(dataset.get("Train").get("Output"))
            y_train = np.argmax(y_train, 1)
            x_val = np.array(dataset.get("Validation").get("Input"))
            y_val = np.array(dataset.get("Validation").get("Output"))
            y_val = np.argmax(y_val, 1)

            fig_data_train = go.Figure()
            fig_data_val = go.Figure()
            colors_data = self.get_colorpalette("hls", self.config.model.output_dimension)
            for i in range(self.config.model.output_dimension):
                fig_data_train.add_trace(go.Scatter(x=x_train[np.where(y_train==i)][:, 0], y=x_train[np.where(y_train==i)][:, 1], name="y={}".format(i), mode="markers", marker={"color": colors_data[i]}))
                fig_data_val.add_trace(go.Scatter(x=x_val[np.where(y_val==i)][:, 0], y=x_val[np.where(y_val==i)][:, 1], name="y={}".format(i), mode="markers", marker={"color": colors_data[i]}),)
            fig_data_train.update_layout(yaxis=dict(scaleanchor='x'))
            fig_data_train.update_xaxes(title="x1")
            fig_data_train.update_yaxes(title="x2")
            fig_data_val.update_layout(yaxis=dict(scaleanchor='x'))
            fig_data_val.update_xaxes(title="x1")
            fig_data_val.update_yaxes(title="x2")

            return fig_data_train, fig_data_val

    def get_colorpalette(self, colorpalette, n_colors):
        palette = sns.color_palette(colorpalette, n_colors)
        rgb = ['rgb({},{},{})'.format(*[x*256 for x in rgb]) for rgb in palette]
        return rgb[::-1]


class App3D1D:
    def __init__(self, config: Config, app):
        self.config = config

        self.app = app
        self.layout = html.Div(children=[
            html.Div(children=[
                html.H2(children="Training data", style={"textAlign": "center"}),
                html.Div(children=[
                    html.Div(children=[
                        html.H3(children="Correct data", style={"textAlign": "center"}),
                        dcc.Graph(
                            id="training-data-3d-correct-graph"
                        )
                    ], id="training-data-3d-correct", style={"width": "48%", "margin": "10px"}),
                    html.Div(children=[
                        html.H3(children="Predict data", style={"textAlign": "center"}),
                        dcc.Graph(
                            id="training-data-3d-predict-graph"
                        )
                    ], id="training-data-3d-predict", style={"width": "48%", "margin": "10px"})
                ], id="training-data-3d", style={"display": "flex", "justify-content": "center"}),
                html.Div(children=[
                    html.Div(children=[
                        html.H3(children="x-y plane", style={"textAlign": "center"}),
                        dcc.Graph(
                            id="training-data-xy-graph"
                        )
                    ], id="training-data-xy", style={"width": "32%", "margin": "10px"}),
                    html.Div(children=[
                        html.H3(children="y-z plane", style={"textAlign": "center"}),
                        dcc.Graph(
                            id="training-data-yz-graph"
                        )
                    ], id="training-data-yz", style={"width": "32%", "margin": "10px"}),
                    html.Div(children=[
                        html.H3(children="z-x plane", style={"textAlign": "center"}),
                        dcc.Graph(
                            id="training-data-zx-graph"
                        )
                    ], id="training-data-zx", style={"width": "32%", "margin": "10px"})
                ], id="training-data-projection", style={"display": "flex", "justify-content": "center"})
            ], id="training-data"),
            html.Div(children=[
                html.H2(children="validation data", style={"textAlign": "center"}),
                html.Div(children=[
                    html.Div(children=[
                        html.H3(children="Correct data", style={"textAlign": "center"}),
                        dcc.Graph(
                            id="validation-data-3d-correct-graph"
                        )
                    ], id="validation-data-3d-correct", style={"width": "48%", "margin": "10px"}),
                    html.Div(children=[
                        html.H3(children="Predict data", style={"textAlign": "center"}),
                        dcc.Graph(
                            id="validation-data-3d-predict-graph"
                        )
                    ], id="validation-data-3d-predict", style={"width": "48%", "margin": "10px"})
                ], id="validation-data-3d", style={"display": "flex", "justify-content": "center"}),
                html.Div(children=[
                    html.Div(children=[
                        html.H3(children="x-y plane", style={"textAlign": "center"}),
                        dcc.Graph(
                            id="validation-data-xy-graph"
                        )
                    ], id="validation-data-xy", style={"width": "32%", "margin": "10px"}),
                    html.Div(children=[
                        html.H3(children="y-z plane", style={"textAlign": "center"}),
                        dcc.Graph(
                            id="validation-data-yz-graph"
                        )
                    ], id="validation-data-yz", style={"width": "32%", "margin": "10px"}),
                    html.Div(children=[
                        html.H3(children="z-x plane", style={"textAlign": "center"}),
                        dcc.Graph(
                            id="validation-data-zx-graph"
                        )
                    ], id="validation-data-zx", style={"width": "32%", "margin": "10px"})
                ], id="validation-data-projection", style={"display": "flex", "justify-content": "center"})
            ], id="validation-data")
        ])

        @self.app.callback(
            Output("training-data-3d-correct-graph", "figure"),
            Output("validation-data-3d-correct-graph", "figure"),
            Input("interval-component", "n_intervals")
        )
        def update_dataset_correct(n):
            with open(self.config.resource.result_visualize_data_path, "rt") as f:
                dataset = json.load(f)
            x_train = np.array(dataset.get("Train").get("Input"))
            y_train = np.array(dataset.get("Train").get("Output")).reshape(-1)
            y_train = np.where(y_train>0.5, 1, 0)
            x_val = np.array(dataset.get("Validation").get("Input"))
            y_val = np.array(dataset.get("Validation").get("Output")).reshape(-1)
            y_val = np.where(y_val>0.5, 1, 0)

            fig_data_train = go.Figure(data=[
                go.Scatter3d(x=x_train[np.where(y_train==1)][:, 0], y=x_train[np.where(y_train==1)][:, 1], z=x_train[np.where(y_train==1)][:, 2], name="y=1", mode="markers", marker={"color": "#ff0000", "size": 3}),
                go.Scatter3d(x=x_train[np.where(y_train==0)][:, 0], y=x_train[np.where(y_train==1)][:, 0], z=x_train[np.where(y_train==0)][:, 2], name="y=0", mode="markers", marker={"color": "#0000ff", "size": 3}),
            ])
            fig_data_train.update_layout(scene=dict(
                xaxis_title='x1',
                yaxis_title='x2',
                zaxis_title='x3'
            ))
            fig_data_val = go.Figure(data=[
                go.Scatter3d(x=x_val[np.where(y_val==1)][:, 0], y=x_val[np.where(y_val==1)][:, 1], z=x_val[np.where(y_val==1)][:, 2], name="y=1", mode="markers", marker={"color": "#ff0000", "size": 3}),
                go.Scatter3d(x=x_val[np.where(y_val==0)][:, 0], y=x_val[np.where(y_val==1)][:, 0], z=x_val[np.where(y_val==0)][:, 2], name="y=0", mode="markers", marker={"color": "#0000ff", "size": 3}),
            ])
            fig_data_val.update_layout(scene=dict(
                xaxis_title='x1',
                yaxis_title='x2',
                zaxis_title='x3'
            ))

            return fig_data_train, fig_data_val

        @self.app.callback(
            Output("training-data-3d-predict-graph", "figure"),
            Output("training-data-xy-graph", "figure"),
            Output("training-data-yz-graph", "figure"),
            Output("training-data-zx-graph", "figure"),
            Output("validation-data-3d-predict-graph", "figure"),
            Output("validation-data-xy-graph", "figure"),
            Output("validation-data-yz-graph", "figure"),
            Output("validation-data-zx-graph", "figure"),
            Input("interval-component", "n_intervals")
        )
        def update_dataset_predict(n):
            with open(self.config.resource.result_visualize_data_predict_path, "rt") as f:
                dataset = json.load(f)
            x_train = np.array(dataset.get("Train").get("Input"))
            y_train = np.array(dataset.get("Train").get("Output")).reshape(-1)
            y_train = np.where(y_train>0.5, 1, 0)
            x_val = np.array(dataset.get("Validation").get("Input"))
            y_val = np.array(dataset.get("Validation").get("Output")).reshape(-1)
            y_val = np.where(y_val>0.5, 1, 0)

            fig_data_train = go.Figure(data=[
                go.Scatter3d(x=x_train[np.where(y_train==1)][:, 0], y=x_train[np.where(y_train==1)][:, 1], z=x_train[np.where(y_train==1)][:, 2], name="y=1", mode="markers", marker={"color": "#ff0000", "size": 3}),
                go.Scatter3d(x=x_train[np.where(y_train==0)][:, 0], y=x_train[np.where(y_train==0)][:, 1], z=x_train[np.where(y_train==0)][:, 2], name="y=0", mode="markers", marker={"color": "#0000ff", "size": 3}),
            ])
            fig_data_train.update_layout(scene=dict(
                xaxis_title='x1',
                yaxis_title='x2',
                zaxis_title='x3'
            ))
            fig_data_train_xy = go.Figure(data=[
                go.Scatter(x=x_train[np.where(y_train==1)][:, 0], y=x_train[np.where(y_train==1)][:, 1], name="y=1", mode="markers", marker={"color": "#ff0000"}),
                go.Scatter(x=x_train[np.where(y_train==0)][:, 0], y=x_train[np.where(y_train==0)][:, 1], name="y=0", mode="markers", marker={"color": "#0000ff"})
            ], layout=go.Layout(yaxis=dict(scaleanchor='x')))
            fig_data_train_xy.update_xaxes(title="x1")
            fig_data_train_xy.update_yaxes(title="x2")
            fig_data_train_yz = go.Figure(data=[
                go.Scatter(x=x_train[np.where(y_train==1)][:, 1], y=x_train[np.where(y_train==1)][:, 2], name="y=1", mode="markers", marker={"color": "#ff0000"}),
                go.Scatter(x=x_train[np.where(y_train==0)][:, 1], y=x_train[np.where(y_train==0)][:, 2], name="y=0", mode="markers", marker={"color": "#0000ff"})
            ], layout=go.Layout(yaxis=dict(scaleanchor='x')))
            fig_data_train_yz.update_xaxes(title="x2")
            fig_data_train_yz.update_yaxes(title="x3")
            fig_data_train_zx = go.Figure(data=[
                go.Scatter(x=x_train[np.where(y_train==1)][:, 2], y=x_train[np.where(y_train==1)][:, 0], name="y=1", mode="markers", marker={"color": "#ff0000"}),
                go.Scatter(x=x_train[np.where(y_train==0)][:, 2], y=x_train[np.where(y_train==0)][:, 0], name="y=0", mode="markers", marker={"color": "#0000ff"})
            ], layout=go.Layout(yaxis=dict(scaleanchor='x')))
            fig_data_train_zx.update_xaxes(title="x3")
            fig_data_train_zx.update_yaxes(title="x1")
            fig_data_val = go.Figure(data=[
                go.Scatter3d(x=x_val[np.where(y_val==1)][:, 0], y=x_val[np.where(y_val==1)][:, 1], z=x_val[np.where(y_val==1)][:, 2], name="y=1", mode="markers", marker={"color": "#ff0000", "size": 3}),
                go.Scatter3d(x=x_val[np.where(y_val==0)][:, 0], y=x_val[np.where(y_val==0)][:, 1], z=x_val[np.where(y_val==0)][:, 2], name="y=0", mode="markers", marker={"color": "#0000ff", "size": 3}),
            ])
            fig_data_val.update_layout(scene=dict(
                xaxis_title='x1',
                yaxis_title='x2',
                zaxis_title='x3'
            ))
            fig_data_val_xy = go.Figure(data=[
                go.Scatter(x=x_val[np.where(y_val==1)][:, 0], y=x_val[np.where(y_val==1)][:, 1], name="y=1", mode="markers", marker={"color": "#ff0000"}),
                go.Scatter(x=x_val[np.where(y_val==0)][:, 0], y=x_val[np.where(y_val==0)][:, 1], name="y=0", mode="markers", marker={"color": "#0000ff"})
            ], layout=go.Layout(yaxis=dict(scaleanchor='x')))
            fig_data_val_xy.update_xaxes(title="x1")
            fig_data_val_xy.update_yaxes(title="x2")
            fig_data_val_yz = go.Figure(data=[
                go.Scatter(x=x_val[np.where(y_val==1)][:, 1], y=x_val[np.where(y_val==1)][:, 2], name="y=1", mode="markers", marker={"color": "#ff0000"}),
                go.Scatter(x=x_val[np.where(y_val==0)][:, 1], y=x_val[np.where(y_val==0)][:, 2], name="y=0", mode="markers", marker={"color": "#0000ff"})
            ], layout=go.Layout(yaxis=dict(scaleanchor='x')))
            fig_data_val_yz.update_xaxes(title="x2")
            fig_data_val_yz.update_yaxes(title="x3")
            fig_data_val_zx = go.Figure(data=[
                go.Scatter(x=x_val[np.where(y_val==1)][:, 2], y=x_val[np.where(y_val==1)][:, 0], name="y=1", mode="markers", marker={"color": "#ff0000"}),
                go.Scatter(x=x_val[np.where(y_val==0)][:, 2], y=x_val[np.where(y_val==0)][:, 0], name="y=0", mode="markers", marker={"color": "#0000ff"})
            ], layout=go.Layout(yaxis=dict(scaleanchor='x')))
            fig_data_val_zx.update_xaxes(title="x3")
            fig_data_val_zx.update_yaxes(title="x1")

            return fig_data_train, fig_data_train_xy, fig_data_train_yz, fig_data_train_zx, fig_data_val, fig_data_val_xy, fig_data_val_yz, fig_data_val_zx


class App3D2D:
    def __init__(self, config: Config, app):
        self.config = config

        self.app = app
        self.layout = html.Div(children=[
            html.Div(children=[
                html.H2(children="Training data", style={"textAlign": "center"}),
                html.Div(children=[
                    html.Div(children=[
                        html.H3(children="Correct data", style={"textAlign": "center"}),
                        dcc.Graph(
                            id="training-data-3d-correct-graph"
                        )
                    ], id="training-data-3d-correct", style={"width": "48%", "margin": "10px"}),
                    html.Div(children=[
                        html.H3(children="Predict data", style={"textAlign": "center"}),
                        dcc.Graph(
                            id="training-data-3d-predict-graph"
                        )
                    ], id="training-data-3d-predict", style={"width": "48%", "margin": "10px"})
                ], id="training-data-3d", style={"display": "flex", "justify-content": "center"}),
                html.Div(children=[
                    html.Div(children=[
                        html.H3(children="x-y plane", style={"textAlign": "center"}),
                        dcc.Graph(
                            id="training-data-xy-graph"
                        )
                    ], id="training-data-xy", style={"width": "32%", "margin": "10px"}),
                    html.Div(children=[
                        html.H3(children="y-z plane", style={"textAlign": "center"}),
                        dcc.Graph(
                            id="training-data-yz-graph"
                        )
                    ], id="training-data-yz", style={"width": "32%", "margin": "10px"}),
                    html.Div(children=[
                        html.H3(children="z-x plane", style={"textAlign": "center"}),
                        dcc.Graph(
                            id="training-data-zx-graph"
                        )
                    ], id="training-data-zx", style={"width": "32%", "margin": "10px"})
                ], id="training-data-projection", style={"display": "flex", "justify-content": "center"})
            ], id="training-data"),
            html.Div(children=[
                html.H2(children="validation data", style={"textAlign": "center"}),
                html.Div(children=[
                    html.Div(children=[
                        html.H3(children="Correct data", style={"textAlign": "center"}),
                        dcc.Graph(
                            id="validation-data-3d-correct-graph"
                        )
                    ], id="validation-data-3d-correct", style={"width": "48%", "margin": "10px"}),
                    html.Div(children=[
                        html.H3(children="Predict data", style={"textAlign": "center"}),
                        dcc.Graph(
                            id="validation-data-3d-predict-graph"
                        )
                    ], id="validation-data-3d-predict", style={"width": "48%", "margin": "10px"})
                ], id="validation-data-3d", style={"display": "flex", "justify-content": "center"}),
                html.Div(children=[
                    html.Div(children=[
                        html.H3(children="x-y plane", style={"textAlign": "center"}),
                        dcc.Graph(
                            id="validation-data-xy-graph"
                        )
                    ], id="validation-data-xy", style={"width": "32%", "margin": "10px"}),
                    html.Div(children=[
                        html.H3(children="y-z plane", style={"textAlign": "center"}),
                        dcc.Graph(
                            id="validation-data-yz-graph"
                        )
                    ], id="validation-data-yz", style={"width": "32%", "margin": "10px"}),
                    html.Div(children=[
                        html.H3(children="z-x plane", style={"textAlign": "center"}),
                        dcc.Graph(
                            id="validation-data-zx-graph"
                        )
                    ], id="validation-data-zx", style={"width": "32%", "margin": "10px"})
                ], id="validation-data-projection", style={"display": "flex", "justify-content": "center"})
            ], id="validation-data")
        ])

        @self.app.callback(
            Output("training-data-3d-correct-graph", "figure"),
            Output("validation-data-3d-correct-graph", "figure"),
            Input("interval-component", "n_intervals")
        )
        def update_dataset_correct(n):
            with open(self.config.resource.result_visualize_data_path, "rt") as f:
                dataset = json.load(f)
            x_train = np.array(dataset.get("Train").get("Input"))
            y_train = np.array(dataset.get("Train").get("Output"))
            y_train = np.argmax(y_train, 1)
            x_val = np.array(dataset.get("Validation").get("Input"))
            y_val = np.array(dataset.get("Validation").get("Output"))
            y_val = np.argmax(y_val, 1)

            fig_data_train = go.Figure()
            fig_data_val = go.Figure()
            colors_data = self.get_colorpalette("hls", self.config.model.output_dimension)
            for i in range(self.config.model.output_dimension):
                fig_data_train.add_trace(go.Scatter3d(x=x_train[np.where(y_train==i)][:, 0], y=x_train[np.where(y_train==i)][:, 1], z=x_train[np.where(y_train==i)][:, 2], name="y={}".format(i), mode="markers", marker={"color": colors_data[i], "size": 3}))
                fig_data_val.add_trace(go.Scatter3d(x=x_val[np.where(y_val==i)][:, 0], y=x_val[np.where(y_val==i)][:, 1], z=x_val[np.where(y_val==i)][:, 2], name="y={}".format(i), mode="markers", marker={"color": colors_data[i], "size": 3}))
            fig_data_train.update_layout(scene=dict(
                xaxis_title='x1',
                yaxis_title='x2',
                zaxis_title='x3'
            ))
            fig_data_val.update_layout(scene=dict(
                xaxis_title='x1',
                yaxis_title='x2',
                zaxis_title='x3'
            ))

            return fig_data_train, fig_data_val
        
        @self.app.callback(
            Output("training-data-3d-predict-graph", "figure"),
            Output("training-data-xy-graph", "figure"),
            Output("training-data-yz-graph", "figure"),
            Output("training-data-zx-graph", "figure"),
            Output("validation-data-3d-predict-graph", "figure"),
            Output("validation-data-xy-graph", "figure"),
            Output("validation-data-yz-graph", "figure"),
            Output("validation-data-zx-graph", "figure"),
            Input("interval-component", "n_intervals")
        )
        def update_dataset_predict(n):
            with open(self.config.resource.result_visualize_data_predict_path, "rt") as f:
                dataset = json.load(f)
            x_train = np.array(dataset.get("Train").get("Input"))
            y_train = np.array(dataset.get("Train").get("Output"))
            y_train = np.argmax(y_train, 1)
            x_val = np.array(dataset.get("Validation").get("Input"))
            y_val = np.array(dataset.get("Validation").get("Output"))
            y_val = np.argmax(y_val, 1)

            fig_data_train = go.Figure()
            fig_data_train_xy = go.Figure()
            fig_data_train_yz = go.Figure()
            fig_data_train_zx = go.Figure()
            fig_data_val = go.Figure()
            fig_data_val_xy = go.Figure()
            fig_data_val_yz = go.Figure()
            fig_data_val_zx = go.Figure()
            colors_data = self.get_colorpalette("hls", self.config.model.output_dimension)
            for i in range(self.config.model.output_dimension):
                fig_data_train.add_trace(go.Scatter3d(x=x_train[np.where(y_train==i)][:, 0], y=x_train[np.where(y_train==i)][:, 1], z=x_train[np.where(y_train==i)][:, 2], name="y={}".format(i), mode="markers", marker={"color": colors_data[i], "size": 3}))
                fig_data_train_xy.add_trace(go.Scatter(x=x_train[np.where(y_train==i)][:, 0], y=x_train[np.where(y_train==i)][:, 1], name="y={}".format(i), mode="markers", marker={"color": colors_data[i]}))
                fig_data_train_yz.add_trace(go.Scatter(x=x_train[np.where(y_train==i)][:, 1], y=x_train[np.where(y_train==i)][:, 2], name="y={}".format(i), mode="markers", marker={"color": colors_data[i]}))
                fig_data_train_zx.add_trace(go.Scatter(x=x_train[np.where(y_train==i)][:, 2], y=x_train[np.where(y_train==i)][:, 0], name="y={}".format(i), mode="markers", marker={"color": colors_data[i]}))
                fig_data_val.add_trace(go.Scatter3d(x=x_val[np.where(y_val==i)][:, 0], y=x_val[np.where(y_val==i)][:, 1], z=x_val[np.where(y_val==i)][:, 2], name="y={}".format(i), mode="markers", marker={"color": colors_data[i], "size": 3}))
                fig_data_val_xy.add_trace(go.Scatter(x=x_val[np.where(y_val==i)][:, 0], y=x_val[np.where(y_val==i)][:, 1], name="y={}".format(i), mode="markers", marker={"color": colors_data[i]}))
                fig_data_val_yz.add_trace(go.Scatter(x=x_val[np.where(y_val==i)][:, 1], y=x_val[np.where(y_val==i)][:, 2], name="y={}".format(i), mode="markers", marker={"color": colors_data[i]}))
                fig_data_val_zx.add_trace(go.Scatter(x=x_val[np.where(y_val==i)][:, 2], y=x_val[np.where(y_val==i)][:, 0], name="y={}".format(i), mode="markers", marker={"color": colors_data[i]}))
            fig_data_train.update_layout(scene=dict(
                xaxis_title='x1',
                yaxis_title='x2',
                zaxis_title='x3'
            ))
            fig_data_train_xy.update_xaxes(title="x1")
            fig_data_train_xy.update_yaxes(title="x2")
            fig_data_train_yz.update_xaxes(title="x2")
            fig_data_train_yz.update_yaxes(title="x3")
            fig_data_train_zx.update_xaxes(title="x3")
            fig_data_train_zx.update_yaxes(title="x1")
            fig_data_val.update_layout(scene=dict(
                xaxis_title='x1',
                yaxis_title='x2',
                zaxis_title='x3'
            ))
            fig_data_val_xy.update_xaxes(title="x1")
            fig_data_val_xy.update_yaxes(title="x2")
            fig_data_val_yz.update_xaxes(title="x2")
            fig_data_val_yz.update_yaxes(title="x3")
            fig_data_val_zx.update_xaxes(title="x3")
            fig_data_val_zx.update_yaxes(title="x1")
            return fig_data_train, fig_data_train_xy, fig_data_train_yz, fig_data_train_zx, fig_data_val, fig_data_val_xy, fig_data_val_yz, fig_data_val_zx

    def get_colorpalette(self, colorpalette, n_colors):
        palette = sns.color_palette(colorpalette, n_colors)
        rgb = ['rgb({},{},{})'.format(*[x*256 for x in rgb]) for rgb in palette]
        return rgb[::-1]