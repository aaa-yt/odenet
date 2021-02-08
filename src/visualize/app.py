import sys
import json
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append("../")
from config import Config

class App:
    def __init__(self, config: Config, app):
        self.config = config

        self.app = app
        self.layout = html.Div(children=[
            html.H1(id="text-epoch", style={"textAlign": "center"}),
            html.Div(children=[
                html.Div(children=[
                    html.H2(children="Loss", style={"textAlign": "center"}),
                    dcc.Graph(
                        id="loss-graph"
                    )
                ], id="loss", style={"width": "48%", "margin": "10px"}),
                html.Div(children=[
                    html.H2(children="Accuracy", style={"textAlign": "center"}),
                    dcc.Graph(
                        id="accuracy-graph"
                    )
                ], id="accuracy", style={"width": "48%", "margin": "10px"})
            ], id="learning-curve", style={"display": "flex", "justify-content": "center"}),
            html.Div(children=[
                html.Div(children=[
                    html.H2(children="Parameters", style={"textAlign": "center"}),
                    dcc.Graph(
                        id="parameter-all-graph"
                    )
                ], id="parameter-all", style={"width": "60%", "margin": "10px auto"}),
                html.Div(children=[
                    html.Div(children=[
                        html.H2(children="a(t)", style={"textAlign": "center"}),
                        dcc.Graph(
                            id="parameter-a-graph"
                        )
                    ], id="parameter-a", style={"width": "32%", "margin": "10px"}),
                    html.Div(children=[
                        html.H2(children="W(t)", style={"textAlign": "center"}),
                        dcc.Graph(
                            id="parameter-W-graph"
                        )
                    ], id="parameter-W", style={"width": "32%", "margin": "10px"}),
                    html.Div(children=[
                        html.H2(children="b(t)", style={"textAlign": "center"}),
                        dcc.Graph(
                            id="parameter-b-graph"
                        )
                    ], id="parameter-b", style={"width": "32%", "margin": "10px"})
                ], id="parameter-each", style={"display": "flex", "justify-content": "center"}),
            ], id="parameters")
        ])
        
        @self.app.callback(
            Output("text-epoch", "children"),
            Output("loss-graph", "figure"),
            Output("accuracy-graph", "figure"),
            Input("interval-component", "n_intervals")
        )
        def update_learning_curve(n):
            df = pd.read_csv(self.config.resource.result_visualize_learning_curve_path)
            epoch = df["epoch"].values[-1]
            text = "Epoch: {}".format(epoch)
            if self.config.trainer.regularizer_type == "None":
                fig_loss = go.Figure(data=[
                    go.Scatter(x=df["epoch"], y=df["loss_train"], name="train", mode="lines", marker={"color": "#0000ff"}),
                    go.Scatter(x=df["epoch"], y=df["loss_validation"], name="validation", mode="lines", marker={"color": "#ff0000"})
                ])
            else:
                fig_loss = go.Figure(data=[
                    go.Scatter(x=df["epoch"], y=df["loss_train"], name="train", mode="lines", marker={"color": "#0000ff"}),
                    go.Scatter(x=df["epoch"], y=df["loss_train_regularizer"], name="train regularizer", mode="lines", marker={"color": "#bbbbff"}),
                    go.Scatter(x=df["epoch"], y=df["loss_validation"], name="validation", mode="lines", marker={"color": "#ff0000"}),
                    go.Scatter(x=df["epoch"], y=df["loss_validation_regularizer"], name="validation regularizer", mode="lines", marker={"color": "#ffbbbb"})
                ])
            fig_loss.update_xaxes(title="epoch")
            fig_loss.update_yaxes(title="Loss")

            if self.config.trainer.is_accuracy:
                fig_acc = go.Figure(data=[
                    go.Scatter(x=df["epoch"], y=df["accuracy_train"], name="train", mode="lines", marker={"color": "#0000ff"}),
                    go.Scatter(x=df["epoch"], y=df["accuracy_validation"], name="validation", mode="lines", marker={"color": "#ff0000"})
                ])
                fig_acc.update_xaxes(title="epoch")
                fig_acc.update_yaxes(title="Accuracy")
            else:
                fig_acc = None
            return text, fig_loss, fig_acc
        
        @self.app.callback(
            Output("parameter-all-graph", "figure"),
            Output("parameter-a-graph", "figure"),
            Output("parameter-W-graph", "figure"),
            Output("parameter-b-graph", "figure"),
            Input("interval-component", "n_intervals")
        )
        def update_model_parameter(n):
            with open(self.config.resource.result_visualize_model_path, "rt") as f:
                model = json.load(f)
            a = np.array(model.get("a"))
            W = np.array(model.get("W"))
            b = np.array(model.get("b"))
            N = self.config.model.input_dimension + self.config.model.output_dimension
            t = np.linspace(0., self.config.model.maximum_time, self.config.model.weights_division)

            fig_params = go.Figure()
            fig_a = go.Figure()
            fig_W = go.Figure()
            fig_b = go.Figure()
            colors_a = self.get_colorpalette("Reds", 2 * N)
            colors_W = self.get_colorpalette("Greens", 2 * N * N)
            colors_b = self.get_colorpalette("Blues", 2* N)

            for i in range(N):
                fig_params.add_trace(go.Scatter(x=t, y=a[:, i], name='a{}(t)'.format(i), mode="lines", marker={'color': colors_a[i]}))
                fig_a.add_trace(go.Scatter(x=t, y=a[:, i], name='a{}(t)'.format(i), mode="lines", marker={'color': colors_a[i]}))
            for i in range(N):
                for j in range(N):
                    fig_params.add_trace(go.Scatter(x=t, y=W[:, i, j], name='W{}{}(t)'.format(i, j), mode="lines", marker={'color': colors_W[i * N + j]}))
                    fig_W.add_trace(go.Scatter(x=t, y=W[:, i, j], name='W{}{}(t)'.format(i, j), mode="lines", marker={'color': colors_W[i * N + j ]}))
            for i in range(N):
                fig_params.add_trace(go.Scatter(x=t, y=b[:, i], name='b{}(t)'.format(i), mode="lines", marker={'color': colors_b[i]}))
                fig_b.add_trace(go.Scatter(x=t, y=b[:, i], name='b{}(t)'.format(i), mode="lines", marker={'color': colors_b[i]}))
            fig_params.update_xaxes(title='t')
            fig_a.update_xaxes(title='t')
            fig_W.update_xaxes(title='t')
            fig_b.update_xaxes(title='t')
            fig_a.update_yaxes(title='a(t)')
            fig_W.update_yaxes(title='W(t)')
            fig_b.update_yaxes(title='b(t)')

            return fig_params, fig_a, fig_W, fig_b

    def get_colorpalette(self, colorpalette, n_colors):
        palette = sns.color_palette(colorpalette, n_colors)
        rgb = ['rgb({},{},{})'.format(*[x*256 for x in rgb]) for rgb in palette]
        return rgb[::-1]
