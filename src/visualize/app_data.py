import sys
import json
from datetime import datetime
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np

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
        def update_training_data(n):
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