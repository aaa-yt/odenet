import os
import sys
from datetime import datetime
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
sys.path.append("../")
from config import Config
from visualize.app import App

def start(config: Config):
    return AppManager(config).start()


class AppManager:
    def __init__(self, config: Config):
        self.config = config

        style_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.css")
        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', style_path]
        self.app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

        self.app.layout = html.Div(children=[
            dcc.Location(id="url", refresh=False),
            html.H4(id="text-datetime", style={"textAlign": "right"}),
            html.Div(id="page-content"),
            dcc.Interval(
                id="interval-component",
                interval=10000,
                n_intervals=0
            )
        ])

        if self.config.model.input_dimension == 1 and self.config.model.output_dimension == 1:
            from visualize.app_data import App1D1D
            self.app2 = App1D1D(self.config, self.app)
           
        self.app1 = App(self.config, self.app)

        @self.app.callback(
            Output("page-content", "children"),
            Input("url", "pathname")
        )
        def display_page(pathname):
            if pathname == "/data": return self.app2.layout
            return self.app1.layout
        
        @self.app.callback(
            Output("text-datetime", "children"),
            Input("interval-component", "n_intervals")
        )
        def update_text_datetime(n):
            text = datetime.now().strftime("%Y/%m/%d  %H:%M:%S")
            return text
    
    def start(self):
        self.app.run_server()