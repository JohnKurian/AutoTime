# import dash
# import dash_table
# import pandas as pd
#
# df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/solar.csv')
#
# app = dash.Dash(__name__)
#
import dash_core_components as dcc
import plotly.graph_objs as go


#
# app.layout = dash_table.DataTable(
#         id='datatable-interactivity',
#         columns=[
#             {"name": i, "id": i, "deletable": False, "selectable": True} for i in df.columns
#         ],
#         data=df.to_dict('records'),
#         editable=False,
#         filter_action="native",
#         sort_action="native",
#         sort_mode="multi",
#         column_selectable="single",
#         row_selectable="multi",
#         row_deletable=False,
#         selected_columns=[],
#         selected_rows=[],
#         page_action="native",
#         page_current= 0,
#         page_size= 10,
#     )
#
# if __name__ == '__main__':
#     app.run_server(debug=True)
#


# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    'background': 'white',
    'text': 'black'
}

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

fig1 = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[4, 1, 2])])


fig.update_layout(
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text']
)

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='AutoML',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),

    html.Div(children='AutoML: Automated machine learning service', style={
        'textAlign': 'center',
        'color': colors['text']
    }),

    dcc.Graph(
        id='example-graph-2',
        figure=fig
    ),

    dcc.Graph(
            id='example-graph-3',
            figure=fig1
        )


])

if __name__ == '__main__':
    app.run_server(debug=True)