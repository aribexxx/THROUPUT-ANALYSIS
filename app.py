#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib, os, json, math, datetime, copy, dash
import pandas as pd
import numpy as np
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import plotly.express as px
import constants
from datetime import date
from plotly.subplots import make_subplots

# app initialize
app = dash.Dash(__name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)
server = app.server
app.config["suppress_callback_exceptions"] = True


# Load data
APP_PATH = str(pathlib.Path().parent.resolve())


# In[2]:


# 这里改 data source
df = pd.read_csv(os.path.join(APP_PATH, os.path.join("data", "Query-both-whole.csv")))
df['Datetime'] = list(map(lambda s: datetime.datetime.strptime(s,'%m/%d/%Y'),df.loc[:,'date']))
df_position = pd.read_csv(os.path.join(APP_PATH, os.path.join("data", "location.csv")))
df_part1 = pd.read_csv(os.path.join(APP_PATH, os.path.join("data", "main_part1.csv")))
df_covid = pd.read_csv(os.path.join(APP_PATH, os.path.join("data", "COVID-19.csv")),infer_datetime_format=True,parse_dates=[0])
df_weather= pd.read_csv('./data/Statistics-weather.csv')
df_total_tourist= pd.read_csv('./data/Statistics-arrival-departure.csv')
#大陆cpi 按月
cpi_china = pd.read_csv(os.path.join(APP_PATH, os.path.join("data","chinese_cpi.csv")))
cpi_china['date']=list(map(lambda s: datetime.datetime.strptime(s,'%Y/%m/%d'),cpi_china.loc[:,'Date']))
df_arrive_total = pd.read_csv(os.path.join(APP_PATH, os.path.join("data","mean_Arrival_Departure_By_month.csv")))
# Assign color to legend
colormap = {}
colorlist = constants.colors
for ind, formation_name in enumerate(df["control point"].unique().tolist()):
    colormap[formation_name] = colorlist[ind]


# In[3]:


type2realtype = {
    "Mainland": "Mainland Visitors",
    "HK": "Hong Kong Residents",
    "Other": "Other Visitors",
    "All": "Total"
}
type2number = {
    "Mainland": 0,
    "HK" : 1,
    "Other": 2,
    "All": 3,
    "Arrival": 0,
    "Departure": 1
}


# In[4]:


def build_banner():
    return html.Div(
        id="banner",
        className="banner",
        children=[
            #html.Img(src=app.get_asset_url("logo.png"),width=300,height=150), 
            html.H6("Control point throughput"),
        ],
    )


def build_graph_title(title):
    return html.P(className="graph-title", children=title)

def lastxdays(x):
    d = {}
    d['date'] = [k for k in df_covid["Report date"].unique() if k > pd.Timestamp(2020, 1, 23) and k < pd.Timestamp(2020, 9, 23)]
    types = ['Elderly','Child','Total']
    temp = []
    for k in d['date']:
        temp.append(len(df_covid.loc[(df_covid['Age'] >= 60) & (df_covid['Report date'] <= k)                                      & (df_covid['Report date'] > k - np.timedelta64(x,'D'))]))
    d["Last {}days {} Cases".format(str(x),types[0])] = temp
    temp = []
    for k in d['date']:
        temp.append(len(df_covid.loc[(df_covid['Age'] <= 14) & (df_covid['Report date'] <= k)                                      & (df_covid['Report date'] > k - np.timedelta64(x,'D'))]))
    d["Last {}days {} Cases".format(str(x),types[1])] = temp
    temp = []
    for k in d['date']:
        temp.append(len(df_covid.loc[(df_covid['Report date'] <= k)                                      & (df_covid['Report date'] > k - np.timedelta64(x,'D'))]))
    d["Last {}days {} Cases".format(str(x),types[2])] = temp
    return pd.DataFrame(data=d)


# In[5]:


app.layout = html.Div(
    children=[
        html.Div(
            id="top-row",
            children=[
                html.Div(
                    className="row",
                    id="top-row-header",
                    children=[
                        html.Div(
                            id="header-container",
                            children=[
                                build_banner(),
                                html.P(
                                    id="instructions",
                                    children='''This demo is a visual analysis tool that contains six figures demonstrating
                                    the relations between the throughput at control points of Hong Kong and various potential factors.''', #修改一下
                                ),
                                build_graph_title("Select Operator"),
                                dcc.Dropdown(
                                    id="operator-select-EE",
                                    options=[
                                        {"label": i, "value": i}
                                        for i in df["Arrival / Departure"].unique().tolist()
                                    ],
                                    multi=True,
                                    value=[
                                        df["Arrival / Departure"].unique().tolist()[0],
                                        df["Arrival / Departure"].unique().tolist()[1]
                                    ],
                                    style={"margin-bottom":"8px"}
                                ),
                                dcc.Dropdown(
                                    id="operator-select-type",
                                    options=[
                                        {"label": "Mainland", "value": "Mainland"},
                                        {"label": "HK", "value": "HK"},
                                        {"label": "Other", "value": "Other"},
                                        {"label": "Sum of above", "value": "All"},
                                    ],
                                    multi=True,
                                    value=[
                                        "Mainland"
                                    ],
                                    style={"margin-bottom":"10px"}
                                ),
                                build_graph_title("COVID-19 restriction"),
                                dcc.RadioItems(
                                    id="COVID-selector",
                                    options=[
                                        {"label": "Before", "value": "prior to"},
                                        {"label": "After", "value": "post"}
                                    ],
                                    value="prior to",
                                    labelStyle={'display': 'inline-block'},
                                    style = {"color": "white"}
                                ),
                            ],
                        )
                    ],
                ),
                html.Div(
                    className="row",
                    id="top-row-graphs",
                    children=[
                        # Well map
                        html.Div(
                            id="well-map-container",
                            children=[
                                build_graph_title("Throughput at control point"),
                                dcc.RadioItems(
                                    id="mapbox-view-selector",
                                    options=[
                                        {"label": "sum", "value": "sum"},
                                        {"label": "average", "value": "average"}
                                    ],
                                    value="sum",
                                ),
                                html.Div(
                                    children=[
                                        html.Label(children=["From:  "],style={"color": "white","display":"inline-block"}),
                                        dcc.Input(
                                            id="start-date-input",
                                            type="text",
                                            placeholder="MM/DD/YYYY",
                                            value="01/01/2019",
                                            style={"display":"inline-block","margin-left":"5px","margin-right":"5px","width":110}
                                        ),
                                        html.Label(children=["To:  "],style={"color": "white","display":"inline-block"}),
                                        dcc.Input(
                                            id="end-date-input",
                                            type="text",
                                            placeholder="MM/DD/YYYY",
                                            value="09/30/2020",
                                            style={"display":"inline-block","margin-left":"5px","margin-right":"5px","width":110}
                                        ),
                                        html.Button('Search', id='button-search', style={"backgroundColor": "white","margin-left":"10px"}),
                                    ],
                                    style={"margin-bottom":"10px"}
                                ),  
                                dcc.Graph(
                                    id="well-map",
                                    figure={
                                        "layout": {
                                            "paper_bgcolor": "#192444",
                                            "plot_bgcolor": "#192444",
                                        }
                                    },
                                    config={"scrollZoom": True, "displayModeBar": True},
                                ),
                            ],
                        ),
                        # Ternary map
                        html.Div(
                            id="ternary-map-container",
                            children=[
                                html.Div(
                                    id="ternary-header",
                                    children=[
                                        build_graph_title(
                                            "Overview: throughput over time"
                                        ),
                                    ],
                                ),
                                dcc.Graph(
                                    id="ternary-map",
                                    figure={
                                        "layout": {
                                            "paper_bgcolor": "#192444",
                                            "plot_bgcolor": "#192444",
                                        }
                                    },
                                    config={
                                        "scrollZoom": True,
                                        "displayModeBar": False,
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            className="row",
            id="bottom-row",
            children=[
                # Formation bar plots
                html.Div(
                    id="form-bar-container",
                    className="six columns",
                    children=[
                        build_graph_title("POLAR SCATTER GRAPH OF COMMERCIAL IMPACT"),
                        dcc.DatePickerRange(
                            id='my-picker-range',   
                            min_date_allowed=date(2019, 4, 4),
                            max_date_allowed=date(2020, 9, 30),
                            start_date=date(2019, 4, 4),
                            end_date=date(2020, 9, 30),
                            style={"margin-bottom":"5px"},
                        ),
                        dcc.Dropdown(
                            id="operator-select-variables",
                            options=[
                                {"label": "URBAN CPI OF PCR", "value": "PCR_URBAN"},
                                {"label": "COUNTRYSIDE CPI OF PCR", "value": "PCR_COUNTRYSIDE"},
                                {"label": "TOTAL CPI OF PCR", "value": "PCR_TOTAL"},
                            ],
                            multi=True,
                            value="",
                            style={"margin-bottom":"8px"},
                        ),
                        dcc.Graph(id="form-by-bar"),
                    ],
                ),
                html.Div(
                    # Selected well productions
                    id="well-production-container",
                    className="six columns",
                    children=[
                        build_graph_title("WEATHER IMPACT"),
                        dcc.DatePickerRange(
                            id='my-date-picker-range',   
                            min_date_allowed=date(2019, 4, 4),
                            max_date_allowed=date(2020, 9, 30),
                            start_date=date(2019, 4, 4),
                            end_date=date(2020, 9, 30),
                            style={"margin-bottom":"5px"},
                        ),
                        html.Div(id='output-container-date-picker-range'),
                        dcc.Graph(
                            id="3d-scatter", 
                            figure={
                                "layout": {
                                    "paper_bgcolor": "#192444",
                                    "plot_bgcolor": "#192444",
                                }
                            }
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            className="row",
            id="appending",
            children=[
                # Stacked bar plots
                html.Div(
                    id="stack-container",
                    className="six columns",
                    children=[
                        build_graph_title("STACK PERCENTAGE GRAPH - BEFORE COVID-19"),
                        dcc.Dropdown(
                            id="operator-select-variable",
                            options=[
                                {"label": "Mainland public holiday", "value": "Festival"},
                                {"label": "Day of week", "value": "DayofWeek"},
                                {"label": "Precipitation", "value": "Weather"},
                                {"label": "PCR GDP Growth rate", "value": "PCR_GDP_GR"},
                                {"label": "HK GDP Growth rate", "value": "HK_GDP_GR"},
                                {"label": "PCR CPI Growth rate", "value": "PCR_CPI_GR"},
                                {"label": "HK CPI Growth rate", "value": "HK_CPI_GR"},
                            ],
                            multi=False,
                            value="Festival",
                            style={"margin-bottom":"8px"},
                        ),
                        dcc.Graph(id="form-by-stack"),
                    ],
                ),
                html.Div(
                    id="parallel-container",
                    className="six columns",
                    children=[
                        build_graph_title("COVID-19 IMPACT"),
                        html.Label(children=["Last X days:  "],style={"color": "black","display":"inline-block"}),
                        dcc.Slider(
                            id='my-slider',
                            min=1,
                            max=30,
                            step=1,
                            value=10,
                            marks={
                                1: '1',
                                10: '10',
                                20: '20',
                                30: '30',
                            },
                        ),
                        dcc.Graph(id="parallel-coordinate-plot"),
                    ],
                )
            ],
        ),
    ]
)


# In[6]:


# Update street map
@app.callback(
    Output("well-map", "figure"),
    [
        Input("operator-select-type", "value"),
        Input("operator-select-EE", "value"),
        Input("mapbox-view-selector", "value"),
        Input("button-search", "n_clicks")
    ],
    [
        State("start-date-input", "value"),
        State("end-date-input", "value"),
    ],
)
def generate_well_map(selected_type,selected_EE, selected_method, _, date1,date2):
    """
    Generate well map based on selected data.

    :param dff: dataframe for generate plot.
    :param selected_data: Processed dictionary for plot generation with defined selected points.
    :param style: mapbox visual style.
    :return: Plotly figure object.
    """
    date1, date2 = list(map(lambda s: datetime.datetime.strptime(s,'%m/%d/%Y'), (date1,date2)))
    layout = go.Layout(
        clickmode="event+select",
        dragmode="lasso",
        showlegend=True,
        autosize=False,
        width=450,
        height=400,
        hovermode="closest",
        margin=dict(l=0, r=0, t=0, b=0),
        mapbox=go.layout.Mapbox(
            bearing=0,
            center=go.layout.mapbox.Center(lat=22.3433953, lon=114.0565349), #换香港 22.3433953,114.0565349
            pitch=0,
            zoom=9,
            style='open-street-map',
        ),
        legend=dict(
            bgcolor="#1f2c56",
            orientation="h",
            font=dict(color="white"),
            x=0,
            y=0,
            yanchor="bottom",
        ),
    )

    text_list = list(df_position['Spot'])
    if len(selected_EE) == 1:
        df_temp = df.loc[(df["Datetime"]>=date1)&(df["Datetime"]<=date2)&(df["Arrival / Departure"]==selected_EE[0]), ["control point"]+[type2realtype[i] for i in selected_type]]
    elif len(selected_EE) == 2:
        df_temp = df.loc[(df["Datetime"] >= date1) & (df["Datetime"] <= date2), ["control point"]+[type2realtype[i] for i in selected_type]]
    else:
         df_temp = df.loc[(df["Datetime"] >= date1) & (df["Datetime"] <= date2)&(df["Arrival / Departure"]==None), ["control point"]+[type2realtype[i] for i in selected_type]]
    num_list = []
    for t in df_position['Spot']:
        temp = df_temp.loc[df_temp["control point"] == t]
        if not len(temp):
            num_list.append(0)
            continue
        if selected_method == 'sum':
            num_list.append(temp.sum(numeric_only=True).values.sum())
        else:
            num_list.append(round(temp.mean(numeric_only=True).values.sum(),4))

    text_list = ["Throughput: " + str(num_list[i]) + "<br>" + text_list[i] for i in range(len(text_list))]
    maxsize, minsize = max(num_list), min(num_list)
    if maxsize == minsize:
        size_list = [10]*len(text_list)
    else:
        size_list = [int(6 + 15 * (i-minsize)/(maxsize-minsize)) for i in num_list]
    data = [go.Scattermapbox(
        lat=[lat],
        lon=[lon],
        mode="markers",
        marker={"color": c, "size": size},
        text=[text],
        name="Control point",
        showlegend=False,
        customdata=[text[text.find("<br>")+4:]]
    ) for c, lat, lon, size, text in zip(colorlist,df_position["lat"],df_position["lon"],size_list,text_list)] 
    return {"data": data, "layout": layout}


# In[7]:


# Update overview map
@app.callback(Output("ternary-map", "figure"),
              [
                  Input("operator-select-type", "value"),
                  Input("operator-select-EE", "value"),
                  Input("COVID-selector", "value"),
                  Input('well-map', 'selectedData'),
              ])
def generate_overview_map(selected_type,selected_EE,selected_COVID, selection):

    layout_individual = dict(
        autosize=False,
        width=600,
        height=485,
        automargin=True,
        margin=dict(l=30, r=30, b=20, t=40),
        hovermode="closest",
        legend=dict(orientation="h"),
        title="Throughout Over Time",
        mapbox=dict(
            style="open-street-map",
            center=dict(lon=-78.05, lat=42.54),
            zoom=7,
        ),
    )
    data = []
    selection = [j['customdata'] for j in selection['points']] if selection else None
    if selected_type == None or selected_EE == None:
        annotation = dict(
            text="No data available",
            x=1,
            y=1,
            align="center",
            showarrow=False,
            xref="paper",
            yref="paper",
        )
        layout_individual["annotations"] = [annotation]
    else:
        i = 0
        for mytype in selected_type:
            for myEE in selected_EE:
                if not selection:
                    temp = df.loc[df["Arrival / Departure"]==myEE][["Datetime",type2realtype[mytype]]]
                    temp.set_index("Datetime", inplace = True)
                    temp1 = temp.sum(level="Datetime")
                    xdata, ydata = temp1.index, temp1[type2realtype[mytype]]
                else:
                    
                    temp = df.loc[(df["Arrival / Departure"]==myEE) & ([j in selection for j in df["control point"]])][["Datetime",type2realtype[mytype]]]
                    temp.set_index("Datetime", inplace = True)
                    temp1 = temp.sum(level="Datetime")
                    xdata, ydata = temp1.index, temp1[type2realtype[mytype]]
                if selected_COVID == "prior to":
                    data.append(
                         go.Scatter(
                        dict(
                            mode="lines",
                            name="{} {}".format(mytype,myEE),
                            x=xdata[xdata<=np.datetime64('2020-01-30T00:00:00.000000000')],
                            y=ydata[xdata<=np.datetime64('2020-01-30T00:00:00.000000000')],
                            line=dict(shape="spline",smoothing=0, color=colorlist[type2number[myEE]*4+type2number[mytype]]),
                            stackgroup='one' # define stack group
                        )
                         )
                    )
                else:
                    data.append(
                        go.Scatter(
                        dict(
                            mode="lines",
                            name="{} {}".format(mytype,myEE),
                            x=xdata[xdata>np.datetime64('2020-01-30T00:00:00.000000000')],
                            y=ydata[xdata>np.datetime64('2020-01-30T00:00:00.000000000')],
                            line=dict(shape="spline", smoothing=0, color=colorlist[type2number[myEE]*4+type2number[mytype]]),
                            stackgroup='one' # define stack group
                        )
                        )
                    )
                    
                i += 1
    figure = dict(data=data, layout=layout_individual)
    return figure


# In[8]:


# Update street map
@app.callback(
    Output("form-by-stack", "figure"),
    [
        Input('well-map', 'selectedData'),
        Input('operator-select-variable', 'value')
    ]
)

def generate_percentage_map(selection, variable):
    if not variable:
        variable = "Festival"
    selection = [j['customdata'] for j in selection['points']] if selection else None
    selectedcolumn = ['Arrival / Departure', 'Hong Kong Residents', 'Mainland Visitors', 'Other Visitors', 'Total', (variable if variable else 'Festival')]
    if selection:
        temp = df_part1.loc[[str(j) in selection for j in df_part1["Control Point"]]][selectedcolumn]
    else:
        temp = copy.deepcopy(df_part1.loc[:,selectedcolumn])
    top_labels = [str(j) if type(j) != np.float64 else str(round(j*100,3))+'%' for j in sorted(temp[variable].unique())]
    temp.set_index(["Arrival / Departure",variable], inplace = True)
    temp = round(temp.mean(level=["Arrival / Departure",variable]),2)
    x_data, y_data = [], []
    for i in ["Arrival","Departure"]:
        for j in zip(['Hong Kong Residents', 'Mainland Visitors', 'Other Visitors', 'Total'],['HK', 'Mainland', 'Other', 'Total']):
            newlist = temp.loc[i][j[0]]
            newlist = newlist.sort_index()
            x_data.append((round(newlist/newlist.sum()*100,2) if newlist.sum() else newlist).values)
            y_data.append("{} {}".format(j[1],i))
    x_data.reverse()
    y_data.reverse()
    x_data = np.array(x_data)
    bar_fig = []
    for index, prop in enumerate(top_labels):
        bar_fig.append(
            go.Bar(
                name=prop,
                orientation="h",
                y=y_data,
                x=x_data[:,index],
            )
        )
    figure = go.Figure(
        data=bar_fig,
        layout=dict(
            barmode="stack",
            colorway=constants.conti_colors,
            yaxis_type="category",
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=True,
            yaxis=dict(title="Category"),
            paper_bgcolor="rgb(255,255,255)",
            plot_bgcolor="rgb(255,255,255)",
            transition_duration=500,
        ),
    )
    return figure


# In[9]:


# Update street map
@app.callback(Output("parallel-coordinate-plot", "figure"),
              [
                  Input("operator-select-type", "value"),
                  Input("operator-select-EE", "value"),
                  Input('well-map', 'selectedData'),
                  Input('my-slider', 'value'),
              ]
)

def generate_parallel_coordinates_plot(selected_type, selected_EE, selection, selected_days):
    selection = [j['customdata'] for j in selection['points']] if selection else None
    if (not selected_type) or (not selected_EE):
         x_data = pd.DataFrame(data={
             "date": [0],
             "Last {}days Elderly Cases".format(str(selected_days)): [4],
             "Last {}days Child Cases".format(str(selected_days)): [4],
             "Last {}days Total Cases".format(str(selected_days)): [4],
             "Throughput": [4]
         })
    else:
        if not selection:
            temp = df.loc[[i in selected_EE for i in df["Arrival / Departure"]]]             [["Datetime"]+[type2realtype[i] for i in selected_type]]
        else:
            temp = df.loc[np.array([i in selected_EE for i in df["Arrival / Departure"]]) &                           np.array([j in selection for j in df["control point"]])][["Datetime"]+                             [type2realtype[i] for i in selected_type]]
        temp.set_index("Datetime", inplace = True)
        temp1 = temp.sum(level="Datetime")
        temp1 = temp1.sum(axis=1)
        d = {m:math.log(n+1,10) for m, n in zip(temp1.index, temp1)}
        x_data = lastxdays(selected_days)
        x_data['Throughput'] = [d[i] for i in x_data['date']]
    fig = px.parallel_coordinates(x_data, color="Throughput", labels={"Throughput": "Log Throughput",
                    x_data.columns[1]: x_data.columns[1], x_data.columns[2]: x_data.columns[2],
                    x_data.columns[3]: x_data.columns[3],},
                                 color_continuous_scale=px.colors.sequential.Viridis,
                                 color_continuous_midpoint=4)
    return fig


# In[10]:


index_cpi_all=cpi_china['total_cpi'].tolist()
index_cpi_urban=cpi_china['urban_cpi'].tolist()
index_cpi_countryside=cpi_china['countryside_cpi'].tolist()
date_cpi=cpi_china['Date'].tolist()
departure=np.round(np.log(df_arrive_total.num_departure.values),2)
arrive=np.round(np.log(df_arrive_total.num_arrive.values),2)
inflow=df_arrive_total.num_inflow
date_arrive=df_arrive_total.Date

@app.callback(
    Output('form-by-bar', 'figure'),
    
    [
         Input('my-picker-range', 'start_date'),
         Input('my-picker-range', 'end_date'),
         Input('operator-select-variables', 'value'),
         
    ],    

)
def get_polarfigure(dateStart,dateEnd,selected):
    index_year_start=dateStart.find('-')
    index_month_start=dateStart.find('-',5)
    index_year_end=dateEnd.find('-')
    index_month_end=dateEnd.find('-',5)
    dateStart=dateStart[0:index_year_start]+'/'+dateStart[index_year_start+1:index_month_start]+'/1'
    dateEnd=dateEnd[0:index_year_end]+'/'+dateEnd[index_year_end+1:index_month_end]+'/1'
    
    fig = make_subplots(rows=2, cols=2, specs=[[{'type': 'polar'}]*2]*2)
    fig.add_trace(go.Scatterpolar(
              r = departure,
              theta = date_arrive,
              name = "log mean departure",
              fill='toself'
            ),1,1)
    fig.add_trace(go.Scatterpolar(
              r = arrive,
              theta = date_arrive,
              name = "log mean arrive",
              fill='toself'
            ),1,2)
    fig.add_trace(go.Scatterpolar(
              r = inflow,
              theta = date_arrive,
              name = "mean net inflow",
              fill='toself'
            ),2,1)

        # Common parameters for all traces
        # fig.update_traces(fill='toself')

    
    timeStamp=[]
    for i in range(len(date_cpi)):
        if date_cpi[i]==dateStart:
            timeStamp.append(i)
        if date_cpi[i]==dateEnd:
            timeStamp.append(i)
    if ('PCR_URBAN' in selected): 
        fig.update_traces(r = departure[timeStamp[0]:timeStamp[1]+1],
              theta = date_arrive[timeStamp[0]:timeStamp[1]+1],row=1,col=1)
        fig.update_traces(r = arrive[timeStamp[0]:timeStamp[1]+1],
              theta = date_arrive[timeStamp[0]:timeStamp[1]+1],row=1,col=2)
        fig.update_traces(r = inflow[timeStamp[0]:timeStamp[1]+1],
              theta = date_arrive[timeStamp[0]:timeStamp[1]+1],row=2,col=1)
        fig.add_trace(go.Scatterpolargl(
              r = index_cpi_urban[timeStamp[0]:timeStamp[1]+1],
              theta = date_cpi[timeStamp[0]:timeStamp[1]+1],
              name = "urban cpi(range from 100 to 110)",
              marker=dict(size=5, color="rgb(127,60,141)")
            ),2,2)

        # Common parameters for all traces
        # fig.update_traces(fill='toself')

        fig.update_layout(
            polar4 = dict(
              bgcolor = "rgb(223, 223, 223)",
              angularaxis = dict(
                linewidth = 2,
                showline=True,
                linecolor='black',
                tickfont_size = 7,
              ),
              radialaxis = dict(
                side = "counterclockwise",
                showline = True,
                linewidth = 2,
                gridcolor = "white",
                gridwidth = 2,
                tickfont_size=8,
                range=[101,106.3]
              )
            ),
#             paper_bgcolor = "rgb(223, 223, 223)"
        )
    if ('PCR_COUNTRYSIDE' in selected): 
        fig.update_traces(r = departure[timeStamp[0]:timeStamp[1]+1],
              theta = date_arrive[timeStamp[0]:timeStamp[1]+1],row=1,col=1)
        fig.update_traces(r = arrive[timeStamp[0]:timeStamp[1]+1],
              theta = date_arrive[timeStamp[0]:timeStamp[1]+1],row=1,col=2)
        fig.update_traces(r = inflow[timeStamp[0]:timeStamp[1]+1],
              theta = date_arrive[timeStamp[0]:timeStamp[1]+1],row=2,col=1)
        fig.add_trace(go.Scatterpolargl(
              r = index_cpi_countryside[timeStamp[0]:timeStamp[1]+1],
              theta = date_cpi[timeStamp[0]:timeStamp[1]+1],
              name = "countryside cpi(range from 100 to 110)",
              marker=dict(size=5, color="rgb(127,60,141)")
            ),2,2)

        # Common parameters for all traces
        # fig.update_traces(fill='toself')

        fig.update_layout(
            polar4 = dict(
              bgcolor = "rgb(223, 223, 223)",
              angularaxis = dict(
                linewidth = 2,
                showline=True,
                linecolor='black',
                tickfont_size = 7,
              ),
              radialaxis = dict(
                side = "counterclockwise",
                showline = True,
                linewidth = 2,
                gridcolor = "white",
                gridwidth = 2,
                tickfont_size=8,
                range=[101,106.3]
              )
            ),
#             paper_bgcolor = "rgb(223, 223, 223)"
        )
    if ('PCR_TOTAL' in selected):
        fig.update_traces(r = departure[timeStamp[0]:timeStamp[1]+1],
              theta = date_arrive[timeStamp[0]:timeStamp[1]+1],row=1,col=1)
        fig.update_traces(r = arrive[timeStamp[0]:timeStamp[1]+1],
              theta = date_arrive[timeStamp[0]:timeStamp[1]+1],row=1,col=2)
        fig.update_traces(r = inflow[timeStamp[0]:timeStamp[1]+1],
              theta = date_arrive[timeStamp[0]:timeStamp[1]+1],row=2,col=1)

        fig.add_trace(go.Scatterpolargl(
              r = index_cpi_all[timeStamp[0]:timeStamp[1]+1],
              theta = date_cpi[timeStamp[0]:timeStamp[1]+1],
              name = "total cpi(range from 100 to 110)",
              marker=dict(size=5, color="rgb(254,136,177)")
            ),2,2)

        # Common parameters for all traces
        # fig.update_traces(fill='toself')

        fig.update_layout(
            polar4 = dict(
              bgcolor = "rgb(223, 223, 223)",
              angularaxis = dict(
                linewidth = 2,
                showline=True,
                linecolor='black',
                tickfont_size = 7,
              ),
              radialaxis = dict(
                side = "counterclockwise",
                showline = True,
                linewidth = 2,
                gridcolor = "white",
                gridwidth = 2,
                tickfont_size=8,
                range=[101,106.3]
              )
            ),
#             paper_bgcolor = "rgb(223, 223, 223)"

        )
    fig.update_layout(
    #     title = "Commercial impact",
        font_size = 15,
#         automargin=True,
#         margin=dict(l=30, r=30, b=20, t=40),
#         hovermode="closest",
        showlegend = True,
        height = 650,
        legend=dict(
            orientation="h",
            yanchor="top",
#             xanchor="right",
            font=dict(
                size=10
            )
        ),
        polar1 = dict(
          bgcolor = "rgb(223, 223, 223)",
          angularaxis = dict(
            linewidth = 2,
            showline=True,
            linecolor='black',
            tickfont_size = 7,
          ),
          radialaxis = dict(
            side = "counterclockwise",
            showline = True,
            linewidth = 2,
            gridcolor = "white",
            gridwidth = 2,
            showticklabels=False
#             tickfont_size=1,
          )
        ),
        polar2 = dict(
          bgcolor = "rgb(223, 223, 223)",
          angularaxis = dict(
            linewidth = 2,
            showline=True,
            linecolor='black',
            tickfont_size = 7,
          ),
          radialaxis = dict(
            side = "counterclockwise",
            showline = True,
            linewidth = 2,
            gridcolor = "white",
            gridwidth = 2,
            showticklabels=False
#             tickfont_size=1,
          )
        ),
        polar3 = dict(
          bgcolor = "rgb(223, 223, 223)",
          angularaxis = dict(
            linewidth = 2,
            showline=True,
            linecolor='black',
            tickfont_size = 7,
          ),
          radialaxis = dict(
            side = "counterclockwise",
            showline = True,
            linewidth = 2,
            gridcolor = "white",
            gridwidth = 2,
            showticklabels=False
#             tickfont_size=1,

          )
        ),
#         paper_bgcolor = "rgb(223, 223, 223)"
    )

    return fig

    


# In[11]:


@app.callback(
    Output('3d-scatter', 'figure'),
    
    [
         Input('operator-select-EE','value'),
         Input('my-date-picker-range', 'start_date'),
         Input('my-date-picker-range', 'end_date')
    ],    

)

def get_figure(selectEE,date1,date2):
    weather= copy.deepcopy(df_weather)
    total_tourist= copy.deepcopy(df_total_tourist)
    weatherdates=pd.to_datetime(weather['Date'])
    dates=pd.to_datetime(total_tourist['Date'])
    date1=pd.to_datetime(date1);
    date2=pd.to_datetime(date2);
    weather=weather.loc[(weatherdates>date1)&(weatherdates<date2)]
    weather['Date']=pd.to_datetime(weather['Date'])
    total_tourist=total_tourist.loc[(dates>date1)&(dates<date2)]
    optionAorD=''#default is arrival for the option
    if ('Arrival' in selectEE) and len(selectEE) == 1: 
        optionAorD='Arrival'
        total_tourist=total_tourist[total_tourist["Arrival/Departure"] == optionAorD]
    elif ('Departure' in selectEE) and len(selectEE) == 1: 
        optionAorD='Departure'
        total_tourist=total_tourist[total_tourist["Arrival/Departure"] == optionAorD]
    elif len(selectEE) == 2:
        total_tourist=total_tourist
    else:
        total_tourist=total_tourist[total_tourist["Arrival/Departure"] == optionAorD]
      
 
    if ('Arrival' in selectEE or 'Departure' in selectEE): 
        total_tourist['Date']=pd.to_datetime(total_tourist['Date'])
        total_tourist=total_tourist.groupby(total_tourist['Date'], as_index=False).agg({'Mainland Visitors':'sum'})
        cols = ['Date']
        weather=weather.merge(total_tourist, on=cols, how='inner')
        
        
     #weather data fetching
    avg_temp=weather['Average Temperature(°F)']
    avg_wind_speed=weather['Average wind speed(knots)']
    #mainland_visitor=mainland_arrival['Mainland Visitors']  
    percipitation=weather["Precipitation(in)"]
    wind_speed=weather["Average wind speed(knots)"]  
    
    mainland_arrival=total_tourist[['Date','Mainland Visitors']]# select a subset of columns 
    mainland_visitor=mainland_arrival['Mainland Visitors']

     #normalize it to [0,1],need to scale it up with 80
    scale=40
    normalized_size=(mainland_visitor-mainland_visitor.min())/(mainland_visitor.max()-mainland_visitor.min())
        

     #draw Scatter Plot
    figure1=go.Scatter3d(
#         title="circle size represent number of visitors"
        x=weather["Date"], y=percipitation, z=wind_speed,   
        marker=dict(
            size=normalized_size*scale,
            color=avg_temp,
            colorscale='Viridis',
            colorbar=dict(thickness=20,title="Average Temperature(°F) "),
        ),
        #add hoverinfo
        text = ['Num of visitor: {}'.format(x) for x in mainland_visitor],
#         hover template for x,y,z
        hovertemplate =
        '<i>Date</i>: %{x}'+
        '<br>Preciptation: %{y}<br>'+
        'Wind Speed: %{z}<br>'+
        '%{text}',    

        line=dict(
            color='darkblue',
            width=0.3
        )
    )

    fig = go.Figure(data=figure1)
    
#   fig.add_annotation(text="circle size represent number of visitors")
#     fig.add_trace(text="circle size represent number of visitors",textposition="top right")
    
    fig.update_layout(
    title="Size of the circles represents number of visitors",
    font=dict(size=15),
    width=700,
    height=700,
    autosize=False,
    scene=dict(
         xaxis = dict(
        title='Date'),
    yaxis = dict(
        title='Precipitation'),
    zaxis = dict(
        title='Average wind speed(knots)'),
        
        camera=dict(
            up=dict(
                x=0,
                y=0,
                z=0
            ),
            eye=dict(
                x=0,
                y=1,
                z=1,
            ),
            
            
        ),
        aspectratio = dict( x=1, y=1, z=0.7 ),
        aspectmode = 'manual',
         ),
    )
    
    return fig


# In[12]:

########### Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

