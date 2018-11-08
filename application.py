# import codecs
import dash
import dash_core_components as dcc
import dash_html_components as html
# import dash_table_experiments as dt
from dash.dependencies import Input, Output, State, Event
# import colorlover as cl
# import flask
# import plotly.graph_objs as go
# import requests
# from base64 import b64decode
# from flask import send_from_directory
# import numpy as np
# import plotly.figure_factory as ff

import os, glob, datetime, time
import pandas as pd
from mylib import batdispatch


# record time to save file
timeSave = datetime.datetime.now().strftime('%Y/%m/%d/%H/%M/%S')

## Dash app for prototype visuals
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# User Inputs
zone = 'N.Y.C.'
bat_maxkW_ch = 100 #Max charge power capacity (kW)
bat_maxkW_disch = 100 #Max discharge power capacity (kW)
bat_kwh = 200 #Discharge energy capacity (kWh)
bat_eff = .85 #AC-AC Round-trip efficiency (%)
bat_maxkwh_day = 200 #Maximum daily discharged throughput (kWh)
user_refresh = '' #for now, manually change to "Yes" to rerun optimizer

# Path to main LBMP folder
path_to_parent = '2017_NYISO_LBMPs'
folder_list = os.listdir(path_to_parent)
while '.DS_Store' in folder_list:
    folder_list.remove('.DS_Store')

# Path to main optimized outputs folder
path_to_parent2 = 'outputs/tocombine'
folder_list2 = os.listdir(path_to_parent2)
while '.DS_Store' in folder_list2:
    folder_list2.remove('.DS_Store')



def runDaily(zone, path_to_parent, bat_eff, bat_maxkW_ch, bat_maxkW_disch, bat_kwh, bat_maxkwh_day):
    '''runs optimizer for each day'''
    try:
        for folder in folder_list:
            try:
                # os.mkdir('inputs/tocombine/' + folder)
                os.mkdir('outputs/tocombine/' + folder)
            except Exception as e:
                print(e)

            csv_files = glob.glob(path_to_parent + '/' + folder + '/' + '*.csv')
            for file in csv_files:
                filename = os.path.basename(file)
                df_raw = pd.read_csv(file)
                df_raw['Date'] = pd.to_datetime(df_raw['Time Stamp'])
                # Only keep zone of interest:
                df = df_raw[df_raw['Name'] == zone].copy().reset_index()
                # print(df.head())
                # df_oneday = df[df['Date'] < '01/02/2017']

                #Call solver
                optim = batdispatch(df, bat_eff, bat_maxkW_ch, bat_maxkW_disch, bat_kwh, bat_maxkwh_day)
                df_optim = optim.drMarket()
                df_optim.to_csv('outputs/tocombine/' + folder + '/' + 'batdispatch_drMarket_' + filename, index=False) # + timeSave.replace('/', '_') + '.csv', index=False)
    except Exception as e:
        print(e)
    return print('Done with daily run of market simulation')



## Iterating through all csv files to combine into single dataframe
def compileAllOptim(zone, path_to_parent, folder_list, timeSave):
    '''compiles all optimized files into single folder for calculations and graphing'''
    print(folder_list)
    try:
        for folder in folder_list:
            csv_files = glob.glob(path_to_parent + '/' + folder + '/' + '*.csv')
            df_comb = pd.concat(pd.read_csv(f) for f in csv_files)
            # print(df_comb.head())
            df_comb.to_csv('outputs/combined/' + folder + '_combined.csv', index=False)
        # Now combine everything into single DataFrame
        final_files = glob.glob('outputs/combined/' + '*.csv')
        print(final_files)
        df_final = pd.concat(pd.read_csv(f) for f in final_files)
        df_final['Date'] = pd.to_datetime(df_final['Date'])
        df_final = df_final.sort_values('Date')
        df_final.to_csv('outputs/masterfile_optimized_' + timeSave.replace('/', '_') + '.csv', index=False)
        # print(df_final.head(10))
        # print(df_final.tail())

        # Only keep zone of interest:
        df_out = df_final.copy()#.reset_index()

    except Exception as e:
        print(e)
    print('Done with compiling all optimized files into one dataframe')
    return df_out



def compileAllBase(zone, path_to_parent, folder_list, timeSave):
    '''compiles all base LMP files into single folder for calculations and graphing'''
    try:
        for folder in folder_list:
            csv_files = glob.glob(path_to_parent + '/' + folder + '/' + '*.csv')
            df_comb = pd.concat(pd.read_csv(f) for f in csv_files)
            # print(df_comb.head())
            df_comb.to_csv('inputs/tocombine/' + folder + '_combined.csv', index=False)
        # Now combine everything into single DataFrame
        final_files = glob.glob('inputs/tocombine/' + '*.csv')
        df_final = pd.concat(pd.read_csv(f) for f in final_files)
        df_final['Date'] = pd.to_datetime(df_final['Time Stamp'])
        df_final = df_final.sort_values('Date')

        df_final.to_csv('inputs/masterfile_baseline_' + timeSave.replace('/', '_') + '.csv', index=False)
        # print(df_final.head(10))
        # print(df_final.tail())

        # Only keep zone of interest:
        df_out = df_final[df_final['Name'] == zone].copy().reset_index()

    except Exception as e:
        print(e)
    print('Done with compiling all LMP baseline files into one dataframe')
    return df_out


# Run code for visuals
try:
    if user_refresh == 'Yes':
        # Call optimizer
        start = time.time()
        runDaily(zone, path_to_parent, bat_eff, bat_maxkW_ch, bat_maxkW_disch, bat_kwh, bat_maxkwh_day)
        compileAllBase(zone, path_to_parent, folder_list, timeSave)
        compileAllOptim(zone, path_to_parent2, folder_list2, timeSave)
        print('It took %s to run' %(time.time() - start))

    ## Display latest Results
    list_of_files_base = glob.glob('inputs/' + '*.csv')
    latest_file_base = max(list_of_files_base, key=os.path.getctime)
    df_baseline =  pd.read_csv(latest_file_base)

    list_of_files_optim = glob.glob('outputs/' + '*.csv')
    latest_file_optim = max(list_of_files_optim, key=os.path.getctime)
    df_optim =  pd.read_csv(latest_file_optim)

    ## Output requested per Requirement doc
    df_optim['Total revenue generation ($)'] = -1 * df_optim['Power output (kW)'] * df_optim['LBMP ($/MWHr)']
    df_optim.loc[df_optim['Power output (kW)'] >0 , 'Total revenue generation ($)'] = 0

    df_optim['Total charging cost ($)'] = 1 * df_optim['Power output (kW)'] * df_optim['LBMP ($/MWHr)']
    df_optim.loc[df_optim['Power output (kW)'] <0 , 'Total charging cost ($)'] = 0
    annual_rev = df_optim['Total revenue generation ($)'].sum()
    annual_cost = df_optim['Total charging cost ($)'].sum()
    annual_throughput = -1*df_optim['Power output (kW)'][df_optim['Power output (kW)']<0].sum()

    print('The total annual revenue generation in US dollars is %s' %(annual_rev))
    print('The total annual charging cost in US dollars is %s' %(annual_cost))
    print('The total annual throughput in kWh is %s' %(annual_throughput))

    ## Best week
    df_optim['Date'] = pd.to_datetime(df_optim['Date'])
    df_optim = df_optim.set_index(df_optim['Date'])
    df_week_rev = df_optim['Total revenue generation ($)'].resample('W').sum()#, how='sum')
    df_week_cost = df_optim['Total charging cost ($)'].resample('W').sum()
    df_week_best = df_week_rev - df_week_cost
    # print(df_week_best.tail(2))
    # print(df_week_best.max())
    # print(df_week_best.idxmax().date())#.index())
    # df_optim['Date'] = pd.to_datetime(df_optim['Date'])
    df_week = df_optim[(df_optim['Date'] <= df_week_best.idxmax() + datetime.timedelta(days=1)) & (df_optim['Date'] >= df_week_best.idxmax() - datetime.timedelta(days=7))]
    # print(df_week.head())

    ## Monthly profit
    df_month_rev = df_optim['Total revenue generation ($)'].resample('M').sum()#, how='sum')
    df_month_cost = df_optim['Total charging cost ($)'].resample('M').sum()
    df_month = df_month_rev - df_month_cost
    # print(df_month.head())

except Exception as e:
    print(e)

## HTML Layout
style = {'padding': '5px', 'fontSize': '16px'}

app.layout = html.Div(children=[
    html.H1(children='Blueprint Power Challenge'),

    html.H5(children='''
        As solved by Eunice Hameyie using Python, Pyomo and Dash.
    '''),

    html.Div(children='''
        Designed and submitted on Nov 7, 2018.
    '''),

    # html.Br(),
    # html.Button(id='submit-button', n_clicks=0, children='Rerun Optimizer'),
    html.Br(),
    # html.Div(id='live-update-text'),
    html.Span('The total annual revenue generation in US dollars is {0:.2f}'.format(annual_rev), style=style),
    html.Br(),
    html.Span('The total annual charging cost in US dollars is: {0:.2f}'.format(annual_cost), style=style),
    html.Br(),
    html.Span('The total annual throughput in kWh is: {0:0.2f}'.format(annual_throughput), style=style),
    html.Br(),

    dcc.Graph(
        id='bestweek-graph',
        figure={
            'data': [
                {'x': df_week.index, 'y': df_week['Power output (kW)'], 'type': 'line', 'name': 'Hourly Dispatch'},
                {'x': df_week.index, 'y': df_week['LBMP ($/MWHr)'], 'type': 'line', 'name': 'Market Price'},
            ],
            'layout': {
                'title': 'Best Week Visualization'
            }
        }
    ),

    dcc.Graph(
        id='monthlyprofit-graph',
        figure={
            'data': [
                {'x': df_month.index, 'y': df_month, 'type': 'bar', 'name': 'Total Monthly Profit'},
            ],
            'layout': {
                'title': 'Total Monthly Profit Visualization'
            }
        }
    )
])


# @app.callback(Output('live-update-text', 'children'),
#               [Input('submit-button', 'n_clicks')])
# def update_metrics(n_clicks):
#     if n_clicks>0:
#         user_refresh = 'Yes'
#         style = {'padding': '5px', 'fontSize': '16px'}
#
#     return [
#         html.Span('The total annual revenue generation in US dollars is {0:.2f}'.format(annual_rev), style=style),
#         html.Br(),
#         html.Span('The total annual charging cost in US dollars is: {0:.2f}'.format(annual_cost), style=style),
#         html.Br(),
#         html.Span('The total annual throughput in kWh is: {0:0.2f}'.format(annual_throughput), style=style),
#         html.Br(),
#     ]


if __name__ == '__main__':
    app.run_server(debug=True)
