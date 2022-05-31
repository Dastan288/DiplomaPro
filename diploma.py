import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go  
from plotly.subplots import make_subplots
pd.set_option('precision',0)
from sklearn.preprocessing import MinMaxScaler
# import tensorflow as tf
# from tensorflow.keras.layers import Input, Conv1D, Dense, Flatten, Dropout, BatchNormalization,LSTM,SeparableConv1D
# from tensorflow.keras.models import Model,Sequential
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping
import warnings
from tensorflow import keras 
from tensorflow.keras.models import load_model

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
warnings.filterwarnings('ignore')
from pmdarima.arima import auto_arima
from pmdarima.datasets import load_lynx
import numpy as np

from scipy import integrate
from scipy import optimize

# For serialization:
import joblib
import pickle

st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(layout='wide')


with st.sidebar:
    add_radio = st.radio(
        "Choose one",
        ("Data Analysis", "Model Forecast")
    )
st.write("""
# Diploma Project:
# Implementing Machine Learning Models for Epidemy Propogation Forecasting
### Berik Gulina, Kapizov Dastan, Abdykalyk Gulnazym 
""")
confirmed_df = pd.read_csv("C:/Users/Асер/Desktop/Diploma/confirmed_df.csv")
deaths_df = pd.read_csv('C:/Users/Асер/Desktop/Diploma/deaths_df.csv')
recoveries_df = pd.read_csv('C:/Users/Асер/Desktop/Diploma/recoveries_df.csv')

lastupdate_data = pd.read_csv('C:/Users/Асер/Desktop/Diploma/lastupdate_data.csv')
latest_data = pd.read_csv("C:/Users/Асер/Desktop/Diploma/latest_data.csv")
Country_df = pd.read_csv("C:/Users/Асер/Desktop/Diploma/Country_df.csv")

confirmed_df = confirmed_df.rename(columns={"Province/State":"state","Country/Region": "country",'Longitude': 'Long', 'Latitude': 'Lat'})
deaths_df = deaths_df.rename(columns={"Province/State":"state","Country/Region": "country",'Longitude': 'Long', 'Latitude': 'Lat'})
recoveries_df = recoveries_df.rename(columns={"Province/State":"state","Country/Region": "country",'Longitude': 'Long', 'Latitude': 'Lat'})
Country_df = Country_df.rename(columns={"Country_Region": "country",'Longitude': 'Long', 'Latitude': 'Lat'})

Confirmed = confirmed_df
Death = deaths_df
Recovered = recoveries_df

Confirmed.drop(axis=1, inplace=True, columns=['state', 'Lat', 'Long'])
Death.drop(axis=1, inplace=True, columns=['state', 'Lat', 'Long'])
Recovered.drop(axis=1, inplace=True, columns=['state', 'Lat', 'Long'])
India_Confirmed_Data = confirmed_df.loc[confirmed_df['country']=="India"]

India_Deaths_Data = deaths_df.loc[deaths_df['country']=="India"]

India_Recovery_Data = recoveries_df.loc[recoveries_df['country']=="India"]

df_list = [India_Confirmed_Data,India_Deaths_Data,India_Recovery_Data]
cases = ['India_Confirmed_Data', 'India_Deaths_Data', 'India_Recovery_Data', 'Active']
case_color = ['orange','red','green','blue']
case_dict = {cases[i]:case_color[i] for i in range(len(cases))}

time_series_data = pd.DataFrame()

for i in range(len(cases)-1):
    df =  pd.DataFrame(df_list[i][df_list[i].columns[5:]].sum(),columns=[cases[i]])
    time_series_data = pd.concat([time_series_data,df],axis = 1)

# convert_date(time_series_data)
time_series_data.index = pd.to_datetime(time_series_data.index,format='%m/%d/%y')
time_series_data['Active'] = time_series_data['India_Confirmed_Data'] - time_series_data['India_Deaths_Data'] - time_series_data['India_Recovery_Data']
time_series_data= time_series_data.rename_axis('ObservationDate').reset_index()

# st.line_chart(time_series_data['India_Confirmed_Data'])


time_series_data['ObservationDate'] = pd.to_datetime(time_series_data['ObservationDate'])
india_11 = time_series_data[(time_series_data['ObservationDate'] >=pd.to_datetime('20200515')) & (time_series_data['ObservationDate'] <= pd.to_datetime('20200915'))]
india_22 = time_series_data[(time_series_data['ObservationDate'] >=pd.to_datetime('20210216')) & (time_series_data['ObservationDate'] <= pd.to_datetime('20210716'))]
india_33 = time_series_data[(time_series_data['ObservationDate'] >=pd.to_datetime('20211215')) & (time_series_data['ObservationDate'] <= pd.to_datetime('20220201'))]

if add_radio == "Data Analysis":
    fig11=go.Figure()
    fig11.add_trace(go.Scatter(x=time_series_data['ObservationDate'], y=time_series_data["India_Confirmed_Data"],
                        mode='lines+markers', 
                        name='Confirmed Cases'))
    
    fig11.update_layout(title="Growth of confirmed cases in India",
                     xaxis_title="Date",yaxis_title="Number of Cases",legend=dict(x=0,y=1,traceorder="normal"))

    fig21=go.Figure()
    fig21.add_trace(go.Scatter(x=india_11['ObservationDate'], y=india_11["India_Confirmed_Data"],
                        mode='lines+markers',line_color = 'red',
                        name='Confirmed Cases'))
    
    fig21.update_layout(title="First period of analysis",
                     xaxis_title="Date",yaxis_title="Number of Cases",legend=dict(x=0,y=1,traceorder="normal"))

    fig31=go.Figure()
    fig31.add_trace(go.Scatter(x=india_22['ObservationDate'], y=india_22["India_Confirmed_Data"],
                        mode='lines+markers', line_color = 'mediumseagreen',
                        name='Confirmed Cases'))
    
    fig31.update_layout(title="Second period of analysis",
                     xaxis_title="Date",yaxis_title="Number of Cases",legend=dict(x=0,y=1,traceorder="normal"))

    fig41=go.Figure()
    fig41.add_trace(go.Scatter(x=india_33['ObservationDate'], y=india_33["India_Confirmed_Data"],
                        mode='lines+markers',line_color = '#d534eb',
                        name='Confirmed Cases'))
    
    fig41.update_layout(title="Third period of analysis",
                     xaxis_title="Date",yaxis_title="Number of Cases",legend=dict(x=0,y=1,traceorder="normal"))
    
    col1, col2 = st.beta_columns((1, 1))
    with col1:
        st.plotly_chart(fig11)
    with col2:
        st.plotly_chart(fig21)


    col3, col4 = st.beta_columns((1, 1))
    with col3:
        st.plotly_chart(fig31)
    with col4:
        st.plotly_chart(fig41)
    # count_df = time_series_data.iloc[1:]
    # print(count_df)
    # count_df = pd.DataFrame(count_df).reset_index(level = 0).rename(columns = {'index':'category',1:'count'})
    # fig55 = px.bar(count_df, x='count', y='category',
    #             hover_data=['count'], color='count',
    #             labels={}, orientation='h',height=400, width = 650)
    # fig55.update_layout(title_text='<b>Confirmed vs Recovered vs Deaths vs Active</b>',title_x=0.5,showlegend = False) 
    # st.plotly_chart(fig55)




else:

    first_seir = pd.read_csv('C:/Users/Асер/Desktop/Diploma/first.csv')['Predicted']

    second_seir = pd.read_csv('C:/Users/Асер/Desktop/Diploma/second.csv')['Predicted']

    third_seir = pd.read_csv('C:/Users/Асер/Desktop/Diploma/third.csv')['Predicted']

    forth_seir = pd.read_csv('C:/Users/Асер/Desktop/Diploma/four.csv')['Predicted']

    first_lstm = pd.read_csv('C:/Users/Асер/Desktop/Diploma/firstt.csv')['Predicted']

    second_lstm = pd.read_csv('C:/Users/Асер/Desktop/Diploma/secondd.csv')['Predicted']

    third_lstm = pd.read_csv('C:/Users/Асер/Desktop/Diploma/thirdd.csv')['Predicted']

    forth_lstm = pd.read_csv('C:/Users/Асер/Desktop/Diploma/pwh.csv')['Predicted']

    l = first_lstm.iloc[0]
    first_lstm.loc[-1] = l

    first_lstm.index = first_lstm.index + 1
    first_lstm.sort_index(inplace= True)

    l = third_lstm.iloc[0]
    third_lstm.loc[-1] = l

    third_lstm.index = third_lstm.index + 1
    third_lstm.sort_index(inplace= True)

    l = second_lstm.iloc[0]
    second_lstm.loc[-1] = l

    second_lstm.index = second_lstm.index + 1
    second_lstm.sort_index(inplace= True)


    s = first_seir.iloc[0]
    first_seir.loc[-1] = s

    first_seir.index = first_seir.index + 1
    first_seir.sort_index(inplace= True)

    s = third_seir.iloc[0]
    third_seir.loc[-1] = s

    third_seir.index = third_seir.index + 1
    third_seir.sort_index(inplace= True)

    s = second_seir.iloc[0]
    second_seir.loc[-1] = s

    second_seir.index = second_seir.index + 1
    second_seir.sort_index(inplace= True)


    time_series_data = pd.DataFrame()

    for i in range(len(cases)-1):
        df =  pd.DataFrame(df_list[i][df_list[i].columns[5:]].sum(),columns=[cases[i]])
        time_series_data = pd.concat([time_series_data,df],axis = 1)

    # convert_date(time_series_data)
    time_series_data.index = pd.to_datetime(time_series_data.index,format='%m/%d/%y')
    time_series_data['Active'] = time_series_data['India_Confirmed_Data'] - time_series_data['India_Deaths_Data'] - time_series_data['India_Recovery_Data']
    time_series_data= time_series_data.rename_axis('ObservationDate').reset_index()

    # st.line_chart(time_series_data['India_Confirmed_Data'])
    

    time_series_data['ObservationDate'] = pd.to_datetime(time_series_data['ObservationDate'])
    india_11 = time_series_data[(time_series_data['ObservationDate'] >=pd.to_datetime('20200515')) & (time_series_data['ObservationDate'] <= pd.to_datetime('20200915'))]
    india_22 = time_series_data[(time_series_data['ObservationDate'] >=pd.to_datetime('20210216')) & (time_series_data['ObservationDate'] <= pd.to_datetime('20210716'))]
    india_33 = time_series_data[(time_series_data['ObservationDate'] >=pd.to_datetime('20211215')) & (time_series_data['ObservationDate'] <= pd.to_datetime('20220201'))]

    import statsmodels.api as stas


    ##### SEIR Modeling
    seir_pred = first_seir.values
    seir_pred = pd.DataFrame(seir_pred)

    seir_pred.index = india_11["India_Confirmed_Data"].index

    seir_pred = seir_pred.iloc[:,0]

    # print(seir_pred)


    ##### LSTM

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    forecast = first_lstm

    pred_lstm = pd.DataFrame(forecast)

    seir_pred = first_seir.values

    pred_lstm.index = india_11["India_Confirmed_Data"].index


    pred_lstm = pred_lstm.iloc[:,0]

    #print(pred_lstm)

    def __getnewargs__(self):
        return ((self.endog),(self.k_lags, self.k_diff, self.k_ma))
    ARIMA.__getnewargs__ = __getnewargs__

    Y_train_1 = india_11['India_Confirmed_Data']
    Y_test_1 = india_11['India_Confirmed_Data']
    date_train_1 = india_11['ObservationDate']
    date_test_1 = india_11['ObservationDate']

    model = stas.tsa.arima.ARIMA(india_11['India_Confirmed_Data'].iloc[:-24], order = (2, 2, 2))

    # ARIMA first model

    arima_1 = model.fit()

    start = 0
    end = len(Y_train_1) - 1

    pred_arima = arima_1.predict(start = 0, end = end, type = 'levels')
    pred_arima.index = india_11['India_Confirmed_Data'].index[start:end+1]


    trace1 = go.Scatter(
        x = date_train_1,
        y = Y_train_1,
        mode = 'lines',
        line_color = 'red',
        name = 'Real Data'
    )
    trace2 = go.Scatter(
        x = date_test_1[2:],
        y = pred_arima[2:],
        mode = 'lines',
        line_color = '#d534eb',
        name = 'Prediction ARIMA'
    )
    trace4 = go.Scatter(
        x = date_test_1,
        y = pred_lstm,
        mode = 'lines',
        line_color = 'mediumseagreen',
        name = 'Prediction LSTM'
    )
    trace5 = go.Scatter(
        x = date_test_1,
        y = seir_pred,
        mode = 'lines',
        line_color = 'blue',
        name = 'Prediction SEIR'
    )

    layout = go.Layout(
        title = "<b>Confirmed Cases for the first period</b>",
        xaxis = {'title' : "Date"},
        yaxis = {'title' : "Cases"}
    )
    fig1 = go.Figure(data=[trace2, trace4, trace5,trace1], layout=layout)
    # st.plotly_chart(fig1)


    # # ------------------------------ SEIR second ----------------------------


    seir_pred2 = second_seir.values



    seir_pred2 = pd.DataFrame(seir_pred2)

    seir_pred2.index = india_22["India_Confirmed_Data"].index
    seir_pred2 = seir_pred2.iloc[:,0]


    # # -------------------------------- LSTM Second--------------------------------------



    forecast = second_lstm

    pred_lstm2 = pd.DataFrame(forecast)

    pred_lstm2.index = india_22["India_Confirmed_Data"].index


    pred_lstm2 = pred_lstm2.iloc[:,0]



    # # ---------------------------ARIMA Second-------------------------------
    Y_train_2 = india_22['India_Confirmed_Data']
    Y_test_2 = india_22['India_Confirmed_Data']
    date_train_2 = india_22['ObservationDate']
    date_test_2 = india_22['ObservationDate']

    obs = india_22['India_Confirmed_Data']
    train = obs.iloc[:-30]
    test = obs.iloc[-30:]

    model = stas.tsa.arima.ARIMA(train, order = (0, 2, 0))
    arima_2 = model.fit()


    start = 0
    end = len(Y_train_2) - 1
    pred_arima2 = arima_2.predict(start = start, end = end, type = 'levels')

    pred_arima2.index = india_22['India_Confirmed_Data'].index[start:end + 1]


    trace1 = go.Scatter(
        x = date_train_2,
        y = Y_train_2,
        mode = 'lines',
        line_color = 'red',
        name = 'Real Data'
    )
    trace2 = go.Scatter(
        x = date_test_2[2:],
        y = pred_arima2[2:],
        mode = 'lines',
        line_color = '#d534eb',
        name = 'Prediction ARIMA'
    )
    trace4 = go.Scatter(
        x = date_test_2,
        y = pred_lstm2,
        mode = 'lines',
        line_color = 'mediumseagreen',
        name = 'Prediction LSTM'
    )
    trace5 = go.Scatter(
        x = date_test_2,
        y = seir_pred2,
        mode = 'lines',
        line_color = 'blue',
        name = 'Prediction SEIR'
    )

    layout = go.Layout(
        title = "<b>Confirmed Cases for the second period</b>",
        xaxis = {'title' : "Date"},
        yaxis = {'title' : "Cases"}
    )
    fig2 = go.Figure(data=[trace2, trace4, trace5,trace1], layout=layout)
    # st.plotly_chart(fig2)



    # #-----------------------------Third Analysis----------------------------


    # # ---------------------------------SEIR Third----------------------------

    seir_pred3 = third_seir.values


    seir_pred3 = pd.DataFrame(seir_pred3)

    seir_pred3.index = india_33["India_Confirmed_Data"].index
    seir_pred3 = seir_pred3.iloc[:,0]



    # #---------------------------------LSTM Third ---------------------------
    # lstm_3 = load_model("C:/Users/Асер/Desktop/Diploma/models/lstm/3.h5")
    # print(india_33[["India_Confirmed_Data"]])


    pred_lstm3 = pd.DataFrame(third_lstm)


    pred_lstm3.index = india_33["India_Confirmed_Data"].index


    pred_lstm3 = pred_lstm3.iloc[:,0]





    # #----------------------------------Arima Third----------------------------



    obs = india_33['India_Confirmed_Data']
    train = obs.iloc[:-8]
    test = obs.iloc[-8:]

    model = stas.tsa.arima.ARIMA(train, order = (0, 2, 0))
    arima_3 = model.fit()




    Y_train_3 = india_33['India_Confirmed_Data']
    Y_test_3 = india_33['India_Confirmed_Data']
    date_train_3 = india_33['ObservationDate']
    date_test_3 = india_33['ObservationDate']

    start = 0
    end = len(Y_train_3) - 1
    pred_arima3 = arima_3.predict(start = start , end = end, type = 'levels')
    pred_arima3.index = obs.index[start:end+1]

    trace1 = go.Scatter(
        x = date_train_3,
        y = Y_train_3,
        mode = 'lines',
        line_color = 'red',
        name = 'Real Data'
    )
    trace2 = go.Scatter(
        x = date_test_3[2:],
        y = pred_arima3[2:],
        mode = 'lines',
        line_color = '#d534eb',
        name = 'Prediction ARIMA'
    )
    trace4 = go.Scatter(
        x = date_test_3,
        y = pred_lstm3,
        mode = 'lines',
        line_color = 'mediumseagreen',
        name = 'Prediction LSTM'
    )
    trace5 = go.Scatter(
        x = date_test_3,
        y = seir_pred3,
        mode = 'lines',
        line_color = 'blue',
        name = 'Prediction SEIR'
    )

    layout = go.Layout(
        title = "<b>Confirmed Cases for the third period</b>",
        xaxis = {'title' : "Date"},
        yaxis = {'title' : "Cases"}
    )
    fig3 = go.Figure(data=[trace2, trace4, trace5,trace1], layout=layout)
    # st.plotly_chart(fig3)


    # #------------------------------Whole Period-------------------------------
    Y_train = time_series_data['India_Confirmed_Data']
    Y_test = time_series_data['India_Confirmed_Data']
    date_train = time_series_data['ObservationDate']
    date_test = time_series_data['ObservationDate']


    # #--------------------------------SEIR Whole------------------------------
    seir_pred4 = forth_seir.values



    seir_pred4 = pd.DataFrame(seir_pred4)
    seir_pred4 = seir_pred4[:-2]

    seir_pred4.index = time_series_data["India_Confirmed_Data"].index
    seir_pred4 = seir_pred4.iloc[:,0]



    # #--------------------------------LSTM Whole--------------------------------




    pred_lstm4 = pd.DataFrame(forth_lstm)

    pred_lstm4.index = time_series_data["India_Confirmed_Data"].index[:-1]


    pred_lstm4 = pred_lstm4.iloc[:,0]



    # #--------------------------------ARIMA Whole------------------------------

    obs = time_series_data['India_Confirmed_Data']
    train = obs.iloc[:-160]
    test = obs.iloc[-160:]
    model = stas.tsa.arima.ARIMA(train, order = (2, 2, 5))

    arima4 = model.fit()

    start = 0
    end = len(Y_train) - 1
    pred_arima4 = arima4.predict(start = start , end = end, type = 'levels')
    pred_arima4.index = obs.index[start :end+1]

    trace1 = go.Scatter(
        x = date_train,
        y = Y_train,
        mode = 'lines',
        line_color = 'red',
        name = 'Real Data'
    )
    trace2 = go.Scatter(
        x = date_test[2:],
        y = pred_arima4[2:],
        mode = 'lines',
        line_color = '#d534eb',
        name = 'Prediction ARIMA'
    )
    trace4 = go.Scatter(
        x = date_test,
        y = pred_lstm4,
        mode = 'lines',
        line_color = 'mediumseagreen',
        name = 'Prediction LSTM'
    )
    trace5 = go.Scatter(
        x = date_test,
        y = seir_pred4,
        mode = 'lines',
        line_color = 'blue',
        name = 'Prediction SEIR'
    )

    layout = go.Layout(
        title = "<b>Confirmed Cases for the whole period</b>",
        xaxis = {'title' : "Date"},
        yaxis = {'title' : "Cases"}
    )
    fig4 = go.Figure(data=[trace2, trace4, trace5,trace1], layout=layout)
    # st.plotly_chart(fig4)


    col1, col2 = st.beta_columns((1, 1))
    with col1:
        st.plotly_chart(fig1)
    with col2:
        st.plotly_chart(fig2)


    col3, col4 = st.beta_columns((1, 1))
    with col3:
        st.plotly_chart(fig3)
    with col4:
        st.plotly_chart(fig4)

    # with st.sidebar:
            
    #     d = st.date_input(
    #     "Select the prediction date")

    #     print(time_series_data[time_series_data['ObservationDate'] == f'{d}'].index.iloc[0])

    #     print(arima_1.predict(start = 250, end = 250))

    #     print(time_series_data.head())
