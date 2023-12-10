import streamlit as st
from datetime import datetime
from Home import MWT_data
import Home as e_app
import plotly.figure_factory as ff
import plotly.graph_objects as go
from tensorflow import keras
import numpy as np
import calendar
import random

def long_term_projection():
    st.title("Long Term Projection")
    # Your long-term projection page content goes here

if __name__ == "__main__":
    long_term_projection()

st.sidebar.title("Settings")

st.sidebar.divider()

selected_on_calendar = 0

def plot_gauge(
    indicator_number, indicator_color, indicator_suffix, indicator_title, max_bound
):
    fig = go.Figure(
        go.Indicator(
            value=indicator_number,
            mode="gauge+number",
            domain={"x": [0, 1], "y": [0, 1]},
            number={
                "suffix": indicator_suffix,
                "font.size": 35,
            },
            gauge={
                "axis": {"range": [0, max_bound], "tickwidth": 5},
                "bar": {"color": indicator_color},
            },
            title={
                "text": indicator_title,
                "font": {"size": 28},
            },
            
        )
    )
    fig.update_layout(
        # paper_bgcolor="lightgrey",
        height=280,
        margin=dict(l=1, r=1, t=1, b=1, pad=1),
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_metric(label, value, prefix="", suffix="", show_graph=False, color_graph=""):
    fig = go.Figure()

    fig.add_trace(
        go.Indicator(
            mode = "number+delta",
            value=value,
            gauge={"axis": {"visible": False}},
            number={
                "prefix": prefix,
                "suffix": suffix,
                "font": {"size": 28, "color": "white"}
                
            },
            title={
                "text": label,
                "font": {"size": 24, "color": "white"}
            },
        )
    )

    if show_graph:
        fig.add_trace(
            go.Scatter(
                y=random.sample(range(0, 101), 30),
                hoverinfo="skip",
                fill="tozeroy",
                fillcolor=color_graph,
                line={
                    "color": color_graph,
                },
            )
        )

    fig.update_xaxes(visible=False, fixedrange=True)
    fig.update_yaxes(visible=False, fixedrange=True)
    fig.update_layout(
        # paper_bgcolor="lightgrey",
        margin=dict(t=0, b=0),
        showlegend=False,
        plot_bgcolor="#0e1117",
        height=230,
    )

    st.plotly_chart(fig, use_container_width=True)

plot_top_left = st.columns([10])[0]

bottom_info_left, bottom_info_right = st.columns([9,1])

given_values = [744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016, 8760]


st.sidebar.subheader('Choose a month', 
               divider='gray',
               help="This changes the line chart above based on the month you selected. The 6 metrics below will also change based on your selected month")
months_list = list(calendar.month_name)[1:]
# Display the date input
month_option = st.sidebar.selectbox(
        ' ',
        months_list,
        key="month_selectbox")

# Create the dictionary of month names with assigned values
month_values = {month: value for month, value in zip(calendar.month_name[1:], given_values)}
start_key_value = {'Start': 0}
# Merge the new key-value pair with the existing dictionary
updated_dict_month_values = {**start_key_value, **month_values}

st.sidebar.subheader('Choose a model', 
               divider='gray',
               help="This changes the line chart projection result above based on the deep learning model you selected. It also changes the 6 metrics below based on your selection")
#st.sidebar.title("Choose a model")
option = st.sidebar.selectbox(
        ' ',
        ("MLP", "LSTM", "GRU"),
        key="long_term_models")

    


from sklearn.preprocessing import MinMaxScaler

# Feature scaling
scaler = MinMaxScaler()
MWT_data['MinMax'] = scaler.fit_transform(MWT_data['MWT'].values.reshape(-1, 1))

# Plotly figure for 'MinMax' Time Series
fig_minmax = go.Figure()
fig_minmax.add_trace(go.Scatter(x=MWT_data.index, y=MWT_data['MinMax'], mode='lines', name='Time Series Data', line=dict(color='dodgerblue')))
fig_minmax.update_layout(
    title='Normalized MWT Time Series Plot',
    xaxis_title='Date',
    yaxis_title='MWT'
)

# Split the data into training and testing DataFrames
data = MWT_data['MinMax']

train_size = int(len(data) * 0.8)  # 80% of the data for training

train = data.iloc[:train_size]  # Select the first 80% for training
test = data.iloc[train_size:]   # Select the remaining 20% for testing

# Plotly figure for Train-Test Split
fig_train_test = go.Figure()
fig_train_test.add_trace(go.Scatter(x=train.index, y=train, mode='lines', name='Training Data', line=dict(color='dodgerblue')))
fig_train_test.add_trace(go.Scatter(x=test.index, y=test, mode='lines', name='Testing Data', line=dict(color='indianred')))
fig_train_test.update_layout(
    title='80/20 Train-Test Split',
    xaxis_title='Time',
    yaxis_title='Value'
)


look_back = 24  # Number of previous hours to use for prediction
trainX, trainY = e_app.create_dataset(train, look_back)
testX, testY = e_app.create_dataset(test, look_back)

if option == "MLP":
    # Load your Keras model
    model = keras.models.load_model('best_model_mlp.h5')
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    trainPredict = trainPredict.flatten()
    testPredict = testPredict.flatten()


elif option == "LSTM":
    # Load your Keras model
    model = keras.models.load_model('best_model_lstm_latest.h5')
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    trainPredict = trainPredict.flatten()
    testPredict = testPredict.flatten()

    
elif option == "GRU":
    # Load your Keras model
    model = keras.models.load_model('best_model_gru.h5')
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    trainPredict = trainPredict.flatten()
    testPredict = testPredict.flatten()


       

df_total_predictions = np.array([])
df_total_predictions = np.append(trainPredict, df_total_predictions)

for _ in range(48):
    df_total_predictions = np.append(df_total_predictions, 0)
    
df_total_predictions = np.append(df_total_predictions, testPredict)

month_selected = updated_dict_month_values[month_option]
month_max_month = list(updated_dict_month_values.values())
month_day_index = list(updated_dict_month_values.values()).index(month_selected)

min_plot_value = month_max_month[month_day_index - 1]

# Compute the max value slider indicator
max_day_slider_val = int((month_selected - month_max_month[month_day_index - 1]) / 24)


#st.sidebar.title("Change the range of days")
st.sidebar.subheader('Select range of days', 
               divider='gray',
               help="This changes the line chart projection result above based on the deep learning model you selected. It also changes the 6 metrics below based on your selection")
selected_value_hour = st.sidebar.slider(
    " ",
    value=(1, max_day_slider_val),
    max_value=max_day_slider_val,
    min_value=1,
    key="day_range_slider")


with bottom_info_right:
    
    st.divider()
    min_user_slider_input = (min_plot_value + (selected_value_hour[0] * 24)) - 24
    max_user_slider_input = month_selected - ((max_day_slider_val - selected_value_hour[1]) * 24)

    number = df_total_predictions[min_user_slider_input:max_user_slider_input].sum()
    truncated_number = float(f'{number:.3f}')

    total_saved_mwt =  data[min_user_slider_input:max_user_slider_input].sum() - truncated_number
    truncated_saved_mwt = float(f'{total_saved_mwt:.3f}')

    percentage_saved_mwt = (total_saved_mwt / data[min_user_slider_input:max_user_slider_input].sum()) * 100
    truncated_percentage_saved_mwt = float(f'{percentage_saved_mwt:.3f}')

    bottom_info_right.metric("Predicted MWT", 
                              truncated_number,
                              help="This indicator displays the Predicted MWT value with decimal places. The value of MWT was from the gauge indicator.")
    bottom_info_right.metric("Saved MWT", 
                              truncated_saved_mwt, 
                              truncated_percentage_saved_mwt,
                              help="This indicator displays the Saved MWT value from the projected value of your selected deep-learning model. The value was from the difference between the actual and predicted output of the deep learning model. The arrow up (green) indicates that the selected range of days has a saved MWT; the arrow down indicates a loss of MWT in that range of days.")
    
    st.divider()

with plot_top_left:
    st.subheader('One Month Projection Line Chart', 
               help="This line chart display the Actual and Projection value based on your selected deep learning model. It also display the Time (range of days) and value of MWT.",
               divider='gray')
    
    if option == "MLP":
        # Plotting actual vs predicted data in Plotly
        fig_january = go.Figure()

        fig_january.add_trace(go.Scatter(x=data.index[min_user_slider_input:max_user_slider_input], y=data[min_user_slider_input:max_user_slider_input], mode='lines', name='Actual Data', line=dict(color='dodgerblue')))
        fig_january.add_trace(go.Scatter(x=data.index[min_user_slider_input:max_user_slider_input], y=df_total_predictions[min_user_slider_input:max_user_slider_input], mode='lines', name='Predicted Data', line=dict(color='indianred')))

        fig_january.update_layout(
            
            xaxis_title='Time',
            yaxis_title='Value',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Display Plotly figure using st.plotly_chart()
        st.plotly_chart(fig_january, use_container_width=True)
    elif option == "LSTM":
        min_calendar_val = selected_on_calendar - 24
         # Plotting actual vs predicted data in Plotly
        fig_january = go.Figure()

        fig_january.add_trace(go.Scatter(x=data.index[min_user_slider_input:max_user_slider_input], y=data[min_user_slider_input:max_user_slider_input], mode='lines', name='Actual Data', line=dict(color='dodgerblue')))
        fig_january.add_trace(go.Scatter(x=data.index[min_user_slider_input:max_user_slider_input], y=df_total_predictions[min_user_slider_input:max_user_slider_input], mode='lines', name='Predicted Data', line=dict(color='indianred')))

        fig_january.update_layout(
            title='(One Month Prediction)',
            xaxis_title='Time',
            yaxis_title='Value',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Display Plotly figure using st.plotly_chart()
        st.plotly_chart(fig_january, use_container_width=True)

    elif option == "GRU":
        min_calendar_val = selected_on_calendar - 24
        # Plotting actual vs predicted data in Plotly
        fig_january = go.Figure()

        fig_january.add_trace(go.Scatter(x=data.index[min_user_slider_input:max_user_slider_input], y=data[min_user_slider_input:max_user_slider_input], mode='lines', name='Actual Data', line=dict(color='dodgerblue')))
        fig_january.add_trace(go.Scatter(x=data.index[min_user_slider_input:max_user_slider_input], y=df_total_predictions[min_user_slider_input:max_user_slider_input], mode='lines', name='Predicted Data', line=dict(color='indianred')))
        
        fig_january.update_layout(
            title='(One Month Prediction)',
            xaxis_title='Time',
            yaxis_title='Value',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Display Plotly figure using st.plotly_chart()
        st.plotly_chart(fig_january, use_container_width=True)

with bottom_info_left:
    
    st.subheader('One Month Metric Indicators of MWT', 
               help="These metric indicators display different categories of MWT numerical values based on your selected month and deep learning model. The values of the metric indicators were based on the result of the projection in the line chart above.",
               divider='gray')
    indicator_total_actual_mwt, indicator_total_predict_mwt, gauge_total_actual_mwt, gauge_total_predict_mwt, = st.columns([3,3,3,3])
    
    total_mwt_actual_indicator_gauge_val = data[min_user_slider_input:max_user_slider_input].sum()
    total_mwt_predicted_indicator_gauge_val = df_total_predictions[min_user_slider_input:max_user_slider_input].sum()
   
    total_mwt_actual_indicator_val = data[month_max_month[month_day_index - 1]:month_selected].sum()
    total_mwt_predicted_indicator_val = df_total_predictions[month_max_month[month_day_index - 1]:month_selected].sum() 

    with indicator_total_actual_mwt:
        total_mwt_day = MWT_data.loc[:, 'MWT']
        plot_metric("Total Actual MWT",
            total_mwt_actual_indicator_val,
            prefix="",
            suffix=" MW",
            show_graph=True,
            color_graph="rgba(89, 92, 255, 0.6)",)
    with indicator_total_predict_mwt:
        plot_metric("Total Predicted MWT",
            total_mwt_predicted_indicator_val,
            prefix="",
            suffix=" MW",
            show_graph=True,
            color_graph="rgba(255, 82, 82, 0.6)",)

    max_total_mwt_actual_month = data[min_plot_value:month_selected].sum()
    max_total_mwt_predicted_month = df_total_predictions[min_plot_value:month_selected].sum()

    with gauge_total_actual_mwt:
        plot_gauge(total_mwt_actual_indicator_gauge_val, "#0400ff", " MW", "Actual MWT value", max_total_mwt_actual_month)
        
    with gauge_total_predict_mwt:
        if total_mwt_actual_indicator_gauge_val >= total_mwt_predicted_indicator_val:
            plot_gauge(total_mwt_predicted_indicator_gauge_val, "#47ff7b", " MW", "Predicted MWT value", max_total_mwt_predicted_month)
        else:
            plot_gauge(total_mwt_predicted_indicator_gauge_val, "#FF2B2B", " MW", "Predicted MWT value", max_total_mwt_predicted_month)
