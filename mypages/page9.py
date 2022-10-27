

def page9():

    '''
    plot the demand forecasting reulsts 

    '''
    ############################################################################################
    # import and setup

    # basic library
    import pandas as pd
    import numpy as np
    import warnings
    warnings.filterwarnings(action='ignore')
    
    # plot
    import plotly.express as px
    import plotly.figure_factory as ff
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # support
    import streamlit as st

    # import  data 
    @st.cache
    def load_data():
        df_final_merge = pd.read_csv("data/demand_forecasting_final_merge.csv")
        df_accuracy = pd.read_csv("data/demand_forecasting_accuracy.csv")
        return df_final_merge, df_accuracy
    df_final_merge, df_accuracy = load_data()

    ############################################################################################
    # pre=process and filter the data
    st.write("## Demand Forecasting")
    state_list = df_final_merge['windeliverystate'].unique().tolist()
    selected_state = st.selectbox("Choose the State", state_list, state_list.index('CA'))
    df_final_state = df_final_merge[df_final_merge['windeliverystate'] == selected_state]
    st.write('--------------------------------')

    ############################################################################################
    # plot time series
    st.write("#### Actual vs. Forecast of Sales Quantity for " + selected_state)
    st.write(
    '''
    Here we show the actual monthly sales associate with back testing results (6 months from 12/2021 to 05/2022) and future sales forecast (6 months from 06/2022 to 11/2022). The algorithm of forecasting integrated both trend and seasonality. 
    
    
    '''
    )
    fig = px.line(df_final_state, x='date', y=df_final_state.columns[2:4])
    fig.update_layout(width=1000,height=600)
    st.plotly_chart(fig, use_container_width=False, config= dict(displayModeBar = False))



    ############################################################################################
    # plot accuracy
    col1, col2, col3, col4 = st.columns([3,1,4,2])

    fig, ax = plt.subplots(figsize=(3,5))
    sns.set_theme(style="whitegrid")
    # sns.set(rc={'figure.figsize':(3,5)})
    ax = sns.boxplot(y=df_accuracy['accuracy'], data=df_accuracy, whis=np.inf)
    ax = sns.stripplot(y=df_accuracy['accuracy'], data=df_accuracy, color=".3")
    ax.set_ylim([0, 2]) 

    with col1:
        st.write("#### Accuracy of Back Test")
        st.pyplot(fig, use_container_width=False)

    with col3:
        st.write("#### Defination of Accuracy")
        st.text(" \n")
        st.write("The **accuracy** here is defined by the sum of all the predicted values from 12/2020 to 05/2021 (total of 6 months), devided by the actual sales from the same period. For example, the accuracy = 1.2 means the predicted results are 120% of the actual results, which is 20% overpredicted. And if accuracy = 1, means a predict equals to actual."
        )



