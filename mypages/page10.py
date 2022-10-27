

def page10():

    '''
    Present the similar items results from item-based collaborate filtering

    '''

    ############################################################################################
    # insert logo




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

    # @st.cache
    # def load_data():
    df_similar_item_final = pd.read_csv('data/df_similar_item_final.csv')
    #     return df_similar_item_final
    # df_similar_item_final = load_data()

    st.write("## Similar Items")

    # get the varietal and year selectbox
    df_similar_item_final[['verietal','year']] = df_similar_item_final['varietal_year'].str.split('_',expand=True)
    varietal_list = df_similar_item_final['verietal'].unique().tolist()
    year_list = sorted(df_similar_item_final['year'].unique().tolist(), reverse=True)
    varietal_list.sort()
    year_list.sort

    varietal_selected = st.selectbox("Choose the Varietal", varietal_list, varietal_list.index('CAB SAUV'))
    year_selected = st.selectbox("Choose the Year", year_list, year_list.index('2018'))
    

    ############################################################################################
    # pre=process and filter the data

    df_top10_filtered = df_similar_item_final[
        (df_similar_item_final['year'] == year_selected) & (df_similar_item_final['verietal'] == varietal_selected)
    ]


    st.write('--------------------------------')
    st.write(f"#### Top 10 Similar Items to {varietal_selected} produced in {year_selected}")
    # st.table(df_top10_filtered[['varietal_year','rank','varietal_year_similar']])
    fig = ff.create_table(df_top10_filtered[['varietal_year','rank','varietal_year_similar']], index=False)
    fig.update_layout(
    width=1000,
    )
    st.plotly_chart(fig, use_container_width=False, config= dict(displayModeBar = False))
