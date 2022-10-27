

def page4():
    '''
    Modeling of Market Basket Analyssi
    Use association ruls to calculate frequently bought item pairs for all sellers

    Input:
    - Raw Data

    Output
    - A table containing all merged frequently bought item pairs for each seller

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
    import plotly.graph_objects as go
    import plotly.figure_factory as ff
    
    # support
    import streamlit as st

    # input the reaw merge files
    @st.cache
    def load_data():
        df_results = pd.read_csv("data/df_results_market_basket.csv")

        # seller name
        df_seller_name_list = pd.read_csv("data/df_seller_name_list.csv")

        return df_results, df_seller_name_list
    df_results, df_seller_name_list = load_data()
    

    ############################################################################################
    # select the seller and pre-processing data

    st.write("## Market Basket Analysis")


    # select box of seller id
    ID = list(set(df_results['seller_id'].tolist()))
    ID.sort()
    seller_id = st.selectbox("Choose Seller ID", ID, ID.index(10412))

    # filter to get the seller name
    seller_name = df_seller_name_list[df_seller_name_list['accountnum'] == seller_id]['domain'].values[0]
    st.write(f"#### Seller Selected: {seller_name}")


    # slider to get lift cutoff
    lift_cutoff = st.slider("Choose Lift Cutoff", 1.0, 3.0, 1.2, 0.01)

    # plot a heatmap to know how strong the association is based on lift values
    rules_clean = df_results[(df_results['seller_id'] == seller_id) & (df_results['lift'] > lift_cutoff)]
    rules_clean['lift'] = np.round(rules_clean['lift'],2)
    pivot_support = rules_clean.pivot(index='antecedents', columns='consequents', values='lift')

    ############################################################################################
    # plot the heat map

    st.write('--------------------------------')
    st.write("#### Top 10 Frequently Bought Together Varietals")
    # st.write(rules_clean.sort_values(by='lift',ascending=False)[0:10].reset_index(drop=True))

    df_top_10_freq = rules_clean.sort_values(by='lift',ascending=False)[0:10].reset_index(drop=True)
    
    for col in df_top_10_freq.columns:
        if df_top_10_freq[col].dtype == "float64":
            df_top_10_freq[col] = np.round(df_top_10_freq[col],2)

    fig =  ff.create_table(df_top_10_freq, height_constant=30)
    fig.update_layout(width=1000)
    st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))


    ############################################################################################
    # plot the heat map

    st.write('--------------------------------')
    st.write("#### Frequently Bought Heatmap")

    # plotly plot
    fig = px.imshow(
        pivot_support, 
        text_auto=True,
        color_continuous_scale=[[0.0, 'white'], [1.0, '#24477F']],
    )

    fig.update_layout(
        width=1500, 
        height=1000, 
        # title='Frequently Bought Varietals',
        margin=dict(l=0, r=0, t=50 ,b=0),
        legend=dict(
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
            orientation="h",
            ),
    )


    st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))