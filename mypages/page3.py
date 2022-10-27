

def page3():

    '''
    this page is to present the state undersold analysis 
    input:
        - seller id
        - varietal
        - self-rank, state-rank or peer rank
    
    output:
        - map view indicate top and bottom state
        - boost volume
    
    '''

    ############################################################################################
    # helper functions
    
    # prepare the data
    def prepare_data(df, metric, mode):
        largest_index = df[metric].nlargest(3).index
        smallest_index = df[metric].nsmallest(3).index

        df['performance'] = 'NONE'
        df['performance'] = np.where(df.index.isin(largest_index),"h_over", df['performance'])
        df['performance'] = np.where(df.index.isin(smallest_index),"h_under", df['performance'])
        df[metric] = np.round(df[metric],1)

        # crate lables
        if mode == 'percent':
            df['annotation'] = df['windeliverystate'] + " (" + df[metric].astype(int).astype(str) + "%)"
        if mode == 'volume':
            df['annotation'] = df['windeliverystate'] + " (" + df[metric].astype(int).astype(str) + ")"
    
        df['annotation'] = np.where(df['performance'] == "NONE", df['windeliverystate'],df['annotation'])

        # merge with full states for ploting purpse
        df = df.merge(df_clean_50_states[['true_state']], left_on='windeliverystate',right_on='true_state',how='right')

        # fill na
        df['annotation'].fillna(df['true_state'],inplace=True)
        df['performance'].fillna('NONE',inplace=True)

        return df

    # plot
    def plot_choropleth_map(df):

        fig = px.choropleth(
            df, 
            locations='true_state',
            locationmode="USA-states", 
            color='performance', 
            scope="usa",
            color_discrete_map={
                "h_over":"lightgreen",
                "h_under":"lightpink",
                "NONE":"white"
            },
        )

        fig.add_scattergeo(
            locations=df['true_state'],    ###codes for states,
            locationmode='USA-states',
            text=df[['annotation']],
            mode='text',
            texttemplate = "<b>%{text}",
            textfont=dict(
                family="Roboto",
                size=12,
                color="Black"
            )
        )

        fig.update_layout(
            width=1000,
            height=600,
            margin=dict(l=0,r=0,t=0,b=0),
            legend=dict(
                yanchor="bottom",
                y=1,
                xanchor="center",
                x=0.5,
                orientation="h",
                )
        )
    
        # fig.show()
        return fig
    


    
    def plot_boost(df, metric):

        df_seller_insight= df[df[metric] > 0]

        df_seller_agg_varietal = df_seller_insight.groupby('winvarietal')[metric].sum().sort_values(ascending=False)[0:3]
        top_3_varietal = df_seller_agg_varietal.index.to_list()
        df_seller_filtered = df_seller_insight[df_seller_insight['winvarietal'].isin(top_3_varietal)]

        df_seller_agg_state = df_seller_filtered.groupby('windeliverystate')[metric].sum().sort_values(ascending=False)[0:5]
        top_5_state = df_seller_agg_state.index.to_list()
        df_seller_filtered = df_seller_filtered[df_seller_filtered['windeliverystate'].isin(top_5_state)].reset_index(drop=True)

        fig = px.bar(
            df_seller_filtered, 
            x="windeliverystate",
            y=metric,
            barmode="group",
            color="winvarietal",
            color_discrete_map={
                top_3_varietal[0]: "#042440",
                top_3_varietal[1]: "#E37222",
                top_3_varietal[2]: "#BFAE5A",
                },
        )

        fig.update_layout(
            width=1000,
            height=400,
            margin=dict(l=0,r=0,t=50,b=0),
            xaxis={'categoryorder':'total descending'},
            legend=dict(
                yanchor="top",
                y=-0.15,
                xanchor="center",
                x=0.5,
                orientation="h",
            ),
        )

        fig.update_xaxes(
            title_text = "State",
            #title_font = {"size": 20},
            #tickfont=dict(size=16),
            # title_standoff = 15
        )

        fig.update_yaxes(
            title_text = "Sales",
            #title_font = {"size": 20},
            #tickfont=dict(size=16),
            # title_standoff = 15
        )

        # fig.show()
        return fig




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

    # support
    import tqdm
    from scipy import stats
    import streamlit as st

    # import the merged data 
    @st.cache
    def load_data():
        df_final_insight = pd.read_csv("data/df_final_insight.csv")
        
        # seller name
        df_seller_name_list = pd.read_csv("data/df_seller_name_list.csv")

        return df_final_insight, df_seller_name_list
    df_final_insight, df_seller_name_list = load_data()

    # clean 50 states to match on 
    clean_50_states = ['CA', 'IA', 'IL', 'MN', 'NE', 'TX', 'IN', 'KS', 'OH', 'MO', 'WA',
       'NC', 'SD', 'FL', 'ND', 'AR', 'MI', 'WI', 'GA', 'MS', 'ID', 'PA',
       'LA', 'KY', 'CO', 'OR', 'MT', 'OK', 'TN', 'NY', 'AZ', 'AL', 'VA',
       'SC', 'NM', 'MD', 'NJ', 'UT', 'HI', 'WY', 'DE', 'ME', 'CT', 'MA',
       'VT', 'NV', 'WV', 'NH', 'RI', 'AK']

    df_clean_50_states = pd.DataFrame({
        'true_state': clean_50_states
    })



    ############################################################################################
    # select the seller and pre-processing data
    
    st.write("## State Undersold Analysis")
    ID = list(set(df_final_insight['accountnum'].tolist()))
    ID.sort()
    V = list(set(df_final_insight['winvarietal'].tolist()))
    kind = ["State Ranking", "Self Ranking", "Peer Ranking"]    



    seller_id = st.selectbox("Choose Seller ID", ID, ID.index(10412))
    # filter to get the seller name
    seller_name = df_seller_name_list[df_seller_name_list['accountnum'] == seller_id]['domain'].values[0]
    st.write(f"#### Seller Selected: {seller_name}")


    varietal = st.selectbox("Choose Varietal", V, V.index('PINOT NOIR'))
    KIND = st.selectbox("Choose Ranking Method", kind, 0)

    # filter the data by seller id and varietal
    df_insight = df_final_insight[
        (df_final_insight['accountnum'] == seller_id) &
        (df_final_insight['winvarietal'] == varietal)]

    
    if KIND==kind[0]:
        df_insight_percent = prepare_data(df_insight, 'percentile_diff_insight_1','percent')
        df_insight_volume = prepare_data(df_insight, 'undersold_volume_insight_1','volume')
    elif KIND==kind[1]:    
        df_insight_percent = prepare_data(df_insight, 'percentile_diff_insight_2','percent')
        df_insight_volume = prepare_data(df_insight, 'undersold_volume_insight_2','volume')
    elif KIND==kind[2]:    
        df_insight_percent = prepare_data(df_insight, 'percentile_diff_insight_3','percent')
        df_insight_volume = prepare_data(df_insight, 'undersold_volume_insight_3','volume')




    ############################################################################################
    # plot percentile map

    st.write('--------------------------------')
    col1, col2 = st.columns(2)
    with col1:

        st.write("#### Undersold States By Percentile")
        fig = plot_choropleth_map(df_insight_percent)
        st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))

    ############################################################################################
    # plot volume map
    # st.write('--------------------------------')
    with col2:
        st.write("#### Undersold States By Volume")
        fig = plot_choropleth_map(df_insight_volume)
        st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))


    ############################################################################################
    # plot the sales boost
    
    st.write('--------------------------------')     
    st.write("#### Sales Boost by State")
    

    df_seller = df_final_insight[df_final_insight['accountnum'] == seller_id]

    if KIND==kind[0]:
        fig = plot_boost(df_seller, 'sales_boost_insight_1')
        st.plotly_chart(fig, use_container_width=False, config= dict(displayModeBar = False))
    elif KIND==kind[1]:
        fig = plot_boost(df_seller, 'sales_boost_insight_2')
        st.plotly_chart(fig, use_container_width=False, config= dict(displayModeBar = False))
    elif KIND==kind[2]:
        fig = plot_boost(df_seller, 'sales_boost_insight_3')
        st.plotly_chart(fig, use_container_width=False, config= dict(displayModeBar = False))