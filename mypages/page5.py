
def page5():
    '''
    customer cohor analysis

    input:
    - retention data
    output:
    - cohor chart
    '''
    ############################################################################################
    # helper function

    def generate_cohort_matrix(df_retention_all, seller_id, period):
        '''
        plot cohort matrix based on the retention dataset
        '''

        df_retention = df_retention_all[df_retention_all['seller_id'] == seller_id]
        df_retention = df_retention[df_retention['cohort_type'] == period]
        
        cohort_matrix_volume = df_retention.pivot_table(
            index = 'acquisition_cohort',
            columns = 'periods',
            values = 'customers'
        )
        cohort_matrix_percent = cohort_matrix_volume.divide(cohort_matrix_volume.iloc[:,0], axis=0)*100

        for col in cohort_matrix_percent.columns:
            if cohort_matrix_percent[col].dtype == "float64":
                cohort_matrix_percent[col] = np.round(cohort_matrix_percent[col],0)
        

        return cohort_matrix_volume, cohort_matrix_percent

    def plot_cohort(cohort_matrix):
        cohort_matrix.index = [i.replace("-", "/") for i in cohort_matrix.index.tolist()]
        # plotly plot
        fig = px.imshow(
            cohort_matrix, 
            text_auto=True,
            color_continuous_scale=[[0.0, 'white'], [1.0, '#24477F']],
        )

        fig.update_layout(
            width=1000, 
            height=800, 
            # title='Cohort analysis',
            margin=dict(l=0, r=0, t=50 ,b=0),
            legend=dict(
                yanchor="top",
                y=-0.15,
                xanchor="center",
                x=0.5,
                orientation="h",
            ),
        )

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
    import plotly.figure_factory as ff

    # support
    import operator as op
    import streamlit as st

    @st.cache
    def load_data():
        df_retention_all = pd.read_csv("data/df_retention_all.csv")
        # seller name
        df_seller_name_list = pd.read_csv("data/df_seller_name_list.csv")

        return df_retention_all, df_seller_name_list
    df_retention_all, df_seller_name_list = load_data()

    ############################################################################################
    # select the seller and pre-processing data

    st.write("## Cohort Analysis")

    # seller_id select box
    v = df_retention_all['seller_id'].tolist()
    v.sort()
    seller_id = st.selectbox("Choose Seller ID", v, v.index(10412))
    # filter to get the seller name
    seller_name = df_seller_name_list[df_seller_name_list['accountnum'] == seller_id]['domain'].values[0]
    st.write(f"#### Seller Selected: {seller_name}")


    kind = ["year", "quarter", "month"]   

    # year, quarter or month selection
    plot_type = st.selectbox("Choose Ranking Method", kind, 1)

    # get the matrix
    cohort_matrix_volume, cohort_matrix_percent = generate_cohort_matrix(df_retention_all, seller_id, plot_type)

    ############################################################################################
    # plot volume

    st.write('--------------------------------')
    st.write("#### Cohort by Sales Volume")
    
    fig_volume = plot_cohort(cohort_matrix_volume)
    st.plotly_chart(fig_volume, use_container_width=True, config= dict(displayModeBar = False))

    ############################################################################################
    # plot percent
    st.write('--------------------------------')
    st.write("#### Cohort by Retention Rate")

    fig_percent = plot_cohort(cohort_matrix_percent)
    st.plotly_chart(fig_percent, use_container_width=True, config= dict(displayModeBar = False))
    
