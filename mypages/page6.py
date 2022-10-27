

from cProfile import label


def page6():
    '''
    ploting buyer segmentation results

    Input:
    - buyer data merged 

    Output:
    - spider charts
    - cluster summary table
    - PCA plot with clusters 
    '''


    ############################################################################################
    # import and setup
    
    # basic library
    import pandas as pd
    import numpy as np
    import warnings
    warnings.filterwarnings(action='ignore')

    # plot
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.figure_factory as ff
    import plotly.graph_objects as go
    import plotly.express as px

    # support
    import datetime as dt
    from datetime import timedelta
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    import streamlit as st
    
    # import data 
    df_buyer_merge_snake_plot = pd.read_csv("data/df_buyer_spider_plot.csv")
    df_buyer_cluster_summary = pd.read_csv("data/df_buyer_cluster_summary.csv")

    st.write("## Buyer Segmentation")
    st.write('--------------------------------')

    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Buyer Data Highlights")
        st.metric(label="Total Buyers in Data", value = df_buyer_cluster_summary['Count'].sum())
        st.metric(label="Total Seller in Data", value = 769)        
        st.metric(label="Total Time Span of Data", value = "2018 - 2022")



            

    with col2:
        st.write(
            '''
            #### Segmentation Features
            Showing results of the buyer segmentation using RFM analysis and other key features:
            - Recency: Days since last purchase (Days)
            - Frequency: Unique count of transaction id (Counts)
            - Monetary: All retail price multiplied by sales quantity ($)
            - Wine Age: Difference between wine vintage and purchases date (Years)
            - Varietal: Total unique varietals that the buyer bought (Counts)
            - Wine Type: Total unique wine types that the buyer bought (Counts)
            - Sellers: Total unique sellers that the buyer bought (Counts)
            - Tenure: Time between first and last day of purchase (Days)
            ''')

    st.write('--------------------------------')

    ############################################################################################
    # Summary Table
    # Table View of the undersold anlaysis
    st.write("#### Summary Table of All Clusters and Features")   
    fig = ff.create_table(df_buyer_cluster_summary, height_constant=30)
    fig.update_layout(width=1000)
    st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))


    col1, col2, col3= st.columns(3)
    with col1:    

        st.write("####  Clusters Interpretations")
        st.write(" ")
        st.write("- **Cluster 1: Potential Loyalist** - New buyers who buys frequently and has potential to become a loyal customer")
        st.write("- **Cluster 2: Champion Buyer** - Loyal customers who buy a lot and buy most often ")
        st.write("- **Cluster 3: High-end Single Buyers** - Buyers focus buying wines with a higher price from one seller")
        st.write("- **Cluster 4: Anniversary and Holiday Shopper** - Buyers who buy less often, but with more expensive wines")
        st.write("- **Cluster 5: Almost Lost Small Spenders** - Buyers who have not bought for a long time ")
        st.write("- **Cluster 6: High-end Multiple Buyers** - High-value Buyers who bought wines from multiple sellers")

        

    with col2:
        st.write("#### Segment Counts")
        df_buyer_cluster_summary.sort_values(by='Cluster', inplace=True)
        df_buyer_cluster_summary.reset_index(drop=True)
        labels = "Cluster_" + df_buyer_cluster_summary['Cluster'].astype(np.int64).astype(str)
        values = df_buyer_cluster_summary['Count'].astype(np.int64)
        colors=px.colors.qualitative.G10

        fig = go.Figure(
            data=[go.Pie(
                labels=labels, 
                values=values, 
                hole=.7,
                )])

        fig.update_traces(
            marker=dict(colors=colors),)


        fig.update_layout(
                margin=dict(l=0,r=0,t=0,b=0),

            )
        st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))


    with col3:
        st.write("#### Segment Revenue ($USD Million)")
        labels = "Cluster_" + df_buyer_cluster_summary['Cluster'].astype(np.int64).astype(str)
        values = df_buyer_cluster_summary['Count'].astype(np.int64) * df_buyer_cluster_summary['Monetary'].astype(np.int64) / 1000000
        colors=px.colors.qualitative.G10

        fig = go.Figure(
            data=[go.Pie(
                labels=labels, 
                values=values, 
                hole=.7,

                )])

        fig.update_traces(
            marker=dict(colors=colors),)


        fig.update_layout(
                margin=dict(l=45,r=45,t=45,b=45),
                showlegend=False
            )
        st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))

    ############################################################################################
    # snake plot
    st.write('--------------------------------')
    st.write("#### Snake Plot of K-Means = 6 Clusters")    

    df_buyer_merge_snake_plot['Cluster'] = df_buyer_merge_snake_plot['Cluster'] + 1
    fig = plt.figure(figsize=(16, 8))
    sns.set(font_scale=0.9)
    sns.pointplot(
        data=df_buyer_merge_snake_plot, 
        x='Metric', 
        y='Value', 
        hue='Cluster',
        order=['Recency', 'Frequency', 'Monetary', 'Wine Age','Varietal', 'Wine Type', 'Sellers','Tenure']
        )
    # plt.title('Snake Plot of K-Means = 6')
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.legend(title='Segments', loc='upper left')
    st.pyplot(fig)



    