

def page7():
    '''
    ploting seller segmentation results

    Input:
    - seller data merged 

    Output:
    - spider charts
    - cluster summary table
    - PCA plot with clusters 
    '''


    ############################################################################################
    # helper function 

    def rfm_values(df):
        df_new = df.groupby(['Cluster']).agg({
            'Recency': 'median',
            'Frequency': 'median',
            'Monetary': 'median',
            'itemid': 'median',
            'winvarietal': 'median',
            'product_winbrand': 'median',
            'windeliverystate': 'median',
            'winretailvalue': 'median',
            'sparkling': 'median',
            'seller_tenure': 'median',
            'popular_percent':['median','count'],
    
        }).round(0)
        return df_new
    





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
    @st.cache
    def load_data():
        df_seller_merge = pd.read_csv("data/df_seller_merge.csv")
        return df_seller_merge
    df_seller_merge = load_data()
    df_cluster_3 = df_seller_merge.drop(columns=['CustomerID'],axis=1)


    ############################################################################################
    # data prepare and kmeans clustering

    # make all logs
    for col in df_cluster_3.columns:
        df_cluster_3[col] = np.log(df_cluster_3[col] + 1)


    # standard scaler
    scaler = StandardScaler()
    scaler.fit(df_cluster_3)
    df_cluster_3_scaled = scaler.transform(df_cluster_3)
    
    def kmeans(normalised_df_rfm, clusters_number, original_df_rfm):
        kmeans = KMeans(n_clusters = clusters_number, random_state = 1)
        kmeans.fit(normalised_df_rfm)
    
        # Extract cluster labels
        cluster_labels = kmeans.labels_
    
        # Create a cluster label column in original dataset
        df_new = original_df_rfm.assign(Cluster = cluster_labels)
        return df_new

    df_rfm_k6 = kmeans(df_cluster_3_scaled, 6, df_seller_merge)
    
    cluster_cols = [
        'Recency',
        'Frequency',
        'Monetary',
        'itemid',
        'winvarietal',
        'product_winbrand',
        'windeliverystate',
        'winretailvalue',
        'WINERYDELV',
        'sparkling',
        'seller_tenure',
        'popular_percent'
        ]
    
    
    new_cols = [
        'Recency',
        'Frequency',
        'Monetary',
        'Items',
        'Varietals',
        'Brands',
        'States',
        'Price',
        'Winery Delivery',
        'Sparkling',
        'Tenure',
        'Popular Percent'
        ]
    
    normalised_df_rfm = pd.DataFrame(
        df_cluster_3_scaled, 
        index=df_rfm_k6.index, 
        columns=cluster_cols
    )

    ############################################################################################
    # summary table
    st.write("## Seller Segmentation")
    st.write('--------------------------------')

    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Seller Data Highlights")
        st.write(" ")
        st.metric(label="Total Seller in Data", value = len(df_seller_merge))
        st.metric(label="Average Transactions USD$ Million Per Seller", value = int(df_seller_merge['Monetary'].mean()/1000000))        
        st.metric(label="Total Time Span of Data", value = "2018 - 2022")

    with col2:
        st.write(
            '''
            #### Segmentation Features
            - **Recency**: Days since last shipments 
            - **Frequency**: Unique count of transaction id
            - **Monetary**: All retail price multiply by sales quantity
            - **Items**: Unique item id count
            - **Varietals**: Unique varietals count
            - **Brands**: Unique brands count
            - **States**: Unique state count
            - **Price**: Average retail price
            - **Sparkling**: Sales quantity of sparkling
            - **Tenure**: Time between first and last day of shipping     
            - **Popular**: Percentage of popular wines this seller sells
            ''')
    st.write('--------------------------------')
    st.write("#### Summary Table")

    df_cluster_summary = rfm_values(df_rfm_k6)
    df_rfm_k6['Cluster'] = df_rfm_k6['Cluster'] + 1
    df_cluster_summary['Percent'] = df_rfm_k6.groupby('Cluster').size()/df_rfm_k6.shape[0]
    df_cluster_summary.columns = [
        'Recency',
        'Frequency',
        'Monetary',
        'Items',
        'Varietals',
        'Brands',
        'States',
        'Price',
        'Sparkling',
        'Tenure',
        'Popular',
        'Count',
        'Percent'
        ]
    
    df_cluster_summary = df_cluster_summary.reset_index()
    df_cluster_summary['Cluster'] = df_cluster_summary['Cluster'] + 1
    df_cluster_summary['Cluster'] = df_cluster_summary['Cluster'].astype(str)
    df_cluster_summary['Percent'] = np.round(df_cluster_summary['Percent'],2)
    df_cluster_summary['Monetary'] = df_cluster_summary['Monetary'].astype(np.int64)
    df_cluster_summary['Items'] = df_cluster_summary['Items'].astype(np.int64)
    

    # Table View of the undersold anlaysis
    fig =  ff.create_table(df_cluster_summary, height_constant=30)
    fig.update_layout(width=1000)
    st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))

    col1, col2, col3= st.columns(3)
    with col1:    

        st.write("####  Clusters Interpretations")
        st.write(" ")
        st.write("- **Cluster 1: Concentrated Small Seller** - New buyers who buys frequently and has potential to become a loyal customer")
        st.write("- **Cluster 2: Concentrated Big Seller** - Loyal customers who buy a lot and buy most often ")
        st.write("- **Cluster 3: Almost lost small seller** - Buyers focus buying wines with a higher price from one seller")
        st.write("- **Cluster 4: Champion Seller** - Buyers who buy less often, but with more expensive wines")
        st.write("- **Cluster 5: Versatile Big Seller** - Buyers who have not bought for a long time ")
        st.write("- **Cluster 6: Almost lost high-end seller** - High-value Buyers who bought wines from multiple sellers")


    with col2:
        st.write("#### Segment Counts")
        df_cluster_summary.sort_values(by='Cluster', inplace=True)
        df_cluster_summary.reset_index(drop=True)
        labels = "Cluster_" + df_cluster_summary['Cluster'].astype(np.int64).astype(str)
        values = df_cluster_summary['Count'].astype(np.int64)
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
        labels = "Cluster_" + df_cluster_summary['Cluster'].astype(np.int64).astype(str)
        values = df_cluster_summary['Count'].astype(np.int64) * df_cluster_summary['Monetary'].astype(np.int64) / 1000000
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
    st.write("#### Snake Plot Showing K-Means")

    normalised_df_rfm.columns = new_cols
    normalised_df_rfm['Cluster'] = df_rfm_k6['Cluster']
    normalised_df_rfm['CustomerID'] = df_seller_merge['CustomerID']
    df_melt = pd.melt(normalised_df_rfm.reset_index(), 
                        id_vars=['CustomerID', 'Cluster'],
                        value_vars=new_cols, 
                        var_name='Metric', 
                        value_name='Value')
    
    fig = plt.figure(figsize=(20, 10))
    sns.set(font_scale=0.9)
    sns.pointplot(data=df_melt, x='Metric', y='Value', hue='Cluster')
    # fig.set_xticklabels(fig.get_xticklabels(), rotation=30)
    plt.xticks(rotation=30)
    # plt.title('Snake Plot of K-Means = 6')
    plt.xlabel('Metric')
    plt.ylabel('Value')
    st.pyplot(fig, use_container_width=True)
    



