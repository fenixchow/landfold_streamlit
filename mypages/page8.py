


def page8():

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

    # support
    import streamlit as st
    from streamlit_plotly_events import plotly_events
    
   
    # import the merged data 
    @st.cache
    def load_data():
        df_deploy_w_pred = pd.read_csv("data/df_depoly_w_pred.csv")
        df_feat_importance =  pd.read_csv("data/df_churn_deploy_feat_importance.csv")
        
        # seller name
        df_seller_name_list = pd.read_csv("data/df_seller_name_list.csv")

        return df_deploy_w_pred, df_feat_importance, df_seller_name_list
    df_deploy_w_pred, df_feat_importance, df_seller_name_list = load_data()
    
    ############################################################################################
    # select seller id and pre-processing
    st.write("## Chrun Deploy Results")
    st.write("Showing the churn customers in different risk level for targeting")

    v = df_deploy_w_pred['accountnum'].unique().tolist()
    v.sort()
    seller_id = st.selectbox("Choose Seller ID", v, v.index(10412))
    # filter to get the seller name
    seller_name = df_seller_name_list[df_seller_name_list['accountnum'] == seller_id]['domain'].values[0]
    st.write(f"#### Seller Selected: {seller_name}")


    df_deploy_w_pred_seller = df_deploy_w_pred[df_deploy_w_pred['accountnum'] == seller_id]


    st.write('--------------------------------')

   ############################################################################################
    # risk level pivot table  
    st.write("#### Different Risk Level of Churn Customers")
    df_deploy_w_pred_pivot = df_deploy_w_pred_seller.pivot_table(
        values=['Recency','Frequency','Monetary','Sales Quantity','Distinct Items','Distinct Varietals'],
        index='risk',
        # columns=['Recency','Frequency','Monetary','Sales Quantity','Distinct Items','Distinct Varietals'],
        aggfunc='mean'
        ).reset_index()
    
    
    df_deploy_w_pred_pivot = df_deploy_w_pred_pivot[['risk','Recency','Frequency','Monetary','Sales Quantity','Distinct Items','Distinct Varietals']]
    
    df_deploy_w_pred_pivot.set_index('risk',inplace=True)
    # df_deploy_w_pred_pivot.reindex(["Low Risk", "Medium Risk", "High Risk", "Very High Risk"],inpla)
    df_deploy_w_pred_pivot = df_deploy_w_pred_pivot.loc[["Low Risk", "Medium Risk", "High Risk", "Very High Risk"],]
    df_deploy_w_pred_pivot.loc['Overall Average',] = df_deploy_w_pred_seller['Recency'].mean()
    df_deploy_w_pred_pivot.loc['Overall Average', 'Frequency'] = df_deploy_w_pred_seller['Frequency'].mean()
    df_deploy_w_pred_pivot.loc['Overall Average', 'Monetary'] = df_deploy_w_pred_seller['Monetary'].mean()
    df_deploy_w_pred_pivot.loc['Overall Average', 'Sales Quantity'] = df_deploy_w_pred_seller['Sales Quantity'].mean()
    df_deploy_w_pred_pivot.loc['Overall Average', 'Distinct Items'] = df_deploy_w_pred_seller['Distinct Items'].mean()
    df_deploy_w_pred_pivot.loc['Overall Average', 'Distinct Varietals'] = df_deploy_w_pred_seller['Distinct Varietals'].mean()
    
    for col in df_deploy_w_pred_pivot.columns:
        df_deploy_w_pred_pivot[col] = df_deploy_w_pred_pivot[col].astype(np.int64)
    
    fig = ff.create_table(df_deploy_w_pred_pivot, index=True)
    
    fig.update_layout(
        width=1000,
        height=300,
        )
    
    # # Make text size larger
    # for i in range(len(fig.layout.annotations)):
    #     fig.layout.annotations[i].font.size = 12
    
    st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))
    


    ############################################################################################
    # plot feature importance
    st.write('--------------------------------')
    col1, col2 = st.columns(2)
   

    fig = px.bar(
        df_feat_importance, 
        y='variable', 
        x=df_feat_importance['scaled_importance'], 
        text_auto='.2f',
        # title="Feature Importance of Churn Prediction",
        orientation='h',
        )
    
    fig.update_xaxes(title='Count', visible=False, showticklabels=True)
    fig.update_yaxes(title='', visible=True, showticklabels=True)
    
    fig.update_layout(
        width=1000,
        height=600,
        yaxis={'categoryorder':'total ascending'},
        margin = dict(pad=20),
        )
    
    fig.update_traces(
        marker_color='#24477F')

    with col1:
        st.write("#### Feature Importance of Churn")
        st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))
    
    ############################################################################################
    # plot risk bar chart
    
    df_risk_bar = df_deploy_w_pred_seller.groupby('risk').size().to_frame().reset_index().rename(columns={0:"count"})

    fig = px.bar(
        df_risk_bar, 
        y='risk', 
        x='count', 
        text_auto='int',
        # title="Customer Counts by Churn Risk",
        color='risk',
        orientation='h',
        category_orders={"risk": ["Low Risk", "Medium Risk", "High Risk", "Very High Risk"]},
        color_discrete_sequence=['#41C572','#F4D646','#FBCBCB','#F05050']
        )
    
    fig.update_layout(
        width=1000,
        height=600,
        xaxis=dict(title='', visible=True, showticklabels=True),
        yaxis=dict(title='', visible=True, showticklabels=True),
        )
    
    #fig.update_xaxes(title='', visible=True, showticklabels=True)
    #fig.update_yaxes(title='', visible=True, showticklabels=True)
    with col2:
        st.write("#### Customer Counts by Churn Risk")
        st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))
    
 
    ############################################################################################
    # scatter plot
    
    df_deploy_w_pred_seller['yes'] = np.round(df_deploy_w_pred_seller['yes'],4)

    fig = px.scatter(
        df_deploy_w_pred_seller,
        x='Monetary', 
        y='yes',
        color = 'risk',
        category_orders={"risk": ["Low Risk", "Medium Risk", "High Risk", "Very High Risk"]},
        color_discrete_sequence=['#41C572','#F4D646','#FBCBCB','#F05050'],
        # title="Churn Risk vs. Customer Value",
    )
    
    fig.update_layout(
        # width = 1000,
        # height = 600,
        xaxis_range=[0,5000],
        # plot_bgcolor='rgba(0,0,0,0)'
    )
    
    fig.update_xaxes(title='Customer Value', visible=True, showticklabels=True)
    fig.update_yaxes(title='Probability of Chrun', visible=True, showticklabels=True)
    
    st.write('--------------------------------')
    st.write("#### Churn Risk vs. Customer Value")
    st.write('''
    Business stratagy is to focus on the customers with high probability of churn with ralatively high values, which will maximize the **Return On Investment** (ROI) for marketing campaign \n
    **Select the data by clicking on the data points**
    ''')

    # generate the compositie index for the selected plotly event
 
    try:
        # plot the figure with click events on
        selected_points = plotly_events(fig)
        selected_point=selected_points[0]

        # generate the unique values from the selection for filter
        selected_index_x = selected_point['x']
        selected_index_y = selected_point['y']
        df_selected_index = df_deploy_w_pred_seller.loc[
            df_deploy_w_pred_seller['yes']==selected_index_y  ,:]  
        df_selected_index = df_selected_index.loc[
            df_selected_index['Monetary']==int(selected_index_x)  ,:]
        st.write(df_selected_index)

    except:
        pass