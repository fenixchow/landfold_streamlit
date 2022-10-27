

def show_page(params):

    '''
    this is the deploy calculation of BPR, including a compare with
    a seller's historical sales vs. its top 10 recommended items
    by compare these two we can figure out the sales gap and
    undersold varietals 
    
    '''


    # ############################################################################################
    # # import and setup

    # # basic library
    # import pandas as pd
    # import numpy as np
    # import warnings
    # warnings.filterwarnings(action='ignore')

    # # plot
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # import plotly.express as px
    # import plotly.graph_objects as go
    # import plotly.figure_factory as ff
    
    # # support 
    # import streamlit as st


    # # hide row index of a streamlit dataframe table
    # # CSS to inject contained in a string
    # # CSS to inject contained in a string
    # hide_dataframe_row_index = """
    #             <style>
    #             .row_heading.level0 {display:none}
    #             .blank {display:none}
    #             </style>
    #             """

    # # Inject CSS with Markdown
    # st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)




    # # import dataset
    # @st.cache(allow_output_mutation=True)
    # def load_data():
    #     df = pd.read_csv("data/base_sin_v6.csv")
    #     df['ConsultationDate'] = pd.to_datetime(df['ConsultationDate'])
    #     return df
    # df = load_data()

    # # standard color
    # colors=px.colors.qualitative.G10

    # ########################################################################
    # # Data Summary
    # st.write("## Feature Engineering")
    # st.write('''Use various methodology of feature engineering to find out the potental fraud patterns, including but not limited to''')
    # st.write('''
    #     - Duplicate Claim
    #     - Overclaimed Amount
    #     - Excess Quantity
    #     - Incorrect Signature
    #     - Late Submission
    #     - Missing Info
    #     - Not Covered
    #     - Non Refundable
    #     - Rectification Deadline Exceeded
    #     - Refill Too Soon
    #     - Service Inconsistant
    #     - Treatment Before Coverage    
    # ''')
    # st.write("We group all these claim items with 0-99% approval rate as **soft fraud**, and **labeled them as 1**")
    # st.write("We group all other valid claim items are **labeled as 0*")
    # st.write("We can frame this problem as a **supervised machine learning with binary classification**")
    # st.write('--------------------------------')
    # st.write('#### Fraud Type 1: Duplicate Claim')
    # st.write('The hypothesis is that if a beneficiary made too many claims with huge amount, it raise a concern on the fraud.')














    # ############################################################################################
    # # select the seller and pre-processing data
    # st.write("## BPR Undersold Analysis")
    # ID = list(set(df_historical_sales['custaccount'].tolist()))
    # ID.sort()
    # seller_id = st.selectbox("Choose Seller ID", ID, ID.index(10412))

    # # filter to get the seller name
    # seller_name = df_seller_name_list[df_seller_name_list['accountnum'] == seller_id]['domain'].values[0]


    # ############################################################################################
    # # top 10 recommendation
    # st.write('--------------------------------')
    # st.write(f"#### Top 10 Recommendation for {seller_name}")
    # st.write("Compare the top 10 recommended varietals vs. their actual sales to identify undersold gaps")
    # undersold = df_undersold[df_undersold['supplier'] == seller_id]
    # undersold = undersold[['supplier','winvarietal_product','recommend_rank','sales_rank','rank_difference']].\
    #     sort_values(by='rank_difference',ascending=False).reset_index(drop=True)

    # performance = []
    # # give the perform bucket based on rank difference
    # for i in range(undersold.shape[0]):
    
    #     rank_diff = undersold.loc[i,'rank_difference']
    #     if rank_diff > 10:
    #         performance.append('Heavily Underperform')
    #     elif rank_diff > 0:
    #         performance.append('Slighly Underperform')
    #     elif rank_diff >= -5:
    #         performance.append('Slightly Outperform')
    #     else:
    #         performance.append('Heavily Outperform')
    # undersold['performance'] = performance

    # # Table View of the undersold anlaysis
    # undersold_plotly = undersold.copy()
    # undersold_plotly.columns = ['Supplier',' Varietal','Recommend Rank','Sales Rank','Rank Difference','Performance']

    # # plot
    # fig =  ff.create_table(undersold_plotly, height_constant=30)
    # fig.update_layout(width=1000, margin=dict(l=0,r=0,t=0,b=0),)
    # st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))



    # ############################################################################################
    # # historical sales
    # st.write('--------------------------------')
    # st.write("#### Historical Sales by Varietal")

    # # get the sales count for the seller
    # cnt_sales = df_historical_sales[df_historical_sales['custaccount'] == seller_id].reset_index(drop=True)

    # # show top 50 sales 
    # varietal_count = min(cnt_sales.shape[0],50)

    # # sort the sales from the top
    # cnt_sales = cnt_sales.sort_values(by='salesqty',ascending=False)[0:varietal_count]

    # fig = px.treemap(
    #     cnt_sales, 
    #     # path= [cnt_sales['winvarietal_product']],
    #     path=[px.Constant("<br>"), 'winvarietal_product'],
    #     values=cnt_sales['salesqty'],
    #     color=cnt_sales['salesqty'],
    #     color_continuous_scale=[[0.0, 'white'], [1.0, '#24477F']],
    #     width=1000, 
    #     height=600,
    #     custom_data='',
    #     # title='Historical Sales by Varietal',
    # )

    # # switch on legend
    # # fig.update(layout_coloraxis_showscale=False)

    # # remove the background
    # fig.data[0].customdata[-1][0] = 100
    # fig.data[0].marker.colors[-1] = 0
    # fig.update_layout(margin=dict(l=0,r=0,t=0,b=0),)
    
    # # with expd:
    # st.plotly_chart(fig, use_container_width=False, config= dict(displayModeBar = False))


    # ############################################################################################      
    # # top 10 preformnace 
    # st.write('--------------------------------')    
    # col1, col2 = st.columns(2)



    # # get the top 10 recommend for current seller
    # top_10_recommend = df_top_10_recommendation[df_top_10_recommendation['supplier_id'] == seller_id]

    # # merge the top 10 recommend with the performance bucket for color
    # top_10_recommend = top_10_recommend.merge(undersold[['winvarietal_product','performance']],\
    #     left_on='varietal',right_on='winvarietal_product',how='left').fillna('Not sold by this seller').drop('winvarietal_product',axis=1)

    # # plot the bar chart
    # fig = px.bar(
    #     top_10_recommend, 
    #     x="sales_qty", 
    #     y="varietal", 
    #     color='performance', 
    #     orientation='h',
    #     hover_data=["varietal", "sales_qty"],
    #     width=1000, 
    #     height=400,
    #     # title='Top 10 Recommendations by Performance',
    #     color_discrete_map = {
    #         'Heavily Underperform': 'darkred',
    #         'Slighly Underperform':'lightpink',
    #         'Slightly Outperform':'lightgreen',
    #         'Heavily Outperform':'green',
    #         'Not sold by this seller':'lightgray'
    #     }
    # )

    # fig.update_layout(
    #     # title="sales_qty & varietal",
    #     barmode='stack', 
    #     yaxis={'categoryorder':'total ascending'}, 
    #     margin=dict(l=0,r=0,t=50,b=0),
    #     legend=dict(
    #         yanchor="top",
    #         y=-0.15,
    #         xanchor="center",
    #         x=0.5,
    #         orientation="h",
    #     ),
    #     xaxis_title=None,
    #     yaxis_title=None,
    # )
    
    # # fig.update_yaxes(visible=True, showticklabels=True)
    # # fig.update_xaxes(visible=False, showticklabels=True)

    # with col1:
    #     st.write("#### Top 10 Recommendation Performance")
    #     st.write("highlight the top 10 recommended varietals vs. sales rank difference")
    #     st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))




    # ############################################################################################      
    # # boost sales   
    
    


    # # bar chart for the boosted sales
    # boost_sales = df_undersold[df_undersold['supplier'] == seller_id]
    # boost_sales = boost_sales[['item','salesqty','expected_qty']]
    # boost_sales.columns = ['Varietal','Original Sales','Expected Sales']
    # # boost_sales

    # # build groupd (clustered) bar chart
    # fig = go.Figure(data=[
    #     go.Bar(
    #         name='Original Sales', 
    #         x=boost_sales['Varietal'], 
    #         y=boost_sales['Original Sales'],
    #         marker_color = "#D9C666"
    #         ),
    #     go.Bar(
    #         name='Expected Sales', 
    #         x=boost_sales['Varietal'], 
    #         y=boost_sales['Expected Sales'],
    #         marker_color = '#24477F'
    #         )
    # ])

    # # Change the bar mode
    # fig.update_layout(
    #     barmode='group', 
    #     width=1000, 
    #     height=400, 
    #     # title='Boosted Sales Volume', 
    #     margin=dict(l=0,r=0,t=50,b=0),
    #     legend=dict(
    #         yanchor="top",
    #         y=-0.15,
    #         xanchor="center",
    #         x=0.5,
    #         orientation="h",
    #         ),
    #     # xaxis=dict(title="Varietal",), 
    #     yaxis=dict(title="Original Sales & Expected Sales",), 
    #     )

    # with col2:
    #     st.write("#### Boosted Sales")
    #     st.write("Assuming 30% conversion rate and average buying quantity for each buyer")
    #     st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))