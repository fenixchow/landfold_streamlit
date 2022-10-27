
def show_page(params):
    '''
    Shwoing the Land Score

    '''

    ########################################################################
    # helper function
    def plot_map(feature):
        df_map = gdf_zip_code.merge(df_score[['zip_code',feature]], left_on='ZCTA5CE20', right_on='zip_code',how='inner')
        df_map.drop_duplicates(inplace=True)
        df_map[feature].fillna(0,inplace=True)
        fig = px.choropleth_mapbox(
            df_map,
            geojson = df_map.geometry,
            locations = df_map.index,
            color = df_map[feature],
            color_continuous_scale="reds",
            hover_data = ['zip_code'],
            center={"lat": 31.9686, "lon": -99.9018},
            mapbox_style="carto-positron", 
            zoom = 5,
            width = 800,
            height = 800,
            )
        return fig







    ########################################################################
    # import and setup

    # basic library
    import pandas as pd
    import numpy as np

    # plot
    import plotly.express as px


    # support
    import streamlit as st
    import geopandas as gpd


    # import dataset
    @st.cache(allow_output_mutation=True)
    def load_data():
        df_score = pd.read_csv("data/df_score_merge_tx.csv")
        gdf_zip_code = gpd.read_file('data/zip_code_simple.shp')
        df_score['zip_code'] = df_score['zip_code'].astype(np.int64)
        gdf_zip_code['ZCTA5CE20'] = gdf_zip_code['ZCTA5CE20'].astype(np.int64)   
        return df_score, gdf_zip_code
    df_score, gdf_zip_code = load_data()



    ########################################################################
    # make selection

    with st.sidebar:

        st.write("#### Data Source")
        st.write("- www.census.gov")
        st.write("- www.greatschools.org")
        st.write("- www.txdot.gov")
        st.write("- www.realtor.com")
        st.write('--------------------------------')

        selected_radio_button_1 = st.radio("Select Column", options = [
            "Population", 
            "Population Density",
            "Median Income",
            "School Rating",
            "School Funding",
            "Freeway", 
            "Construction",
            "Days on Market", 
            "Number of Views", 
            "Housing Price $/sqft"], 
            )



    ########################################################################
    # plot the map selected
    st.write(f"## {selected_radio_button_1} by Zip Code")

    if selected_radio_button_1 == 'Population':
        fig = plot_map('total_population_score')
        st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))

    if selected_radio_button_1 == 'Population Density':
        fig = plot_map('population_density_score')
        st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))

    if selected_radio_button_1 == 'Median Income':
        fig = plot_map('median_income_score')
        st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))

    if selected_radio_button_1 == 'School Rating':
        fig = plot_map('school_rating_score')
        st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))

    if selected_radio_button_1 == 'School Funding':
        fig = plot_map('school_funding_score')
        st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))

    if selected_radio_button_1 == 'Freeway':
        fig = plot_map('freeway_score')
        st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))

    if selected_radio_button_1 == 'Construction':
        fig = plot_map('construction_score')
        st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))

    if selected_radio_button_1 == 'Days on Market':
        fig = plot_map('median_days_on_market_score')
        st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))

    if selected_radio_button_1 == 'Number of Views':
        fig = plot_map('number_of_veiws_score')
        st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))

    if selected_radio_button_1 == 'Housing Price $/sqft':
        fig = plot_map('home_price_score')
        st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))











    # ########################################################################
    # # helper functions 

    # def donut_chart_claim(df,col):

    #     labels = df[col].value_counts().index.tolist()
    #     values = df[col].value_counts().values.tolist()

    #     fig = go.Figure(
    #         data=[go.Pie(labels=labels, values=values, hole=.7,)],
    #         )

    #     fig.update_traces(
    #         marker=dict(colors=colors),)


    #     fig.update_layout(
    #             margin=dict(l=0,r=0,t=0,b=0),
    #             width = 380,
    #             height = 380,
    #             legend=dict(
    #                 yanchor="top",
    #                 y=0.60,
    #                 xanchor="left",
    #                 x=0.30)                
    #         )

    #     return fig
        

    # def donut_chart_beneficiary(df,col):
    #     distribution = df.groupby(['BeneficiaryID'])[col].first().value_counts()
    #     labels = distribution.index.tolist()
    #     values = distribution.values.tolist()
    #     fig = go.Figure(
    #         data=[go.Pie(
    #             labels=labels, 
    #             values=values, 
    #             hole=.7,
    #             textinfo='none')],
    #         )
    #     fig.update_traces(
    #         marker=dict(colors=colors),)
            
    #     fig.update_layout(
    #             margin=dict(l=0,r=0,t=5,b=0),
    #             width = 400,
    #             height = 400,
    #         )          
    #     return fig

    # def donut_chart_general(df,col):

    #     labels = df[col].value_counts().index.tolist()
    #     values = df[col].value_counts().values.tolist()

    #     fig = go.Figure(
    #         data=[go.Pie(labels=labels, values=values, hole=.7,)],
    #         )

    #     fig.update_traces(
    #         marker=dict(colors=colors),)


    #     fig.update_layout(
    #             margin=dict(l=0,r=0,t=0,b=0),
    #             width = 380,
    #             height = 380,          
    #         )

    #     return fig
        

    # def horizonal_bar_top(df, col, x_axis_title, y_axis_title):
    #     top_15 = df[col].value_counts()[0:15]
    #     fig = px.bar(top_15, orientation='h')
    #     fig.update_traces(marker_color='#203370')                    
    #     fig.update_layout(
    #             margin=dict(l=0,r=0,t=0,b=0),
    #             width = 600,
    #             height = 380,
    #             bargap=0.2,
    #             yaxis={'categoryorder':'total ascending'},
    #             showlegend=False,
    #             xaxis_title=x_axis_title, 
    #             yaxis_title=y_axis_title,                    
    #             )    
    #     return fig

    # ########################################################################
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
    # from streamlit_plotly_events import plotly_events

    

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
    #     return df
    # df = load_data()

    # # standard color
    # colors=px.colors.qualitative.G10

    # ########################################################################
    # # Data Summary
    # st.write("## Basic Exploratory Data Analysis (EDA)")
    # st.write('--------------------------------')

    # col1, col2, col3, col4 = st.columns([2,2,2,2])
    # with col1:
    #     st.metric(label="Total Claim Items", value = df.shape[0])
    #     st.metric(label="100% Approved Claim Items", value = df[df['claim_class'] == 'valid'].shape[0])
    #     st.metric(label="Partialy Approved Claim Items", value = df[df['claim_class'] == 'partially invalid'].shape[0])
    #     st.metric(label="Denied Claim Items", value = df[df['claim_class'] == 'invalid'].shape[0])
    #     st.metric(label="Time Span", value = "12/2021 - 03/2022")


    # with col2:
    #     st.write("**Claim Items Approval**")
    #     fig = donut_chart_claim(df,'claim_class')
    #     st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))


    # with col3:
    #     st.write("**Claim Items Status**")
    #     fig = donut_chart_claim(df,'ClaimStatus')
    #     st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))



    # with col4:
    #     st.write("**Claim Items Type**")
    #     fig = donut_chart_claim(df,'Reimbursement Type')
    #     st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))

    # ########################################################################
    # # Beneficiary Summary
    # st.write('--------------------------------')
    # st.write('#### Beneficiary Stats')
    # selected_radio_button_1 = st.radio("Select Column", options = ["Age", "Gender","Marital Status","Dependency","Family of Causes", "Family of Benefits"], horizontal=True)

    # col1, col2 = st.columns([1,3])
    # with col1:
    #     st.metric(label = 'Total Beneficiaries', value = df['BeneficiaryID'].nunique())
    #     st.metric(label = 'Total Claims', value = df['CalimKey'].nunique())
    #     st.metric(label = 'Average Claim Amount Per Beneficiary (TND)', value = int(np.mean((df.groupby(['BeneficiaryID','CalimKey'])['TotalAmountClaimed'].sum()))))
    #     st.metric(label = 'Average Claim Items Per Claim', value = int(np.mean((df.groupby('BeneficiaryID')['CalimKey'].count()))))

    # with col2:

    #     if selected_radio_button_1 == "Age":
    #         labels = df.groupby('BeneficiaryID')['Age'].mean()

    #         fig = px.histogram(labels, nbins=20)
    #         fig.update_traces(marker_color='#4BA38C')            
    #         fig.update_layout(
    #                 margin=dict(l=0,r=0,t=5,b=0),
    #                 width = 600,
    #                 height = 380,
    #                 bargap=0.2,
    #                 showlegend=False,
    #                 xaxis_title="Age", 
    #                 yaxis_title="Beneficiary Count",      
    #                 )

    #         st.plotly_chart(fig, use_container_width=False, config= dict(displayModeBar = False))


    #     if selected_radio_button_1 == "Gender":
    #         fig = donut_chart_beneficiary(df,'Gender')
    #         st.plotly_chart(fig, use_container_width=False, config= dict(displayModeBar = False))

            
    #     if selected_radio_button_1 == "Marital Status":
    #         fig = donut_chart_beneficiary(df,'MaritalStatus')
    #         st.plotly_chart(fig, use_container_width=False, config= dict(displayModeBar = False))


    #     if selected_radio_button_1 == "Dependency":
    #         fig = donut_chart_beneficiary(df,'Dependency')
    #         st.plotly_chart(fig, use_container_width=False, config= dict(displayModeBar = False))


    #     if selected_radio_button_1 == "Family of Causes":
    #         df_foc_gender = df.loc[:,["FamilyOfCauses","Gender"]].value_counts().to_frame().reset_index()[0:7]
    #         df_foc_gender.columns = ['Family of Causes','Gender','Claim Count']
    #         fig = px.bar(df_foc_gender, x="Family of Causes", y="Claim Count", 
    #                     color="Gender", barmode="group")
    #         fig.update_layout(
    #                 margin=dict(l=0,r=0,t=5,b=0),
    #                 width = 600,
    #                 height = 380,
    #                 # bargap=0.2,
    #                 )
    #         st.plotly_chart(fig, use_container_width=False, config= dict(displayModeBar = False))


    #     if selected_radio_button_1 == "Family of Benefits":
    #         df_foc_gender = df.loc[:,["FamilyOfBenefits","Gender"]].value_counts().to_frame().reset_index()[0:7]
    #         df_foc_gender.columns = ['Family of Benefits','Gender','Claim Count']
    #         fig = px.bar(df_foc_gender, x="Family of Benefits", y="Claim Count", 
    #                     color="Gender", barmode="group")
    #         fig.update_layout(
    #                 margin=dict(l=0,r=0,t=5,b=0),
    #                 width = 600,
    #                 height = 380,
    #                 # bargap=0.2,
    #                 )
    #         st.plotly_chart(fig, use_container_width=False, config= dict(displayModeBar = False))


    # ########################################################################
    # # Provider Summary
    # st.write('--------------------------------')
    # st.write('#### Provider Stats')
    # selected_radio_button_2 = st.radio("Select Column", options = ["Doctor Type","Doctor Specialty","Doctor Name","Provider Type", "Provider Name",], horizontal=True)
    # col1, col2, col3 = st.columns([1,3,1])
    # with col1:
    #     st.metric(label = 'Unique Providers', value = df['Provider'].nunique())
    #     st.metric(label = 'Unique Doctors', value = df['ProfID'].nunique())
    #     st.metric(label = 'Average Claims Count Per Privider', value = int(np.mean(df.groupby('Provider')['CalimKey'].nunique())))
    #     st.metric(label = 'Average Claims Amount Per Privider', value = int(np.mean(df.groupby('Provider')['TotalAmountClaimed'].sum())))

    # with col2:
    #     if selected_radio_button_2 == 'Provider Type':
    #         fig = donut_chart_general(df,'ProviderType')
    #         st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))

    #     if selected_radio_button_2 == 'Provider Name':
    #         fig = horizonal_bar_top(df,'Provider','Claim Item Count','Provider')
    #         st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))

    #     if selected_radio_button_2 == 'Doctor Type':
    #         fig = donut_chart_general(df,'ProfessionalType')
    #         st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))

    #     if selected_radio_button_2 == 'Doctor Specialty':
    #         fig = horizonal_bar_top(df[df['ProfessionalSpecialty']!="NONE"],'ProfessionalSpecialty','Claim Item Count','Doctor Specialty')
    #         st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))

    #     if selected_radio_button_2 == 'Doctor Name':
    #         fig = horizonal_bar_top(df[df['ProfName']!="NONE"],'ProfName','Claim Item Count','Doctor Name')
    #         st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))




    # ########################################################################
    # # Insurance Summary
    # st.write('--------------------------------')
    # st.write('#### Insurance Stats')
    # selected_radio_button_3 = st.radio("Select Column", options = ["Payer","Contract", "Policy Type", "Product Class", "Product"], horizontal=True)

    # # get average payer shares
    # df['payer_share'] = df['PayerShare']/df['InvoicedAmount']
    # df['payer_share'].replace([np.inf, -np.inf], np.nan, inplace=True)
    # df['payer_share'] = df['payer_share'].fillna(0)
    # col1, col2, col3 = st.columns([1,3,1])
    # with col1:
    #     st.metric(label = 'Unique Payers', value = df['Payer'].nunique())
    #     st.metric(label = 'Unique Contracts', value = df['Contract'].nunique())
    #     st.metric(label = 'Unique Products', value = df['Product'].nunique())
    #     st.metric(label = 'Average Payer Share %', value = int(100*(df['payer_share'].mean())))

    # with col2:
    #     if selected_radio_button_3 == 'Payer':
    #         fig = donut_chart_general(df,'Payer')
    #         st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))


    # with col2:
    #     if selected_radio_button_3 == 'Contract':
    #         fig = horizonal_bar_top(df[df['Contract']!="NONE"],'Contract','Claim Item Count','Contract')
    #         st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))

    # with col2:
    #     if selected_radio_button_3 == 'Policy Type':
    #         fig = donut_chart_general(df,'PolicyType')
    #         st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))


    # with col2:
    #     if selected_radio_button_3 == 'Product Class':
    #         fig = donut_chart_general(df,'ProductClass')
    #         st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))


    # with col2:
    #     if selected_radio_button_3 == 'Product':
    #         fig = horizonal_bar_top(df[df['Product']!="NONE"],'Product','Claim Item Count','Product')
    #         st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))



    # ########################################################################
    # # Claim Summary
    # st.write('--------------------------------')
    # st.write('#### Claim Stats')
    # selected_radio_button_4 = st.radio("Select Column", options = [
    #     "Service","Item", "SpecAssessment","GeneralAssessment","Assessment","Unit Price","Quantity","Total Amount"], horizontal=True)

    # col1, col2, col3 = st.columns([1,3,1])
    # with col1:
    #     st.metric(label = 'Unique Services', value = df['ServiceID'].nunique())
    #     st.metric(label = 'Unique Service Items', value = df['ItemCode'].nunique())
    #     st.metric(label = 'Unique Assessment', value = df['SpecAssessmentCode'].nunique())
    #     st.metric(label = 'Average Service Items Per Claim', value = int(round(df.groupby('CalimKey')['ItemCode'].size().mean(),0)))


    # with col2:
    #     if selected_radio_button_4 == 'Service':
    #         fig = horizonal_bar_top(df[df['Service']!="NONE"],'Service','Claim Item Count','Service')
    #         st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))

    #     if selected_radio_button_4 == 'Item':
    #         fig = horizonal_bar_top(df[df['ItemName']!="NONE"],'ItemName','Claim Item Count','Service Item')
    #         st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))

    #     if selected_radio_button_4 == 'SpecAssessment':
    #         fig = horizonal_bar_top(df[df['SpecAssessment']!="NONE"],'SpecAssessment','Claim Item Count','SpecAssessment')
    #         st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))

    #     if selected_radio_button_4 == 'GeneralAssessment':
    #         fig = horizonal_bar_top(df[df['GeneralAssessment']!="NONE"],'GeneralAssessment','Claim Item Count','GeneralAssessment')
    #         st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))

    #     if selected_radio_button_4 == 'Assessment':
    #         fig = horizonal_bar_top(df[df['Assessment']!="NONE"],'Assessment','Claim Item Count','Assessment')
    #         st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))

    #     if selected_radio_button_4 == 'Unit Price':
    #         labels = df['UnitPriceClaimed'].clip(0,1000)
    #         fig = px.histogram(labels, nbins=200)
    #         fig.update_traces(marker_color='#4BA38C')            
    #         fig.update_layout(
    #                 margin=dict(l=0,r=0,t=5,b=0),
    #                 width = 600,
    #                 height = 380,
    #                 bargap= 0.2,
    #                 showlegend=False,
    #                 xaxis_title="Unit Price", 
    #                 yaxis_title="Claim Count",      
    #                 )
    #         st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))


    #     if selected_radio_button_4 == 'Quantity':
    #         labels = df['QuantityClaimed'].clip(0,100)
    #         fig = px.histogram(labels, nbins=200)
    #         fig.update_traces(marker_color='#4BA38C')            
    #         fig.update_layout(
    #                 margin=dict(l=0,r=0,t=5,b=0),
    #                 width = 600,
    #                 height = 380,
    #                 bargap= 0.2,
    #                 showlegend=False,
    #                 xaxis_title="Quantity", 
    #                 yaxis_title="Claim Count",      
    #                 )
    #         st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))

    #     if selected_radio_button_4 == 'Total Amount':
    #         labels = df['TotalAmountClaimed'].clip(0,1000)
    #         fig = px.histogram(labels, nbins=200)
    #         fig.update_traces(marker_color='#4BA38C')            
    #         fig.update_layout(
    #                 margin=dict(l=0,r=0,t=5,b=0),
    #                 width = 600,
    #                 height = 380,
    #                 bargap= 0.2,
    #                 showlegend=False,
    #                 xaxis_title="Total Amount", 
    #                 yaxis_title="Claim Count",      
    #                 )
    #         st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))















    # ########################################################################
    # # model performance
    # st.write("## BPR Model Results")
    # st.write('--------------------------------')
    # st.write("#### Model Performance")
    # st.write("The entire dataset is split into **70% train** and **30% test** by time. We also filtered the data by more than 10 items per user, and more than 10 users per item in order to be more statistically robust. We used **Beyesian Personal Ranking (BPR)** model that give the best backtesting results with popular metrics such as **MAP, nDCG, Precision and Recall**.")

    # for col in df_model_performance.columns[3:]:
    #     df_model_performance[col] = np.round(df_model_performance[col],2)

    # fig =  ff.create_table(df_model_performance, height_constant=30)
    # fig.update_layout(width=1000)
    # st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))





    # ########################################################################
    # # pre-processing data
    
    # # hit count
    # df_hit_count = df_test_stats.groupby('hit').size().to_frame().reset_index().rename(columns={0:'total_hit'})
    # df_hit_count['hit_of_5'] = df_hit_count['hit'].astype(str) + '/5'

    # # get the rank hit, where top 2 recommendation overlap with top 2 buy
    # df_hit_count['rank_hit'] = df_test_stats.groupby('hit')['is_correct'].sum()

    # # get the cumulative count and percentage
    # total_cusmomer_count = df_test_stats.shape[0]
    # cumulative_sum_list = []
    # cumulative_sum = 0
    # for i in range(df_hit_count.shape[0]-1,-1,-1):
    #     cumulative_sum += df_hit_count.loc[i,'total_hit']
    #     cumulative_sum_list.append(cumulative_sum)
    # cumulative_sum_list.reverse()
    # df_hit_count['cumulative_count'] = cumulative_sum_list
    # df_hit_count['percentage'] = df_hit_count['cumulative_count'] / total_cusmomer_count

    # df_hit_count.sort_values(by='hit',ascending=False,inplace=True)
    # # df_hit_count = df_hit_count[['hit_of_5','total_hit','rank_hit','cumulative_count','percentage']]
    # df_hit_count.columns = ['Hit Count','Total Hit','Hit of 5','Rank Hit','Cumulative Count','Cumulative Percent']

    # ########################################################################
    # # summary table

    # st.write('--------------------------------')
    # st.write("#### Summary Results")
    # st.write("Summary table of the back testing results of Bayesian Personalized Ranking (BPR) based on top 5 recommended varietals for each customer")

    # df_hit_count['Cumulative Percent'] = np.round(df_hit_count['Cumulative Percent'],3)

    # fig =  ff.create_table(df_hit_count, height_constant=30)
    # fig.update_layout(width=1000)
    # st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))

    # ########################################################################
    # # bar chart hit rate

    # # use two cols to place the charts
    # st.write('--------------------------------')
    # col1, col2 = st.columns(2)



    # fig = go.Figure(data=[
    #     go.Bar(
    #         name='Total Hit', 
    #         x=df_hit_count['Hit of 5'], 
    #         y=df_hit_count['Total Hit'],
    #         marker_color = '#24477F'
    #         ),
    #     go.Bar(
    #         name='Rank Hit', 
    #         x=df_hit_count['Hit of 5'], 
    #         y=df_hit_count['Rank Hit'],
    #         marker_color = "#D9C666"
    #     )
    # ])

    # # Change the bar mode
    # fig.update_layout(
    #     barmode='group', 
    #     width=1000, 
    #     height=450, 
    #     # title='Count of Hit',
    #     margin=dict(l=0,r=0,t=50,b=0),
    #     xaxis=dict(
    #         # autorange="reversed",
    #         # title="this is x label"
    #         ),
    #     yaxis=dict(
    #         # title="this is y label"
    #         ),
    #     legend=dict(
    #         yanchor="top",
    #         y=-0.15,
    #         xanchor="center",
    #         x=0.5,
    #         orientation="h",
    #         ),
    #     # config=dict(displayModeBar=False)
    # )
    

    # with col1:
    #     st.write("#### Count of Hit")
    #     st.write("Bar chart comparing the number of total hit vs. number of rank hit")
    # # with st.expander("", True):
    #     st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))

    
    # ########################################################################
    # # line chart cumulative

    # df_hit_count.sort_values(by='Hit Count',ascending=True,inplace=True)

    # fig = go.Figure(
    # data=go.Scatter(
    #     x=df_hit_count['Hit of 5'], 
    #     y=np.round(df_hit_count['Cumulative Percent']*100,1),
    #     mode="lines+markers+text",
    #     line=dict(
    #         color='#24477F', 
    #         width=2),
    #     text = df_hit_count['Cumulative Percent']*100,
    #     textposition="top left",
    #     texttemplate="%{y}"
    #     ))

    # fig.update_layout(
    #     width=1000, 
    #     height=400, 
    #     # title='Cumulative Percentage of Hit', 
    #     margin=dict(l=0,r=0,t=50,b=0),
    #     xaxis=dict(
    #         autorange="reversed",
    #         showline=True,
    #         showgrid=False,
    #         showticklabels=True,
    #         # title="this is y label"
    #         ),

    #     yaxis=dict(
    #         showgrid=True,
    #         zeroline=False,
    #         showline=True,
    #         showticklabels=False,
    #         # title="this is y label"
    #         ), 
    #     legend=dict(
    #         yanchor="top",
    #         y=-0.15,
    #         xanchor="center",
    #         x=0.5,
    #         orientation="h",
    #         ),
    #     plot_bgcolor='white'    
    # )
    

    # with col2:
    #     st.write("#### Cumulative Percentage")
    #     st.write("Line chart showing the cumulative hit counts for entire population")
    #     st.plotly_chart(fig, use_container_width=True, config= dict(displayModeBar = False))
        
    
    

    # ########################################################################
    # # Backtest Results Check Scatter Plot
    # st.write('--------------------------------')
    # st.write("#### Back Testing Check")
    # st.write('''
    # Plot the average precision vs. hit rate (out of 5), each data point represent one user, you can select the data point and check the back tested results (train vs. test vs. predict) \n
    # **Select the data by clicking on the data points** 
    # ''')
    # df_test_stats['is_correct'] = df_test_stats['is_correct'].astype(str)
    # fig = px.scatter(
    #     df_test_stats, 
    #     x="hit", 
    #     y="average_precision",
    #     color="is_correct",
    #     color_discrete_map = {'0':'red','1':'green'},
    #     hover_data=["user_id"]
    #     )

    # # fig.update_layout(
    # #     width=1000, 
    # #     height=1000, 
    # #     margin=dict(l=0,r=0,t=0,b=0),
    # # )
    
    # ########################################################################


    # # load extra data for QC
    # @st.cache(allow_output_mutation=True)
    # def load_qc_data():
    #     df_train_final = pd.read_csv("data/train_final_simple.csv")
    #     df_test_final = pd.read_csv("data/test_final_simple.csv")
    #     df_pred_final = pd.read_csv("data/df_final_results_simple.csv")
    #     return df_train_final, df_test_final, df_pred_final
    # df_train_final, df_test_final,df_pred_final = load_qc_data()


    # # generate the compositie index for the selected plotly event
    # df_test_stats['qc_index'] = df_test_stats['hit'].astype(str) + "_" + df_test_stats['average_precision'].astype(str)

    # # st.write(df_test_stats)

    # try:
    #     # plot the figure with click events on
    #     selected_points = plotly_events(fig)
    #     selected_point = selected_points[0]

    #     # generate the unique values from the selection for filter
    #     selected_index = str(selected_point['x']) + "_" + str(selected_point['y'])
    #     df_selected_index = df_test_stats.loc[df_test_stats['qc_index']==selected_index,:]

    #     # generate the user_id selected for plotting 
    #     user_id_list = df_selected_index['user_id'].unique().tolist()[0:1]
        
    #     # final plot
    #     col1, col2, col3 = st.columns(3)
    #     with col1:
    #         st.write("**Train Data**")
    #         df_train_final_filtered = df_train_final[df_train_final['user_id'].isin(user_id_list)]
    #         df_train_final_filtered.sort_values(by='purchase_frequency',ascending=False, inplace=True)
    #         df_train_final_filtered.set_index('user_id', inplace=True)
    #         st.dataframe(df_train_final_filtered)


    #     with col2:
    #         st.write("**Test Data**")
    #         df_test_final_filtered = df_test_final[df_test_final['user_id'].isin(user_id_list)]
    #         df_test_final_filtered.sort_values(by='purchase_frequency',ascending=False,inplace=True)
    #         df_test_final_filtered.set_index('user_id', inplace=True)
    #         st.dataframe(df_test_final_filtered)

    #     with col3:
    #         st.write("**Top 5 Recommended**")
    #         df_pred_final_filtered = df_pred_final[df_pred_final['user_id'].isin(user_id_list)]
    #         df_pred_final_filtered.set_index('user_id', inplace=True)
    #         st.dataframe(df_pred_final_filtered)
    # except:
    #     pass