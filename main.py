

from xml.sax.handler import feature_external_ges
import yaml
from yaml import SafeLoader

import streamlit as st
import streamlit_authenticator as stauth
from  streamlit_authenticator import Authenticate

import hydralit_components as hc
from collections import OrderedDict

from mypages import land_score
from mypages import back_testing


def show_page(params):

    st.set_page_config(page_title="LandFold App", layout="wide")

    hide_streamlit_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibllity: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)


    reduce_header_height_style = """
        <style>
            div.block-container {padding-top:0rem;}
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            footer {visibility: hidden;}
            footer:after {
                content:'Copyright @ 2022'; 
                visibility: visible;
                display: block;
                position: relative;
                #background-color: red;
                padding: 5px;
                top: 2px;
            }
        </style>
        """
    st.markdown(reduce_header_height_style, unsafe_allow_html=True)

    # logo
    st.image("img/logo.png")

    # main menu
    main_menu_items = OrderedDict()
    main_menu_items["Land Score"] = land_score
    main_menu_items["Back Testing"] = back_testing

    # main menu color
    over_theme = {'menu_background': '#002A31'}

    # main menu icon
    chart_menu = []
    for item_name in main_menu_items:
        chart_menu.append({'icon': "far fa-chart-bar", 'label': item_name})

    # set the menu
    menu_id = hc.nav_bar(
        key="TopMenu",
        menu_definition=chart_menu,
        home_name=None,
        hide_streamlit_markers=True,
        override_theme=over_theme
    )

    # show page
    selected_page = main_menu_items[menu_id]
    selected_page.show_page({})



if __name__ == "__main__":
    show_page({})

        



























# menu_item = [
#     "01. Basic EDA", 
#     "02. Feature Engineering", 
#     "03. Machine Learning",
#     "04. Anomaly Detection",
#     "05. Clustering Analysis",
#     # "05. Cohort Analysis",
#     # "06. Buyer Segments",
#     # "07. Seller Segments",
#     # "08. Churn Deploy",
#     # "09. Demand Forecasting",
#     # "10. Similar Items"
#     ]

# with st.sidebar:

#     # put the logo in the middle
#     col1, col2, col3 = st.columns([1,5,1])
#     with col1:
#         st.write(' ')
#     with col2:
#             st.image("img/logo.png")
#     with col3:
#         st.write(' ')


#     selected = option_menu(
#                 menu_title="Nextcare Claim Screening",
#                 options=menu_item,
#                 icons=["arrow-return-right","arrow-return-right","arrow-return-right","arrow-return-right","arrow-return-right","arrow-return-right","arrow-return-right","arrow-return-right","arrow-return-right","arrow-return-right"],
#                 menu_icon="bar-chart-line-fill",
#                 default_index=0, #optional
#                 orientation="vertical",
#                 styles={"nav-link":{"font-size": "12px"}, "menu-title":{"font-size": "16px"}}
#                 )

# if selected==menu_item[0]:
#     page1.page1()
# if selected==menu_item[1]:
#     page2.page2()
# if selected==menu_item[2]:
#     page3.page3()
# # if selected==menu_item[3]:
# #     page4.page4()
# # if selected==menu_item[4]:
# #     page5.page5()
# # if selected==menu_item[5]:
# #     page6.page6()
# # if selected==menu_item[6]:
# #     page7.page7()
# # if selected==menu_item[7]:
# #     page8.page8()
# # if selected==menu_item[8]:
# #     page9.page9()
# # if selected==menu_item[9]:
# #     page10.page10()
