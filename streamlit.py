##Streamlit application: 
import streamlit as st
import pandas as pd
import numpy as np

import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="Thesis",
    page_icon="👋",
)

st.write("# Welcome to my thesis demo! 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    An interactive application that displays the results of my project and provides information on the tools used in it
    ### Want to learn more?
    - Check out [streamlit.io](https://streamlit.io)
    - Jump into our [documentation](https://docs.streamlit.io)
    - Ask a question in our [community
        forums](https://discuss.streamlit.io)
    ### See more complex demos
    - Use a neural net to [analyze the Udacity Self-driving Car Image
        Dataset](https://github.com/streamlit/demo-self-driving)
    - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
"""
)
with st.sidebar:
    selected = option_menu(
        menu_title = "Main Menu",
        options =["Home", "Info","Result"], 
    )
if selected == "Home":
    st.write("# Welcome to my thesis demo! 👋")
if selected == "Info":
    st.markdown(
    """
    An interactive application that displays the results of my project and provides information on the tools used in it
    ### Want to learn more?
    - Check out [streamlit.io](https://streamlit.io)
    - Jump into our [documentation](https://docs.streamlit.io)
    - Ask a question in our [community
        forums](https://discuss.streamlit.io)
    ### See more complex demos
    - Use a neural net to [analyze the Udacity Self-driving Car Image
        Dataset](https://github.com/streamlit/demo-self-driving)
    - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
"""
)
    
