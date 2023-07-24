import streamlit as st
import neo4j
import time
from neo4j import GraphDatabase
from py2neo import Graph
import pandas as pd
from numpy.random import randint
from pyspark.sql.types import *
from pyspark.sql import functions as F
from sklearn.metrics import roc_curve, auc
from collections import Counter
from cycler import cycler
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import streamlit_authenticator as stauth
matplotlib.use('Agg')
import yaml
from yaml.loader import SafeLoader
@st.cache_resource(experimental_allow_widgets=True)
#---Neo4j driver connector---
class Neo4jConnection:
    
    def __init__(self, uri, user, pwd):
        self.__uri = uri
        self.__user = user
        self.__pwd = pwd
        self.__driver = None
        try:
            self.__driver = GraphDatabase.driver(self.__uri, auth=(self.__user, self.__pwd))
        except Exception as e:
            print("Failed to create the driver:", e)
        
    def close(self):
        if self.__driver is not None:
            self.__driver.close()
        
    def query(self, query, parameters=None, db=None):
        assert self.__driver is not None, "Driver not initialized!"
        session = None
        response = None
        try: 
            session = self.__driver.session(database=db) if db is not None else self.__driver.session() 
            response = list(session.run(query, parameters))
        except Exception as e:
            print("Query failed:", e)
        finally: 
            if session is not None:
                session.close()
        return response

def display_result():
        
        def stateful_button(*args, key=None, **kwargs):
                if key is None:
                    raise ValueError("Must pass key")

                if key not in st.session_state:
                    st.session_state[key] = False

                if st.button(*args, **kwargs):
                    st.session_state[key] = not st.session_state[key]

                return st.session_state[key]


        with open('D:\huy\streamlit\Authen.yaml') as file:
            config = yaml.load(file,Loader=SafeLoader)
        
        authenticator = stauth.Authenticate(
            config['credentials'],
            config['cookie']['name'],
            config['cookie']['key'],
            config['cookie']['expiry_days'],
            config['preauthorized']
        )

    
        name, authenticator_status, username = authenticator.login('Login', 'main')

        if authenticator_status == False:
            warn_error = st.error('Username/password is incorrect')
            time.sleep(1)
            warn_error.empty()

        if authenticator_status == None:
            warn_missing = st.warning('Please enter your username and password')
            time.sleep(3)
            warn_missing.empty()

        if authenticator_status:
            progress_text = "Operation in progress. Please wait."
            my_bar = st.progress(0, text=progress_text)
            for percent_complete in range(100):
                time.sleep(0.05)
                my_bar.progress(percent_complete + 1, text=progress_text)
            my_bar.empty()
            authenticator.logout("Log out","sidebar")
            st.title(f'**Welcome {name}**')
            con = Neo4jConnection(uri="bolt://localhost:7687",user="huy", pwd="12345huy@")
            tab1,tab2,tab3= st.tabs(["Data preprocessing","Machine learning prepare","Evaluation"])
            with tab1:

                dashboard_url = "http://neodash.graphapp.io/..."

                with st.expander("**Interactive graph**"):
                    st.write("NeoDash Dashboard")
                    
                    css = """
                    iframe {
                         width: 100%;
                         height: calc(100vh - 150px);  
                    }   
                    """
                    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
                         
                    iframe_html = f"""
                         <iframe   
                              src="{dashboard_url}"        
                              width="100%"       
                              scrolling="yes"         
                              frameborder="0">
                         </iframe> 
                    """
                         
                    st.write(iframe_html,unsafe_allow_html=True)
                                        
                with st.expander("**Information about the database**"):
                    st.markdown(":blue[Total number of Authors and relationships in the database:]")
                    query_rela='''
                        MATCH (n:Author) 
                        WITH count(n) as count
                        RETURN "Authors" as label,count
                        UNION ALL
                        MATCH ()-[r:CO_AUTHOR_EARLY]->() 
                        WITH count(r) as count
                        RETURN "Co-author relationships" as label, count
                        '''
                    result_rela = con.query(query_rela, db='mltest')
                    df1= pd.DataFrame(result_rela)
                    df1.set_axis(['Type','Count'],axis='columns', inplace=True)
                    st.write(df1)
                    
                    st.markdown(":orange[Total articles published by year]")
                    query = """
                            MATCH (article:Article)
                            RETURN article.year AS year, count(*) AS count
                            ORDER BY year
                            """
                    by_year = con.query(query, db='mltest')
                    df = pd.DataFrame(by_year)
                    df.set_axis(['YEAR', 'COUNT'], axis='columns', inplace=True)
                    st.bar_chart(df, x='YEAR',y='COUNT', height=700)
                st.subheader(":blue[Create training and testing dataset]")
                st.markdown("**Notation:** a category is represented by the label column, where a value of 1 shows a link between two nodes and a value of 0 represents an absence of a link.")
                col1,col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Train dataset**")
                    train_data = pd.read_csv(r'D:\huy\streamlit\data\train_data_graph.csv')
                    st.write(train_data)
                    if stateful_button("Down sample train data", key='down_train_data'):
                        st.write("**The content of train dataset after down sampling**")
                        st.write(train_data.groupby(["label"])["label"].count())
                with col2:
                    st.markdown("**Test dataset**")
                    test_data = pd.read_csv(r'D:\huy\streamlit\data\test_data_graph.csv')
                    st.write(test_data)
                    if stateful_button("Down sample test data",key="down_test_data"):
                        st.write("**The content of test dataset after down sampling**")
                        st.write(test_data.groupby(["label"])["label"].count())
            with tab2: 
                st.subheader(":red[Adding features to the datasets]")
                st.write(":green[**First model**]")
                cl1,cl2 = st.columns(2)
                with cl1:
                    if stateful_button('Create trainning set of model 1', key="train_data"):
                        st.write("Train dataset")
                        data_train_model1 = pd.read_csv(r'D:\huy\streamlit\data\train_data_basic_model.csv')
                        st.write(data_train_model1.loc[:,'node1':'commonAuthor'])
                with cl2:
                    if stateful_button("Create testing set of model 1", key="test_data"):
                        st.write("Test dataset")
                        data_test_model1 = pd.read_csv(r'D:\huy\streamlit\data\test_data_basicmodel.csv')
                        st.write(data_test_model1.loc[:,'node1':'commonAuthor'])
                st.write("---")
                st.write(":violet[**Adding graphy features to the model**]")
                col_graph_1, col_graph_2 = st.columns(2)
                with col_graph_1:
                    if stateful_button('Create trainning set of graphy feature ', key="train_graph_data"):
                        st.write("Train dataset")
                        data_train_model2 = pd.read_csv(r'D:\huy\streamlit\data\train_data_basic_model.csv')
                        st.write(data_train_model2.loc[:,'node1':'TotalNeighbors'])
                with col_graph_2:
                    if stateful_button("Create testing set of graphy feature", key="test_graph_data"):
                        st.write("Test dataset")
                        data_test_model2 = pd.read_csv(r'D:\huy\streamlit\data\test_data_basicmodel.csv')
                        st.write(data_test_model2.loc[:,'node1':'TotalNeighbors'])
                st.write("---")
                st.write(":blue[**Ađing community feature to the model**]")
                col_commu_1, col_commu_2 = st.columns(2)
                with col_commu_1:
                    if stateful_button('Create trainning set of community feature ', key="train_commu_data"):
                        st.write("Train dataset")
                        data_train_model3 = pd.read_csv(r'D:\huy\streamlit\data\train_data_commu.csv')
                        st.write(data_train_model3)
                with col_commu_2:
                    if stateful_button("Create testing set of community feature ", key="test_commu_data"):
                        st.write("Test dataset")
                        data_test_model3 = pd.read_csv(r'D:\huy\streamlit\data\test_data_commu.csv')
                        st.write(data_test_model3)
            with tab3:
                def create_roc_plot():
                    plt.style.use('classic')
                    fig = plt.figure(figsize=(14,9))
                    plt.xlim([0,1])
                    plt.ylim([0,1])
                    plt.ylabel('True positive rate')
                    plt.xlabel('False positive rate')
                    plt.rc('axes', prop_cycle=(cycler('color', ['r','b',"c","m","y","k"])))
                    plt.plot([0,1],[0,1], linestyle='--', label = 'Random score (AUC =0.50)')
                    return fig
                
                def add_curve(plt, title, fpr, tpr, roc):   
                    plt.plot(fpr, tpr, label=f"{title} (AUC={roc.iloc[0]:.2f})")
                
                with st.container():
                    st.write("**Initial model**")
                    if stateful_button('Evaluate first model result', key="first_model_evaluate"):
                        first_model_col1, first_model_col2 = st.columns(2)
                        with first_model_col1:
                            first_model = pd.read_csv(r'D:\huy\streamlit\data\first_model_graph.csv')
                            fig = create_roc_plot()
                            add_curve(plt, "Common Authors", first_model["fpr"], first_model["tpr"],first_model["roc_auc"])
                            plt.legend(loc='lower right')
                            st.pyplot(fig)
                            st.caption("_ROC CURVE chart of the first model_")
                        with first_model_col2:
                            first_model_result = pd.read_csv(r'D:\huy\streamlit\data\first_model_result.csv')
                            st.write(first_model_result)
                            st.caption("_Evaluation scores of the initial model_")
                with st.container():
                    st.write("**Model after adding graphy features**")
                    if stateful_button('Evaluate model with graphy features result', key="second_model_evaluate"):
                        second_model_col1, second_model_col2 = st.columns(2)
                        with second_model_col1:
                            graphy_model = pd.read_csv(r'D:\huy\streamlit\data\graphy_model_graph.csv')
                            fig = create_roc_plot()
                            add_curve(plt, "Common Authors", first_model["fpr"], first_model["tpr"],first_model["roc_auc"])
                            add_curve(plt, "Graphy",graphy_model["fpr"], graphy_model["tpr"],graphy_model["roc_auc"])
                            plt.legend(loc='lower right')
                            st.pyplot(fig)
                            st.caption("_ROC CURVE chart of the model after adding graphy feature_")
                        with second_model_col2:
                            graphy_model_result = pd.read_csv(r'D:\huy\streamlit\data\graphy_model_result.csv')
                            st.write(graphy_model_result)
                            st.caption("_Evaluation scores of the model after adding graphy feature _")
                with st.container():
                    st.write("**Model with community feature**")
                    if stateful_button('Evaluate model after adding community result', key="third_model_evaluate"):
                        third_model_col1, third_model_col2 = st.columns(2)
                        with third_model_col1:
                            community_louvain_model = pd.read_csv(r'D:\huy\streamlit\data\community_model_louvain_graph.csv')
                            community_fuzzy_model = pd.read_csv(r'D:\huy\streamlit\data\community_model_fuzzy_graph.csv')
                            fig = create_roc_plot()
                            add_curve(plt, "Common Authors", first_model["fpr"], first_model["tpr"],first_model["roc_auc"])
                            add_curve(plt, "Graphy",graphy_model["fpr"], graphy_model["tpr"],graphy_model["roc_auc"])
                            add_curve(plt, "Communities",community_louvain_model["fpr"], community_louvain_model["tpr"],community_louvain_model["roc_auc"])
                            add_curve(plt, "Fuzzy + Louvain",community_fuzzy_model["fpr"], community_fuzzy_model["tpr"],community_fuzzy_model["roc_auc"])
                            plt.legend(loc='lower right')
                            st.pyplot(fig)
                            st.caption("_ROC CURVE chart of the model after adding community feature_")
                        with third_model_col2:
                            graphy_model_result = pd.read_csv(r'D:\huy\streamlit\data\louvain_model_result_new.csv')
                            st.write(graphy_model_result)
                            st.caption("_Evaluation scores of the model after adding community feature_")
      


                    

        