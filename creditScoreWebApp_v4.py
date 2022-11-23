import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import pydeck as pdk
import plotly.figure_factory as ff
import matplotlib.pyplot as plt

#######Functions###############

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

@st.experimental_singleton()
def load_and_calculate():
    if st.session_state.selectCountry!='None':
        fnR = 'data/'+st.session_state.selectCountry+'/'+st.session_state.selectCountry+'_Remoteness.csv'
        fnP = 'data/' + st.session_state.selectCountry + '/' + st.session_state.selectCountry + '_Poverty.csv'
        fnI = 'data/' + st.session_state.selectCountry + '/' + st.session_state.selectCountry + '_Impact.csv'
        fnC = 'data/' + st.session_state.selectCountry + '/' + st.session_state.selectCountry + '_Connectivity.csv'
        #fnSC = 'data/' + st.session_state.selectCountry + '/' + st.session_state.selectCountry + '_School_Connectivity.csv'
        st.session_state.df_remoteness = pd.read_csv(fnR)
        st.session_state.df_poverty = pd.read_csv(fnP)
        st.session_state.df_impact = pd.read_csv(fnI)
        st.session_state.df_connectivity = pd.read_csv(fnC)
        #st.session_state.df_school_connectivity = pd.read_csv(fnSC)

        st.session_state.df_remoteness.drop(columns=st.session_state.df_remoteness.columns[0], axis=1, inplace=True)
        st.session_state.df_poverty.drop(columns=st.session_state.df_poverty.columns[0], axis=1, inplace=True)
        st.session_state.df_impact.drop(columns=st.session_state.df_impact.columns[0], axis=1, inplace=True)
        st.session_state.df_connectivity.drop(columns=st.session_state.df_connectivity.columns[0], axis=1, inplace=True)
        #st.session_state.df_school_connectivity.drop(columns=st.session_state.df_school_connectivity.columns[0], axis=1, inplace=True)

        copycols = list(st.session_state.df_remoteness.columns[:4])
        st.session_state.df_school_dup = st.session_state.df_remoteness[copycols].copy()

        #The first 3 columns are school id, lon,lat
        st.session_state.attributesR = list(st.session_state.df_remoteness.columns[4:])
        st.session_state.attributesP = list(st.session_state.df_poverty.columns[4:])
        st.session_state.attributesI = list(st.session_state.df_impact.columns[4:])
        st.session_state.attributesC = list(st.session_state.df_connectivity.columns[3:])

        st.session_state.df_remoteness['Remoteness_raw'] = st.session_state.df_remoteness[st.session_state.attributesR].sum(axis=1)
        st.session_state.df_poverty['Poverty_raw'] = st.session_state.df_poverty[st.session_state.attributesP].sum(axis=1)
        st.session_state.df_impact['Impact_raw'] = st.session_state.df_impact[st.session_state.attributesI].sum(axis=1)
        st.session_state.df_connectivity['Connectivity_raw'] = st.session_state.df_connectivity[st.session_state.attributesC].sum(axis=1)
        #Normalize
        minR = st.session_state.df_remoteness['Remoteness_raw'].describe()['min']
        maxR = st.session_state.df_remoteness['Remoteness_raw'].describe()['max'] - minR
        st.session_state.df_remoteness['Remoteness'] = st.session_state.df_remoteness.apply(lambda row: (row['Remoteness_raw'] - minR) / (maxR), axis=1)

        minP = st.session_state.df_poverty['Poverty_raw'].describe()['min']
        maxP = st.session_state.df_poverty['Poverty_raw'].describe()['max'] - minP
        st.session_state.df_poverty['Poverty'] = st.session_state.df_poverty.apply(lambda row: (row['Poverty_raw'] - minP) / (maxP), axis=1)

        minI = st.session_state.df_impact['Impact_raw'].describe()['min']
        maxI = st.session_state.df_impact['Impact_raw'].describe()['max'] - minI
        st.session_state.df_impact['Impact'] = st.session_state.df_impact.apply(lambda row: (row['Impact_raw'] - minI) / (maxI), axis=1)

        minC = st.session_state.df_connectivity['Connectivity_raw'].describe()['min']
        maxC = st.session_state.df_connectivity['Connectivity_raw'].describe()['max'] - minC
        st.session_state.df_connectivity['Connectivity'] = st.session_state.df_connectivity.apply(lambda row: (row['Connectivity_raw'] - minC) / (maxC), axis=1)

        st.session_state.df_school_dup['Remoteness'] = st.session_state.df_remoteness['Remoteness']
        st.session_state.df_school_dup['Poverty'] = st.session_state.df_poverty['Poverty']
        st.session_state.df_school_dup['Impact'] = st.session_state.df_impact['Impact']
        st.session_state.df_school_dup['Connectivity'] = st.session_state.df_connectivity['Connectivity']
        #st.session_state.df_school_dup['CreditScore'] = st.session_state.df_school_dup[['Remoteness','Poverty','Impact','Connectivity']].sum(axis=1)
        st.session_state.df_school_dup['CreditScore'] = st.session_state.df_school_dup[
                                                            'Remoteness'] * st.session_state.Wr + \
                                                        st.session_state.df_school_dup[
                                                            'Poverty'] * st.session_state.Wp + \
                                                        st.session_state.df_school_dup[
                                                            'Impact'] * st.session_state.Wi + \
                                                        st.session_state.df_school_dup[
                                                            'Connectivity'] * st.session_state.Wc

        st.session_state.df_remoteness = st.session_state.df_remoteness.sort_values(by=['Remoteness'], ascending=False)
        st.session_state.df_poverty = st.session_state.df_poverty.sort_values(by=['Poverty'], ascending=False)
        st.session_state.df_impact = st.session_state.df_impact.sort_values(by=['Impact'], ascending=False)
        st.session_state.df_connectivity = st.session_state.df_connectivity.sort_values(by=['Connectivity'], ascending=False)

        st.session_state.df_school_dup = st.session_state.df_school_dup.sort_values(by=['CreditScore'], ascending=False)

        #Initialize attribute weights
        for att in st.session_state.attributesR:
            if 'W'+att not in st.session_state:
                st.session_state['W'+att] = 1.0
        for att in st.session_state.attributesP:
            if 'W' + att not in st.session_state:
                st.session_state['W'+att] = 1.0
        for att in st.session_state.attributesI:
            if 'W' + att not in st.session_state:
                st.session_state['W'+att] = 1.0
        for att in st.session_state.attributesC:
            if 'W' + att not in st.session_state:
                st.session_state['W'+att] = 1.0

def recalculateScore():
    if not st.session_state.df_school_dup.empty:
        st.session_state.df_school_dup['CreditScore'] = st.session_state.df_school_dup['Remoteness'] * st.session_state.Wr + st.session_state.df_school_dup['Poverty'] * st.session_state.Wp + st.session_state.df_school_dup[
        'Impact'] * st.session_state.Wi + st.session_state.df_school_dup['Connectivity'] * st.session_state.Wc

        st.session_state.df_school_dup = st.session_state.df_school_dup.sort_values(by=['CreditScore'], ascending=False)

def recalculateScoreR():
    if not st.session_state.df_school_dup.empty and not st.session_state.df_remoteness.empty:
        for index,row in st.session_state.df_remoteness.iterrows():
            st.session_state.df_remoteness.loc[index,'Remoteness_raw'] = 0.0
            for att in st.session_state.attributesR:
                st.session_state.df_remoteness.loc[index,'Remoteness_raw'] += row[att]*st.session_state['W'+att]
        # Normalize
        minR = st.session_state.df_remoteness['Remoteness_raw'].describe()['min']
        maxR = st.session_state.df_remoteness['Remoteness_raw'].describe()['max'] - minR
        st.session_state.df_remoteness['Remoteness'] = st.session_state.df_remoteness.apply(
                    lambda row: (row['Remoteness_raw'] - minR) / (maxR), axis=1)

        st.session_state.df_remoteness = st.session_state.df_remoteness.sort_values(by=['Remoteness'], ascending=False)

        #Recalculate Credit Score
        for index,row in st.session_state.df_school_dup.iterrows():
            ids = row['giga_id_school']
            st.session_state.df_school_dup.loc[index,'Remoteness'] = float(st.session_state.df_remoteness.loc[st.session_state.df_remoteness['giga_id_school']==ids]['Remoteness'])

        recalculateScore()

def recalculateScoreP():
    if not st.session_state.df_school_dup.empty and not st.session_state.df_poverty.empty:
        for index,row in st.session_state.df_poverty.iterrows():
            st.session_state.df_poverty.loc[index,'Poverty_raw'] = 0.0
            for att in st.session_state.attributesP:
                st.session_state.df_poverty.loc[index,'Poverty_raw'] += row[att]*st.session_state['W'+att]
        # Normalize
        minP = st.session_state.df_poverty['Poverty_raw'].describe()['min']
        maxP = st.session_state.df_poverty['Poverty_raw'].describe()['max'] - minP
        st.session_state.df_poverty['Poverty'] = st.session_state.df_poverty.apply(
                    lambda row: (row['Poverty_raw'] - minP) / (maxP), axis=1)

        st.session_state.df_poverty = st.session_state.df_poverty.sort_values(by=['Poverty'], ascending=False)

        #Recalculate Credit Score
        for index,row in st.session_state.df_school_dup.iterrows():
            ids = row['giga_id_school']
            st.session_state.df_school_dup.loc[index,'Poverty'] = float(st.session_state.df_poverty.loc[st.session_state.df_poverty['giga_id_school']==ids]['Poverty'])

        recalculateScore()

def recalculateScoreI():
    if not st.session_state.df_school_dup.empty and not st.session_state.df_impact.empty:
        for index,row in st.session_state.df_impact.iterrows():
            st.session_state.df_impact.loc[index,'Impact_raw'] = 0.0
            for att in st.session_state.attributesI:
                st.session_state.df_impact.loc[index,'Impact_raw'] += row[att]*st.session_state['W'+att]
        # Normalize
        minI = st.session_state.df_impact['Impact_raw'].describe()['min']
        maxI = st.session_state.df_impact['Impact_raw'].describe()['max'] - minI
        st.session_state.df_impact['Impact'] = st.session_state.df_impact.apply(
                    lambda row: (row['Impact_raw'] - minI) / (maxI), axis=1)

        st.session_state.df_impact = st.session_state.df_impact.sort_values(by=['Impact'], ascending=False)

        #Recalculate Credit Score
        for index,row in st.session_state.df_school_dup.iterrows():
            ids = row['giga_id_school']
            st.session_state.df_school_dup.loc[index,'Impact'] = float(st.session_state.df_impact.loc[st.session_state.df_impact['giga_id_school']==ids]['Impact'])

        recalculateScore()

def recalculateScoreC():
    if not st.session_state.df_school_dup.empty and not st.session_state.df_connectivity.empty:
        for index,row in st.session_state.df_connectivity.iterrows():
            st.session_state.df_connectivity.loc[index,'Connectivity_raw'] = 0.0
            for att in st.session_state.attributesC:
                st.session_state.df_connectivity.loc[index,'Connectivity_raw'] += row[att]*st.session_state['W'+att]
        # Normalize
        minC = st.session_state.df_connectivity['Connectivity_raw'].describe()['min']
        maxC = st.session_state.df_connectivity['Connectivity_raw'].describe()['max'] - minC
        st.session_state.df_connectivity['Connectivity'] = st.session_state.df_connectivity.apply(
                    lambda row: (row['Connectivity_raw'] - minC) / (maxC), axis=1)

        st.session_state.df_connectivity = st.session_state.df_connectivity.sort_values(by=['Connectivity'], ascending=False)

        #Recalculate Credit Score
        for index,row in st.session_state.df_school_dup.iterrows():
            ids = row['giga_id_school']
            st.session_state.df_school_dup.loc[index,'Connectivity'] = float(st.session_state.df_connectivity.loc[st.session_state.df_connectivity['giga_id_school']==ids]['Connectivity'])

        recalculateScore()

###############################


#######Title and setup#########
st.set_page_config(layout="wide")
st.title("Credit Score App")
###############################

#######State variables#########
if 'df_school_dup' not in st.session_state:
    st.session_state.df_school_dup = pd.DataFrame()
if 'df_remoteness' not in st.session_state:
    st.session_state.df_remoteness = pd.DataFrame()
if 'df_poverty' not in st.session_state:
    st.session_state.df_poverty = pd.DataFrame()
if 'df_impact' not in st.session_state:
    st.session_state.df_impact = pd.DataFrame()
if 'df_connectivity' not in st.session_state:
    st.session_state.df_connectivity = pd.DataFrame()
#if 'df_school_connectivity' not in st.session_state:
#    st.session_state.df_school_connectivity = pd.DataFrame()
if 'selectCountry' not in st.session_state:
    st.session_state.selectCountry = 'None'
if 'attributesR' not in st.session_state:
    st.session_state.attributesR = []
if 'attributesP' not in st.session_state:
    st.session_state.attributesP = []
if 'attributesI' not in st.session_state:
    st.session_state.attributesI = []
if 'attributesC' not in st.session_state:
    st.session_state.attributesC = []
###############################

#######Side bar Categories#########
with st.sidebar:
    with st.expander("Category Weights",expanded=True):
        slider_Wr = st.slider('Remoteness weight', 0.0, 1.0, 1.0, key='Wr', on_change = recalculateScore(),help='Remoteness is a proxy for cost/difficulty of the connection')
        slider_Wp = st.slider('Poverty weight', 0.0, 1.0, 1.0, key='Wp', on_change = recalculateScore(), help='Poverty or children poverty indicators')
        slider_Wi = st.slider('Impact weight', 0.0, 1.0, 1.0, key='Wi', on_change = recalculateScore(), help='Impact of connectivity in the region around the schools')
        slider_Wc = st.slider('Connectivity weight', 0.0, 1.0, 1.0, key='Wc', on_change = recalculateScore(), help='Connectivity status around the school and in the country')
###############################

######Country selectBox#############
country = st.selectbox(
    'Choose a Country',
    ('None','Mongolia', 'Botswana', 'Nigeria'), key='selectCountry')#,on_change=load_and_calculate())
###############################

######All these only happen if I select a country#########
if country and country!='None':
    load_and_calculate()

    #########Side Bar Attributes###########
    with st.sidebar:
        with st.expander("Attribute weights:"):
            st.text('Remoteness')
            for att in st.session_state.attributesR:
                st.slider(att + ' weight', 0.0, 1.0, 1.0, key='W' + att, on_change=recalculateScoreR())
            st.text('Poverty')
            for att in st.session_state.attributesP:
                st.slider(att + ' weight', 0.0, 1.0, 1.0, key='W' + att, on_change=recalculateScoreP())
            st.text('Impact')
            for att in st.session_state.attributesI:
                st.slider(att + ' weight', 0.0, 1.0, 1.0, key='W' + att, on_change=recalculateScoreI())
            st.text('Connectivity')
            for att in st.session_state.attributesC:
                st.slider(att + ' weight', 0.0, 1.0, 1.0, key='W' + att, on_change=recalculateScoreC())

    ############################


    ######2 Columns#######
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Ranked Data')
        st.dataframe(st.session_state.df_school_dup)

    with col2:
        st.subheader('Credit Score Map')
        tooltip = {
            "html": "School: {giga_id_school}</br> Credit Score: {CreditScore} </br>"
        }
        st.pydeck_chart(pdk.Deck(
            tooltip=tooltip,
            map_provider="mapbox",
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=pdk.ViewState(
                latitude=np.mean(st.session_state.df_school_dup['latitude']),
                longitude=np.mean(st.session_state.df_school_dup['longitude']),
                zoom=5,
                pitch=50,
            ),
            layers=[
                pdk.Layer(
                    'GridCellLayer',
                    data=st.session_state.df_school_dup,
                    get_position=['longitude', 'latitude'],
                    cellSize=10000,
                    elevationScale=50000,
                    get_fill_color=['255', '0', "bin_internet_availability > 0 ? 255 : 10"],
                    #get_fill_color=[255, 140, 100*'bin_internet_availability', 140],
                    getElevation='CreditScore',
                    pickable=True,
                    auto_highlight=True,
                ),
            ],
        ))
    ###################

    #######For downloading#########
    csv = convert_df(st.session_state.df_school_dup)

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name=st.session_state.selectCountry+'_CreditsScores.csv',
        mime='text/csv',
    )
    ###################




    #########Exploring the Category Scores##################
    st.header('Explore the category scores:')

    option = st.selectbox(
        'Choose a Score',
        ( 'None','Remoteness', 'Poverty', 'Impact', 'Connectivity'))

    if option and option != 'None':
        colA, colB = st.columns(2)
        with colA:
            st.subheader('Ranked Category Score distributions')
            if option=='Remoteness':
                st.dataframe(st.session_state.df_remoteness)
            elif option=='Poverty':
                st.dataframe(st.session_state.df_poverty)
            elif option=='Impact':
                st.dataframe(st.session_state.df_impact)
            elif option=='Connectivity':
                st.dataframe(st.session_state.df_connectivity)

        with colB:
            st.subheader('Category Score Map')
            tooltip2 = {
                "html": "School: {giga_id_school}</br> " + option + " Score: {" + option + "} </br>"
            }
            st.pydeck_chart(pdk.Deck(
                tooltip=tooltip2,
                map_provider="mapbox",
                map_style="mapbox://styles/mapbox/light-v9",
                initial_view_state=pdk.ViewState(
                    latitude=np.mean(st.session_state.df_school_dup['latitude']),
                    longitude=np.mean(st.session_state.df_school_dup['longitude']),
                    zoom=5,
                    pitch=50,
                ),
                layers=[
                    pdk.Layer(
                        'GridCellLayer',
                        data=st.session_state.df_school_dup,
                        get_position=['longitude', 'latitude'],
                        cellSize=10000,
                        elevationScale=50000,
                        get_fill_color=[255, 140, option, 140],
                        getElevation=option,
                        pickable=True,
                        auto_highlight=True,
                    ),
                ],
            ))

    ##############################

    #####Explore the attributes########
    st.header('Explore the attributes:')

    attrs = ['None'] + st.session_state.attributesR + st.session_state.attributesP + st.session_state.attributesI + st.session_state.attributesC


    option2 = st.selectbox(
        'Choose an Attribute',
        tuple(attrs))

    if option2 and option2 != 'None':
        colA, colB = st.columns(2)
        with colA:
            st.subheader('Attribute distributions')
            fig, ax = plt.subplots()
            if option2 in st.session_state.attributesR:
                st.session_state.df_remoteness.hist(ax=ax, column=option2)
                st.pyplot(fig)
            elif option2 in st.session_state.attributesP:
                st.session_state.df_poverty.hist(ax=ax, column=option2)
                st.pyplot(fig)
            elif option2 in st.session_state.attributesI:
                st.session_state.df_impact.hist(ax=ax, column=option2)
                st.pyplot(fig)
            elif option2 in st.session_state.attributesC:
                st.session_state.df_connectivity.hist(ax=ax, column=option2)
                st.pyplot(fig)

        with colB:
            st.subheader('Attribute Map')
            tooltip2 = {
                "html": "School: {giga_id_school}</br> " + option2 + " Value: {" + option2 + "} </br>"
            }

            if option2 in st.session_state.attributesR:
                st.pydeck_chart(pdk.Deck(
                    tooltip=tooltip2,
                    map_provider="mapbox",
                    map_style="mapbox://styles/mapbox/light-v9",
                    initial_view_state=pdk.ViewState(
                        latitude=np.mean(st.session_state.df_remoteness['latitude']),
                        longitude=np.mean(st.session_state.df_remoteness['longitude']),
                        zoom=5,
                        pitch=50,
                    ),
                    layers=[
                        pdk.Layer(
                            'GridCellLayer',
                            data=st.session_state.df_remoteness,
                            get_position=['longitude', 'latitude'],
                            cellSize=10000,
                            elevationScale=50000,
                            get_fill_color=[255, 140, option2, 140],
                            getElevation=option2,
                            pickable=True,
                            auto_highlight=True,
                        ),
                    ],
                ))
            elif option2 in st.session_state.attributesP:
                st.pydeck_chart(pdk.Deck(
                    tooltip=tooltip2,
                    map_provider="mapbox",
                    map_style="mapbox://styles/mapbox/light-v9",
                    initial_view_state=pdk.ViewState(
                        latitude=np.mean(st.session_state.df_poverty['latitude']),
                        longitude=np.mean(st.session_state.df_poverty['longitude']),
                        zoom=5,
                        pitch=50,
                    ),
                    layers=[
                        pdk.Layer(
                            'GridCellLayer',
                            data=st.session_state.df_poverty,
                            get_position=['longitude', 'latitude'],
                            cellSize=10000,
                            elevationScale=50000,
                            get_fill_color=[255, 140, option2, 140],
                            getElevation=option2,
                            pickable=True,
                            auto_highlight=True,
                        ),
                    ],
                ))
            elif option2 in st.session_state.attributesI:
                st.pydeck_chart(pdk.Deck(
                    tooltip=tooltip2,
                    map_provider="mapbox",
                    map_style="mapbox://styles/mapbox/light-v9",
                    initial_view_state=pdk.ViewState(
                        latitude=np.mean(st.session_state.df_impact['latitude']),
                        longitude=np.mean(st.session_state.df_impact['longitude']),
                        zoom=5,
                        pitch=50,
                    ),
                    layers=[
                        pdk.Layer(
                            'GridCellLayer',
                            data=st.session_state.df_impact,
                            get_position=['longitude', 'latitude'],
                            cellSize=10000,
                            elevationScale=50000,
                            get_fill_color=[255, 140, option2, 140],
                            getElevation=option2,
                            pickable=True,
                            auto_highlight=True,
                        ),
                    ],
                ))
            elif option2 in st.session_state.attributesC:
                st.pydeck_chart(pdk.Deck(
                    tooltip=tooltip2,
                    map_provider="mapbox",
                    map_style="mapbox://styles/mapbox/light-v9",
                    initial_view_state=pdk.ViewState(
                        latitude=np.mean(st.session_state.df_connectivity['latitude']),
                        longitude=np.mean(st.session_state.df_connectivity['longitude']),
                        zoom=5,
                        pitch=50,
                    ),
                    layers=[
                        pdk.Layer(
                            'GridCellLayer',
                            data=st.session_state.df_connectivity,
                            get_position=['longitude', 'latitude'],
                            cellSize=10000,
                            elevationScale=50000,
                            get_fill_color=[255, 140, option2, 140],
                            getElevation=option2,
                            pickable=True,
                            auto_highlight=True,
                        ),
                    ],
                ))

    ####################################