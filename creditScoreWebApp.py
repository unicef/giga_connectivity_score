import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import pydeck as pdk

fields = ['school_id','lon','lat','Remoteness','Poverty','Impact','Connectivity']#'name',


@st.experimental_singleton()
def load_and_treat_file(fn):
    if fn != None:
        df_school = pd.read_csv(fn)

        st.session_state.df_school_dup = df_school.copy()

        for colname in st.session_state.df_school_dup.columns:
            if colname not in fields:
                st.session_state.df_school_dup.drop(colname, inplace=True, axis=1)

        st.session_state.df_school_dup['CreditScore'] = st.session_state.df_school_dup['Remoteness'] * st.session_state.Wr + st.session_state.df_school_dup['Poverty'] * st.session_state.Wp + st.session_state.df_school_dup[
        'Impact'] * st.session_state.Wi + st.session_state.df_school_dup['Connectivity'] * st.session_state.Wc

        st.session_state.df_school_dup = st.session_state.df_school_dup.sort_values(by=['CreditScore'], ascending=False)


def recalculateScore():
    if not st.session_state.df_school_dup.empty:
        st.session_state.df_school_dup['CreditScore'] = st.session_state.df_school_dup['Remoteness'] * st.session_state.Wr + st.session_state.df_school_dup['Poverty'] * st.session_state.Wp + st.session_state.df_school_dup[
        'Impact'] * st.session_state.Wi + st.session_state.df_school_dup['Connectivity'] * st.session_state.Wc

        st.session_state.df_school_dup = st.session_state.df_school_dup.sort_values(by=['CreditScore'], ascending=False)

st.set_page_config(layout="wide")
st.title("Credit Score App")


if 'df_school_dup' not in st.session_state:
    st.session_state.df_school_dup = pd.DataFrame()

with st.sidebar:
    with st.expander("Category Weights",expanded=True):
        slider_Wr = st.slider('Remoteness weight', 0.0, 1.0, 1.0, key='Wr', on_change = recalculateScore())
        slider_Wp = st.slider('Poverty weight', 0.0, 1.0, 1.0, key='Wp', on_change = recalculateScore())
        slider_Wi = st.slider('Impact weight', 0.0, 1.0, 1.0, key='Wi', on_change = recalculateScore())
        slider_Wc = st.slider('Connectivity weight', 0.0, 1.0, 1.0, key='Wc', on_change = recalculateScore())

    with st.expander("Attribute weights:"):
        st.text('Remoteness')
        slider_Wele = st.slider('Electricity weight', 0.0, 1.0, 1.0, key='Wele', on_change=recalculateScore())
        slider_Wfn = st.slider('Fiber node dist weight', 0.0, 1.0, 1.0, key='Wfn', on_change=recalculateScore())
        st.text('Poverty')
        slider_Wwa = st.slider('Water weight', 0.0, 1.0, 1.0, key='Wwa', on_change=recalculateScore())
        slider_Wpi = st.slider('Poverty index weight', 0.0, 1.0, 1.0, key='Wpi', on_change=recalculateScore())
        st.text('Impact')
        slider_Wns = st.slider('# students weight', 0.0, 1.0, 1.0, key='Wns', on_change=recalculateScore())
        slider_Wpop = st.slider('Population weight', 0.0, 1.0, 1.0, key='Wpop', on_change=recalculateScore())
        st.text('Connectivity')
        slider_Wpc = st.slider('Previously connected weight', 0.0, 1.0, 1.0, key='Wpc', on_change=recalculateScore())
        slider_Wnc = st.slider('Nearby connectivity weight', 0.0, 1.0, 1.0, key='Wnc', on_change=recalculateScore())

fn = st.file_uploader("Choose a country file")
if fn:
    load_and_treat_file(fn)

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(st.session_state.df_school_dup)

    with col2:
        tooltip = {
            "html": "School: {school_id}</br> Credit Score: {CreditScore} </br>"
        }
        st.pydeck_chart(pdk.Deck(
            #map_style=None,
            tooltip=tooltip,
            map_provider="mapbox",
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=pdk.ViewState(
                latitude=np.mean(st.session_state.df_school_dup['lat']),
                longitude=np.mean(st.session_state.df_school_dup['lon']),
                zoom=5,
                pitch=50,
            ),
            layers=[
                pdk.Layer(
                    'ColumnLayer',
                    data=st.session_state.df_school_dup,
                    get_position=['lon', 'lat'],
                    get_weight='CreditScore',
                    elevation_scale=20,
                    radius=2000,
                    get_fill_color=[180, 0, 200, 140],
                    pickable=True,
                    auto_highlight=True,
                ),
            ],
        ))

