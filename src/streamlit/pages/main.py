import streamlit as st

st.markdown("# Main page 🎈")
st.sidebar.markdown("# Main page 🎈")


import pandas as pd



data = [
    {
        "id": 1,
        "candidate": "Roberto mathews",
        "health_index": {"bmi": 22, "blood_pressure": 130},
    },
    {"candidate": "Shane wade", "health_index": {"bmi": 28, "blood_pressure": 160}},
    {
        "id": 2,
        "candidate": "Bruce tommy",
        "health_index": {"bmi": 31, "blood_pressure": 190},
    },
]
df = pd.json_normalize(data, max_level=1)
edited_df = st.data_editor(df)

st.header('Models with highest recall')
st.header('Models with highest precision')
st.header('_Streamlit_ is :blue[cool] :sunglasses:')

pd.concat()