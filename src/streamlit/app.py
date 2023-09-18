# Contents of ~/my_app/streamlit_app.py
import json
from pathlib import Path
import plotly.express as px
import streamlit as st
import pandas as pd

import streamlit as st
from BirdMOT.detection.evaluate import EvaluationController


def main_page():
    st.markdown("# Main page 🎈")
    st.sidebar.markdown("# Main page 🎈")

def page2():
    st.markdown("# Page 2 ❄️")
    st.sidebar.markdown("# Page 2 ❄️")

def page3():
    st.markdown("# Page 3 🎉")
    st.sidebar.markdown("# Page 3 🎉")

page_names_to_funcs = {
    "Main Page": main_page,
    "Page 2": page2,
    "Page 3": page3,
}




with open('/media/data/BirdMOT/local_data/configs/experiments/yolov8n_setup_comparison_320_640.json') as json_file:
    data = json.load(json_file)

st.markdown(data['description'])

assembly_config = Path("/media/data/BirdMOT/local_data/configs/dataset_assembly/dataset_assembly2_rc_4good_tracks_in_val.json")
assert assembly_config.exists(), f"Assembly config {assembly_config} does not exist"
with open(assembly_config) as json_file:
    assembly_config = json.load(json_file)

def basic_comparison(evaluation_result):
    return dict(
        sahi_setup_name=evaluation_result['sahi_setup_name'],
        fps=evaluation_result['data']['prediction_results']['durations_in_seconds']['fps'],
        bbox_mAP=evaluation_result['data']['eval_results']['bbox_mAP'],
        bbox_mAP_s=evaluation_result['data']['eval_results']['bbox_mAP_s'],
        bbox_mAP_m=evaluation_result['data']['eval_results']['bbox_mAP_m'],
        bbox_mAP_l=evaluation_result['data']['eval_results']['bbox_mAP_l'],
    )

device = 0
evaluation_controller = EvaluationController()
evaluations= [evaluation_controller.find_or_create_evaluation(one_experiment_config,assembly_config , device=device, train_missing=True) for one_experiment_config in data['experiments']]


df = pd.DataFrame(evaluations)




st.subheader("Define a custom colorscale")
df = px.data.iris()
fig = px.scatter(
    df,
    x="bbox_mAP",
    y="fps",
    color="sepal_length",
    color_continuous_scale="reds",
)

tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
with tab1:
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
with tab2:
    st.plotly_chart(fig, theme=None, use_container_width=True)

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
