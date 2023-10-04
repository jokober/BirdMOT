# Contents of ~/my_app/streamlit_app.py
import json
from pathlib import Path
import plotly.express as px
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from BirdMOT.detection.evaluate import EvaluationController
from operator import add

st.set_page_config(layout="wide")


def dataframe_with_selections(df):
    df_with_selections = df.copy()
    df_with_selections.insert(0, "Select", False)

    # Get dataframe row-selections from user with st.data_editor
    edited_df = st.data_editor(
        df_with_selections,
        hide_index=True,
        column_config={"Select": st.column_config.CheckboxColumn(required=True)},
        disabled=df.columns,
    )

    # Filter the dataframe using the temporary column, then drop the column
    selected_rows = edited_df[edited_df.Select]
    return selected_rows.drop('Select', axis=1)


def basic_comparison(evaluation_result):
    print(evaluation_result)
    return {
        "setup_name": evaluation_result['setup_name'],
        "fps": evaluation_result['data']['prediction_result']['data']['durations_in_seconds']['prediction']['fps'],
        "bbox_mAP": evaluation_result['data']['eval_results']['bbox_mAP'],
        "bbox_mAP50_coco": evaluation_result['data']['eval_results_pycocotools']['bbox_mAP50'],
        "bbox_mAP50_sahi": evaluation_result['data']["eval_results"]['bbox_mAP50'],
        "bbox_mAP_fo@40": evaluation_result['data']["eval_results_fiftyone"]['mAP'],
        "bbox_mAP_s": evaluation_result['data']['eval_results']['bbox_mAP_s'],
        "bbox_mAP_m": evaluation_result['data']['eval_results']['bbox_mAP_m'],
        "bbox_mAP_l": evaluation_result['data']['eval_results']['bbox_mAP_l'],
        "bbox_AR@100": evaluation_result['data']['eval_results_pycocotools']['bbox_AR@100'],
        "AR_s": evaluation_result['data']["eval_results"]['bbox_AR_s'],
        "AR_m": evaluation_result['data']["eval_results"]['bbox_AR_m'],
        "AR_l": evaluation_result['data']["eval_results"]['bbox_AR_l'],
    }


@st.cache_data(ttl=3600)
def load_sahi_setup_comparison_config():
    with open('/media/data/BirdMOT/local_data/configs/experiments/yolov8n_setup_comparison_320_640.json') as json_file:
        data = json.load(json_file)

    assembly_config = Path(
        #    "/media/data/BirdMOT/local_data/configs/dataset_assembly/dataset_assembly2_rc_4good_tracks_in_val.json")
        "/media/data/BirdMOT/local_data/configs/dataset_assembly/dataset_assembly4_without_random100.json")
    assert assembly_config.exists(), f"Assembly config {assembly_config} does not exist"
    with open(assembly_config) as json_file:
        assembly_config = json.load(json_file)

    return data, assembly_config


@st.cache_data(ttl=3600)
def load_sahi_yolov8_full_resolution_config():
    with open('/media/data/BirdMOT/local_data/configs/experiments/yolov8n_full_resolution.json') as json_file:
        data = json.load(json_file)

    assembly_config = Path(
        #    "/media/data/BirdMOT/local_data/configs/dataset_assembly/dataset_assembly2_rc_4good_tracks_in_val.json")
        "/media/data/BirdMOT/local_data/configs/dataset_assembly/dataset_assembly4_without_random100.json")
    assert assembly_config.exists(), f"Assembly config {assembly_config} does not exist"
    with open(assembly_config) as json_file:
        assembly_config = json.load(json_file)

    return data, assembly_config
@st.cache_data(ttl=3600)
def load_evaluations(experiment_config, assembly_config, device):
    evaluation_controller = EvaluationController()
    return [evaluation_controller.find_or_create_evaluation(one_experiment_config, assembly_config, device=device,
                                                            train_missing=True) for one_experiment_config in
            experiment_config['experiments']]

sahi_setup_comparison_experiments, sahi_setup_comparison_assembly_config = load_sahi_setup_comparison_config()
full_res_comparison_experiments, full_res_assembly_config = load_sahi_yolov8_full_resolution_config()


device = "cpu"

evaluations = []
evaluations.extend(load_evaluations(sahi_setup_comparison_experiments, sahi_setup_comparison_assembly_config, device))


full_res_evals = load_evaluations(full_res_comparison_experiments, full_res_assembly_config, device)
evaluations.extend(full_res_evals)

filtered_evaluations = [basic_comparison(evaluation) for evaluation in evaluations]

# evaluations= [{'bla':'blubb', 'a':1}]
df = pd.DataFrame(filtered_evaluations)

st.markdown(f"**Experiment Description:** {sahi_setup_comparison_experiments['description']}")
st.markdown("## All Data (Selection)" )
dft = dataframe_with_selections(df)

st.markdown(
    """
    - SF: slicing aided fine-tuning

    - SAHI: slicing aided inference

    - FI: full image inference

    - PO: overlapping patches 
    """
)

st.markdown("## Selection Based Data" )
btab1, btab2 = st.tabs(["Dataframe Widget", "Latex Table"])
with btab1:
    st.dataframe(dft)
with btab2:
    st.markdown(
        dft.to_latex(buf=None, columns=None, header=True, index=True, na_rep='NaN', formatters=None, float_format=None,
                    sparsify=None, index_names=True, bold_rows=False, column_format=None, longtable=None, escape=None,
                    encoding=None, decimal='.', multicolumn=None, multicolumn_format=None, multirow=None, caption=None,
                    label=None, position=None))

with st.expander("Json Data"):
    for i, row in dft.iterrows():
        one_evaluation = [it for it in evaluations if it['setup_name'] == row[0]][0]
        st.write(one_evaluation)

# st.markdown(
#     """
#     🎯 **Meaning of the metrics:**
#     **C75:** Results at 0.75 IOU threshod
#     **C50:** Results at 0.75 IOU threshold
#     **Loc:** Results after ignoring localization errors
#     **Sim:** Results after ignoring supercategory false positives
#     **Oth:** Results after ignoring all category confusions
#     **BG:** Results after ignoring all false positives
#     **FN:** Results after ignoring all false negatives
#
#     📈 **Possible model improvements:**
#     **C75-C50 and C50-Loc**=Potential gain with more accurate bounding box prediction
#     **Loc-Sim**=Potential gain after fixing supercategory confusions
#     **Loc-Oth**=Potential gain after fixing category confusions
#     **Oth-BG**=Potential gain after fixing all false positives
#     **BG-FN**=Potential gain after fixing all false negatives
#     """
# )

fig = px.scatter(
    dft,
    x="fps",
    y="bbox_mAP",
    text="setup_name",
    # color="sepal_length",
    # color_continuous_scale="reds",
)
fig.update_traces(textposition="bottom right")

tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
with tab1:
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
with tab2:
    st.plotly_chart(fig, theme=None, use_container_width=True)



fig = go.Figure()
for i, row in dft.iterrows():
    one_evaluation = [it for it in evaluations if it['setup_name'] == row[0]][0]
    recall = one_evaluation['data']["eval_results_fiftyone"]['recall']
    thresholds = one_evaluation['data']["eval_results_fiftyone"]['iou_thresholds']

    fig.add_trace(go.Scatter(x=thresholds, y=recall, name=row[0], mode='lines'))

fig.update_layout(
    title="Recall over IoU Threshold",
    xaxis_title='iou Threshold',
    #  yaxis_title='Precision',
    width=700, height=500
)
st.plotly_chart(fig, use_container_width=True)


fig = go.Figure()
for i, row in dft.iterrows():
    one_evaluation = [it for it in evaluations if it['setup_name'] == row[0]][0]
    precision = one_evaluation['data']["eval_results_fiftyone"]['precision']
    thresholds = one_evaluation['data']["eval_results_fiftyone"]['iou_thresholds']

    fig.add_trace(go.Scatter(x=thresholds, y=precision, name=row[0], mode='lines'))

fig.update_layout(
    title="Precision over IoU Threshold",
    xaxis_title='iou Threshold',
    #  yaxis_title='Precision',
    width=700, height=500
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("## Single Datapoint Evaluation" )


one_evaluation = [it for it in evaluations if it['setup_name'] == 'yolov8n_SAHI'][0]
# fpr = one_evaluation['data']["eval_results_fiftyone"]['fpr']
tpr = one_evaluation['data']["eval_results_fiftyone"]['tpr']
fnr = one_evaluation['data']["eval_results_fiftyone"]['fnr']
tp_iou_results = one_evaluation['data']["eval_results_fiftyone"]['tp_iou_results']
fn_iou_results = one_evaluation['data']["eval_results_fiftyone"]['fn_iou_results']
fp_iou_results = one_evaluation['data']["eval_results_fiftyone"]['fp_iou_results']
wrong_predictions = list(map(add, fn_iou_results, fp_iou_results))
thresholds = one_evaluation['data']["eval_results_fiftyone"]['iou_thresholds']
recall = one_evaluation['data']["eval_results_fiftyone"]['recall']
precision = one_evaluation['data']["eval_results_fiftyone"]['precision']



# Recall IoU Curve
fig = go.Figure()
fig.add_trace(go.Scatter(x=thresholds, y=precision, name="Precision", mode='lines'))
fig.add_trace(go.Scatter(x=thresholds, y=recall, name="Reacall", mode='lines'))
print(recall)
print(precision)
fig.update_layout(
    title="Recall and Precision at every IoU Threshold",
    xaxis_title='iou Threshold',
    #  yaxis_title='Precision',
    width=700, height=500
)
st.plotly_chart(fig, use_container_width=True)

# TP and wrong predictions IoU Curve
fig = go.Figure()
fig.add_trace(go.Scatter(x=thresholds, y=tp_iou_results, name="True Positives", mode='lines'))
fig.add_trace(go.Scatter(x=thresholds, y=wrong_predictions, name="Wrong Predictions (fn + fp)", mode='lines'))
fig.update_layout(
    title="True Positives count and Wrong Predictions count at every IoU Threshold",
    xaxis_title='iou Threshold',
    #  yaxis_title='Precision',
    width=700, height=500
)
st.plotly_chart(fig, use_container_width=True)

"eval_results_pycocotools"
one_evaluation['data']["eval_results_pycocotools"]
"eval_results"
one_evaluation['data']["eval_results"]