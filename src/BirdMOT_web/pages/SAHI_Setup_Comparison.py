# Contents of ~/my_app/streamlit_app.py
import json
from operator import add

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from BirdMOT.detection.evaluate import EvaluationController
from BirdMOT.helper.config import get_list_of_dataset_assemblies, get_list_of_experiments, get_dataset_assembly_by_name, \
    get_experiment_by_name
from BirdMOT_web.utils.charts import improve_text_position

st.set_page_config(layout="wide")


def dataframe_with_selections(df):
    df_with_selections = df.copy()
    df_with_selections.insert(0, "Select", False)

    # Get dataframe row-selections from user with st.data_editor
    edited_df = st.data_editor(
        df_with_selections,
        hide_index=True,
        column_order=(
            'Select', "Setup Name", "bbox_mAP", "bbox_mAP50_coco", "bbox_mAP_s", "bbox_mAP_m", "bbox_mAP_l", "AR_s",
            "AR_m",
            "AR_l", "bbox_AR@100", "fps"),
        column_config={"Select": st.column_config.CheckboxColumn(required=True)},
        disabled=df.columns,
    )

    # Filter the dataframe using the temporary column, then drop the column
    selected_rows = edited_df[edited_df.Select]
    return selected_rows.drop('Select', axis=1)


@st.cache_data(ttl=3600)
def basic_comparison(evaluation_result):
    print(evaluation_result)
    return {
        "Setup Name": evaluation_result['setup_name'],
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
def load_evaluations(experiment_config, assembly_config, device):
    evaluation_controller = EvaluationController()
    return [evaluation_controller.find_or_create_evaluation(one_experiment_config, assembly_config, device=device,
                                                            train_missing=False) for one_experiment_config in
            experiment_config['experiments']]


device = "cpu"

selected_assembly = st.selectbox(
    "Select an assembly",
    get_list_of_dataset_assemblies()
)

st.write('You selected:', selected_assembly)

selected_experiments = st.multiselect(
    'Select experiments to compare',
    get_list_of_experiments())

st.write('You selected:', selected_experiments)

assembly_config_path = get_dataset_assembly_by_name(selected_assembly)
assert assembly_config_path.exists(), f"Assembly config {assembly_config_path} does not exist"
with open(assembly_config_path) as json_file:
    assembly_config = json.load(json_file)

evaluations = []
for exp_name in selected_experiments:
    with open(get_experiment_by_name(exp_name)) as json_file:
        experiments_data = json.load(json_file)
    evaluations.extend(load_evaluations(experiments_data, assembly_config, device))

filtered_evaluations = [basic_comparison(evaluation) for evaluation in evaluations]

# evaluations= [{'bla':'blubb', 'a':1}]
df = pd.DataFrame(filtered_evaluations)

# st.markdown(f"**Experiment Description:** {sahi_setup_comparison_experiments['description']}")
st.markdown("## All Models (Selection)")

prefix_filter = st.text_input('Model Name Prefix Filter', '')
df = df[df['Setup Name'].str.contains(prefix_filter)]
selected_models_df = dataframe_with_selections(df)

st.markdown(
    """
    - SF: slicing aided fine-tuning (slicing during training)

    - SAHI: slicing aided inference (slicing during inference)

    - FI: full image inference

    - PO: overlapping patches 
    """
)

st.markdown("## Selected Models Comparison")
btab1, btab2 = st.tabs(["Dataframe Widget", "Latex Table"])
with btab1:
    st.dataframe(selected_models_df, column_order=(
        'Select', "Setup Name", "bbox_mAP", "bbox_mAP50_coco", "bbox_mAP_s", "bbox_mAP_m", "bbox_mAP_l", "AR_s", "AR_m",
        "AR_l", "bbox_AR@100", "fps"), )
with btab2:
    st.markdown(
        selected_models_df.to_latex(buf=None, columns=None, header=True, index=True, na_rep='NaN', formatters=None,
                                    float_format=None,
                                    sparsify=None, index_names=True, bold_rows=False, column_format=None,
                                    longtable=None, escape=True,
                                    encoding=None, decimal='.', multicolumn=True, multicolumn_format=None,
                                    multirow=None, caption=None,
                                    label=None, position=None))

with st.expander("Json Data"):
    for i, row in selected_models_df.iterrows():
        selected_model_evaluation = [it for it in evaluations if it['setup_name'] == row[0]][0]
        st.write(selected_model_evaluation)

fig = px.scatter(
    selected_models_df,
    x="fps",
    y="bbox_AR@100",
    text="Setup Name",
    template="ggplot2"
    # color="sepal_length",
    # color_continuous_scale="reds",
)
fig.update_traces(textposition=improve_text_position(df['bbox_mAP50_coco']))
st.plotly_chart(fig, theme=None, use_container_width=True)

fig = px.scatter(
    selected_models_df,
    x="fps",
    y="bbox_AR@100",
    text=None,
    template="ggplot2"
    # color="sepal_length",
    # color_continuous_scale="reds",
)
fig.update_traces(textposition=improve_text_position(df['bbox_mAP50_coco']))
st.plotly_chart(fig, theme=None, use_container_width=True)

fig = px.scatter(
    selected_models_df,
    x="fps",
    y="bbox_mAP",
    text="Setup Name",
    template="ggplot2"
    # color="sepal_length",
    # color_continuous_scale="reds",
)
fig.update_traces(textposition=improve_text_position(df['bbox_mAP']))
st.plotly_chart(fig, theme=None, use_container_width=True)

with st.expander("Table of selected models sorted by AR@100"):
    st.dataframe(selected_models_df.sort_values(by=['bbox_AR@100'], ascending=False),
                 column_order=("Setup Name", "bbox_AR@100", "AR_s", "AR_m", "AR_l", "bbox_mAP"))

with st.expander("Table of selected models sorted by AR_s"):
    st.dataframe(selected_models_df.sort_values(by=['AR_s'], ascending=False),
                 column_order=("Setup Name", "AR_s", "AR_m", "AR_l", "bbox_AR@100", "bbox_mAP"))

with st.expander("Table of selected models sorted by AR_m"):
    st.dataframe(selected_models_df.sort_values(by=['AR_m'], ascending=False),
                 column_order=("Setup Name", "bbox_AR@100", "AR_s", "AR_m", "AR_l", "bbox_mAP"))

with st.expander("Table of selected models sorted by AR_l"):
    st.dataframe(selected_models_df.sort_values(by=['AR_l'], ascending=False),
                 column_order=("Setup Name", "bbox_AR@100", "AR_s", "AR_m", "AR_l", "bbox_mAP"))

with st.expander("Recall over IoU Threshold"):
    fig = go.Figure()
    for i, row in selected_models_df.iterrows():
        selected_model_evaluation = [it for it in evaluations if it['setup_name'] == row[0]][0]
        recall = selected_model_evaluation['data']["eval_results_fiftyone"]['recall']
        thresholds = selected_model_evaluation['data']["eval_results_fiftyone"]['iou_thresholds']

        fig.add_trace(go.Scatter(x=thresholds, y=recall, name=row[0], mode='lines'))

    fig.update_layout(
        title="Recall over IoU Threshold",
        xaxis_title='IoU Threshold',
        #  yaxis_title='Precision',
        width=700, height=500
    )
    st.plotly_chart(fig, use_container_width=True)

with st.expander("Precision over IoU Threshold"):
    fig = go.Figure()
    for i, row in selected_models_df.iterrows():
        selected_model_evaluation = [it for it in evaluations if it['setup_name'] == row[0]][0]
        precision = selected_model_evaluation['data']["eval_results_fiftyone"]['precision']
        thresholds = selected_model_evaluation['data']["eval_results_fiftyone"]['iou_thresholds']

        fig.add_trace(go.Scatter(x=thresholds, y=precision, name=row[0], mode='lines'))

    fig.update_layout(
        title="Precision over IoU Threshold",
        xaxis_title='iou Threshold',
        #  yaxis_title='Precision',
        width=700, height=500
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("## Single Model Evaluation")
single_selected_model = dataframe_with_selections(selected_models_df)
if len(single_selected_model) > 1:
    st.error('More than one model was selected', icon="🚨")
else:
    single_selected_model["Setup Name"]
    selected_model_evaluation = \
        [it for it in evaluations if it['setup_name'] == single_selected_model["Setup Name"].to_list()[0]][0]
    # fpr = one_evaluation['data']["eval_results_fiftyone"]['fpr']
    tpr = selected_model_evaluation['data']["eval_results_fiftyone"]['tpr']
    fnr = selected_model_evaluation['data']["eval_results_fiftyone"]['fnr']
    tp_iou_results = selected_model_evaluation['data']["eval_results_fiftyone"]['tp_iou_results']
    fn_iou_results = selected_model_evaluation['data']["eval_results_fiftyone"]['fn_iou_results']
    fp_iou_results = selected_model_evaluation['data']["eval_results_fiftyone"]['fp_iou_results']
    wrong_predictions = list(map(add, fn_iou_results, fp_iou_results))
    thresholds = selected_model_evaluation['data']["eval_results_fiftyone"]['iou_thresholds']
    recall = selected_model_evaluation['data']["eval_results_fiftyone"]['recall']
    precision = selected_model_evaluation['data']["eval_results_fiftyone"]['precision']

    with st.expander("Recall and Precision at every IoU Threshold"):
        # Recall IoU Curve
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=thresholds, y=precision, name="Precision", mode='lines'))
        fig.add_trace(go.Scatter(x=thresholds, y=recall, name="Reacall", mode='lines'))
        print(recall)
        print(precision)
        fig.update_layout(
            # title="Recall and Precision at different IoU Thresholds",
            xaxis_title='IoU Threshold',
            #  yaxis_title='Precision',
            template='seaborn',
            width=700, height=500
        )
        st.plotly_chart(fig, theme=None, use_container_width=True)

    with st.expander("True Positives count and Wrong Predictions count at every IoU Threshold"):
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

    with st.expander("SAHI Analysis Results"):
        st.markdown("""
        🎯 **Meaning of the metrics:**  
        **C75:** Results at 0.75 IOU threshod  
        **C50:** Results at 0.75 IOU threshold  
        **Loc:** Results after ignoring localization errors  
        **Sim:** Results after ignoring supercategory false positives  
        **Oth:** Results after ignoring all category confusions  
        **BG:** Results after ignoring all false positives  
        **FN:** Results after ignoring all false negatives
        
        📈 **Possible model improvements:**  
        **C75-C50 and C50-Loc**=Potential gain with more accurate bounding box prediction  
        **Loc-Sim**=Potential gain after fixing supercategory confusions  
        **Loc-Oth**=Potential gain after fixing category confusions  
        **Oth-BG**=Potential gain after fixing all false positives  
        **BG-FN**=Potential gain after fixing all false negatives
        """)
        "Bar Plot"
        image = Image.open(selected_model_evaluation['data']['analysis_results']['bbox']['overall']['bar_plot'])
        st.image(image, caption='Bar Plot')

        "BBox All Area"
        image = Image.open(selected_model_evaluation['data']['analysis_results']['bbox']['overall']['curves'][0])
        st.image(image, caption='BBox All Area')

        "BBox Small Area"
        image = Image.open(selected_model_evaluation['data']['analysis_results']['bbox']['overall']['curves'][1])
        st.image(image, caption='BBox Small Area')

        "BBox Medium Area"
        image = Image.open(selected_model_evaluation['data']['analysis_results']['bbox']['overall']['curves'][2])
        st.image(image, caption='BBox Medium Area')

        "BBox Large Area"
        image = Image.open(selected_model_evaluation['data']['analysis_results']['bbox']['overall']['curves'][3])
        st.image(image, caption='BBox Large Area')

        "GT Area Group Members"
        image = Image.open(
            selected_model_evaluation['data']['analysis_results']['bbox']['overall']['gt_area_group_numbers'])
        st.image(image, caption="GT Area Group Members")

        "GT Area History"
        image = Image.open(
            selected_model_evaluation['data']['analysis_results']['bbox']['overall']['gt_area_histogram'])
        st.image(image, caption="GT Area History")

    with st.expander("Get the Model"):
        sahi_prediction_params = selected_model_evaluation['one_experiment_config']['sahi_prediction_params']
        f"Weights: {selected_model_evaluation['sahi_prediction_params']['model_path']}"  # ToDo: Add download link

    with st.expander("json"):
        selected_model_evaluation
