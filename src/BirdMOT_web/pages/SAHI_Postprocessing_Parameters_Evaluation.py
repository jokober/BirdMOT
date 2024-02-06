import json
from pathlib import Path
import plotly.express as px
import streamlit as st
import pandas as pd

import streamlit as st
from BirdMOT.detection.evaluate import EvaluationController

st.set_page_config(page_title="SAHI Postprocessing Parameters Evaluation")


st.markdown("# SAHI Postprocessing Parameters Evaluation")

st.markdown("""## SAHI Postprocess Match Metric
- IOU: intersection over union
- IOS: is basically intersection over smaller region, ie intersection area of two box / area of smaller box. It is used as the match criteria while performing NMS and NMM.

In the original sahi paper only standard NMS and IOU were utilized. The second paper was never published...


 
""")

st.markdown("""## SAHI Postprocess Type
- NMM: non-maximum merging, in other words, it merges the lower scored box instead of suppressing it as in NMS . It is useful when the model confidence threshold is large (as 0.3 to 0.5) and the predictions are confident. It merges the box predictions, corresponding to the same instance, coming from different slices.
- GREEDYNMM: 
- NMS: 

""")



def basic_comparison(evaluation_result):
    print(evaluation_result)
    return dict(
        sahi_setup_name=evaluation_result['sahi_setup_name'],
        postprocess_type=evaluation_result['sahi_prediction_params']['postprocess_type'],
        postprocess_match_metric=evaluation_result['sahi_prediction_params']['postprocess_match_metric'],
        postprocess_match_threshold=evaluation_result['sahi_prediction_params']['postprocess_match_threshold'],
        bbox_mAP=evaluation_result['data']['eval_results']['bbox_mAP'],
        bbox_mAP_s=evaluation_result['data']['eval_results']['bbox_mAP_s'],
        bbox_mAP_m=evaluation_result['data']['eval_results']['bbox_mAP_m'],
        bbox_mAP_l=evaluation_result['data']['eval_results']['bbox_mAP_l'],
    )

@st.cache_data(ttl=3600)
def load_data():
    with open(
            '/media/data/BirdMOT/local_data/configs/experiments/yolov8n_postprocessing_comparison_320_640.json') as json_file:
        data = json.load(json_file)



    assembly_config = Path(
        "/media/data/BirdMOT/local_data/configs/dataset_assembly/dataset_assembly2_rc_4good_tracks_in_val.json")
    assert assembly_config.exists(), f"Assembly config {assembly_config} does not exist"
    with open(assembly_config) as json_file:
        assembly_config = json.load(json_file)

    return data, assembly_config

data, assembly_config = load_data()
st.markdown(data['description'])

device = "cpu"
evaluation_controller = EvaluationController()
evaluations = [evaluation_controller.find_or_create_evaluation(one_experiment_config, assembly_config, device=device,
                                                               train_missing=False) for one_experiment_config in
               data['experiments']]
# evaluation_controller.find_or_create_evaluation(data['experiments'][0],assembly_config , device=device, train_missing=True)
# evaluations= [evaluation for evaluation in evaluation_controller.state['evaluations']]
evaluations = [basic_comparison(evaluation) for evaluation in evaluations]
print(evaluations)
# evaluations= [{'bla':'blubb', 'a':1}]
df = pd.DataFrame(evaluations)

btab1, btab2 = st.tabs(["Dataframe Widget", "Latex Table"])
with btab1:
    st.dataframe(df)
with btab2:
    st.markdown(
        df.to_latex(buf=None, columns=None, header=True, index=True, na_rep='NaN', formatters=None, float_format=None,
                    sparsify=None, index_names=True, bold_rows=False, column_format=None, longtable=None, escape=None,
                    encoding=None, decimal='.', multicolumn=None, multicolumn_format=None, multirow=None, caption=None,
                    label=None, position=None))

st.markdown(
    """
    SF: slicing aided fine-tuning
    SAHI: slicing aided inference
    FI: full image inference
    PO: overlapping patches 
    """
)

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
    df,
    x="postprocess_type",
    y="bbox_mAP",
    text="postprocess_match_metric",
    # color="sepal_length",
    # color_continuous_scale="reds",
)
fig.update_traces(textposition="bottom right")

tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
with tab1:
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
with tab2:
    st.plotly_chart(fig, theme=None, use_container_width=True)

