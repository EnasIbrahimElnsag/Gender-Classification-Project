import gradio as gr
import joblib


# Load trained model and scaler
nb_model = joblib.load("nb_model.pkl")
scaler = joblib.load("scaler.pkl")


def predict_gender(
    long_hair,
    forehead_width,
    forehead_height,
    nose_wide,
    nose_long,
    thin_lips,
    distance_nose_to_lip_long
):

    input_data = [[
        long_hair,
        forehead_width,
        forehead_height,
        nose_wide,
        nose_long,
        thin_lips,
        distance_nose_to_lip_long
    ]]

    input_data = scaler.transform(input_data)

    prediction = nb_model.predict(input_data)[0]

    if prediction == 1:
        return "👨 MALE"
    else:
        return "👩 FEMALE"



css = """

.gradio-container{
    max-width:1100px !important;
    margin:auto !important;
    background:white !important;
}

footer{
    display:none;
}

h1{
    text-align:center;
    color:#ff7a00 !important;
    font-size:46px !important;
    font-weight:bold !important;
}


h2,h3,h4,p,li,span,strong{
    color:#222 !important;
}


button{
    background:#ff7a00 !important;
    color:white !important;
    border:none !important;
    border-radius:12px !important;
    font-size:18px !important;
    font-weight:bold !important;
}


button:hover{
    background:#e56d00 !important;
}


.result textarea{
    text-align:center !important;
    font-size:32px !important;
    font-weight:bold !important;
}


.info-card{

    background:#fff3e6;
    padding:20px;
    border-radius:16px;
    margin-bottom:20px;
    border-left:6px solid #ff7a00;

}


.info-card *{
    color:#222 !important;
}

"""


with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="orange"),
    css=css
) as demo:


    gr.Markdown("""
# 🧠 Gender Classification Using Machine Learning
""")


    gr.HTML("""

<div class="info-card">

<h3>🏆 Best Performing Model</h3>

<p><b>Naive Bayes</b></p>

<p>📊 Accuracy: <b>95.83%</b></p>

<p>
This application predicts gender based on facial characteristics
using the best-performing machine learning model after comparing
multiple classification algorithms.
</p>

</div>

""")


    gr.Markdown("""
### 👥 Team Members

- Enas Ibrahim Ali Elnsag
- Malak Tamer Mohamed Ali
- Salma Amer Ahmed Abdel Fattah
- Fatma Mohamed Helmy Mohamed
- Mariem Medhat Afifi
""")


    with gr.Row():

        long_hair = gr.Radio([0,1], label="Long Hair")

        forehead_width = gr.Number(
            label="Forehead Width (cm)"
        )

        forehead_height = gr.Number(
            label="Forehead Height (cm)"
        )



    with gr.Row():

        nose_wide = gr.Radio(
            [0,1],
            label="Nose Wide"
        )

        nose_long = gr.Radio(
            [0,1],
            label="Nose Long"
        )

        thin_lips = gr.Radio(
            [0,1],
            label="Thin Lips"
        )


        distance_nose_to_lip_long = gr.Radio(
            [0,1],
            label="Distance Nose To Lip Long"
        )



    predict_btn = gr.Button(
        "🚀 Predict Gender"
    )


    result = gr.Textbox(
        label="Prediction Result",
        lines=2,
        elem_classes="result"
    )


    predict_btn.click(
        fn=predict_gender,

        inputs=[

            long_hair,
            forehead_width,
            forehead_height,
            nose_wide,
            nose_long,
            thin_lips,
            distance_nose_to_lip_long

        ],

        outputs=result

    )



demo.launch()