import streamlit as st
import pandas as pd
import joblib
import base64


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)


# =========================================================
# BACKGROUND IMAGE
# =========================================================

def set_background(image_file):

    with open(image_file, "rb") as file:
        encoded = base64.b64encode(
            file.read()
        ).decode()

    css = f"""
    <style>

    .stApp {{
        background-image:
        linear-gradient(
            rgba(0,0,0,0.60),
            rgba(0,0,0,0.60)
        ),
        url("data:image/jpg;base64,{encoded}");

        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    .main-title {{
        font-size: 45px;
        font-weight: bold;
        text-align: center;
        color: white;
    }}

    .subtitle {{
        font-size: 20px;
        text-align: center;
        color: white;
        margin-bottom: 30px;
    }}

    .prediction-box {{
        padding: 25px;
        border-radius: 15px;
        background-color: rgba(255,255,255,0.90);
        text-align: center;
        margin-top: 25px;
    }}

    </style>
    """

    st.markdown(
        css,
        unsafe_allow_html=True
    )


set_background("titanic.jpg")


# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load(
    "titanic_best_model.pkl"
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🚢 Titanic Survival Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning Based Survival Prediction</div>',
    unsafe_allow_html=True
)


# =========================================================
# PASSENGER INFORMATION
# =========================================================

st.subheader("Passenger Information")


col1, col2 = st.columns(2)


with col1:

    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3]
    )

    sex = st.selectbox(
        "Sex",
        ["male", "female"]
    )

    age = st.number_input(
        "Age",
        min_value=0.0,
        max_value=100.0,
        value=30.0
    )

    sibsp = st.number_input(
        "Siblings / Spouses",
        min_value=0,
        max_value=10,
        value=0
    )


with col2:

    parch = st.number_input(
        "Parents / Children",
        min_value=0,
        max_value=10,
        value=0
    )

    fare = st.number_input(
        "Fare",
        min_value=0.0,
        max_value=600.0,
        value=30.0
    )

    embarked = st.selectbox(
        "Port of Embarkation",
        ["S", "C", "Q"]
    )


# =========================================================
# PREDICTION
# =========================================================

if st.button(
    "Predict Survival",
    use_container_width=True
):

    input_data = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [sex],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Embarked": [embarked]
    })


    prediction = model.predict(
        input_data
    )[0]


    probability = model.predict_proba(
        input_data
    )[0]


    survival_probability = probability[1]

    death_probability = probability[0]


    st.markdown(
        '<div class="prediction-box">',
        unsafe_allow_html=True
    )


    if prediction == 1:

        st.success(
            "Passenger is predicted to SURVIVE."
        )

    else:

        st.error(
            "Passenger is predicted NOT TO SURVIVE."
        )


    st.write(
        f"Survival Probability: "
        f"{survival_probability:.2%}"
    )


    st.write(
        f"Not Survival Probability: "
        f"{death_probability:.2%}"
    )


    st.progress(
        float(survival_probability)
    )


    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Titanic Survival Prediction | "
    "Machine Learning + Streamlit"
)
