import streamlit as st
import pickle
import numpy as np

# 1. Page Configuration and Custom CSS Styling (White Background, Bold Black Text, Rectangle Boxes)
st.set_page_config(page_title="AI Basketball Predictor", page_icon="🏀", layout="centered")

st.markdown("""
    <style>
    /* Force background color to white and all text to black/bold */
    .stApp {
        background-color: #FFFFFF !important;
    }
    h1, h2, h3, p, label, .stSelectbox div {
        color: #000000 !important;
        font-weight: bold !important;
        font-family: 'Arial', sans-serif;
    }
    /* Style the dropdown selector boxes to be sharp rectangles */
    div[data-baseweb="select"] > div {
        border-radius: 0px !important;
        border: 2px solid #000000 !important;
        background-color: #FFFFFF !important;
    }
    /* Primary Submit Button Custom Styling */
    div.stButton > button:first-child {
        background-color: #FFD700 !important;
        color: #000000 !important;
        font-weight: bold !important;
        border-radius: 0px !important;
        border: 2px solid #000000 !important;
        width: 100%;
        padding: 10px;
    }
    div.stButton > button:first-child:hover {
        background-color: #FFC000 !important;
        color: #000000 !important;
    }
    /* Custom colored labels for Made/Missed representation */
    .made-label {
        background-color: #22C55E;
        color: white;
        padding: 4px 8px;
        border-radius: 0px;
        font-weight: bold;
    }
    .missed-label {
        background-color: #EF4444;
        color: white;
        padding: 4px 8px;
        border-radius: 0px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Load the saved AI brain
with open("basketball_model.pkl", "rb") as file:
    model = pickle.load(file)

# 3. Main Interface Design
st.title("🏀 AI Basketball Shot Predictor")
st.write("Enter the results of a player's last 3 shots to calculate future streak probabilities.")
st.write("---")

st.subheader("Enter Shot History:")

# Dropping custom styled rectangle labels inside the selection menus
shot_a = st.selectbox("Shot 1 (Oldest)", options=[1, 0], format_func=lambda x: "🟢 Made It" if x == 1 else "🔴 Missed It")
shot_b = st.selectbox("Shot 2", options=[1, 0], format_func=lambda x: "🟢 Made It" if x == 1 else "🔴 Missed It")
shot_c = st.selectbox("Shot 3 (Most Recent)", options=[1, 0], format_func=lambda x: "🟢 Made It" if x == 1 else "🔴 Missed It")

st.write("---")

# 4. Prediction Execution, Sound Effects & Animations
if st.button("Predict Next Shot Probability 🚀"):
    input_data = np.array([[shot_a, shot_b, shot_c]])
    probabilities = model.predict_proba(input_data)
    make_probability = probabilities[0][1] * 100
    
    st.metric(label="Probability of Making the Next Shot", value=f"{make_probability:.1f}%")
    
    if make_probability >= 50:
        # Success Layout: Smiling face emoji, Streamlit native party poppers explosion, and celebratory sound
        st.markdown("<h1 style='text-align: center; font-size: 100px; margin: 0;'>😊🎉</h1>", unsafe_allow_html=True)
        st.success("YAY!! High probability! The math suggests they are in a rhythm.")
        st.balloons() # Native party popper explosion effect
        
        # Audio autoplay trick using open-source arcade win/yay sound effect
        st.components.v1.html(
            """
            <audio autoplay>
                <source src="https://mixkit.co" type="audio/wav">
            </audio>
            """,
            height=0
        )
    else:
        # Failure Layout: Sad face emoji, warning message, and low-frequency "nahh" error sound
        st.markdown("<h1 style='text-align: center; font-size: 100px; margin: 0;'>😞</h1>", unsafe_allow_html=True)
        st.error("NAHH!! Low probability! The streak data suggests a potential miss.")
        
        # Audio autoplay trick using open-source fail/nahh buzzer sound effect
        st.components.v1.html(
            """
            <audio autoplay>
                <source src="https://mixkit.co" type="audio/wav">
            </audio>
            """,
            height=0
        )