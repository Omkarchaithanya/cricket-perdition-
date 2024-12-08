import streamlit as st
import pickle
import pandas as pd

# Teams and Cities
teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
         'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
         'Rajasthan Royals', 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

# Load the model pipeline
try:
    pipe = pickle.load(open('pipe.pkl', 'rb'))
except FileNotFoundError:
    st.error("The file 'pipe.pkl' is missing. Ensure it exists in the script's directory.")
    st.stop()

# App Title
st.title('IPL Win Predictor')

# Columns for inputs
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

selected_city = st.selectbox('Select host city', sorted(cities))

target = st.number_input('Target', min_value=1)

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score', min_value=0)
with col4:
    overs = st.number_input('Overs completed', min_value=0.1, max_value=20.0, step=0.1)
with col5:
    wickets = st.number_input('Wickets out', min_value=0, max_value=10, step=1)

# Predict Button
if st.button('Predict Probability'):
    # Validate inputs
    if score > target:
        st.error("Score cannot exceed the target.")
    else:
        # Calculate derived metrics
        runs_left = target - score
        balls_left = 120 - (overs * 6)
        if balls_left <= 0:
            st.error("Overs completed cannot exceed 20.")
            st.stop()
        
        wickets_left = 10 - wickets
        crr = score / overs
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

        # Prepare input DataFrame
        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets_left],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        # Model input validation
        required_features = ['batting_team', 'bowling_team', 'city', 'runs_left', 
                              'balls_left', 'wickets', 'total_runs_x', 'crr', 'rrr']
        if not all(feature in input_df.columns for feature in required_features):
            st.error("Model input features mismatch. Check the model training script.")
            st.stop()

        # Predict probabilities
        try:
            result = pipe.predict_proba(input_df)
            loss = result[0][0]
            win = result[0][1]

            # Display probabilities
            st.header(f"{batting_team} - {round(win * 100)}%")
            st.header(f"{bowling_team} - {round(loss * 100)}%")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
