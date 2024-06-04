import pandas as pd
import streamlit as st
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import io


page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {   
  background-image: url("https://i.ibb.co/L884RFs/Untitled-design-38.png");
  background-size: auto 70%;
  background-repeat: no-repeat;
  background-position: right;
  background-color: black;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)
# Streamlit app title
st.write("""
# Pokémon Dataset Analysis
Upload a dataset and adjust the parameters to predict if a Pokémon is Legendary or not.
""")

# Sidebar for user input parameters
st.sidebar.header('User Input Parameters')


def user_input_features():
    total = st.sidebar.slider('Total', 180, 780, 318)
    hp = st.sidebar.slider('HP', 1, 255, 45)
    attack = st.sidebar.slider('Attack', 5, 190, 49)
    defense = st.sidebar.slider('Defense', 5, 230, 49)
    sp_atk = st.sidebar.slider('Sp. Atk', 10, 194, 65)
    sp_def = st.sidebar.slider('Sp. Def', 20, 230, 65)
    speed = st.sidebar.slider('Speed', 5, 180, 45)
    generation = st.sidebar.slider('Generation', 1, 6, 1)
    data = {
        'Total': total,
        'HP': hp,
        'Attack': attack,
        'Defense': defense,
        'Sp. Atk': sp_atk,
        'Sp. Def': sp_def,
        'Speed': speed,
        'Generation': generation
    }
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()

st.subheader('User Input Parameters')
st.write(df)

# File uploader for dataset
upl = st.file_uploader('Upload Pokémon dataset (CSV)', type=['csv'])

if upl is not None:
    target = pd.read_csv(upl)
    st.write('Dataset:')
    st.write(target.head())

    # Ensure the necessary columns are present
    required_columns = ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation', 'Legendary']
    if all(col in target.columns for col in required_columns):

        # Separate features and target
        X = target[required_columns[:-1]]
        y = target['Legendary'].astype(int)  # Convert boolean to integer for compatibility

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Build the pipeline
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('reducer', PCA(n_components=0.9)),
            ('classifier', RandomForestClassifier(random_state=42))
        ])

        # Fit the pipeline to the training data
        pipe.fit(X_train, y_train)

        # Evaluate the model
        accuracy = pipe.score(X_test, y_test)
        st.write(f'Accuracy on the test set: {accuracy:.1%}')

        # Make predictions on the user input features
        prediction = pipe.predict(df)
        prediction_proba = pipe.predict_proba(df)

        # Display prediction results
        st.subheader('Prediction')
        st.write('Legendary' if prediction[0] == 1 else 'Not Legendary')

        st.subheader('Prediction Probability')
        st.write(prediction_proba)

        # Add predictions to the original DataFrame
        target['Prediction'] = pipe.predict(X)

    else:
        st.error(f"Please ensure your CSV file contains the following columns: {', '.join(required_columns)}")
else:
    st.write("Please upload a CSV file to proceed.")


