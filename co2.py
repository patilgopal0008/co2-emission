import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from PIL import Image


# Load the dataset
@st.cache_data()
def load_data():
    return pd.read_csv(r"C:\Users\ADMIN\Downloads\co2_emission.csv")  

df = load_data()
fv = df.drop(columns=["CO2 Emissions(g/km)"])
cv = df["CO2 Emissions(g/km)"]

numeric_features = fv.select_dtypes(include=['float64', 'int64']).columns
categorical_features = fv.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore',sparse_output=False,drop='first'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

x_train,x_test,y_train,y_test = train_test_split(fv,cv,test_size = 0.2,random_state=7)
x_trainp = pipeline.fit_transform(x_train)
x_testp = pipeline.transform(x_test)

svm = SVR(kernel='rbf',gamma='scale',C=100)
model = svm.fit(x_trainp,y_train)
preeed = model.predict(x_testp)
R2_scr = r2_score(y_test,preeed)

background_image_url = "https://media.istockphoto.com/id/1135446074/photo/air-pollution-crisis-in-city-from-diesel-vehicle-exhaust-pipe-on-road.jpg?s=2048x2048&w=is&k=20&c=yyPvXvc_-249kyvdJH8vgYOZTSlZ9OAKIqONQTB-BVU="

background_image_style = f"""
    <style>
        .stApp {{
            background-image: url("{background_image_url}");
            background-size: cover;
            color: white;
        }}
    </style>
"""

# Display the background image and set text color using HTML and CSS
st.markdown(background_image_style, unsafe_allow_html=True)





st.markdown("<h1 style='color: white;'>CO2 Emissions Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='color: white;'>Input Features</h2>", unsafe_allow_html=True)

inputs = {}
for feature in fv.columns:
    if fv[feature].dtype == 'object':
        inputs[feature] = st.selectbox(feature, fv[feature].unique())
    else:
        inputs[feature] = st.text_input(feature)

# Function to preprocess input and make prediction
def preprocess_input(input_dict):
    input_df = pd.DataFrame([input_dict])
    input_transformed = pipeline.transform(input_df)
    return input_transformed

# Function to predict CO2 emissions
def predict_co2_emissions(input_data):
    prediction = model.predict(input_data)
    return prediction[0]



# Predict CO2 emissions for input data
if st.button("Predict CO2 Emissions") :
    input_transformed = preprocess_input(inputs)
    prediction = predict_co2_emissions(input_transformed)
    st.success(f"Predicted CO2 Emissions: {prediction:.2f} g/km")
