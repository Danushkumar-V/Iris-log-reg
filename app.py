import streamlit as st
from streamlit_lottie import st_lottie
import pandas as pd
import pickle
import requests
from PIL import Image


model = pickle.load(open('logreg.pkl','rb'))

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def dataframe(head):
    b=pd.DataFrame(head, columns= ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
    return b

setosa = Image.open('flavia-bon-Sw-tMGNlU3o-unsplash.jpg')
vericolor = Image.open('iris_versicolor.jpg')
virginica = Image.open('Iris_virginica.jpg')

lottie_hello = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_vw0rgf1c.json")

st.title('IRIS dataset `Logistic regression`')

st_lottie(
        lottie_hello,
        speed=1,
        reverse=True,
        loop=True,
        quality="low", # medium ; high# canvas
        height=400,
        width=400,
        key=None,
    )

st.write(
    """
    # Let's predict !
    """
)
#'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'

SepalLengthCm = st.text_input('Enter the sepal length (CM):',0.0)
SepalWidthCm = st.text_input('Enter the sepal width (CM):',0.0)
PetalLengthCm = st.text_input('Enter the petal length (CM):',0)
PetalWidthCm = st.text_input('Enter the petal width (CM):',0)

data = [[SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]]
newdf = dataframe(data)

predict_value = model.predict(newdf)
result = st.button("Predict")
if result :
    if predict_value == "Iris-setosa":
        st.image(setosa, caption='Iris-setosa')
    elif predict_value == "Iris-versicolor":
        st.image(vericolor, caption = 'Iris-versicolor')
    elif predict_value == "Iris-virginica":
        st.image(virginica, caption = 'Iris-virginica')
