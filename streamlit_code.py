import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
import pickle

with open('lasso_model.pickle','rb') as modelFile:
     model = pickle.load(modelFile)

st.write(
'''
## Spotify Popularity Predictor

In this app, you will be able to enter details for a particular song and then receive the projected popularity for that song.

''')

st.write(
'''
## Original DataFrame

We are pulling our data to build the model from the Spotify API

''')

data = pd.read_csv('df_MVP.csv')
st.dataframe(data)


st.write(
'''
## Datapoint Inputs

In this section, input your datapoints.

''')

df = pd.read_csv('cleaned_data.csv')

X= [ 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

poly = PolynomialFeatures(degree=2)

target = df['Popularity']
features = df.drop('Popularity',axis=1)
features.drop('Unnamed: 0',axis=1,inplace=True)
st.write(features.columns)
scaler = StandardScaler()
scaler.fit(features)

poly = PolynomialFeatures(degree=2)
X_poly_features = poly.fit_transform(features)


order_on_album = st.slider('Track Number on Album', df['Track_Number'].min(), df['Track_Number'].max(),1)
number_of_songs_on_album = st.slider('Number of Songs on Album', df['Total_Tracks'].min(),df['Total_Tracks'].max(),1)
length_of_song = st.number_input('Length of Song in milliseconds', value = 200)

year = st.number_input('Year Released', value=2017)
month = st.number_input('Month Released (in integer form)', value=1)
mode = st.number_input('Mode',value=1)

danceability = st.slider('Danceability', df['danceability'].min(), df['danceability'].max(),.01)
energy = st.number_input('Energy',value=.6)
key = st.number_input('Key',value=1)
loudness = st.slider('Loudness', df['loudness'].min(), df['loudness'].max(),.01)
speechiness = st.slider('Speechiness', df['speechiness'].min(), df['speechiness'].max(),.01)
acousticness = st.slider('Acousticness', df['acousticness'].min(), df['acousticness'].max(),.01)
instrumentalness = st.slider('Instrumetalness', df['instrumentalness'].min(), df['instrumentalness'].max(),.01)
liveness = st.slider('Liveness', df['liveness'].min(), df['liveness'].max(),.01)
valence = st.slider('Valence', df['valence'].min(), df['valence'].max(),.01)
tempo = st.slider('Tempo', df['tempo'].min(), df['tempo'].max(),.01)

milli = length_of_song*1000


input_data = pd.DataFrame({'Danceability':[danceability],'Energy':[energy],'Key':[key],'Loudness':[loudness],'mode':[mode],'Speechiness':[speechiness],'acousticness':[acousticness],'instrumentalness':[instrumentalness],'liveness':[liveness],'valence':[valence],'tempo':[tempo],'duration':[length_of_song],'Track_Number':[order_on_album], 'Total_Tracks':[number_of_songs_on_album],'Month_Released':[month], 'Year_Released':[year]})
input_data = scaler.transform(input_data)
poly_data = poly.transform(input_data)

pred = model.predict(poly_data)[0]


st.write(
f'Predicted Popularity of Song: {int(pred):,}'
)

