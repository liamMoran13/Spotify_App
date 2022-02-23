import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


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

X = df[['Track_Number','Total_Tracks','Year_Released','Month_Released']]
y = df['Popularity']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.25,random_state=13)

lr = LinearRegression()
lr.fit(X_train, y_train)

order_on_album = st.slider('Order of Song on Album', int(df['Track_Number'].min()), int(df['Track_Number'].max()),1)
number_of_songs_on_album = st.slider('Number of Songs on Album', int(df['Total_Tracks'].min()),int(df['Total_Tracks'].max()),1)

year = st.number_input('Year Released', value=2017)
month = st.number_input('Month Released (in integer form)', value=1)

input_data = pd.DataFrame({'Track_Number':[order_on_album], 'Total_Tracks':[number_of_songs_on_album], 'Year_Released':[year],'Month_Released':[month]})
pred = lr.predict(input_data)[0]

st.write(
f'Predicted Popularity of Song: {int(pred):,}'
)

