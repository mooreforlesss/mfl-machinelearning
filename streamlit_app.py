import streamlit as st
import pandas as pd

st.title('Machine Learning App')

st.write('This app builds a machine learning model')

with st.expander('Data'):
  st.write('**Raw Data')
  df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
  df

  st.write('**X**')
  X = df.drop('species', axis=1)
  X

  st.write('**y**')
  y = df.species
  y

with st.expander('Data visualization'):
  st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')

#data preparations
with st.sidebar:
  st.header('Input Features')
  island = st.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
  bill_length = st.slider('Bill Length (mm)', 32.1, 59.6, 43.5)
  bill_depth = st.slider('Bill Depth (mm)', 13.1, 21.5, 17.2)
  flipper_length = st.slider('Flipper Lenght (mm)', 172.0, 231.0, 201.0)
  body_mass = st.slider('Body Mass (g)', 2700.0, 6300.0, 4207.0)
  gender = st.selectbox('Gender', ('male', 'female'))

  #Create a DataFrame for the input features
  data = {'island': island,
          'bill_length': bill_length,
          'bill_depth': bill_depth,
          'flipper_length': flipper_length,
          'body_mass': body_mass,
          'sex': gender}
  input_df = pd.DataFrame(data, index=[0])
  input_penguins = pd.concat([input_df, X], axis=0)

with st.expander('Input Features'):
  st.write('**Input Penguin***')
  input_df
  st.write('***Combined penguines data***')
  input_penguins

#Encode
encode = ['island', 'sex']
df_penguins = pd.get_dummies(input_penguins[', prefix=encode)

