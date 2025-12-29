from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
from sklearn.datasets import fetch_california_housing
st.title('🏠House Price prediction using ML')

st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRvQdqIasHkDTf5733FK14z5mPQ18VPhg_R_Q&s')

data = fetch_california_housing()

X = pd.DataFrame(data['data'],
            columns = data['feature_names'])

final_X = X.iloc[:,:-2]
scaler = StandardScaler()
scaled_X = scaler.fit_transform(final_X)

st.sidebar.title('select house features: ')
st.sidebar.image('https://cdn.dribble.com/userupload/20000742/file/original-aaf23458355a15
all_value = []
for i in final_X:
  min_value = final_X[i].min()
  max_value = final_X[i].max()
  result = st.sidebar.slider(f'select {i} value',min_value.max_value)
  all_value.append(result)

user_X = scaler.transform([all_values])

@st.cache_data
def ml_model(X,y):
  model = RandomForestRegressor()
  model.fit(scaled_X,y)
  return model

model = ml_model(scaaled_X,y)
House_price = model.predict(user_X)[0]

final_price = round(house_price = 100000,2)

with st.spinner('predicting House Price'):
   import time
   time.sleep(2)

st.success(f'Estimated House Price is: $ {final_price}')
st.markdown('''**Design and Developed by: YASHI SHUKLA**''')
