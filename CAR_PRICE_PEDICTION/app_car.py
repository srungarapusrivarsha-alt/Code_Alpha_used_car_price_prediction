import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor 
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import numpy as np
data=pd.read_csv(r"C:\Users\sriva\Downloads\car data.csv")

data['Fuel_Type']=data['Fuel_Type'].map({'Petrol':0, 'Diesel':1, 'CNG':2})
data['Transmission']=data['Transmission'].map({'Manual':00, 'Automatic':11})
data['Selling_type']=data['Selling_type'].map({'Dealer':1, 'Individual':2})

x=data[['Year','Present_Price', 'Driven_kms','Fuel_Type', 'Selling_type', 'Transmission', 'Owner']]
y=data['Selling_Price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)

model=XGBRegressor()

model.fit(x_train,y_train)

train_output=model.predict(x_train)

test_output = model.predict(x_test)

train_mae=mean_absolute_error(y_train,train_output)
train_mse=mean_squared_error(y_train,train_output)
train_rmse=np.sqrt(train_mse)
train_r2=r2_score(y_train,train_output)

test_mae = mean_absolute_error(y_test, test_output)
test_mse = mean_squared_error(y_test, test_output)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, test_output)


st.title("🚗 CAR PRICE PREDICTION")

st.write("Predict the selling price of a used car.")

st.header("🚘 Enter Car Details")


year = st.number_input("Manufacturing Year",min_value=1990,max_value=2026,value=2018,step=1)


present_price = st.number_input("Present Price (Lakhs)",min_value=0.1,value=5.0)


driven_kms = st.number_input("Driven Kilometers",min_value=0,value=30000,step=30)

fuel_type = st.number_input("Fuel Type (Petrol = 0, Diesel = 1, CNG = 2)", min_value=0, max_value=2,value=0,step=1)

selling_type = st.number_input("Selling Type (Dealer = 1, Individual = 2)",min_value=1,max_value=2,value=1,step=1)

transmission = st.number_input("Transmission (Manual = 0, Automatic = 1)",min_value=0,max_value=1,value=0,step=1
)

owner = st.number_input("Previous Owners",min_value=0, max_value=3, value=0, step=1
)

if st.button("Predict Car Price"):

    input_data = pd.DataFrame(
        [[
            year,
            present_price,
            driven_kms,
            fuel_type,
            selling_type,
            transmission,
            owner
        ]],
        columns=x.columns
    )

    prediction = model.predict(input_data)

    st.success("Predicted Car Price")

    st.subheader(
        f"₹ {prediction[0]:.2f} Lakhs"
    )