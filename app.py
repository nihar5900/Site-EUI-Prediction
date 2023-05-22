import streamlit as st
import numpy as np
from preprocess import process,get_prediction
import joblib

st.set_page_config(page_title="Site Energy Intensity Prediction App",
                   page_icon="🏘", layout="wide")

model=joblib.load(r'model/cat_tuned_model.pkl')
dictionary=joblib.load(r'data/dictionary.pkl')

option_facility_type=list(dictionary['facility_type'].keys())
option_state_factor=list(dictionary['state_factor'].keys())
option_building_class=list(dictionary['building_class'].keys())

st.markdown("<h1 style='text-align: center;'>Site Energy Intensity Prediction APP</h1>", unsafe_allow_html=True)
def main():
    with st.form('predictin_form'):
        st.subheader("Enter the below Features:")

        build_year=st.number_input("Building Year: ",1600,2015,format="%d")
        heating_degree_days=st.number_input("Heating Degree Days: ",398.00,7929.00,format="%.2f")
        snowfall_inches=st.number_input("Snow Fall(inches): ",0.00,128.30,format="%.2f")
        snowdepth_inches=st.number_input("Snow Depth(inches): ",0.00,1292.00,format="%.2f")
        mean_spring_temp=st.number_input("Standard Spring Temperaure: ",3.4,76.00,format="%.2f")
        std_spring_temp=st.number_input("Standard Spring Temperaure: ",12.00,30.00,format="%.2f")
        floor_energy_star_rating=st.number_input("Floor Energy Star Rating: ",-9.00,55.00,format="%.2f")

        facility_type=st.selectbox("Facility Type: ",options=option_facility_type)
        state_factor=st.selectbox("State Factor: ",options=option_state_factor)
        building_class=st.selectbox("Building Class: ",options=option_building_class)
        
        submit=st.form_submit_button("Predict")
    if submit:
        build_year=build_year
        heating_degree_days=heating_degree_days
        snowfall_inches=snowfall_inches
        snowdepth_inches=snowdepth_inches
        mean_spring_temp=mean_spring_temp
        std_spring_temp=std_spring_temp
        floor_energy_star_rating=floor_energy_star_rating

        facility_type=facility_type
        state_factor=state_factor
        building_class=building_class

        value_1=[build_year,heating_degree_days,snowfall_inches,snowdepth_inches,mean_spring_temp,std_spring_temp,floor_energy_star_rating]
        value_2=[facility_type,state_factor,building_class]
        value_3=process(value_2)

        final_values=value_1+value_3
        final_values=np.array([final_values]).reshape(1,18)


        st.write(f"The predicted severity is:  {get_prediction(data=final_values, model=model)[0]}")

if __name__=='__main__':
    main()