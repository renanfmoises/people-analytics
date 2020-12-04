import joblib

import pandas as pd
import numpy as np
import pytz
import streamlit as st

from people_analytics.data import get_data, clean_df, scale_data
from people_analytics.predict import get_model, predict

from PIL import Image
image = Image.open('./images/people_header.png')
st.image(image, use_column_width=True)

st.markdown(
    '''
    # BEST PHARMA - *PEOPLE*
    '''
    )

st.markdown(
    '''
    **Attrition prediction within company.**

    ___

    Please provide a valid **.csv** file. As per downloaded from Best Pharma People software.


    '''
    )


# CSV Uploader
st.set_option('deprecation.showfileUploaderEncoding', False)

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    #Get data from uploaded file
    df = get_data(path = uploaded_file)

    # Create a copy for retrieval, if users wants so
    df_copy = df.copy()

    # st.write(df)

    # Clean data, passing it through the pipeline
    df_cleaned = clean_df(df)

    # Scale data with RobustScaler and StandardScaler
    df_scaled = scale_data(df_cleaned)

    # Call model from .joblib file and save it into 'model' variable
    model = get_model()

    # Use model to predict preprocessed data and save it to y_pred model
    y_pred = model.predict(df_scaled)


    # Implement best threshold with predic_proba
    threshold = .30
    predicted_proba = model.predict_proba(df_scaled)
    predicted = (predicted_proba[:, 1] >= threshold).astype('int')


    # Pass predicted to a pd.DataFrame for easy presentation within Streamlit
    y_pred_df = pd.DataFrame(predicted).rename(columns = {0: 'PredAttrition'})


    # Map target variable to str for better readability
    y_pred_df = y_pred_df['PredAttrition'].map({1: 'True', 0: 'False'})

    # Merge y_pred DF with data, for acessing employee information
    df_merged = df_copy.merge(y_pred_df, left_index = True, right_index = True)


    # Filter data for showing only employees where Attrition is True
    df_merged_attr = df_merged[df_merged['PredAttrition'] == 'True']

    # Select and rearrenge cols for better presentation in Streamlit
    cols_to_streamlit_show = [
        'EmployeeNumber',
        'Age',
        'JobRole',
        'Department',
        'MonthlyIncome',
        'OverTime',
        'YearsAtCompany'
    ]

    st.markdown('''
        ---
        ## Results from analysis:
        ''')

    st.warning('⚠️ The following employees may leave Best Pharma in the next few months ⚠️')

    st.write(df_merged_attr[cols_to_streamlit_show].assign(hack = '').set_index('hack'))


    # st.write(pd.Series(predicted).value_counts())




