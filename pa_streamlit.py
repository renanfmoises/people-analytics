# import joblib

import pandas as pd
import numpy as np
import pytz
import streamlit as st

# from people_analytics.data import get_data, clean_df, scale_data
# from people_analytics.predict import get_model, predict

st.markdown('# PEOPLE ANALYTICS')



# # CSV Uploader
# st.set_option('deprecation.showfileUploaderEncoding', False)

# uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# if uploaded_file is not None:
#     # data = pd.read_csv(uploaded_file)
#     # st.write(data)

#     df = get_data(path = uploaded_file)
#     df_copy = df.copy()

#     st.write(df)

#     df_cleaned = clean_df(df)
#     df_scaled = scale_data(df_cleaned)

#     # Call model from .joblib file
#     model = get_model()

#     y_pred = model.predict(df_scaled)


#     threshold = .12
#     predicted_proba = model.predict_proba(df_scaled)
#     predicted = (predicted_proba[:, 1] >= threshold).astype('int')

#     st.write(pd.Series(predicted).value_counts())
#     y_pred_df = pd.DataFrame(predicted).rename(columns = {0: 'Attrition'})


#     y_pred_df = y_pred_df['Attrition'].map({1: 'True', 0: 'False'})

#     df_merged = df_copy[['EmployeeNumber']].merge(y_pred_df, left_index = True, right_index = True)
#     df_merged_attr = df_merged[df_merged['Attrition'] == 'True']


#     st.write(df_merged_attr)




