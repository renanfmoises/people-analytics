import joblib
import warnings

from people_analytics.data import get_data, clean_df, scale_data

# PATH = '/Users/renanfmoises/code/renanfmoises/people-analytics/rs_xgboost.joblib'

MODEL_LOCAL_PATH = '/Users/renanfmoises/code/renanfmoises/people-analytics/final_model_softvoting_logr_rf_xgb.joblib.joblib'

MODEL_RELATIVE_PATH = './final_model_softvoting_logr_rf_xgb.joblib'

def get_model():
    model = joblib.load(MODEL_RELATIVE_PATH)
    return model

def predict():
    '''
    Load .joblib model from file and predict according to new data
    '''

    y_pred = model.predict(df)
    print(y_pred)

    return y_pred

if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)

    df = get_data()
    df = clean_df(df)
    df = scale_data(df)
    model = get_model()

    y_pred = model.predict(df)
    print(y_pred)
