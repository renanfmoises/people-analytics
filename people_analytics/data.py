import pandas as pd
from google.cloud import storage

# from people_analytics.utils import simple_time_tracker
from people_analytics.transformers import *

PATH = '/Users/renanfmoises/code/renanfmoises/people-analytics/rs_xgboost.joblib'
FILE_PATH = "/Users/renanfmoises/code/renanfmoises/people-analytics/raw_data/people_analytics.csv"

# @simple_time_tracker
def get_data(path):
    """
    Function to get the training data locally or from google cloud bucket
    """
    # # client = storage.Client()
    # if local:
    #     path = FILE_PATH
    #     df = pd.read_csv(path)
    # else:
        # df = pd.read_csv(uploaded_file)
    df = pd.read_csv(path)
    return df

def clean_df(df):

    pipe = make_pipeline(
                            ColumnsRename(),
                            DropColumns(),
                            IncomeBlwDptJLAvg(),
                            EduFieldJobRole(),
                            PromotedLastTwoYears(),
                            IncomePerYearsWorked(),
                            IncomePerAge(),
                            SalaryHikeBelowMedian(),
                            MapBooleans(),
                            FinalColumnsDrop()
    )

    df = pipe.fit_transform(df)

    if 'attrition' in df.columns.tolist():
        df.drop('attrition', axis = 1, inplace = True)

    return df

def scale_data(df):

    to_stand_scale = [
    #     'attrition',
    #     'below_median_dpt_joblevel_monthly_income',
    #     'below_median_pct_salary_hike',
        'daily_rate',
        'distance_from_home',
        'education',
        'environment_satisfaction',
    #     'gender_male',
        'hourly_rate',
    #     'job_role_diff_edu_field',
        'job_satisfaction',
        'monthly_rate',
    #     'over_time',
    #     'promoted_last_2_years',
        'relationship_satisfaction'
    ]

    to_robust_scale = [
        'business_travel',
        'num_companies_worked',
        'stock_option_level',
        'training_times_last_year',
        'years_in_current_role',
        'years_with_curr_manager',
        'm_income_per_total_years_worked',
        'm_income_per_age',
    ]

    df[to_robust_scale] = RobustScaler().fit_transform(df[to_robust_scale])

    df[to_stand_scale] = StandardScaler().fit_transform(df[to_stand_scale])

    return df

if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)

    df = get_data()
    df = clean_df(df)
    df = scale_data(df)

    print(df.head(3))

