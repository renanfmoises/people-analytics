# Basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, \
                                          classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from collections import Counter
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from scipy.stats import loguniform
import re

# Preprocessing
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV


class ColumnsRename(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        assert isinstance(X, pd.DataFrame)

        def camel_case_split(str):
            return '_'.join(re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', str))

        new_columns = []
        for str in X.columns.tolist():
            new_columns.append(camel_case_split(str).lower())

        X.columns = new_columns

        return X


class DropColumns(BaseEstimator, TransformerMixin):
    '''
    Searches for employee department and job level and classify its monthly income
    as below his/hers department average for his/hers specific job level.
    '''

    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        assert isinstance(X, pd.DataFrame)

        X.drop(columns = ['employee_count', 'over', 'standard_hours', 'employee_number'], inplace = True)

        return X


class IncomeBlwDptJLAvg(BaseEstimator, TransformerMixin):
    '''
    Searches for employee department and job level and classify its monthly income
    as below his/hers department average for his/hers specific job level.
    '''

    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        assert isinstance(X, pd.DataFrame)

        _minc_gb = X.groupby(['department', 'job_level'])['monthly_income'].median()

        departments = X['department'].unique().tolist()
        for department in departments:
            job_levels = X.loc[X['department'] == department, 'job_level'].unique().tolist()

            for job_level in job_levels:
                    X.loc[((X['department'] == department) & \
                           (X['job_level'] == job_level)), \
                              'below_median_dpt_joblevel_monthly_income'] = 0

                    X.loc[((X['department'] == department) & \
                           (X['job_level'] == job_level) & \
                           (X['monthly_income'] < _minc_gb[department, job_level])), \
                              'below_median_dpt_joblevel_monthly_income'] = 1

#                 X['below_median_dpt_joblevel_monthly_income'].astype(int)

        return X


class EduFieldJobRole(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        assert isinstance(X, pd.DataFrame)

        X['edu_field_job_role'] = X['education_field'] + '_' + X['job_role']

        edu_field_job_role_map = {
         'Human Resources_Human Resources': 0,
         'Human Resources_Manager': 0,
         'Life Sciences_Healthcare Representative': 0,
         'Life Sciences_Human Resources': 1,
         'Life Sciences_Laboratory Technician': 0,
         'Life Sciences_Manager': 1,
         'Life Sciences_Manufacturing Director': 0,
         'Life Sciences_Research Director': 0,
         'Life Sciences_Research Scientist': 0,
         'Life Sciences_Sales Executive': 1,
         'Life Sciences_Sales Representative': 1,
         'Marketing_Manager': 0,
         'Marketing_Sales Executive': 0,
         'Marketing_Sales Representative': 0,
         'Medical_Healthcare Representative': 0,
         'Medical_Human Resources': 1,
         'Medical_Laboratory Technician': 0,
         'Medical_Manager': 1,
         'Medical_Manufacturing Director': 0,
         'Medical_Research Director': 0,
         'Medical_Research Scientist': 0,
         'Medical_Sales Executive': 1,
         'Medical_Sales Representative': 1,
         'Other_Healthcare Representative': 0,
         'Other_Human Resources': 0,
         'Other_Laboratory Technician': 0,
         'Other_Manager': 0,
         'Other_Manufacturing Director': 0,
         'Other_Research Director': 0,
         'Other_Research Scientist': 0,
         'Other_Sales Executive': 0,
         'Other_Sales Representative': 0,
         'Technical Degree_Healthcare Representative': 0,
         'Technical Degree_Human Resources': 0,
         'Technical Degree_Laboratory Technician': 0,
         'Technical Degree_Manager': 0,
         'Technical Degree_Manufacturing Director': 0,
         'Technical Degree_Research Director': 0,
         'Technical Degree_Research Scientist': 0,
         'Technical Degree_Sales Executive': 0,
         'Technical Degree_Sales Representative': 0
        }

        X['job_role_diff_edu_field'] = X['edu_field_job_role'].map(edu_field_job_role_map)

        X.drop('edu_field_job_role', axis = 1, inplace = True)

        return X

class PromotedLastTwoYears(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):

        X['promoted_last_2_years'] = \
        X['years_since_last_promotion'].apply(lambda x: 1 if x <= 2 else 0)

        return X

class IncomePerYearsWorked(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        assert isinstance(X, pd.DataFrame)

        total_workin_years_min = X[X['total_working_years'] == 0]['monthly_income'].min()

        def get_income_per_years_worked(x):
            if x == 0:
                X['m_income_per_total_years_worked'] = total_workin_years_min
            else:
                X['m_income_per_total_years_worked'] = \
                        X['monthly_income'] / X['total_working_years']

        X['total_working_years'].apply(get_income_per_years_worked)

        X.loc[X['m_income_per_total_years_worked'] == np.inf, 'm_income_per_total_years_worked'] = 1
        X['m_income_per_total_years_worked'].replace(np.inf, total_workin_years_min, inplace = True)

        return X


class IncomePerAge(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        assert isinstance(X, pd.DataFrame)
        X['m_income_per_age'] = X['monthly_income'] / X['age']

        return X


class SalaryHikeBelowMedian(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        assert isinstance(X, pd.DataFrame)

        salary_hike_median = X['percent_salary_hike'].median()

        X['below_median_pct_salary_hike'] = \
                X['percent_salary_hike'].apply(lambda x: 1 if x < salary_hike_median else 0)

        return X


class MapBooleans(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        assert isinstance(X, pd.DataFrame)

        # X['attrition'] = X['attrition'].map({'No': 0, 'Yes': 1}) # COME BACK TO THIS IN THE FUTURE!!!

        X['gender_male'] = X['gender'].map({'Female': 0, 'Male': 1})

        X['over_time'] = X['over_time'].map({'No': 0, 'Yes': 1})

        business_travel_map = {
                                'Non-Travel': 0,
                                'Travel_Rarely': 1,
                                'Travel_Frequently': 2
                               }

        X['business_travel'] = X['business_travel'].map(business_travel_map)

        X = X.drop('gender', axis = 1)

        return X


class FinalColumnsDrop(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        assert isinstance(X, pd.DataFrame)

        X = X.drop(columns = [
                               'job_level',
                               'monthly_income',
                               'years_since_last_promotion',
                               'total_working_years',
                               'percent_salary_hike',
                               'years_at_company',
                               'performance_rating',
                               'department', # not numeric
                               'education_field', # not numeric
                               'job_role',
                               'marital_status',
                               'age',
                               'job_involvement',
                               'work_life_balance',
                         ])
        return X


