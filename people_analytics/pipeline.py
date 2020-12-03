

class PeoplePipe(self):
    def __init__(self, X):
        self.X = X

    def get_pipe(self):
        to_robust_scale = [
            'distance_from_home',
            'education',
            'num_companies_worked',
            'stock_option_level',
            'training_times_last_year',
            'years_in_current_role',
            'years_with_curr_manager',
            'm_income_per_total_years_worked',
            'm_income_per_age'
        ]


        to_stand_scale = [set(X.columns.tolist()) - set(to_robust_scale)]
        to_stand_scale = list(to_stand_scale[0])


        df_setup_pipe = make_pipeline(
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


        scaler_pipe = ColumnTransformer([
                                            ('robust_scaler', RobustScaler(), to_robust_scale),
                                            ('standard_scaler', StandardScaler(), to_stand_scale)
                                        ])


        final_pipe = Pipeline(steps = [
            ('df_setup', df_setup_pipe),
            ('scaler', scaler_pipe),
        ])

        return X

    def people_predict(self):

