import pandas as pd

class SumScoreCalculator():
    def __init__(self, data, trained_pipeline):
        self.data = data
        self.trained_pipeline = trained_pipeline

    def calculate_sum_score(self):

        coef_df = self._make_coefficients_df()
        sum_score = self._make_sum_score(coef_df)
        
        return sum_score
    
    def _make_coefficients_df(self):
        
        estimator = self._get_estimator_from_pipeline(self.trained_pipeline)
        feature_names = self.data.columns

        estimator_base_model_name = self._get_base_model_name_from_estimator(estimator)

        # If model doesn't have coeffieicents, make values empty
        if estimator_base_model_name not in ["logisticregression", "svc"]:
            raise ValueError("Model doesn't have coefficients: ", estimator_base_model_name)

        if estimator_base_model_name == "logisticregression":
            coef = estimator.coef_[0]
        else:
            coef = estimator.coef_[0]
        
        # Make df with coefficients
        coef_df = pd.DataFrame({"feature": feature_names, "coef": coef})
        coef_df = coef_df.set_index("feature")
        
        return coef_df
    
    def _make_sum_score(self, coef_df):
        # Sum up responses from self.item_lvl to items with a positive coefficient, and subtract responses from items with a negative coefficient, 
        # ignore magnitude of coefficient
        to_add_up = coef_df[coef_df["coef"] > 0].index.tolist()
        to_subtract = coef_df[coef_df["coef"] < 0].index.tolist()

        # Drop items that have range of values > 6
        to_add_up = [item for item in to_add_up if self.data[item].max() - self.data[item].min() <= 6]
        to_subtract = [item for item in to_subtract if self.data[item].max() - self.data[item].min() <= 6]

        # Sum up responses
        sum_score = self.data[to_add_up].sum(axis=1) - self.data[to_subtract].sum(axis=1)
        
        return sum_score

    def _get_estimator_from_pipeline(self, pipeline):
        return pipeline.steps[-1][1]

    def _get_base_model_name_from_estimator(self, estimator):
        return estimator.__class__.__name__.lower()