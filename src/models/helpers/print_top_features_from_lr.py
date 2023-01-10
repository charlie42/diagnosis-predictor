import pandas as pd

# To import from parent directory
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import util

def print_top_features_from_lr(pipeline, data, n):
    # Get the coefficients from classifier 
    classifier = util.get_estimator_from_pipeline(pipeline)
    coef = classifier.coef_[0]
    # Get the feature names
    feature_names = data.columns
    # Create a dataframe of the coefficients and feature names
    df = pd.DataFrame({"coef": coef, "feature": feature_names})
    # Sort the dataframe by the coefficients
    df = df.sort_values(by="coef", ascending=False)
    # Print the top n features
    print(df.head(n))
    print(df.tail(n))