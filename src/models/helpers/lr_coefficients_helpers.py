import pandas as pd

# To import from parent directory
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import util

def get_coefficients_df_from_lr(pipeline, data):
    # Get the coefficients from estimator 
    estimator = util.get_estimator_from_pipeline(pipeline)
    coef = estimator.coef_[0]
    # Get the feature names
    feature_names = data.columns
    # Create a dataframe of the coefficients and feature names
    df = pd.DataFrame({"coef": coef, "feature": feature_names})
    # Sort the dataframe by the coefficients
    df = df.sort_values(by="coef", ascending=False)
    # Remove 0 coefficients
    df = df[df["coef"] != 0]
    return df

def print_top_features_from_lr(pipeline, data, n):
    # Create a dataframe of the coefficients and feature names
    df = get_coefficients_df_from_lr(pipeline, data)
    # Sort the dataframe by the coefficients
    df = df.sort_values(by="coef", ascending=False)
    # Print the top n features
    print(df.head(n))
    print(df.tail(n))

def save_coefficients_from_lr(diag, pipeline, data, output_dir):
    # Create a dataframe of the coefficients and feature names
    df = get_coefficients_df_from_lr(pipeline, data)
    # Sort the dataframe by the coefficients
    df = df.sort_values(by="coef", ascending=False)
    # Save to file
    ## Create directory if it doesn't exist
    coef_dir = output_dir + "coefficients/"
    util.create_dir_if_not_exists(coef_dir)
    ## Save to file
    df.to_csv(coef_dir + f'{diag}_coefficients.csv', index=False)