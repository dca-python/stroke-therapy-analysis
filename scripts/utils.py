import numpy as np
import pandas as pd

def lowest_frequency_overview(df):
    """Takes a pd.DataFrame and returns the lowest frequency values of all columns,
    sorted by an indicator that weighs it with the amount of unique values,
    to inform the choice of logistic regression predictors"""
    result_df = pd.DataFrame(columns=['col_name', 'least_freq_val', 'fraction', 'n_unique'])
    for col in df.columns:
        value_counts = df[col].value_counts(normalize=True)
        min_val = value_counts.idxmin()
        fraction = round(value_counts.min() * 100, 3)
        n_unique = df[col].nunique()
        fraction_x_n_unique = round(value_counts.min() * n_unique, 3)
        row = pd.DataFrame.from_dict(
            {'col_name': [col],
             'least_freq_val': [min_val],
             'fraction': [fraction],
             'n_unique': [n_unique],
             'fraction_x_n_unique': [fraction_x_n_unique],
             })
        result_df = pd.concat([result_df, row])
        result_df = result_df.sort_values("fraction_x_n_unique")
        result_df = result_df.reset_index(drop=True).drop(columns=["fraction_x_n_unique"])

    return result_df

def format_p_value(p):
    if p < 0.1:
        return '{:.4f}*'.format(p)
    else:
        return '{:.4f}'.format(p)

def calculate_odds_ratios(X, result):
    odds_ratios = pd.DataFrame({
        # 'Variable': X.columns,
        # 'Log Odds': result.params,
        # 'Odds Ratio': result.params.apply(lambda x: round((np.exp(x)), 3)),
        'Odds Ratio in % Diff.': result.params.apply(lambda x: round((np.exp(x) - 1) * 100, 2)),
        'P-value': result.pvalues.apply(format_p_value),
    })
    return odds_ratios
