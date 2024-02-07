import numpy as np
from prettytable import PrettyTable

def print_wald_chi_square(results):
    """
    Prints a table of Wald Chi-Square statistics for each variable in the model.

    Parameters
    ----------
    results : statsmodels.regression.linear_model.RegressionResultsWrapper
        The results of the GEE model.
    """
    print('Wald Chi-Square:')
    table = PrettyTable()
    table.field_names = ['Variable', 'Chi-Square', 'p-value']
    for var in results.model.exog_names:
        table.add_row([var, results.wald_test(var, scalar=True).statistic, results.wald_test(var, scalar=True).pvalue])
    print(table)

def print_sample_info(metadata):
    """
    Prints information about the sample.

    Parameters
    ----------
    metadata : pandas.DataFrame
        The metadata of the sample.
    """
    num_samples = metadata.shape[0]
    num_females = metadata['Sex'].value_counts()[' F']
    mean_age = metadata['Age'].mean()
    std_age = metadata['Age'].std().__round__(3)

    print(f'{num_samples} particpants ({num_females} female, mean age = {mean_age}, standard deviation = {std_age})')