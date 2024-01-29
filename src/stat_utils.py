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

def bootstrap(data, num_repeats, session_id=9):
    """
    Bootstrap the data.

    Parameters
    ----------
    data : pandas.DataFrame
        The data.
    num_repeats : int
        The number of repeats.
    num_items : int, optional
        The number of items in each sample. The default is 25.
    session_id : int, optional
        The session ID. The default is 9.

    Returns
    -------
    mean_diff : array_like
        The distribution of mean differences.
    """
    session_data = data[data['SessionID'] == session_id]
    mean_diff = np.zeros(num_repeats)
    num_items = session_data['BlockID'].nunique()
    for i in range(num_repeats):
        sample_1 = session_data.groupby(['SubjectID', 'Condition']).sample(n=num_items, replace=True)
        sample_1 = sample_1.groupby(['SubjectID', 'Condition']).mean().reset_index()
        sample_2 = session_data.groupby(['SubjectID', 'Condition']).sample(n=num_items, replace=True)
        sample_2 = sample_2.groupby(['SubjectID', 'Condition']).mean().reset_index()
        mean_diff[i] = (sample_1['Correct'] - sample_2['Correct']).mean()


    return mean_diff