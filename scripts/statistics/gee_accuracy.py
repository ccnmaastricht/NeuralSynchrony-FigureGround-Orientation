"""
This script performs generalized estimating equations (GEE) analysis on the behavioral data.
Results are saved in results/statistics/ and correspond to section X of the paper.
"""
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from src.anl_utils import load_data, get_session_data

def is_significant(results, variable, cutoff=0.05):
    """
    Check if a variable is significant.

    Parameters
    ----------
    results : statsmodels.genmod.generalized_estimating_equations.GEEResultsWrapper
        The results of the GEE analysis.
    variable : str
        The variable of interest.
    cutoff : float, optional
        The cutoff for the p-value. The default is 0.05.
        
    Returns
    -------
    significant : bool
        True if the variable is significant, False otherwise.
    """
    # use Wald test p-value
    pvalue = results.wald_test(variable, scalar=True).pvalue
    return pvalue < cutoff


if __name__ == '__main__':
    # Load data
    data_path = 'data/main.csv'
    try:
        data = load_data(data_path)
    except FileNotFoundError:
        print(f"Data file not found: {data_path}")
        exit(1)

    # ignore transfer (final) session
    data = data[data['SessionID'] != 9]

    # define distribution and covariance structure for GEE
    family = sm.families.Binomial()
    covariance_structure = sm.cov_struct.Autoregressive(grid=True)


    # fit full model
    model = smf.gee("Correct ~ ContrastHeterogeneity + GridCoarseness + SessionID + SessionID*ContrastHeterogeneity + SessionID*GridCoarseness", "SubjectID", data, cov_struct=covariance_structure, family=family)
    results_full = model.fit()

    # save results of full model
    results_full.params.to_pickle('results/statistics/gee_full.pkl')

    if is_significant(results_full, 'SessionID:ContrastHeterogeneity'):
        # simple effects of contrast heterogeneity for each session
        for session in range(1, 9):
            session_data = get_session_data(data, session)
            model = smf.gee("Correct ~ ContrastHeterogeneity", "SubjectID", session_data, cov_struct=covariance_structure, family=family)
            results = model.fit()
            results.params.to_pickle(f'results/statistics/gee_contrast_heterogeneity_{session}.pkl')

    if is_significant(results_full, 'SessionID:GridCoarseness'):
        # simple effects of grid coarseness for each session
        for session in range(1, 9):
            session_data = get_session_data(data, session)
            model = smf.gee("Correct ~ GridCoarseness", "SubjectID", session_data, cov_struct=covariance_structure, family=family)
            results = model.fit()
            results.params.to_pickle(f'results/statistics/gee_grid_coarseness_{session}.pkl')