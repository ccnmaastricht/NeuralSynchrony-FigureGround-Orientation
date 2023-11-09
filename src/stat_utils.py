from prettytable import PrettyTable

def print_wald_chi_square(results):
    print('Wald Chi-Square:')
    table = PrettyTable()
    table.field_names = ['Variable', 'Chi-Square', 'p-value']
    for var in results.model.exog_names:
        table.add_row([var, results.wald_test(var, scalar=True).statistic, results.wald_test(var, scalar=True).pvalue])
    print(table)

def print_sample_info(metadata):
    num_samples = metadata.shape[0]
    num_females = metadata['Sex'].value_counts()[' F']
    mean_age = metadata['Age'].mean()
    std_age = metadata['Age'].std().__round__(3)

    print(f'{num_samples} particpants ({num_females} female, mean age = {mean_age}, standard deviation = {std_age})')