import pandas as pd
import matplotlib.pyplot as plt

results_df = pd.read_csv('../out/result.csv', header=None)
results_df.columns = ['nodes', 'id', 'edges', 't [s]']

exp_number = 1
exp_input_df = pd.DataFrame()
for n_parts in range(6, 21, 2):
    for p_liaison in range(20, 45, 5):
        for mwf in range(50, 95, 5):
            id = 'exp_' + str(exp_number)
            exp_input_df = exp_input_df.append({'id': id, 'n_parts': n_parts, 'p_liaison': p_liaison/100, 'mwf': mwf/100}, ignore_index=True)
            exp_number += 1

print(results_df.head())
print(exp_input_df.head())

experiments_df = pd.merge(exp_input_df, results_df, on='id')

print(experiments_df.head())

experiments_df.to_excel('../out/full_exp_data.xlsx')

experiments_df.plot(x='mwf', y='t [s]')
plt.show()