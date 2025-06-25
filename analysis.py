import os
import pandas as pd

df = pd.read_csv(os.path.join('results', 'results.csv'))
# some cleaning
df['flowers'] =  df['flowers'].str.replace(r"[\"\'\[\]]", "", regex=True)

FLOWERS = 'versicolor, virginica'
# C
mask = (df['condense'] == False) & (df['flowers'] == FLOWERS)
index = ['k','p']
cols = ['empirical_error', 'true_error']
temp = df[mask].pivot_table(index=index, values=cols).reset_index()
temp['diff'] = temp['empirical_error'].sub(temp['true_error']).abs()
temp.sort_values(by='true_error')

# D
mask = (df['condense'] == True) & (df['flowers'] == FLOWERS)
index = ['k','p']
cols = ['train_init_size', 'condensed_size']
temp = df[mask].pivot_table(index=index, values=cols).reset_index()
temp

mask = (df['condense'] == True) & (df['flowers'] == FLOWERS)
index = ['k','p']
cols = ['empirical_error', 'true_error']
temp = df[mask].pivot_table(index=index, values=cols).reset_index()
temp['diff'] = temp['empirical_error'].sub(temp['true_error']).abs()
temp.sort_values(by='true_error')

# E
# CHANGE FLOWERS to 'setosa, virginica'
