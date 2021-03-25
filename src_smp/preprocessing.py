import pandas as pd
# above .data file is comma delimited
forest_cover = pd.read_csv('../data/covtype.data', delimiter=",", header=None)

forest_cover.drop(forest_cover.columns[list(range(10, 54))], axis=1, inplace=True)
forest_cover = forest_cover[forest_cover[54].isin([2, 4])]
forest_cover.to_pickle('../data/forest_c.pkl')

sat_trn = pd.read_csv('../data/sat_trn.csv', delimiter=";", header=None)
sat_tst = pd.read_csv('../data/sat_tst.csv', delimiter=";", header=None)
sat_full = pd.concat([sat_trn, sat_tst], axis=0)
print(sat_full)
print(sat_full.groupby(sat_full.columns[-1]).count())
#sat_full.to_pickle('../data/sat_full.pkl')