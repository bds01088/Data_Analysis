import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix

# value = np.random.standard_normal(500)
# cm = plt.cm.get_cmap('Spectral')
# plt.figure(figsize = (12,6))

# n, bins, patches = plt.hist(value, bins = 30, color = 'green')
# print ("1.bins : " + str(bins))
# print ("2.the length of bins : " + str(len(bins)))

# bin_centers = 0.5*(bins[:-1] + bins[1:])
# print ("3.bin_centers : " + str(bin_centers))

# col = (bin_centers - min(bin_centers)) / (max(bin_centers) - min(bin_centers))
# print("4.col : " + str(col))

# for c, p in zip(col, patches) :
#     plt.setp(p, 'facecolor', cm(c))

# plt.xlabel('value')
# plt.ylabel('frequency')
# plt.title("Histogram 3")
# plt.show()

# import seaborn as sns
# data = sns.load_dataset("flights")
# # print (data.info)
# pivoted_data = data.pivot("year", "month", "passengers")
# print (pivoted_data.info)

# sns.set(context = "poster", font = "monospace")
# sns.heatmap(pivoted_data, annot = True, fmt = "d", linewidth = 3)
# plt.show()


value = np.random.randn(500, 4)
df = pd.DataFrame(value, columns=['value1','value2','value3','value4'])
scatter_matrix(df, alpha = 0.2, figsize=(6,6), diagonal = 'hist')
plt.show()