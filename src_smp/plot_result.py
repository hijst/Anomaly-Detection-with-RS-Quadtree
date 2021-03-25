import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

# Granularity test (k=25)
# X = [1, 2, 3, 5, 7, 8, 9, 10, 15, 25]
# Y = [0.968, 0.967, 0.969, 0.967, 0.969, 0.968, 0.969, 0.974, 0.857, 0.772]

# Number of trees test (g=5)

# X = [1, 5, 10, 15, 25, 50, 100]
# Y = [0.792, 0.691, 0.830, 0.969, 0.967, 0.967, 0.970]

# Contamination test (actual contamination is 0.601)

X = [5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]
Y = [0.90, 0.949, 0.933, 0.974, 0.973, 0.959, 0.971, 0.960, 0.964, 0.961, 0.960]

# Sample size test (k=25, g=5, c=0.065)

#X = [32, 64, 128, 256, 512, 1024]
#Y = [0.970, 0.947, 0.971, 0.972, 0.878, 0.748]

plt.figure()
ax = plt.axes()
ax.plot(X, Y)
plt.title("Shuttle data scores (k=25, g=5)")
plt.xlabel("Contamination")
plt.ylabel("ROC-AUC")
plt.xlim(5, 10)
plt.ylim(0.5, 1)
#plt.savefig('../output/shuttle_contamination.pdf')
plt.show()

