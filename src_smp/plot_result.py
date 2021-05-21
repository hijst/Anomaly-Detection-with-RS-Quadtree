import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

# Granularity test (k=25)
# X = [1, 2, 3, 5, 7, 8, 9, 10, 15, 25]
# Y = [0.968, 0.967, 0.969, 0.967, 0.969, 0.968, 0.969, 0.974, 0.857, 0.772]

# Number of trees test (g=5)

# X = [1, 5, 10, 15, 25, 50, 100]
# Y = [0.792, 0.691, 0.830, 0.969, 0.967, 0.967, 0.970]

# Contamination test (actual contamination is 0.601)

# X = [5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]
# Y = [0.90, 0.949, 0.933, 0.974, 0.973, 0.959, 0.971, 0.960, 0.964, 0.961, 0.960]

# Sample size test (k=25, g=5, c=0.065)

#X = [32, 64, 128, 256, 512, 1024]
#Y = [0.970, 0.947, 0.971, 0.972, 0.878, 0.748]

# Running time test (c=0.01$, $g=10$, $k=25$, $n_samples=256)

# X = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
# Y = [0.98, 1.60, 2.98, 5.56, 10.74, 20.98, 43.12, 87.74, 190.73, 426.30, 996.144]

# Running time test on number of dimensions (c=0.01$, $g=10$, $k=25$, $n_samples=256)

X = [2, 3, 5, 10, 20, 50, 100]
Y = [11.21, 12.25, 13.81, 16.63, 20.92, 21.95, 23.12]

plt.figure()
ax = plt.axes()
ax.plot(X, Y)
# ax.set_xscale('log',  base=2)
# ax.set_yscale('log',  base=2)
plt.title("Running time of RSF as number of dimensions increases")
plt.xlabel("Number of dimensions")
plt.ylabel("Running time")
plt.xlim(2, 100)
plt.ylim(0, 25)
plt.savefig('../output/running_time_dims.pdf')
plt.show()

