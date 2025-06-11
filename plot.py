import matplotlib.pyplot as plt
'''
# univ bf
ks = [100, 300, 500, 700, 1000, 1500, 2000]
distance_errors = [0.19, 0.22, 0.16, 0.17, 0.15, 0.16, 0.16]
execution_times = [0.3288, 0.4581, 0.3497, 0.3393, 0.3382, 0.3600, 0.4419]
'''
#==== image[IMG_2977.jpeg] ====
ks = [50, 100, 300, 500, 700, 1000, 1500, 2000]
distance_errors = [0.41, 0.41, 0.41, 0.44, 0.41, 0.40, 0.42, 0.40]
execution_times = [0.1687, 0.1926, 0.1860, 0.2017, 0.2180, 0.2136, 0.2996, 0.2428]

plt.figure()
plt.plot(ks, distance_errors, marker='o', color='red', label='Distance error (m)')
plt.plot(ks, execution_times, marker='x', color='blue', label='Execution time (sec)')
plt.xlabel('clustering number')
plt.ylabel('Value')
plt.xticks(ks)
plt.legend()
plt.grid(True)
plt.savefig('./plot.png', dpi=300, bbox_inches='tight')
#plt.show()
