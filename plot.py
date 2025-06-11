import matplotlib.pyplot as plt
'''
# univ bf
# ====== [IMG_3342.JPG] ========
ks = [100, 300, 500, 700, 1000, 1500, 2000]
distance_errors = [0.19, 0.22, 0.16, 0.17, 0.15, 0.16, 0.16]
execution_times = [0.3288, 0.4581, 0.3497, 0.3393, 0.3382, 0.3600, 0.4419]
'''
'''
# univ bovw
# ====== [IMG_3342.JPG] ========
ks = [50, 100, 300, 500, 700, 1000, 1500, 2000]
distance_errors = [0.242, 0.216, 0.1233, 0.12,  0.140,0,0,0.1333 ]
execution_times = [0.53, 0.487, 0.5029, 0.466, 0.497,0,0,0.5423 ]

'''

'''
# park bf
#==== image[IMG_2977.jpeg] ====
ks = [50, 100, 300, 500, 700, 1000, 1500, 2000]
distance_errors = [0.41, 0.41, 0.41, 0.44, 0.41, 0.40, 0.42, 0.40]
execution_times = [0.1687, 0.1926, 0.1860, 0.2017, 0.2180, 0.2136, 0.2996, 0.2428]
'''

'''
# park bf 5000-15000
#==== image[IMG_2977.jpeg] ====
ks = [ 5000, 7000, 9000, 11000, 13000, 15000]
distance_errors = [ 0.43, 0.41, 0.40, 0.41, 0.41, 0.42]
execution_times = [ 0.1916, 0.2691, 0.1813, 0.2390, 0.2485, 0.2024]
'''

'''
# park bovw
#==== image[IMG_2977.jpeg] ====
ks = [50, 100, 300, 500, 700, 1000, 1500, 2000]
distance_errors = [0, 0, 0, 0.423, 0.420, 0.413, 0, 0]
execution_times = [0, 0, 0, 0.205, 0.172, 0.167, 0, 0]
'''
'''
# park bovw - 5000-15000
#==== image[IMG_2977.jpeg] ====
ks = [ 5000, 7000, 9000, 11000, 13000, 15000]
distance_errors = [0.4125, 0.4217, 0.4150, 0.4117, 0.4100, 0.4183]
execution_times = [0.2652, 0.2917, 0.3362, 0.3784, 0.4114, 0.4540]
'''

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
