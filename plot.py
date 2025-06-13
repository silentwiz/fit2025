import matplotlib.pyplot as plt
import numpy as np

                                    ######################
                                    #### univ dataset ####
                                    ######################

# for 100~2000 cluster
ks = [100, 300, 500, 700, 1000, 1500, 2000]

# for 5000 ~~> cluster
ks = [ 5000, 7000, 9000, 11000, 13000, 15000]


'''
# univ bf window
# ====== [IMG_3342.JPG] ========
ks = [100, 300, 500, 700, 1000, 1500, 2000]
distance_errors = [0.19, 0.22, 0.16, 0.17, 0.15, 0.16, 0.16]
execution_times = [0.3288, 0.4581, 0.3497, 0.3393, 0.3382, 0.3600, 0.4419]
#0.173 m
#0.374 sec
'''



'''
# univ bf 5000-15000 window
#==== image[IMG_2977.jpeg] ====
ks = [ 5000, 7000, 9000, 11000, 13000, 15000]
distance_errors = [ 0.43, 0.41, 0.40, 0.41, 0.41, 0.42]
execution_times = [ 0.1916, 0.2691, 0.1813, 0.2390, 0.2485, 0.2024]
# 0.413 m
# 0.222 sec
'''


'''
# univ bovw window
# ====== [IMG_3342.JPG] ========
ks = [100, 300, 500, 700, 1000, 1500, 2000]
distance_errors = [0.216, 0.1233, 0.12,  0.140,0,0,0.1333 ]
execution_times = [0.487, 0.5029, 0.466, 0.497,0,0,0.5423 ]
#0.147 m
#0.499 sec
'''


'''
# univ bovw 5000-15000 window
#==== image[IMG_2977.jpeg] ====
ks = [ 5000, 7000, 9000, 11000, 13000, 15000]
distance_errors = [ 0.43, 0.41, 0.40, 0.41, 0.41, 0.42]
execution_times = [ 0.1916, 0.2691, 0.1813, 0.2390, 0.2485, 0.2024]
# 0.413 m
# 0.222 sec
'''


#univ distance_error
proposal =  [0.19, 0.22, 0.16, 0.17, 0.15, 0.16, 0.16]
image_retrieval  =  [0.216, 0.1233, 0.12,  0.140,0,0,0.1333 ]


'''
#univ execution times
proposal = [0.3288, 0.4581, 0.3497, 0.3393, 0.3382, 0.3600, 0.4419]
image_retrieval = [0.487, 0.5029, 0.466, 0.497,0,0,0.5423 ]
'''


                                    ######################
                                    #### park dataset ####
                                    ######################
'''
# park bf window
#==== image[IMG_2977.jpeg] ====
ks = [100, 300, 500, 700, 1000, 1500, 2000]
distance_errors = [0.41, 0.41, 0.44, 0.41, 0.40, 0.42, 0.40]
execution_times = [0.1926, 0.1860, 0.2017, 0.2180, 0.2136, 0.2996, 0.2428]
#0.413 m
#0.222 sec
'''


'''
# park bf 5000-15000 window
#==== image[IMG_2977.jpeg] ====
ks = [ 5000, 7000, 9000, 11000, 13000, 15000]
distance_errors = [ 0.43, 0.41, 0.40, 0.41, 0.41, 0.42]
execution_times = [ 0.1916, 0.2691, 0.1813, 0.2390, 0.2485, 0.2024]
# 0.413 m
# 0.222 sec
'''

'''
# park bovw window
#==== image[IMG_2977.jpeg] ====
ks = [100, 300, 500, 700, 1000, 1500, 2000]
distance_errors = [ 0, 0, 0.423, 0.420, 0.413, 0, 0]
execution_times = [ 0, 0, 0.205, 0.172, 0.167, 0, 0]
# 0.4187 m
# 0.1813 sec
'''


'''
# park bovw - 5000-15000 window
#==== image[IMG_2977.jpeg] ====
ks = [ 5000, 7000, 9000, 11000, 13000, 15000]
distance_errors = [0.4125, 0.4217, 0.4150, 0.4117, 0.4100, 0.4183]
execution_times = [0.2652, 0.2917, 0.3362, 0.3784, 0.4114, 0.4540]
#0.415 m
#0.356 sec
'''


'''
#park distance error
proposal =  [ 0.43, 0.41, 0.40, 0.41, 0.41, 0.42]
image_retrieval  = [0.4125, 0.4217, 0.4150, 0.4117, 0.4100, 0.4183]
'''


#park executation time
proposal =  [ 0.1916, 0.2691, 0.1813, 0.2390, 0.2485, 0.2024]
image_retrieval  =  [0.2652, 0.2917, 0.3362, 0.3784, 0.4114, 0.4540]



'''
                ### func for distance error and execution times###
for i, dis_e in enumerate(distance_errors):
    if dis_e == 0:
        distance_errors[i] = np.nan

for i, exe_t in enumerate(execution_times):
    if exe_t == 0:
        execution_times[i] = np.nan
'''

for i, dis_e in enumerate(proposal):
    if dis_e == 0:
        proposal[i] = np.nan

for i, exe_t in enumerate(image_retrieval):
    if exe_t == 0:
        image_retrieval[i] = np.nan


'''
plt.figure()
plt.plot(ks, distance_errors, marker='o', color='red', label='Distance error (m)')
plt.plot(ks, execution_times, marker='x', color='blue', label='Execution time (sec)')
plt.xlabel('clustering number')
plt.ylabel('Value')
plt.xticks(ks)
plt.legend(fontsize=17)
plt.grid(True)
plt.savefig('./plot.png', dpi=300, bbox_inches='tight')
#plt.show()
'''


plt.figure()
plt.plot(ks, proposal, marker='o', color='red', label='Proposal')
plt.plot(ks, image_retrieval, marker='x', color='blue', label='Image Retribe')
plt.xlabel('Clustering number', fontsize=20)
plt.ylabel('Distansce error (m)', fontsize=20)
#plt.ylabel('Execution times (sec)', fontsize=20)
plt.xticks(ks, fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=20)
plt.grid(True)
plt.savefig('./plot.png', dpi=300, bbox_inches='tight')
#plt.show()


'''
plt.figure()
plt.plot(ks, distance_errors, marker='o', color='red', label='Distance error (m)')
plt.plot(ks, execution_times, marker='x', color='blue', label='Execution time (sec)')
plt.xlabel('clustering number', fontsize=17)
plt.ylabel('Distance error (m)', fontsize=17)
plt.xticks(ks)
plt.legend(fontsize=17)
plt.grid(True)
plt.savefig('./plot.png', dpi=300, bbox_inches='tight')
#plt.show()

plt.figure()
plt.plot(ks, distance_errors, marker='o', color='red', label='Distance error (m)')
plt.plot(ks, execution_times, marker='x', color='blue', label='Execution time (sec)')
plt.xlabel('clustering number', fontsize=17)
plt.ylabel('Execution time (sec)', fontsize=17)
plt.xticks(ks)
plt.legend(fontsize=17)
plt.grid(True)
plt.savefig('./plot.png', dpi=300, bbox_inches='tight')
#plt.show()
'''