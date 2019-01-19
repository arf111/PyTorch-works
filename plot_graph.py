import matplotlib.pyplot as plt 

real_weight = 0.7
cartoon_weight = 0.6
fig = plt.figure()
fig.suptitle('SN on D',fontsize=15)
plt.plot([1,2,3,4,5],[2,4,6,8,10])
fig.savefig('FID({:.1f},{:.1f}).jpg'.format(real_weight,cartoon_weight))