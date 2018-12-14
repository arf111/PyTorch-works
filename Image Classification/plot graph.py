import matplotlib.pyplot as plt 
import numpy as np

fid_lis = [n for n in range(20,201,20)]
lis = [n for n in range(20, 201, 20)]
# f1 = plt.figure(1)
# f2 = plt.figure(2)
# ax1 = f1.add_subplot(111)
# ax1.plot(range(0,10))
# ax2 = f2.add_subplot(111)
# ax2.plot(range(10,20))
# ax1.savefig('test.jpg')
# ax2.savefig('test2.jpg')

for i in range(2):
    plt.figure()
    plt.xlabel('epochs(k)')
    if i == 0:
        plt.ylabel('FID')
    else:
        plt.ylabel('Inception')
    plt.plot(lis, fid_lis)
    plt.savefig('test {0}.jpg'.format(i))