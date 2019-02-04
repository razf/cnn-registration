from __future__ import print_function
import Registration
import matplotlib.pyplot as plt
from utils.utils import *
import cv2

# designate image path here
from src import RegistrationRANSAC

# IX_path = '../img/ix.jpg'
# IY_path = '../img/NET.jpg'
IX_path = '../img/90deg/1a.jpg'
IY_path = '../img/90deg/2b.jpg'
IX_path = '../img/sat2/ref.jpg'
IY_path = '../img/sat2/new.jpg'
# IX_path = '../img/satellite/ref.jpg'
# IY_path = '../img/satellite/new.jpg'

IX = cv2.imread(IX_path)
IY = cv2.imread(IY_path)

#initialize
reg = RegistrationRANSAC.CNN_RANSAC()
# reg.draw_matches(IX,IY, '../img/satellite/')
#register
reg.draw_matches(IX, IY, '../img/sat2',SIFT=True)

# X, Y, Z = reg.register(IX, IY)
#generate regsitered image using TPS
# registered = tps_warp(Y, Z, IY, IX.shape)
#
# #cv2.imwrite(r'../img/ref.jpg', IX)
# cv2.imwrite(r'../img/registered.jpg', registered)

print("DONE")
# cb = checkboard(IX, registered, 11)
#
#
#
# plt.subplot(131)
# plt.title('reference')
# plt.imshow(cv2.cvtColor(IX, cv2.COLOR_BGR2RGB))
# plt.subplot(132)
# plt.title('registered')
# plt.imshow(cv2.cvtColor(registered, cv2.COLOR_BGR2RGB))
# plt.subplot(133)
# plt.title('checkboard')
# plt.imshow(cv2.cvtColor(cb, cv2.COLOR_BGR2RGB))
#
# plt.show()






    
