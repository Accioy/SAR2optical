import matplotlib.pyplot as plt
import sys
import numpy as np
import os
from keras.models import Model,load_model
from keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
import scipy.misc

model=load_model('generator150.h5')

rootpath=os.path.abspath('./')
path_sar = os.path.join(rootpath,'sar_ims')
sar_list=os.listdir(path_sar)
sar_list.sort()
imgs_sar = []
for img_path in sar_list:
    print(img_path)
    img_sar = img_to_array(load_img(os.path.join(path_sar,img_path),color_mode='grayscale',target_size=(256,256)))
    imgs_sar.append(img_sar)
imgs_sar = np.array(imgs_sar)/127.5 - 1.

fake_A = model.predict(imgs_sar)
fake_A = fake_A*0.5+0.5

# gen_imgs = [imgs_sar,fake_A]
r, c = 15, 2
# titles = ['Condition', 'Generated']
# fig= plt.figure(figsize=(1,1),dpi=256)

# for i in range(r): #batch
#     plt.imshow(fake_A[i])
#     plt.axis('off')
#     plt.savefig("images/test"+sar_list[i]+".png")
# plt.close()


for i in range(r):
    scipy.misc.imsave('images/out'+sar_list[i]+'.jpg', fake_A[i])