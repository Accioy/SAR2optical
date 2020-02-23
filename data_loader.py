import scipy
# from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array, load_img
class DataLoader():
    def __init__(self, img_res=(128, 128)):
        self.img_res = img_res

    def load_data(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"
        path_sar = os.path.join('/home1/dataset/sar-op/subset_spring_90/',data_type,'sar','s1_90')
        sar_list=os.listdir(path_sar)
        sar_list.sort()
        path_op=os.path.join('/home1/dataset/sar-op/subset_spring_90/',data_type,'op','s2_90')
        op_list=os.listdir(path_op)
        op_list.sort()
        
        state = np.random.get_state()
        batch_images = np.random.choice(sar_list, size=batch_size)
        
        np.random.set_state(state)
        batch_labels = np.random.choice(op_list, size=batch_size)

        imgs_sar = []
        imgs_op = []
        
        for (sar_path,op_path) in zip(batch_images,batch_labels):
            print(sar_path,op_path)
            img_sar = img_to_array(load_img(os.path.join(path_sar,sar_path),color_mode='grayscale'))
            img_op=img_to_array(load_img(os.path.join(path_op,op_path),color_mode='rgb'))

            # img_A = scipy.misc.imresize(img_A, self.img_res)
            # img_B = scipy.misc.imresize(img_B, self.img_res)

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_sar = np.fliplr(img_sar)
                img_op = np.fliplr(img_op)

            imgs_sar.append(img_sar)
            imgs_op.append(img_op)

        imgs_sar = np.array(imgs_sar)/127.5 - 1.
        imgs_op = np.array(imgs_op)/127.5 - 1.
        print('load_data_shape:',imgs_sar.shape)
        return imgs_op,imgs_sar

    def load_batch(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"
        path_sar = os.path.join('/home1/dataset/sar-op/subset_spring_90/',data_type,'sar','s1_90')
        sar_list=os.listdir(path_sar)
        sar_list.sort()
        path_op=os.path.join('/home1/dataset/sar-op/subset_spring_90/',data_type,'op','s2_90')
        op_list=os.listdir(path_op)
        op_list.sort()
        self.n_batches = int(len(sar_list) / batch_size)

        for i in range(self.n_batches-1):
            batch_sar = sar_list[i*batch_size:(i+1)*batch_size]
            batch_op = op_list[i*batch_size:(i+1)*batch_size]
            imgs_sar, imgs_op = [], []
            for (img_sar,img_op) in zip(batch_sar,batch_op):
                img_sar = img_to_array(load_img(os.path.join(path_sar,img_sar),color_mode='grayscale'))
                img_op=img_to_array(load_img(os.path.join(path_op,img_op),color_mode='rgb'))


                if not is_testing and np.random.random() > 0.5:
                        img_sar = np.fliplr(img_sar)
                        img_op = np.fliplr(img_op)

                imgs_sar.append(img_sar)
                imgs_op.append(img_op)

            imgs_sar = np.array(imgs_sar)/127.5 - 1.
            imgs_op = np.array(imgs_op)/127.5 - 1.
            # print(imgs_sar.shape)
            yield imgs_op,imgs_sar


    # def imread(self, path):
    #     return scipy.misc.imread(path, mode='RGB').astype(np.float)

if __name__ == '__main__':
    os.makedirs('test_dataloder/', exist_ok=True)
    r, c = 3,2
    datalo=DataLoader()
    imgs_A, imgs_B = datalo.load_data(batch_size=3, is_testing=True)

    # gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])

    # Rescale images 0 - 1
    # gen_imgs = 0.5 * gen_imgs + 0.5
    imgs_B =0.5 * imgs_B + 0.5
    imgs_A=0.5 * imgs_A + 0.5
    print('sample',imgs_B.shape,imgs_A.shape)
    gen_imgs = [imgs_B,imgs_A]

    titles = ['Condition', 'Original']
    fig, axs = plt.subplots(r, c)
    for i in range(r): #batch
        for j in range(c):
            if j ==0:
                axs[i,j].imshow(gen_imgs[j][i][:,:,0],cmap='gray')
            else:
                axs[i,j].imshow(gen_imgs[j][i])
            axs[i, j].set_title(titles[j])
            axs[i,j].axis('off')
    fig.savefig("test_dataloder/testload.png")
    plt.close()
        