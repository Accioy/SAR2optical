# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
# from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

def get_sample_files(classpath):
    # classpath: ex: iamges/train, which includes different classes samples with each class in its folder
    rootpath=os.path.abspath('.')
    classpath=os.path.join(rootpath,classpath)
    class_list=os.listdir(classpath)
    class_list.sort()
    number_of_class=len(class_list)
    samples=[]
    for i in class_list:
        filesdir=os.path.join(classpath,i)
        files=os.listdir(filesdir)
        for j in range(len(files)):
            files[j]=os.path.join(filesdir,files[j])
        samples+=files
    samples.sort()
    return class_list,number_of_class,samples

def write_list(list1,filename):
    with open(filename+'.txt','w+') as f:
        for i in list1:
            f.write(i+"\r\n")





def load_grayscale_image(img_path, target_size=(256, 256)):
    im = load_img(img_path,color_mode='grayscale', target_size=target_size)
    return img_to_array(im) #converts image to numpy array

def load_rgb_image(img_path, target_size=(256, 256)):
    im = load_img(img_path,color_mode='rgb', target_size=target_size)
    return img_to_array(im) #converts image to numpy array
# 建立一个数据迭代器
def GET_DATASET_chunks(x,y, chunk_size=64):
    print('chunk_size:',chunk_size)
    batch_num = int(len(x) / chunk_size)
    max_len = batch_num * chunk_size
    x = np.array(x[:max_len])
    y = np.array(y[:max_len])
    x_batches = np.split(x, batch_num)
    y_batches = np.split(y, batch_num)
    for i in range(len(x_batches)):
        x = np.array(list(map(load_grayscale_image, x_batches[i])))
        y = np.array(list(map(load_rgb_image, y_batches[i])))
        yield x,y

def data_generator():
    sar_path="/home1/dataset/sar-op/subset_spring_90/train/sar"
    op_path="/home1/dataset/sar-op/subset_spring_90/train/op"
    
    train_data_gen=ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        cval=0.0,
        horizontal_flip=True,
        vertical_flip=True,
        rescale=1./255
    )


    seed = 14
    train_image_generator=train_data_gen.flow_from_directory(directory=sar_path,target_size=(256, 256), 
    color_mode='grayscale',class_mode=None,batch_size=32, shuffle=True, seed=seed)
    train_label_generator=train_data_gen.flow_from_directory(directory=op_path,target_size=(256, 256), 
    color_mode='rgb',class_mode=None,batch_size=32, shuffle=True, seed=seed)
    

    return train_image_generator,train_label_generator


def test_generator():
    val_sar_path="/home1/dataset/sar-op/subset_spring_90/validation/sar"
    val_op_path="/home1/dataset/sar-op/subset_spring_90/validation/op"
    test_data_gen=ImageDataGenerator(
        rescale=1./255
    )
    seed = 14
    val_image_generator=test_data_gen.flow_from_directory(directory=val_sar_path,target_size=(256, 256), 
    color_mode='grayscale',class_mode=None,batch_size=32, shuffle=True, seed=seed)
    val_label_generator=test_data_gen.flow_from_directory(directory=val_op_path,target_size=(256, 256), 
    color_mode='rgb',class_mode=None,batch_size=32, shuffle=True, seed=seed)
    return val_image_generator,val_label_generator




def get_data():
    return 

if __name__ == '__main__':
    # get_generator()
    print('success!')



