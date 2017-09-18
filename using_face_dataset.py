#coding:utf-8
"""
read_data_sets
"""
from PIL import Image
import numpy as np

def read_data_sets(dataset_path):
    img = Image.open(dataset_path)
    #np.asarray()将图片转为灰度
    img_ndarray = np.asarray(img,dtype='float64')/256
    #共有400张图片,每张图片大小57*47=2679
    faces = np.empty((400,2679))
    #将图片存入faces中
    for row in range(20):
        for column in range(20):
            #第一行的第一张脸存入face0,每张脸的大小为57*47
            faces[row*20+column] = np.ndarray.flatten(img_ndarray[row*57:(row+1)*57,column*47:(column+1)*47])

    #将标签存入label中,400个label标签,40类
    label = np.empty(400)
    for i in range(40):
        label[i*10:i*10+10] = i
        label = label.astype(np.int)
    #设置train320,test80
    train_data = np.empty((320,2679))
    train_label = np.empty(320)
    test_data = np.empty((80,2679))
    test_label = np.empty(80)

    #将数据集存入data,label中
    for i in range(40):
        #前8张存入train,第9-10张存入test，依此顺序存400张图片
        train_data[i*8:i*8+8] = faces[i*10:i*10+8]
        train_label[i*8:i*8+8] = label[i*10:i*10+8]
        test_data[i*2:i*2+2] = faces[i*10+8:i*10+10]
        test_label[i*2:i*2+2] = label[i*10+8:i*10+10]

    data = [(train_data,train_label),(test_data,test_label)]
    return data

#(train_data, train_label), (test_data, test_label) = read_data_sets('olivettifaces.gif')