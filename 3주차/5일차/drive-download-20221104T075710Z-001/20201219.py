#mnist.py
import time

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# TensorFlow에서 제공하는 MNIST 데이터 파일 4개를 다운로드하여 data 폴더에 저장하고 읽어옵니다.
# 최초 실행 시에만 데이터를 다운로드하고, 두 번째 이후부터는 저장된 데이터를 읽어 오기만 하기 때문에 시간이 단축됩니다."""
from tensorflow.examples.tutorials.mnist import input_data
# %time
mnist = input_data.read_data_sets("data/", one_hot=True)  # %time을 통해 전체 실행 시간을 남길 수 있습니다.

# images/labels 데이터 구조 확인
print('train 데이터셋(55,000건):', mnist.train.images.shape, mnist.train.labels.shape)
print('test 데이터셋(10,000건):', mnist.test.images.shape, mnist.test.labels.shape)
print('validation 데이터셋(5,000건):', mnist.validation.images.shape, mnist.validation.labels.shape)

# 샘플 이미지 데이터 확인
print('\nlabel :', mnist.train.labels[0])
label = np.argmax(mnist.train.labels[0])  # 가장 큰 값(즉 1이 있는 곳)

im = np.reshape(mnist.train.images[0], [28,28])
plt.imshow(im, cmap='Greys')
plt.title('label:' + str(label))
plt.show()