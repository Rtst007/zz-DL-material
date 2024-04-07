


import tensorflow as tf


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 说明：
#
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1' # 默认，显示所有信息
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2' # 只显示 warning 和 Error
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3' # 只显示 Error
# 如果需要对CPU进行优化，可以访问下面的github，重新编译tensorflow源码以兼容AVX
# https://github.com/lakshayg/tensorflow-build



tensorflow_version = tf.__version__
# gpu_available = tf.test.is_gpu_available()

gpu_available = tf.config.list_physical_devices('GPU')

print("tensorflow version:", tensorflow_version, "\tGPU available:", gpu_available)

a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([1.0, 2.0], name="b")
result = tf.add(a, b, name="add")
print(result)
