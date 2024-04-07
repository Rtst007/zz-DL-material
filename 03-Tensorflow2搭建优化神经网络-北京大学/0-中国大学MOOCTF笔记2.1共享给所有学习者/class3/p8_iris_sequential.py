##
##### 神经网络 八股， 6步法 搭建网络
##

# ------------------------------------------------------------------------
# ----- 步骤 1：  import 所需要的模块
# ------------------------------------------------------------------------
import tensorflow as tf
from sklearn import datasets
import numpy as np

# ------------------------------------------------------------------------
# ----- 步骤 2：  train    test
# ------------------------------------------------------------------------
x_train = datasets.load_iris().data
y_train = datasets.load_iris().target

np.random.seed(116)
np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(y_train)
tf.random.set_seed(116)

# ------------------------------------------------------------------------
# ----- 步骤 3：  models sequential
# ------------------------------------------------------------------------
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())
])


# ------------------------------------------------------------------------
# ----- 步骤 4：  model compile
# ------------------------------------------------------------------------
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

# ------------------------------------------------------------------------
# ----- 步骤 5：  model fit
# ------------------------------------------------------------------------
model.fit(x_train, y_train, batch_size=32, epochs=500, validation_split=0.2, validation_freq=20)

# ------------------------------------------------------------------------
# ----- 步骤 6：  model summary
# ------------------------------------------------------------------------
model.summary()
