import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(), # 讲输入数据 拉直为 一维数组
    tf.keras.layers.Dense(128, activation='relu'), # 定义第一层神经网络，有 128个神经元
    tf.keras.layers.Dense(10, activation='softmax')  # 定义第二层神经网络，有 10个神经元
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), # 第9行的网络softmax输入服从概率分布，因此这里from_logits=False，否则要改为true
              metrics=['sparse_categorical_accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1)
model.summary()
