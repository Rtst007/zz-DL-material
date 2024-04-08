import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy


tf.random.set_seed(100)


n_epoch = 10
BATCH_SIZE = 128
TEST_BATCH_SIZE = 1000

# 封装 DNN 网络
class MnistModel(tf.keras.Model):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(28, activation='relu')
        self.fc2 = layers.Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# 实例化，模型和优化器
model = MnistModel()
optimizer = Adam(learning_rate=0.001)

#######################################################################
#######################################################################
## 训练 阶段
(train_images, train_labels), _ = tf.keras.datasets.mnist.load_data()
train_images = train_images / 255.0  # 归一化
train_labels = train_labels.astype('int64')

for i in range(n_epoch):
    for idx in range(0, len(train_images), BATCH_SIZE):
        batch_images = train_images[idx:idx+BATCH_SIZE]
        batch_labels = train_labels[idx:idx+BATCH_SIZE]

        with tf.GradientTape() as tape:
            output = model(batch_images, training=True)
            loss = SparseCategoricalCrossentropy()(batch_labels, output)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if idx % 10 == 0:
            print(i, idx, loss.numpy())

#######################################################################
#######################################################################
## 测试 阶段
test_images, test_labels = tf.keras.datasets.mnist.load_data()[1]
test_images = test_images / 255.0  # 归一化
test_labels = test_labels.astype('int64')

test_loss, test_acc = 0.0, 0.0
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(TEST_BATCH_SIZE)

num_batches = tf.data.experimental.cardinality(test_dataset).numpy()

for batch_images, batch_labels in test_dataset:
    output = model(batch_images, training=False)
    cur_loss = SparseCategoricalCrossentropy()(batch_labels, output)
    test_loss += cur_loss.numpy()

    pred = tf.argmax(output, axis=-1)
    cur_acc = tf.reduce_mean(tf.cast(tf.equal(pred, batch_labels), tf.float32))
    test_acc += cur_acc.numpy()

print("平均准确率：", test_acc / num_batches, "平均损失：", test_loss / num_batches)
