import numpy as np
import tensorflow as tf
import cv2

class jigsaw_piece_nn_classifier():
    def __init__(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32,(3, 3), padding='same', activation='relu', input_shape=(60,60,3)),
            tf.keras.layers.Conv2D(32,(3, 3), activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

            tf.keras.layers.Conv2D(64,(3, 3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64,(3, 3), activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(2, activation=tf.nn.softmax)
        ])
    def fit_model(self, train_data, label):
        self.model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        self.model.summary()
        # self.model.load_weights('weights')
        self.model.fit(train_data, label, epochs=5)

    def my_evaluate(self, test_data, test_label):
        loss, acc = self.model.evaluate(test_data, test_label)
        print(loss, acc)

    def my_save_weights(self):
        self.model.save('model.h5')
        # self.model.save_weights('weights')

    def my_load_model(self, file_name):
        self.model = tf.keras.models.load_model(file_name)
        
    def predict(self, img):
        img = img / 255.0
        res = self.model.predict(np.array([img]))
        res = np.argmax(res)
        return res == 1

if __name__ == '__main__':
    model = jigsaw_piece_nn_classifier()
    data_set1 = np.load('data_set1.npy')
    data_set2 = np.load('data_set2.npy')
    data_set3 = np.load('data_set3.npy')
    data_set4 = np.load('data_set4.npy')
    label_set1 = np.load('label_set1.npy')
    label_set2 = np.load('label_set2.npy')
    label_set3 = np.load('label_set3.npy')
    label_set4 = np.load('label_set4.npy')
    data = np.concatenate((data_set1,data_set2,data_set3,data_set4),axis=0)
    label = np.concatenate((label_set1, label_set2, label_set3, label_set4),axis=0)
    data = data / 255.0
    train_data = data[0:6000]
    train_label = label[0:6000]
    test_data = data[6000:8000]
    test_label = label[6000:8000]
    print(train_data.shape, train_label.shape)
    model.fit_model(train_data, train_label)
    model.my_save_weights()
    model.my_evaluate(test_data, test_label)

    print(model.predict(test_data[0]))
