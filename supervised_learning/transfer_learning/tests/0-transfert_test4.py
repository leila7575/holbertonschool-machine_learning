#!/usr/bin/env python3

from tensorflow import keras as K
import numpy as np
import matplotlib.pyplot as plt

def preprocess_data(X, Y):
    X_p = K.applications.densenet.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p

if __name__ == "__main__":
    
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    datagen_train = K.preprocessing.image.ImageDataGenerator(rotation_range=10, width_shift_range = 0.1, height_shift_range = 0.1, horizontal_flip=True)
    datagen_test = K.preprocessing.image.ImageDataGenerator()

    train_data_iter = datagen_train.flow(X_train, Y_train, batch_size=200)
    test_data_iter= datagen_test.flow(X_test, Y_test, batch_size=200)
    
    base_model = K.applications.DenseNet121(weights='imagenet', include_top=False)
    base_model.trainable = False

    for layer in base_model.layers[-20:]:
        layer.trainable = True

    input_layer = K.layers.Input(shape=(32, 32, 3))
    rescaled_images = K.layers.Resizing(224, 224)(input_layer)
    pretrained_output = base_model(rescaled_images, training=True)

    x = K.layers.Flatten()(pretrained_output)
    x = K.layers.Dense(1024, activation='relu')(x)
    x = K.layers.Dropout(0.3)(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Dense(512, activation='relu')(x)
    x = K.layers.Dropout(0.3)(x)
    x = K.layers.BatchNormalization()(x)
    output = K.layers.Dense(10, activation='softmax')(x)

    model = K.Model(inputs=input_layer, outputs=output)

    model.compile(optimizer=K.optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = K.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(train_data_iter, validation_data=test_data_iter, epochs=25, batch_size=200, verbose=1, callbacks=[early_stopping])

    model.save("cifar10.h5")

    test_loss, test_acc = model.evaluate(test_data_iter)
    print(f"Test Accuracy: {test_acc:.4f}")

    model.save("cifar10.h5")
    
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
predictions = model.predict(X_test[:6])
for i in range(6):
    plt.subplot(2, 3, i + 1)
    img = (X_test[i] - X_test[i].min()) / (X_test[i].max() - X_test[i].min())
    plt.imshow(img)
    true_label = np.argmax(Y_test[i])
    predicted_label = np.argmax(predictions[i])
    plt.title(f"True: {true_label} Pred: {predicted_label}")
    plt.axis('off')
plt.tight_layout()
plt.show()
