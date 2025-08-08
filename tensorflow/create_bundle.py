import keras


if __name__ == "__main__":
    mnist = keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = keras.models.Sequential(
        [
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10),
        ]
    )

    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

    model.fit(x_train, y_train, epochs=1)

    model.save_weights("model_tf.weights.h5")
    print("Model weights saved into ./model_tf.weights.h5")
