from pathlib import Path

import tensorflow as tf
from flyteplugins.tensorflow.task import Tensorflow
from tensorflow.keras import datasets, layers, models

import flyte

image = (
    flyte.Image.from_debian_base(name="tensorflow")
    .with_pip_packages("tensorflow==2.20.0")
    .with_source_folder(Path(__file__).parent.parent.parent / "plugins/tensorflow", "./tensorflow")
    .with_env_vars({"PYTHONPATH": "./tensorflow/src:${PYTHONPATH}"})
    .with_local_v2()
)

tensorflow_env = flyte.TaskEnvironment(
    name="tensorflow-env",
    resources=flyte.Resources(cpu=(1, 2), memory=("1Gi", "2Gi")),
    plugin_config=Tensorflow(),
    image=image,
)


def train_and_evaluate_mnist(epochs: int = 5, use_gpu: bool = False):
    if not use_gpu:
        tf.config.set_visible_devices([], "GPU")

    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    model = models.Sequential(
        [
            layers.Flatten(input_shape=(28, 28)),
            layers.Dense(128, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    model.fit(x_train, y_train, epochs=epochs, validation_split=0.1)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test accuracy: {test_acc:.4f}")

    predictions = model.predict(x_test[:5])
    print("Predicted labels:", predictions.argmax(axis=1))
    print("True labels:", y_test[:5])

    return model, (x_test, y_test)


@tensorflow_env.task
def tensorflow_train(epochs: int) -> None:
    """
    A Flyte task that trains a simple TensorFlow model.
    """
    a, b = train_and_evaluate_mnist(epochs=epochs)
    print(a)
    print(b)

    print("Training complete.")


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(mode="remote").run(tensorflow_train, epochs=3)
    print("run name:", run.name)
    print("run url:", run.url)
