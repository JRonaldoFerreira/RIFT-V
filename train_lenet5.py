"""Train LeNet-5 on MNIST for 12 epochs and save as .h5."""

import tensorflow as tf

# ── LeNet-5 (classic architecture adapted for MNIST 28×28×1) ──────────
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(6, 5, activation="relu", padding="same", input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(16, 5, activation="relu"),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120, activation="relu"),
    tf.keras.layers.Dense(84, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ── Load MNIST ────────────────────────────────────────────────────────
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train[..., None] / 255.0
x_test = x_test[..., None] / 255.0

# ── Train ─────────────────────────────────────────────────────────────
model.fit(
    x_train, y_train,
    epochs=12,
    batch_size=128,
    validation_data=(x_test, y_test),
)

# ── Evaluate ──────────────────────────────────────────────────────────
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}")

# ── Save ──────────────────────────────────────────────────────────────
model.save("models/lenet5_mnist.h5")
print("Model saved → models/lenet5_mnist.h5")
