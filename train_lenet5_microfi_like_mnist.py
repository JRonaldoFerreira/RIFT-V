# -*- coding: utf-8 -*-
"""
Treina uma LeNet-5 "MicroFI-like" em MNIST e salva o modelo em .h5.

Arquitetura usada:
    Input 32x32x1
    Conv1: 6 filtros 5x5
    AvgPool
    Conv2: 16 filtros 5x5
    AvgPool
    Conv3: 120 filtros 5x5
    FC1: 84 neurônios
    FC2: 10 classes

Observação:
- O artigo MicroFI informa LeNet-5 com 3 camadas convolucionais e 2 fully connected.
- Isso corresponde à leitura clássica da LeNet-5 onde a camada C5 é tratada como Conv2D 5x5.
- O paper não divulga todos os hiperparâmetros de treino; este script usa uma configuração reprodutível.
"""

import os
import random
import argparse
import numpy as np
import tensorflow as tf


def set_seed(seed: int = 42) -> None:
    """Fixa sementes para tornar o treinamento mais reprodutível."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_mnist_32x32():
    """
    Carrega MNIST, normaliza para [0, 1] e aplica padding 28x28 -> 32x32.

    A LeNet-5 clássica usa entrada 32x32.
    MNIST original vem como 28x28, então adicionamos 2 pixels de borda em cada lado.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # (N, 28, 28) -> (N, 28, 28, 1)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # Padding: 28x28 -> 32x32
    x_train = np.pad(
        x_train,
        pad_width=((0, 0), (2, 2), (2, 2), (0, 0)),
        mode="constant",
        constant_values=0.0,
    )
    x_test = np.pad(
        x_test,
        pad_width=((0, 0), (2, 2), (2, 2), (0, 0)),
        mode="constant",
        constant_values=0.0,
    )

    return (x_train, y_train), (x_test, y_test)


def build_lenet5_microfi_like(activation: str = "tanh") -> tf.keras.Model:
    """
    Cria uma LeNet-5 com 3 Conv + 2 FC.

    Por padrão uso tanh + AveragePooling, que é mais próximo da LeNet-5 clássica.
    Se quiser uma versão mais moderna, troque activation="relu".
    """
    model = tf.keras.Sequential(
        name="LeNet5_MicroFI_like",
        layers=[
            tf.keras.layers.Input(shape=(32, 32, 1), name="Input_32x32"),

            # C1: 32x32x1 -> 28x28x6
            tf.keras.layers.Conv2D(
                filters=6,
                kernel_size=(5, 5),
                strides=(1, 1),
                padding="valid",
                activation=activation,
                name="Conv1",
            ),

            # S2: 28x28x6 -> 14x14x6
            tf.keras.layers.AveragePooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                name="Pool1",
            ),

            # C3: 14x14x6 -> 10x10x16
            tf.keras.layers.Conv2D(
                filters=16,
                kernel_size=(5, 5),
                strides=(1, 1),
                padding="valid",
                activation=activation,
                name="Conv2",
            ),

            # S4: 10x10x16 -> 5x5x16
            tf.keras.layers.AveragePooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                name="Pool2",
            ),

            # C5: 5x5x16 -> 1x1x120
            # Aqui está a "terceira convolucional" citada como Conv3.
            tf.keras.layers.Conv2D(
                filters=120,
                kernel_size=(5, 5),
                strides=(1, 1),
                padding="valid",
                activation=activation,
                name="Conv3",
            ),

            tf.keras.layers.Flatten(name="Flatten"),

            # F6
            tf.keras.layers.Dense(
                units=84,
                activation=activation,
                name="FC1",
            ),

            # Output: 10 classes MNIST
            tf.keras.layers.Dense(
                units=10,
                activation="softmax",
                name="FC2",
            ),
        ],
    )

    return model


def save_100_image_subset(x_test, y_test, seed: int, out_path: str) -> None:
    """
    Salva uma subamostra de 100 imagens de teste.

    O MicroFI usa 100 imagens MNIST na campanha embarcada por restrição de memória.
    Este arquivo ajuda a manter a campanha do RIFT-V reprodutível.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x_test), size=100, replace=False)

    np.savez_compressed(
        out_path,
        x=x_test[idx].astype("float32"),
        y=y_test[idx].astype("int64"),
        indices=idx.astype("int64"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--activation", type=str, default="tanh", choices=["tanh", "relu"])
    parser.add_argument("--out-h5", type=str, default="lenet5_microfi_like_mnist.h5")
    parser.add_argument("--out-subset", type=str, default="mnist_100_subset_microfi_like.npz")
    args = parser.parse_args()

    set_seed(args.seed)

    (x_train, y_train), (x_test, y_test) = load_mnist_32x32()

    model = build_lenet5_microfi_like(activation=args.activation)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            mode="max",
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath="best_" + args.out_h5,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    print("\n[RESULTADO]")
    print(f"Test loss: {test_loss:.6f}")
    print(f"Test accuracy: {test_acc:.6f}")

    # Salva o modelo final no formato HDF5 (.h5)
    model.save(args.out_h5)
    print(f"\n[OK] Modelo salvo em: {args.out_h5}")

    # Salva uma subamostra de 100 imagens para campanha embarcada/reprodutível
    save_100_image_subset(x_test, y_test, args.seed, args.out_subset)
    print(f"[OK] Subset MNIST com 100 imagens salvo em: {args.out_subset}")

    # Salva um log simples com hiperparâmetros e acurácia
    with open("training_report_lenet5_microfi_like.txt", "w", encoding="utf-8") as f:
        f.write("LeNet-5 MicroFI-like MNIST\n")
        f.write("===========================\n")
        f.write(f"activation: {args.activation}\n")
        f.write(f"epochs_requested: {args.epochs}\n")
        f.write(f"batch_size: {args.batch_size}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"test_loss: {test_loss:.8f}\n")
        f.write(f"test_accuracy: {test_acc:.8f}\n")
        f.write(f"h5_model: {args.out_h5}\n")
        f.write(f"subset_100_images: {args.out_subset}\n")
        f.write("\nModel summary:\n")
        model.summary(print_fn=lambda line: f.write(line + "\n"))

    print("[OK] Relatório salvo em: training_report_lenet5_microfi_like.txt")


if __name__ == "__main__":
    main()
