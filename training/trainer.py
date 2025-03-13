import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"❌ Error setting memory growth: {e}")

# Enable Mixed Precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

IMG_SIZE = (1920, 1080)
BATCH_SIZE = 2
EPOCHS = 50


def parse_filename(filename):
    """Extracts country, latitude, longitude from filenames."""
    base = os.path.splitext(filename)[0]
    parts = base.split("_")
    if len(parts) < 3:
        return None, None, None
    try:
        return parts[0], float(parts[1]), float(parts[2])
    except ValueError:
        return None, None, None


def load_image_and_label(filepath):
    """Loads and preprocesses an image with label extraction."""
    image = tf.io.read_file(filepath)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0  # Normalize

    filename = tf.strings.split(filepath, os.sep)[-1]

    def _parse_fn(fname):
        fname_str = fname.numpy().decode('utf-8')
        _, lat, lon = parse_filename(fname_str)
        return lat if lat is not None else 0.0, lon if lon is not None else 0.0

    lat, lon = tf.py_function(func=_parse_fn, inp=[filename], Tout=[tf.float32, tf.float32])

    return image, tf.stack([lat, lon])


def create_dataset_from_directory(directory, validation_split=0.2):
    """Creates a tf.data dataset from images in 'directory'."""
    all_filepaths = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(".png")]
    train_paths, val_paths = train_test_split(all_filepaths, test_size=validation_split, random_state=42)

    train_ds = tf.data.Dataset.from_tensor_slices(train_paths).map(
        load_image_and_label, num_parallel_calls=tf.data.AUTOTUNE
    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices(val_paths).map(
        load_image_and_label, num_parallel_calls=tf.data.AUTOTUNE
    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


# ----------------------------------------------------
# Haversine Distance Metric (Miles)
# ----------------------------------------------------

class HaversineDistance(tf.keras.metrics.Metric):
    """
    Custom metric to compute the Haversine distance (miles) between true and predicted coordinates.
    """

    def __init__(self, name="haversine_miles", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_distance = self.add_weight(name="total_distance", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Computes the Haversine distance in miles between the true and predicted lat/lon coordinates.
        """
        R = 3958.8

        lat1, lon1 = y_true[:, 0], y_true[:, 1]
        lat2, lon2 = y_pred[:, 0], y_pred[:, 1]

        lat1, lon1, lat2, lon2 = [x * (np.pi / 180.0) for x in [lat1, lon1, lat2, lon2]]

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = tf.sin(dlat / 2.0) ** 2 + tf.cos(lat1) * tf.cos(lat2) * tf.sin(dlon / 2.0) ** 2
        c = 2 * tf.atan2(tf.sqrt(a), tf.sqrt(1 - a))

        distance = R * c

        self.total_distance.assign_add(tf.reduce_sum(distance))
        self.count.assign_add(tf.cast(tf.size(distance), tf.float32))

    def result(self):
        """Returns the average Haversine distance over all updates."""
        return self.total_distance / self.count

    def reset_state(self):
        """Resets metric state at the start of each epoch."""
        self.total_distance.assign(0.0)
        self.count.assign(0.0)


# ----------------------------------------------------
# CNN Model (ResNet-style)
# ----------------------------------------------------
def create_advanced_regression_model(input_shape=(1920, 1080, 3)):
    inputs = keras.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(128, (5, 5), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)

    # Residual Block (ResNet-style)
    res = layers.Conv2D(256, (1, 1), padding='same')(x)  # **FIXED to 256**
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.Add()([x, res])  # Now both have 256 channels
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Additional Conv Blocks
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Flatten & Dense Layers
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(2, dtype=tf.float32)(x)

    return keras.Model(inputs=inputs, outputs=outputs)


# ----------------------------------------------------
# Training
# ----------------------------------------------------
def main():
    data_dir = r"C:\Users\fishd\PycharmProjects\Geoguessr AI\database\temp_storage\completed"

    train_ds, val_ds = create_dataset_from_directory(directory=data_dir)

    model = create_advanced_regression_model()

    model.load_weights(r"C:\Users\fishd\PycharmProjects\Geoguessr AI\training\best_model\627miles.h5")

    checkpoint_dir = "best_model"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "geoguessr_best_model.h5"),
        monitor="val_haversine_miles",
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        verbose=1
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-7),
        loss="mean_absolute_error",
        metrics=[HaversineDistance()]
    )

    model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=[checkpoint_callback]
    )
    model.save("geoguessr_cnn_model.h5", save_weights_only=True)
    print("Model training complete.")


if __name__ == "__main__":
    main()
