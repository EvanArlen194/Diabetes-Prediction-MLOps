import os
import tensorflow as tf
import tensorflow_transform as tft
from diabetes_churn_transform import (
    CATEGORICAL_FEATURES,
    LABEL_KEY,
    NUMERICAL_FEATURES,
    transformed_name,
)

def build_model(show_summary: bool = True) -> tf.keras.Model:
    """
    Builds and compiles a Keras model.

    Args:
        show_summary (bool): Whether to display the model summary.

    Returns:
        tf.keras.Model: A compiled Keras model.
    """
    input_layers = [
        tf.keras.Input(shape=(dim + 1,), name=transformed_name(feature))
        for feature, dim in CATEGORICAL_FEATURES.items()
    ]

    input_layers += [
        tf.keras.Input(shape=(1,), name=transformed_name(feature))
        for feature in NUMERICAL_FEATURES
    ]

    concatenated_inputs = tf.keras.layers.concatenate(input_layers)
    x = tf.keras.layers.Dense(256, activation="relu")(concatenated_inputs)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.models.Model(inputs=input_layers, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy()],
    )

    if show_summary:
        model.summary()

    return model

def gzip_reader_fn(filenames: str) -> tf.data.TFRecordDataset:
    """
    Reads compressed TFRecord files.

    Args:
        filenames (str): The file pattern to read.

    Returns:
        tf.data.TFRecordDataset: The dataset object.
    """
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")


def input_fn(
    file_pattern: str, tf_transform_output: tft.TFTransformOutput, batch_size: int = 64
) -> tf.data.Dataset:
    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()

    return tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=gzip_reader_fn,
        label_key=transformed_name(LABEL_KEY),
    )

def get_serve_tf_examples_fn(
    model: tf.keras.Model, tf_transform_output: tft.TFTransformOutput
) -> tf.function:
    """
    Creates a serving function for the model.

    Args:
        model (tf.keras.Model): The trained model.
        tf_transform_output (tft.TFTransformOutput): The TFT output object.

    Returns:
        tf.function: A serving function.
    """
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples: tf.Tensor) -> dict:
        """
        Parses serialized tf.Example for serving.

        Args:
            serialized_tf_examples (tf.Tensor): Serialized examples.

        Returns:
            dict: Model outputs.
        """
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        return {"outputs": model(transformed_features)}

    return serve_tf_examples_fn

def run_fn(fn_args):
    """
    Runs the TFX training pipeline.

    Args:
        fn_args: Holds arguments used to train the model.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = input_fn(fn_args.train_files, tf_transform_output, batch_size=64)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, batch_size=64)

    model = build_model()

    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), "logs")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq="batch")

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback],
        epochs=10,
    )

    serving_fn = get_serve_tf_examples_fn(model, tf_transform_output)
    signatures = {
        "serving_default": serving_fn.get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        ),
    }

    model.save(fn_args.serving_model_dir, save_format="tf", signatures=signatures)
