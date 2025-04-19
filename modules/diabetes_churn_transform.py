import tensorflow as tf
import tensorflow_transform as tft

CATEGORICAL_FEATURES = {
    "gender": 2,
    "smoking_history": 3,
}

NUMERICAL_FEATURES = [
    "age",
    "hypertension",
    "heart_disease",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
]

LABEL_KEY = "diabetes"


def transformed_name(key: str) -> str:
    """
    Generate the transformed name for a feature.

    Args:
        key (str): The original feature name.

    Returns:
        str: Transformed feature name.
    """
    return f"{key}_xf"


def convert_to_one_hot(tensor: tf.Tensor, num_classes: int) -> tf.Tensor:
    """
    Convert a tensor to a one-hot encoded tensor.

    Args:
        tensor (tf.Tensor): Input tensor with integer values.
        num_classes (int): Number of classes for one-hot encoding.

    Returns:
        tf.Tensor: One-hot encoded tensor.
    """
    one_hot = tf.one_hot(tensor, num_classes)
    return tf.reshape(one_hot, [-1, num_classes])


def process_categorical_features(inputs: dict, outputs: dict) -> None:
    """
    Process categorical features using vocabulary and one-hot encoding.

    Args:
        inputs (dict): Input raw feature tensors.
        outputs (dict): Output dictionary for transformed features.
    """
    for feature_name, num_categories in CATEGORICAL_FEATURES.items():
        integer_values = tft.compute_and_apply_vocabulary(
            inputs[feature_name], top_k=num_categories + 1
        )
        outputs[transformed_name(feature_name)] = convert_to_one_hot(
            integer_values, num_classes=num_categories + 1
        )


def process_numerical_features(inputs: dict, outputs: dict) -> None:
    """
    Process numerical features by scaling them to a [0, 1] range.

    Args:
        inputs (dict): Input raw feature tensors.
        outputs (dict): Output dictionary for transformed features.
    """
    for feature_name in NUMERICAL_FEATURES:
        outputs[transformed_name(feature_name)] = tft.scale_to_0_1(inputs[feature_name])


def process_label(inputs: dict, outputs: dict) -> None:
    """
    Process the label by casting it to an integer type.

    Args:
        inputs (dict): Input raw feature tensors.
        outputs (dict): Output dictionary for transformed features.
    """
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)


def preprocessing_fn(inputs: dict) -> dict:
    """
    Preprocess input features into transformed features.

    Args:
        inputs (dict): Map from feature keys to raw feature tensors.

    Returns:
        dict: Map from feature keys to transformed feature tensors.
    """
    outputs = {}

    process_categorical_features(inputs, outputs)
    process_numerical_features(inputs, outputs)
    process_label(inputs, outputs)

    return outputs
