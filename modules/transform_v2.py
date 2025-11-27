import tensorflow as tf
import tensorflow_transform as tft

LABEL_KEY = "Churn"

NUMERIC_FEATURE_KEYS = [
    'tenure',
    'MonthlyCharges',
    'TotalCharges',
    'SeniorCitizen',
]

CATEGORICAL_FEATURE_KEYS = [
    'gender',
    'Partner',
    'Dependents',
    'PhoneService',
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaperlessBilling',
    'PaymentMethod',
]


def transformed_name(key):
    return key + "_xf"


def _to_float(x):
    """Convert string numeric → float. Replace empty with 0."""
    x = tf.strings.regex_replace(x, r"^\s*$", "0")
    return tf.strings.to_number(x, out_type=tf.float32)


def preprocessing_fn(inputs):
    outputs = {}

    # NUMERIC ------------------------------------------------
    for key in NUMERIC_FEATURE_KEYS:
        raw = tf.squeeze(inputs[key], axis=1)

        # jika string → ubah ke float dulu
        if raw.dtype == tf.string:
            raw = _to_float(raw)

        outputs[transformed_name(key)] = tft.scale_to_z_score(raw)

    # CATEGORICAL -------------------------------------------
    for key in CATEGORICAL_FEATURE_KEYS:
        raw = tf.squeeze(inputs[key], axis=1)
        outputs[transformed_name(key)] = tft.compute_and_apply_vocabulary(raw)

    # LABEL --------------------------------------------------
    label = tf.squeeze(inputs[LABEL_KEY], axis=1)
    outputs[transformed_name(LABEL_KEY)] = tft.compute_and_apply_vocabulary(
        label, vocab_filename="label_vocab"
    )

    return outputs
