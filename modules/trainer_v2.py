"""
Module trainer_v2.py
Berisi fungsi training dan tuning untuk pipeline TFX.
Menggunakan Keras Tuner untuk hyperparameter tuning dan TensorFlow untuk training model.
"""

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow import keras
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.components.tuner.component import TunerFnResult
import keras_tuner as kt
from typing import List, Text, Dict, Optional, Any

# --- Konstanta ---
LABEL_KEY = 'Churn'

def transformed_name(key: Text) -> Text:
    """Menghasilkan nama fitur hasil transformasi."""
    return key + "_xf"

def _gzip_reader_fn(filenames: List[Text]) -> tf.data.TFRecordDataset:
    """Membaca TFRecord dengan kompresi GZIP."""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def _input_fn(
    file_pattern: List[Text],
    tf_transform_output: tft.TFTransformOutput,
    num_epochs: Optional[int] = None,
    batch_size: int = 64
) -> tf.data.Dataset:
    """Membuat input pipeline untuk training/eval."""
    
    # Ambil feature spec dari output transform
    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )

    # Buat dataset dari TFRecord
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=_gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY),
    )
    return dataset

def _build_keras_model(
    hp: kt.HyperParameters,
    tf_transform_output: tft.TFTransformOutput
) -> keras.Model:
    """Bangun model Keras untuk training atau tuning."""
    
    feature_spec = tf_transform_output.transformed_feature_spec()
    inputs = {}

    # Definisikan Input Layer secara dinamis berdasarkan schema
    for key, spec in feature_spec.items():
        if key == transformed_name(LABEL_KEY):
            continue
        
        if isinstance(spec, tf.io.VarLenFeature):
            inputs[key] = keras.Input(
                shape=(1,), name=key, dtype=spec.dtype, sparse=True
            )
        else:
            inputs[key] = keras.Input(
                shape=spec.shape, name=key, dtype=spec.dtype
            )

    # Preprocessing inputs
    processed_inputs = []
    for key, tensor in inputs.items():
        # 1. Handle Sparse Tensor (Categorical)
        if isinstance(tensor, tf.SparseTensor):
            tensor = tf.sparse.to_dense(tensor)
        
        # 2. [FIX UTAMA] Pastikan semua tensor berbentuk 2D (batch, 1)
        # Ini mengatasi error: expected min_ndim=2, found ndim=1
        tensor = tf.reshape(tensor, [-1, 1])
        
        # 3. Cast ke float32 agar bisa masuk ke Neural Network
        if tensor.dtype == tf.int64:
            tensor = tf.cast(tensor, tf.float32)
        
        processed_inputs.append(tensor)

    # Gabungkan semua fitur (sekarang semua sudah pasti 2D)
    concatenated = keras.layers.Concatenate()(processed_inputs)

    # --- Definisi Hyperparameters ---
    units = hp.Int('units', min_value=32, max_value=128, step=32)
    num_layers = hp.Int('num_layers', min_value=1, max_value=3)
    lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    x = concatenated
    for _ in range(num_layers):
        x = keras.layers.Dense(units, activation='relu')(x)
        x = keras.layers.Dropout(0.2)(x)

    outputs = keras.layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        metrics=[keras.metrics.BinaryAccuracy()],
    )
    
    model.summary(print_fn=lambda x: None)
    return model

def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Mengembalikan fungsi parsing untuk serving signature."""
    
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY) # Label tidak ada saat serving
        
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        
        return model(transformed_features)

    return serve_tf_examples_fn

# --- TFX Component Functions ---

def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    """Fungsi utama untuk komponen Tuner."""
    
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Gunakan batch size lebih kecil & epoch sedikit untuk tuning
    train_set = _input_fn(fn_args.train_files, tf_transform_output, num_epochs=5)
    eval_set = _input_fn(fn_args.eval_files, tf_transform_output, num_epochs=5)

    tuner = kt.Hyperband(
        hypermodel=lambda hp: _build_keras_model(hp, tf_transform_output),
        objective=kt.Objective('val_binary_accuracy', direction='max'),
        max_epochs=5,
        factor=3,
        directory=fn_args.working_dir,
        project_name='telco_churn_kt'
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            'x': train_set,
            'validation_data': eval_set,
            'steps_per_epoch': fn_args.train_steps,
            'validation_steps': fn_args.eval_steps,
        }
    )

def run_fn(fn_args: FnArgs):
    """Fungsi utama untuk komponen Trainer."""
    
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(fn_args.train_files, tf_transform_output, num_epochs=20)
    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output, num_epochs=1)

    # Load Best Hyperparameters
    if fn_args.hyperparameters:
        hp = kt.HyperParameters.from_config(fn_args.hyperparameters)
    else:
        hp = kt.HyperParameters()
        hp.Int('units', 64)
        hp.Int('num_layers', 2)
        hp.Choice('learning_rate', [1e-3])

    model = _build_keras_model(hp, tf_transform_output)

    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=fn_args.model_run_dir, 
        update_freq='batch'
    )

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback]
    )

    signatures = {
        'serving_default': _get_serve_tf_examples_fn(
            model, tf_transform_output).get_concrete_function(
                tf.TensorSpec(shape=[None], dtype=tf.string, name='examples'))
    }
    
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)