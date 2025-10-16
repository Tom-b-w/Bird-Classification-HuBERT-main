import json
import os
import pandas as pd
import numpy as np
import random
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import warnings

warnings.filterwarnings("ignore")

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def load_dataset(data_dir):
    return pd.DataFrame([{'path': os.path.join(sp_path, f).replace("\\", "/"), 'class_name': species}
                         for species in os.listdir(data_dir)
                         for sp_path in [os.path.join(data_dir, species)]
                         if os.path.isdir(sp_path)
                         for f in os.listdir(sp_path)
                         if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')) and os.path.getsize(
            os.path.join(sp_path, f)) > 0])


def get_spec(path):
    try:
        y, sr = librosa.load(path)
        return librosa.feature.melspectrogram(y=y, sr=sr)
    except:
        return None


def extract_features(df, max_time_steps=65, n_features=40):
    features, labels = [], []
    for i in range(len(df)):
        spec = get_spec(df['path'].iloc[i])
        if spec is not None:
            features.append(np.resize(spec, (max_time_steps, n_features)))
            labels.append(df['class_label'].iloc[i])
    return np.asarray(features), np.asarray(labels)


class attention(layers.Layer):
    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1), initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1), initializer='zeros', trainable=True)
        super().build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        alpha = K.softmax(K.squeeze(e, axis=-1))
        return K.sum(x * K.expand_dims(alpha, axis=-1), axis=1)

    def get_config(self):
        return super().get_config()


def build_cnn_lstm_model(max_time_steps, n_features, num_classes):
    tf.keras.backend.clear_session()
    input_layer = layers.Input(shape=(max_time_steps, n_features))
    x = layers.Conv1D(128, 3, activation='relu')(input_layer)
    x = layers.Conv1D(64, 3, activation='relu')(x)
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = attention()(x)
    x = layers.Dense(128, 'relu')(x)
    x = layers.Dense(64, 'relu')(x)
    output = layers.Dense(num_classes, 'softmax')(x)
    return Model(inputs=input_layer, outputs=output)


def main():
    MAX_TIME_STEPS, N_FEATURES, EPOCHS, BATCH_SIZE, LR = 65, 40, 40, 32, 0.001

    df_train = load_dataset('./dataset/whole_dataset/train_val/train_data_backup')
    df_val = load_dataset('./dataset/whole_dataset/train_val/validation_data_backup')

    cls_label = {s: i for i, s in enumerate(sorted(df_train['class_name'].unique()))}
    NUM_CLASSES = len(cls_label)

    df_train['class_label'] = df_train['class_name'].map(cls_label)
    df_val['class_label'] = df_val['class_name'].map(cls_label)
    df_train = df_train[df_train['class_label'] != -1]
    df_val = df_val[df_val['class_label'] != -1]

    X_train, y_train = extract_features(df_train, MAX_TIME_STEPS, N_FEATURES)
    X_val, y_val = extract_features(df_val, MAX_TIME_STEPS, N_FEATURES)

    unique_labels = np.unique(y_train)
    class_weight_dict = {int(label): float(weight) for label, weight in
                         zip(unique_labels, compute_class_weight('balanced', classes=unique_labels, y=y_train))} if len(
        unique_labels) > 1 else None

    model = build_cnn_lstm_model(MAX_TIME_STEPS, N_FEATURES, NUM_CLASSES)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR), loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        tf.keras.callbacks.ModelCheckpoint('best_label_model.keras', monitor='val_accuracy', save_best_only=True,
                                           mode='max', verbose=1)
    ]

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE,
                        class_weight=class_weight_dict, callbacks=callbacks, verbose=1)

    train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)

    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    f1_weighted = f1_score(y_val, y_pred, average='weighted')
    f1_macro = f1_score(y_val, y_pred, average='macro')
    f1_micro = f1_score(y_val, y_pred, average='micro')

    print(f"\n训练: {train_loss:.4f}/{train_accuracy:.4f} - 验证: {val_loss:.4f}/{val_accuracy:.4f}")
    print(f"F1 (加权/宏/微): {f1_weighted:.4f}/{f1_macro:.4f}/{f1_micro:.4f}")
    print("\n" + classification_report(y_val, y_pred,
                                       target_names=[s for s, _ in sorted(cls_label.items(), key=lambda x: x[1])]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metrics, ylabel in [(ax1, ['accuracy', 'val_accuracy'], 'Accuracy'), (ax2, ['loss', 'val_loss'], 'Loss')]:
        for m in metrics:
            ax.plot(history.history[m],
                    label=m.replace('val_', '验证').replace('accuracy', '准确率').replace('loss', '损失'))
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('single_label_training_history.png', dpi=300)

    model.save('single_label_bird_classifier.keras')
    with open('single_label_bird_config.json', 'w', encoding='utf-8') as f:
        json.dump({"model_type": "CNN-LSTM-Attention", "num_classes": NUM_CLASSES, "cls_label": cls_label,
                   "performance_metrics": {"train_loss": float(train_loss), "train_accuracy": float(train_accuracy),
                                           "val_loss": float(val_loss), "val_accuracy": float(val_accuracy),
                                           "f1_weighted": float(f1_weighted), "f1_macro": float(f1_macro),
                                           "f1_micro": float(f1_micro)}},
                  f, indent=2, ensure_ascii=False)

    del X_train, X_val, y_train, y_val
    import gc
    gc.collect()

    print(f"\n训练完成，最佳准确率: {val_accuracy:.4f}")
    return {'model': model, 'cls_label': cls_label, 'val_accuracy': val_accuracy, 'f1_weighted': f1_weighted,
            'num_classes': NUM_CLASSES}


if __name__ == '__main__':
    results = main()
    if results:
        print(
            f"\n鸟类种数: {results['num_classes']}, 验证准确率: {results['val_accuracy']:.4f}, F1: {results['f1_weighted']:.4f}")