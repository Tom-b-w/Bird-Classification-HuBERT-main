import os
import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm
import random
import warnings

warnings.filterwarnings("ignore")

from pydub import AudioSegment
from pydub.utils import make_chunks
from IPython.display import Audio

import librosa
import librosa.display
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

from sklearn.metrics import f1_score, classification_report
from scipy.sparse import lil_matrix
from sklearn.utils.class_weight import compute_class_weight

# GPU配置
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"发现 {len(gpus)} 个GPU设备")
    except RuntimeError as e:
        print(e)


# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(42)


# 数据增强类
class AudioAugmentation:
    """针对鸟鸣的音频数据增强"""

    def __init__(self, sampling_rate=22050):
        self.sr = sampling_rate

    def time_stretch(self, audio, rate_range=(0.9, 1.1)):
        """时间拉伸"""
        rate = np.random.uniform(*rate_range)
        return librosa.effects.time_stretch(audio, rate=rate)

    def pitch_shift(self, audio, n_steps_range=(-1, 1)):
        """音调变换"""
        n_steps = np.random.uniform(*n_steps_range)
        return librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=n_steps)

    def add_noise(self, audio, noise_factor_range=(0.001, 0.003)):
        """添加白噪声"""
        noise_factor = np.random.uniform(*noise_factor_range)
        noise = np.random.randn(len(audio))
        return audio + noise_factor * noise

    def volume_change(self, audio, gain_range=(0.8, 1.2)):
        """音量变化"""
        gain = np.random.uniform(*gain_range)
        return audio * gain


# 数据加载函数
def load_dataset(data_dir, dataset_name="dataset"):
    """加载数据集"""
    label = []
    path = []

    if not os.path.exists(data_dir):
        print(f"警告: 目录不存在 {data_dir}")
        return pd.DataFrame()

    for species in os.listdir(data_dir):
        species_folder = os.path.join(data_dir, species)
        if os.path.isdir(species_folder):
            for audio in os.listdir(species_folder):
                if audio.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                    audio_path = os.path.join(species_folder, audio)
                    path.append(audio_path)
                    label.append(species)

    df = pd.DataFrame({'path': path, 'class_name': label})
    print(f"{dataset_name}数据集大小: {df.shape}")
    return df


# 创建类别映射
def create_class_mapping(df_train):
    """根据训练数据创建类别映射"""
    unique_species = sorted(df_train['class_name'].unique())
    cls_label = {species: idx for idx, species in enumerate(unique_species)}
    print(f"发现 {len(unique_species)} 种鸟类:")
    for species, idx in cls_label.items():
        count = len(df_train[df_train['class_name'] == species])
        print(f"  {species}: {idx} (样本数: {count})")
    return cls_label, len(unique_species)


# 改进的MFCC提取函数
def get_mfcc_enhanced(path, augmentation=None, is_training=False, augment_prob=0.3):
    """增强版MFCC特征提取"""
    try:
        y, sr = librosa.load(path, sr=22050)

        # 预处理
        y = librosa.effects.preemphasis(y, coef=0.97)

        # 训练时应用数据增强
        if is_training and augmentation is not None and np.random.random() < augment_prob:
            # 随机选择增强方法
            if np.random.random() < 0.3:
                try:
                    y = augmentation.time_stretch(y)
                except:
                    pass

            if np.random.random() < 0.2:
                try:
                    y = augmentation.pitch_shift(y)
                except:
                    pass

            if np.random.random() < 0.3:
                y = augmentation.add_noise(y)

            if np.random.random() < 0.2:
                y = augmentation.volume_change(y)

        # 音量标准化
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))

        # 提取Mel频谱图和MFCC
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, fmax=8000)
        mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=40)

        # 添加一阶和二阶差分特征
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)

        # 合并特征
        features = np.vstack([mfccs, delta_mfccs, delta2_mfccs])

        return features.T
    except Exception as e:
        print(f"处理音频失败: {path}, 错误: {e}")
        return None


# 数据预处理配置
MAX_TIME_STEPS = 65
N_FEATURES = 120  # 40 MFCC + 40 delta + 40 delta2

# 主数据加载路径配置
train_dir = './dataset/whole_dataset/train_val/train_data'
val_dir = './dataset/whole_dataset/train_val/validation_data'

# 加载训练和验证数据
print("=== 加载数据集 ===")
df_train = load_dataset(train_dir, "训练")
df_prediction = load_dataset(val_dir, "验证")

if not df_train.empty:
    print("训练集样本:")
    print(df_train.sample(min(5, len(df_train))))

# 创建类别映射
cls_label, NUM_CLASSES = create_class_mapping(df_train)


# 添加数字标签
def name_to_label(x):
    cls_name = x['class_name']
    if cls_name in cls_label:
        return cls_label[cls_name]
    else:
        print(f"警告: 未知类别 {cls_name}")
        return 0


df_train['class_label'] = df_train.apply(name_to_label, axis=1)
df_prediction['class_label'] = df_prediction.apply(name_to_label, axis=1)

# 数据平衡处理
print("\n=== 数据平衡处理 ===")
min_samples_per_class = 50
max_samples_per_class = 150

balanced_train_data = []
for class_name in cls_label.keys():
    class_data = df_train[df_train['class_name'] == class_name]
    n_samples = len(class_data)

    if n_samples < min_samples_per_class:
        print(f"警告: 类别 {class_name} 样本数过少 ({n_samples})")
        if n_samples > 0:
            repeat_factor = min_samples_per_class // n_samples + 1
            class_data = pd.concat([class_data] * repeat_factor, ignore_index=True)
            class_data = class_data.iloc[:min_samples_per_class]
    else:
        # 限制最大样本数
        class_data = class_data.sample(n=min(n_samples, max_samples_per_class), random_state=42)

    balanced_train_data.append(class_data)
    print(f"类别 {class_name}: {len(class_data)} 个样本")

df_train = pd.concat(balanced_train_data, ignore_index=True)
df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

# 验证集处理
val_samples_per_class = 30
balanced_val_data = []
for class_name in cls_label.keys():
    class_data = df_prediction[df_prediction['class_name'] == class_name]
    n_samples = len(class_data)

    if n_samples > 0:
        if n_samples < val_samples_per_class:
            repeat_factor = val_samples_per_class // n_samples + 1
            class_data = pd.concat([class_data] * repeat_factor, ignore_index=True)
            class_data = class_data.iloc[:val_samples_per_class]
        else:
            class_data = class_data.sample(n=val_samples_per_class, random_state=42)

        balanced_val_data.append(class_data)

df_prediction = pd.concat(balanced_val_data, ignore_index=True)
df_prediction = df_prediction.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"平衡后 - 训练集: {df_train.shape[0]}, 验证集: {df_prediction.shape[0]}")

# 创建音频增强器
augmentation = AudioAugmentation(sampling_rate=22050)

# 提取训练集特征（带增强）
print("\n=== 提取训练集特征（带数据增强） ===")
X_train = df_train['path']
y_train = df_train['class_label']

temp = []
label = []
for i in tqdm(range(len(X_train))):
    audio_path = X_train.iloc[i]
    mfcc = get_mfcc_enhanced(audio_path, augmentation, is_training=True, augment_prob=0.4)
    if mfcc is not None:
        # 调整到固定大小
        if mfcc.shape[0] < MAX_TIME_STEPS:
            # 填充
            pad_width = MAX_TIME_STEPS - mfcc.shape[0]
            mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
        elif mfcc.shape[0] > MAX_TIME_STEPS:
            # 截断
            mfcc = mfcc[:MAX_TIME_STEPS, :]

        # 确保特征维度正确
        if mfcc.shape[1] != N_FEATURES:
            mfcc = np.resize(mfcc, (MAX_TIME_STEPS, N_FEATURES))

        temp.append(mfcc)
        label.append(y_train.iloc[i])

X_train_processed = np.asarray(temp)
y_train_processed = np.asarray(label)

print(f"训练集特征形状: {X_train_processed.shape}")

# 提取验证集特征（无增强）
print("\n=== 提取验证集特征（无数据增强） ===")
X_prediction = df_prediction['path']
y_prediction = df_prediction['class_label']

temp = []
label = []

for i in tqdm(range(len(X_prediction))):
    audio_path = X_prediction.iloc[i]
    mfcc = get_mfcc_enhanced(audio_path, None, is_training=False, augment_prob=0.0)
    if mfcc is not None:
        # 调整到固定大小
        if mfcc.shape[0] < MAX_TIME_STEPS:
            # 填充
            pad_width = MAX_TIME_STEPS - mfcc.shape[0]
            mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
        elif mfcc.shape[0] > MAX_TIME_STEPS:
            # 截断
            mfcc = mfcc[:MAX_TIME_STEPS, :]

        # 确保特征维度正确
        if mfcc.shape[1] != N_FEATURES:
            mfcc = np.resize(mfcc, (MAX_TIME_STEPS, N_FEATURES))

        temp.append(mfcc)
        label.append(y_prediction.iloc[i])

X_prediction_processed = np.asarray(temp)
y_prediction_processed = np.asarray(label)

print(f"验证集特征形状: {X_prediction_processed.shape}")

# 特征标准化
print("\n=== 特征标准化 ===")
mean = np.mean(X_train_processed, axis=(0, 1), keepdims=True)
std = np.std(X_train_processed, axis=(0, 1), keepdims=True) + 1e-8

X_train_processed = (X_train_processed - mean) / std
X_prediction_processed = (X_prediction_processed - mean) / std

print("特征标准化完成")

# 计算类别权重
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train_processed),
    y=y_train_processed
)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
print(f"类别权重: {class_weight_dict}")

# 清理Keras后端
tf.keras.backend.clear_session()


# 保持原有的注意力机制
class attention(layers.Layer):
    def __init__(self, **kwargs):
        super(attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1),
                                 initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1),
                                 initializer='zeros', trainable=True)
        super(attention, self).build(input_shape)

    def call(self, x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x, self.W) + self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context


# 保持原有的模型架构，但调整输入维度
print("\n=== 构建模型 ===")

# input - 调整为新的特征维度
input_layer = layers.Input(shape=(MAX_TIME_STEPS, N_FEATURES), name='input_layer')

# CNN层 - 保持原有结构
x = layers.Conv1D(128, 3, activation='relu', padding='same')(input_layer)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

# LSTM层 - 保持原有结构
x = layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)(x)
x = layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)(x)

# 注意力机制 - 保持原有
x = attention()(x)

# Dense层 - 保持原有结构
x = layers.Dense(128, 'relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)

x = layers.Dense(64, 'relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)

# 输出层 - 调整为实际类别数
output = layers.Dense(NUM_CLASSES, 'softmax', name='output_layer')(x)

model = Model(inputs=input_layer, outputs=output)
model.summary()

# 编译模型 - 使用更好的配置
print("\n=== 编译模型 ===")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 回调函数
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_cnn_lstm_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
]

# 训练模型
print("\n=== 开始训练 ===")
history = model.fit(
    X_train_processed, y_train_processed,
    validation_data=(X_prediction_processed, y_prediction_processed),
    epochs=50,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)


# 绘制训练历史
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 准确率
    ax1.plot(history.history['accuracy'], label='训练准确率', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='验证准确率', linewidth=2)
    ax1.set_title('模型准确率')
    ax1.set_xlabel('轮次')
    ax1.set_ylabel('准确率')
    ax1.legend()
    ax1.grid(True)

    # 损失
    ax2.plot(history.history['loss'], label='训练损失', linewidth=2)
    ax2.plot(history.history['val_loss'], label='验证损失', linewidth=2)
    ax2.set_title('模型损失')
    ax2.set_xlabel('轮次')
    ax2.set_ylabel('损失')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('cnn_lstm_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


plot_training_history(history)

# 评估模型
print("\n=== 模型评估 ===")
train_loss, train_accuracy = model.evaluate(X_train_processed, y_train_processed, verbose=0)
prediction_loss, prediction_accuracy = model.evaluate(X_prediction_processed, y_prediction_processed, verbose=0)

print(f"训练准确率: {train_accuracy:.4f}")
print(f"验证准确率: {prediction_accuracy:.4f}")

# 预测和分类报告
y_pred = model.predict(X_prediction_processed)
y_pred_classes = np.argmax(y_pred, axis=1)

target_names = [species for species, _ in sorted(cls_label.items(), key=lambda x: x[1])]
print("\n=== 详细分类报告 ===")
print(classification_report(y_prediction_processed, y_pred_classes, target_names=target_names))

# 计算F1分数
from sklearn.metrics import f1_score

f1 = f1_score(y_prediction_processed, y_pred_classes, average='weighted')
print(f"\n加权平均F1分数: {f1:.4f}")

# 保存模型和配置
print("\n=== 保存模型和配置 ===")
model.save('improved_cnn_lstm_model.keras')

config = {
    "model_type": "Improved CNN-LSTM with Attention",
    "cls_label": cls_label,
    "num_classes": NUM_CLASSES,
    "max_time_steps": MAX_TIME_STEPS,
    "n_features": N_FEATURES,
    "final_accuracy": float(prediction_accuracy),
    "final_f1_score": float(f1),
    "improvements": {
        "data_augmentation": True,
        "feature_engineering": "MFCC + Delta + Delta2",
        "data_balancing": True,
        "batch_normalization": True,
        "dropout_regularization": True,
        "class_weighting": True,
        "early_stopping": True,
        "lr_scheduling": True
    }
}

import json

with open('improved_cnn_lstm_config.json', 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print("改进的CNN-LSTM模型训练完成！")
print(f"最终验证准确率: {prediction_accuracy:.4f}")
print(f"最终F1分数: {f1:.4f}")
print("模型已保存到: improved_cnn_lstm_model.keras")
print("配置已保存到: improved_cnn_lstm_config.json")


# ====================== 以下保持原有的预测代码不变 ======================

# 多标签预测部分的函数定义保持不变
def get_class_labels(r):
    x = r['class_name']
    label_names = x.split('_')

    label_nums = []
    for l in label_names:
        if l in cls_label:
            indx = cls_label[l]
            label_nums.append(indx)

    return label_nums


def label_to_sm(labels, n_classes):
    sm = lil_matrix((len(labels), n_classes))
    for i, label in enumerate(labels):
        sm[i, label] = 1
    return sm


def get_f1_score_multilabel(y_true, y_pred):
    y_true_sm = label_to_sm(labels=y_true, n_classes=NUM_CLASSES)
    y_pred_sm = label_to_sm(labels=y_pred, n_classes=NUM_CLASSES)
    metric = f1_score(y_true=y_true_sm, y_pred=y_pred_sm, average='weighted')
    return metric


def get_sm(y_true, y_pred):
    y_true_sm = label_to_sm(labels=y_true, n_classes=NUM_CLASSES)
    y_pred_sm = label_to_sm(labels=y_pred, n_classes=NUM_CLASSES)
    return y_true_sm, y_pred_sm


def get_split(path):
    myaudio = AudioSegment.from_file(path, "wav")
    chunk_length_ms = 1500  # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms)  # Make chunks of 1.5 sec

    paths = []
    for i, chunk in enumerate(chunks):
        chunk_name = f"chunk{i}.wav"
        paths.append(chunk_name)
        chunk.export(chunk_name, format="wav")

    return paths


def get_mfcc_for_prediction(path):
    """用于预测的MFCC提取函数，与训练保持一致"""
    mfcc = get_mfcc_enhanced(path, None, is_training=False, augment_prob=0.0)
    if mfcc is not None:
        # 调整到固定大小
        if mfcc.shape[0] < MAX_TIME_STEPS:
            pad_width = MAX_TIME_STEPS - mfcc.shape[0]
            mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
        elif mfcc.shape[0] > MAX_TIME_STEPS:
            mfcc = mfcc[:MAX_TIME_STEPS, :]

        if mfcc.shape[1] != N_FEATURES:
            mfcc = np.resize(mfcc, (MAX_TIME_STEPS, N_FEATURES))

        # 标准化
        mfcc = (mfcc - mean) / std

        return mfcc
    return np.zeros((MAX_TIME_STEPS, N_FEATURES))


def get_xqs(paths):
    xqs = []
    for i in range(min(6, len(paths))):  # only using first 6 chunks
        t = get_mfcc_for_prediction(paths[i])
        xqs.append(t)
    return xqs


def get_probs(model, xqs):
    probs = []
    for xq in xqs:
        xq = xq.reshape((1, MAX_TIME_STEPS, N_FEATURES))
        p = model.predict(xq, verbose=0)
        probs.append(p)
    return np.array(probs)


def predict(model, xqs, num_species):
    probs = get_probs(model, xqs)
    # aggregate
    s = np.max(probs, axis=0)
    if num_species == 2:
        n = 2
    else:
        n = 3
    labels = np.argsort(s[0])[::-1][:n]
    return list(labels)


def prediction(model, df, num_species):
    y_true = list(df['class_label'])
    y_pred = []
    for i in tqdm(range(len(df))):
        path = df['path'].iloc[i]
        chunk_paths = get_split(path)
        xqs = get_xqs(chunk_paths)
        label = predict(model, xqs, num_species)
        y_pred.append(label)
        # 清理临时文件
        for chunk_path in chunk_paths:
            if os.path.exists(chunk_path):
                os.remove(chunk_path)
    return y_true, y_pred


def evaluate(model, df_mix_2, df_mix_3):
    y_true_2, y_pred_2 = prediction(model, df_mix_2, 2)
    y_true_3, y_pred_3 = prediction(model, df_mix_3, 3)

    y_true = y_true_2 + y_true_3
    y_pred = y_pred_2 + y_pred_3
    score = get_f1_score_multilabel(y_true, y_pred)

    return y_true, y_pred, score

# 清理临时变量和释放内存
del X_train_processed, X_prediction_processed, temp
import gc

gc.collect()

print("\n训练完成，内存已清理")