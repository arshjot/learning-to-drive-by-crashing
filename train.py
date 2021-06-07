# from tensorflow import keras
import tensorflow as tf
import pickle as pkl
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Activation, BatchNormalization, MaxPooling2D, Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from PIL import Image
import numpy as np
import math
from config import Config


def generate_data_generator(generator, X, Y, t_idx):
    """Used to facilitate multiple outputs"""
    genXY = generator.flow(X[t_idx], Y[t_idx], batch_size=config.batch_size)
    while True:
        Xi, Yi = genXY.next()
        Yi1, Yi2, Yi3 = Yi[:, 0], Yi[:, 1], Yi[:, 2]
        yield Xi, [Yi2, Yi3]


config = Config
val_split = 0.2

# load data
print('Loading data')
with open(config.data_file, 'rb') as f:
    data = pkl.load(f)

X = data['images']
y_obs = data['obstacle'].reshape(-1, 1)
y_angle = data['angleness'].reshape(-1, 1)
y_center = data['centerness'].reshape(-1, 1)
y = np.concatenate([y_obs, y_angle, y_center], 1)

del data
import gc
gc.collect()

train_gen = ImageDataGenerator(
    # rotation_range=20,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # channel_shift_range=0.2,
    # brightness_range=[0.8, 1.2],
    horizontal_flip=True)

print('Splitting')
t_idx, v_idx = train_test_split(range(len(X)), test_size=config.val_split, random_state=42)
val_X, val_y = X[v_idx], y[v_idx]

# val_X /= 255.

train_generator = generate_data_generator(train_gen, X, y, t_idx)

del X, y
import gc
gc.collect()

print('Training')
# get model
net = tf.keras.applications.MobileNetV2(
    input_shape=(config.model_img_w, config.model_img_h, 3), include_top=False, weights=None,
    input_tensor=None, pooling=None)

input_tensor = Input(shape=(config.model_img_w, config.model_img_h, 3))

if config.architecture != 'CustomNet':
    x = Conv2D(3, (1, 1), padding='same')(input_tensor)
    out = net(input_tensor)
    out = Flatten()(out)
    out = Dense(2)(out)
else:
    out = BatchNormalization()(input_tensor)
    out = Conv2D(8, (3, 3), activation='relu')(out)
    out = MaxPooling2D((2, 2))(out)
    out = BatchNormalization()(out)
    out = Conv2D(16, (3, 3), activation='relu')(out)
    out = MaxPooling2D((2, 2))(out)
    out = BatchNormalization()(out)
    out = Conv2D(32, (3, 3), activation='relu')(out)
    out = MaxPooling2D((2, 2))(out)
    out = BatchNormalization()(out)
    out = Conv2D(16, (3, 3), activation='relu')(out)
    out = MaxPooling2D((2, 2))(out)
    out = BatchNormalization()(out)
    out = Conv2D(4, (3, 3), activation='relu')(out)
    out = MaxPooling2D((2, 2))(out)
    out = Flatten()(out)
    out = Dropout(0.2)(out)
    out = Dense(512)(out)
    out = Activation('relu')(out)
    out = Dense(2)(out)

out_angleness = Activation('linear', name="angle")(out[:, 0])
out_centerness = Activation('linear', name="center")(out[:, 1])

model = Model(inputs=input_tensor, outputs=[out_angleness, out_centerness])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),  # Optimizer
    # Loss function to minimize
    loss={'angle': tf.keras.losses.MeanSquaredError(),
          'center': tf.keras.losses.MeanSquaredError()},
    # List of metrics to monitor
    metrics={'angle': tf.keras.metrics.MeanAbsoluteError(),
             'center': tf.keras.metrics.MeanAbsoluteError()}
)

my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=config.es_patience, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=f'model_{config.architecture}_best_mse.h5', save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                             patience=config.lr_reduce_patience, verbose=1),
        tf.keras.callbacks.TensorBoard()
]

history = model.fit(
    train_generator,
    validation_data=(val_X, [val_y[:, 1], val_y[:, 2]]),
    steps_per_epoch= math.ceil(len(t_idx) / config.batch_size),
    epochs=config.num_epochs,
    callbacks=my_callbacks
)
