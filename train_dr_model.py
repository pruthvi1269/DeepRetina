import tensorflow as tf

tf.keras.mixed_precision.set_global_policy(
    'mixed_float16'
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import DenseNet121

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

# =========================
# DATASET PATH
# =========================

dataset_path = "dataset"

# =========================
# IMAGE PREPROCESSING
# =========================

train_datagen = ImageDataGenerator(

    rescale=1./255,

    

    rotation_range=20,

    zoom_range=0.2,

    horizontal_flip=True,

    brightness_range=[0.8,1.2]

)

# =========================
# TRAIN DATA
# =========================

train_data = train_datagen.flow_from_directory(

    "dataset/train",

    target_size=(224,224),

    batch_size=32,

    class_mode='categorical'

)

# =========================
# VALIDATION PREPROCESSING
# =========================

val_datagen = ImageDataGenerator(

    rescale=1./255

)

# =========================
# VALIDATION DATA
# =========================

val_data = val_datagen.flow_from_directory(

    "dataset/val",

    target_size=(224,224),

    batch_size=16,

    class_mode='categorical'

)

# =========================
# LOAD EFFICIENTNETB3
# =========================

base_model = DenseNet121(

    weights='imagenet',

    include_top=False,

    input_shape=(224,224,3)

)

base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable=False

# =========================
# BUILD MODEL
# =========================

model = Sequential([

    base_model,

    GlobalAveragePooling2D(),

    Dropout(0.4),

    Dense(

        256,

        activation='relu',

        dtype='float32'

    ),

    Dropout(0.3),

    Dense(

        5,

        activation='softmax',

        dtype='float32'

    )

])

# =========================
# COMPILE MODEL
# =========================

model.compile(

    optimizer=Adam(learning_rate=0.0001),

    loss='categorical_crossentropy',

    metrics=['accuracy']

)


# =========================
# TRAIN MODEL
# =========================

early_stop = EarlyStopping(

    monitor='val_accuracy',

    patience=8,

    restore_best_weights=True

)

reduce_lr = ReduceLROnPlateau(

    monitor='val_loss',

    factor=0.3,

    patience=3,

    min_lr=1e-7

)

import numpy as np

class_weights = compute_class_weight(

    class_weight='balanced',

    classes=np.unique(train_data.classes),

    y=train_data.classes

)

class_weights = dict(
    enumerate(class_weights)
)

history = model.fit(

    train_data,

    validation_data=val_data,

    epochs=100,

    callbacks=[early_stop, reduce_lr],

    class_weight=class_weights

)

# =========================
# SAVE MODEL
# =========================

model.save_weights(
    "dr_weights.weights.h5"
)

print("WEIGHTS SAVED SUCCESSFULLY")