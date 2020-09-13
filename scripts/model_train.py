import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, LSTM, SimpleRNN, Conv1D, Input, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception


input_shape = (200,200, 3)
effnet_layers = Xception(weights=None, include_top=False, input_shape=input_shape)
for layer in effnet_layers.layers:
    layer.trainable = True



image_data_generator=ImageDataGenerator(rescale=1./255,validation_split=0.1)

data=image_data_generator.flow_from_directory('E:\\birdsong\\train_mel\\',target_size=(200,200),batch_size=4,subset='training')
val_data=image_data_generator.flow_from_directory('E:\\birdsong\\train_mel\\',target_size=(200,200),batch_size=4,subset='validation')
dropout_dense_layer = 0.5
classes_to_predict=data.class_indices
model = Sequential()
model.add(effnet_layers)
    
model.add(GlobalAveragePooling2D())
model.add(Dense(256, use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(dropout_dense_layer))

model.add(Dense(len(classes_to_predict), activation="softmax"))

callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.7),
             EarlyStopping(monitor='val_loss', patience=5),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
model.compile(loss="categorical_crossentropy", optimizer='adam',metrics=['top_k_categorical_accuracy','categorical_accuracy'])


model.fit(data,epochs=10,callbacks=[callbacks],validation_data=val_data)
input()