"""
Created on Wed Apr 14 19:12:31 2021
@author: nmsta
"""
import pandas as pd
from keras.utils import Sequence
import numpy as np
from skimage import io
from random import shuffle
from keras.layers import Input, Flatten, Conv2D, MaxPooling2D, Dropout, Dense, BatchNormalization, Activation
from keras.models import Model
import os
from keras.callbacks import ReduceLROnPlateau, TerminateOnNaN, ModelCheckpoint
import keras.backend as K

class IMGenerator(Sequence):
    """Yields crops and response crops given list of paths and responses, length of list, and batch size."""
    def __init__(self, image_paths_and_responses, dataset_length, batch_size, shuffle_data=True):
        self.image_paths_and_responses = image_paths_and_responses
        self.dataset_length = dataset_length
        self.batch_size = batch_size
        self.shuffle_data = shuffle_data
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches (steps) per epoch."""
        return self.dataset_length // self.batch_size
    
    def __getitem__(self, index):
        """Generate one batch of data based on a random batch index."""
        X = [io.imread(path[0]) for path in self.image_paths_and_responses[index*self.batch_size:(index+1)*self.batch_size]]
        y = [response[1] for response in self.image_paths_and_responses[index*self.batch_size:(index+1)*self.batch_size]]
        
        return np.array(X), np.array(y)

    def on_epoch_end(self):
        """Shuffles batch indices at the beginning of each epoch if shuffle_data=True"""
        if self.shuffle_data:
            shuffle(self.image_paths_and_responses)
            
def conv_block(inputs, filters, block_num, dropout, pool=True, factor=2):
    """Convolutional block"""
    bn = BatchNormalization()(inputs)
    act = Conv2D(filters=int(filters*(factor**block_num)), kernel_size=(3, 3), use_bias=False, activation="elu")(bn)
    bn = BatchNormalization()(act)
    act = Conv2D(filters=int(filters*(factor**block_num)), kernel_size=(3, 3), use_bias=False, activation="elu")(bn)
    if pool:
        act = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(act)
    if dropout is not None:
        act = Dropout(dropout)(act)
    
    return act

def dense_block(inputs, nodes, dropout):
    """Fully connected block"""
    
    act = Dense(nodes, activation="elu")(inputs)
    if dropout is not None:
        drop = Dropout(dropout)(act)
    bn = BatchNormalization()(drop)
    act = Dense(nodes, activation="elu")(bn)
    
    return act

def basic_cnn(channels=5, filters=32, classes=1, dropout=0.1, num_blocks=1, dense_nodes=64, input_shape=(None, None), factor=2):
    """Builds a Unet model"""
    
    inputs = Input(shape=(input_shape[0], input_shape[1], channels))
    act = Conv2D(filters=filters, kernel_size=(3, 3), activation="elu")(inputs)
    
    if num_blocks > 1:
        for i in range(num_blocks - 1):
            act = conv_block(act, filters, i, dropout, factor=factor)
    act = conv_block(act, filters, i, pool=False, factor=factor)     
        
    flat = Flatten()(act)
    dense = dense_block(flat, dense_nodes, dropout)
    denseout = Dense(classes)(dense)
    
    if classes > 1:
        actout = Activation("softmax", name="softmax")(denseout)
    else:
        actout = Activation("sigmoid", name="sigmoid")(denseout)
        
    model = Model(inputs=inputs, outputs=actout)
            
    return model  

## Script ##
path_to_file = "/Users/nmsta/OneDrive/Documents/Train_Summary.csv"
save_path = "/Users/nmsta/OneDrive/Documents/Models"
gpu_memory_fraction = 0.99
gpus = "0" 
gpu_num = len(gpus.split(","))
workers = 1
batch_size = 32
epochs = 100
filters = 32
channels = 5
classes = 1
dropout = 0.2
continue_training = False
optimizer = "nadam"
loss = "binary_crossentropy"
num_blocks = 2
dense_nodes = 256
input_shape = (64, 64)
factor = 2

try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
except:
    print('Tensorflow is not loaded, other backend will be used!')
    
if not os.path.exists(save_path):
    os.makedirs(save_path)
    
df = pd.read_csv(path_to_file)
df["Class"] = (df["Response"] >= 0).astype(int)

train = df[df["Dataset"] == "train"]
val = df[df["Dataset"] == "val"]
train_len = len(train)
val_len = len(val)

train_generator = IMGenerator(list(zip(train["Train_Path"], train["Class"])) , train_len, batch_size)
validation_generator = IMGenerator(list(zip(val["Train_Path"], val["Class"])), val_len, batch_size)

model = basic_cnn(channels, filters, classes, dropout, num_blocks, dense_nodes, input_shape, factor)
model_file = save_path + "/ohlc_daily.json"
model_weight_file = save_path + "/ohlc_daily.h5"

model.summary()
print("Positive:", sum(df["Response"] > 0), round(sum(df["Response"] >= 0)/len(df),2))
print("Negative:", sum(df["Response"] < 0), round(sum(df["Response"] < 0)/len(df),2))
print("Train Length:", len(train))
print("Validation Length:", len(val))
print("Num GPUs Available: ", len(gpus))
print('Allocating %.2f of GPU memory!' % gpu_memory_fraction)
print("workers: " +str(workers))
print("batch size: " + str(batch_size))
print("filters: " + str(filters))
print("dropout: " + str(dropout))
print("optimizer: " + str(optimizer))
print("loss: " + str(loss))
print("num classes: " + str(classes))
print("Saving model to " + model_file)

json_string = model.to_json()
json_file = open(model_file, "w")
json_file.write(json_string)
json_file.close()

if continue_training:
    model.load_weights(model_weight_file)
    print(" Continuing training from: " + model_weight_file)

model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

check_pointer = ModelCheckpoint(model_weight_file, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
learning_rate_adj = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=4, verbose=1)

with tf.device('/device:GPU:0'):
    hist = model.fit(train_generator, steps_per_epoch=train_len // batch_size, epochs=epochs, 
                     verbose=1, callbacks=[check_pointer, learning_rate_adj, TerminateOnNaN()], 
                     validation_data=validation_generator, validation_steps=val_len // batch_size, shuffle=False,
                     workers=workers, max_queue_size=10)

del model
try:
    K.clear_session()
except:
    pass

print("minimum checkpoint loss: " + str(min(hist.history.get("val_loss"))))
print("weight file written to: " + model_weight_file)