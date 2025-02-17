# -*- coding: utf-8 -

import re
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
np.random.seed(2)
import gc
tf.keras.backend.clear_session()
gc.collect()
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
#tf.config.run_functions_eagerly(True)
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Flatten, Conv1D
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras import backend as K
import tensorflow.compat.v1 as tf
#tf.config.experimental_run_functions_eagerly(True)

# dropout 0.2// class imbalnce // 2 folds // learning rate 0.0001

import uuid
"""
def CNN1D(input):
    input1 = input
    x = Conv1D(filters=8, kernel_size=3, activation='selu', padding='valid')(input1)
    x = Conv1D(filters=16, kernel_size=3, activation='selu', padding='valid')(x)
    x = layers.MaxPooling1D(2)(x)
    x = Conv1D(filters=16, kernel_size=3, activation='selu', padding='valid')(x)
    x = Conv1D(filters=1, kernel_size=3, activation='selu', padding='valid')(x)
    x = layers.UpSampling1D(2)(x)
    x = Conv1D(filters=1, kernel_size=100, activation='selu', padding='same')(x) # final Conv1D layer to increase size
    
    return x
"""
from tensorflow.keras.layers import (
    LSTM,
     Bidirectional,
    MaxPooling1D,
     Reshape,
    GlobalAveragePooling1D
)

def InceptionModule1D(x, filters=32):
    conv1 = layers.Conv1D(filters=filters, kernel_size=1, padding='same', activation='selu')(x)
    
    conv3 = layers.Conv1D(filters=filters, kernel_size=3, padding='same', activation='selu')(x)
    
    conv5 = layers.Conv1D(filters=filters, kernel_size=5, padding='same', activation='selu')(x)
    
    maxpool = layers.MaxPooling1D(pool_size=3, strides=1, padding='same')(x)
    maxpool_conv = layers.Conv1D(filters=filters, kernel_size=1, padding='same', activation='selu')(maxpool)
    x = layers.Concatenate()([conv1, conv3, conv5, maxpool_conv])
    
    return x

def CNN1D(input ):
    num_modules=3
    filters=32
    
    x = input
    
    for i in range(num_modules):
        x = InceptionModule1D(x, filters=filters)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('selu')(x)
    
    # Global average pooling for reducing dimensionality
    #x = layers.GlobalAveragePooling1D()(x)
    
    return x


def add_pos_2(input, nb):
    input_pos_encoding = tf.constant(nb, shape=[input.shape[1]], dtype="int32") / input.shape[1]
    input_pos_encoding = tf.cast(tf.reshape(input_pos_encoding, [1, input.shape[1]]), tf.float32) # encoding positional calculation
    # Use Keras layer for addition
    output = layers.Add()([input, input_pos_encoding])
    return output

def stack_block_transformer(num_transformer_blocks): #temporal transformer 
    input1 = keras.Input(shape=(100, 1))
    x = input1
    x = CNN1D(x)
    print( "x CNN" , x)
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x,128,2)
        
    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)  # Adaptive Average Pooling
    x = layers.Flatten()(x)  # Flatten the output
  #average pooling reduces the temporal dimensionality, take the average value
    x = layers.Dropout(0.2)(x) # regularization
    x = layers.Dense(10, activation='selu')(x) #This line applies a fully connected (dense) layer with 10 units and a SELU activation function : temporal vector 
    return input1,x

def stack_block_transformer_spatial(num_transformer_blocks,x):
    for _ in range(num_transformer_blocks):
        print("x", x)
        x = transformer_encoder(x,10*18,4)
         
        x = layers.GlobalAveragePooling1D(data_format="channels_first")(x) # Adaptive Average Pooling
        x = layers.Flatten()(x)  # Flatten the output


    return x

def transformer_encoder(inputs,key_dim,num_heads):
    dropout=0.3
    # Normalization and Attention
    print(inputs.shape)
    print("transformer_encoder",inputs.shape)
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=key_dim, num_heads=num_heads
    )(x, x)
    print("x shape after MultiHeadAttention:", x.shape)
    x = layers.Dropout(dropout)(x)
    res = x + inputs
    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Dense(key_dim, activation='softmax')(x)
    
    return x + res 



def multiple_transformer(nb):
    '''
    :param nb: number of features ( indicates the number of parallel branches)
    :return:
    '''
    # initialise with the first input

    num_transformer_blocks = 4  #hyperparameter
    input_, transformer_ = stack_block_transformer(num_transformer_blocks)
    transformers = []
    inputs = []
    transformers.append(transformer_)
    inputs.append(input_)
    for i in range(1,nb ):
        input_i, transformer_i = stack_block_transformer(num_transformer_blocks)
        inputs.append(input_i) 
        transformer_i = add_pos_2(transformer_i,i)
        transformers.append(transformer_i)
    x = layers.concatenate(transformers, axis=-1)
   # x = tf.expand_dims(x, -1) #-1 denotes the last dimension
    x = layers.Reshape((x.shape[1], 1))(x)
    x = stack_block_transformer_spatial(num_transformer_blocks,x)
    x = Dropout(0.1)(x)
    x = layers.Dense(100, activation='selu')(x)
    x = Dropout(0.1)(x)
    x = layers.Dense(20, activation='selu')(x)
    x = Dropout(0.1)(x)
    answer = layers.Dense(1, activation='sigmoid')(x)
    model = Model(inputs, answer)
    opt = optimizers.RMSprop(learning_rate=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy']) #,experimental_run_tf_function=False)
    print(model.summary())
    return model


def multiple_transformer_5_level(nb):
    '''
    Model for severity prediction , 5 classes output
    :param nb:  number of parallel branch
    :return:
    '''

  # initialise with the first input

    num_transformer_blocks = 1  #hyperparameter
    input_, transformer_ = stack_block_transformer(num_transformer_blocks)
    transformers = []
    inputs = []
    transformers.append(transformer_)
    inputs.append(input_)
    for i in range(1,nb ):
        input_i, transformer_i = stack_block_transformer(num_transformer_blocks)
        inputs.append(input_i)
        transformer_i = add_pos_2(transformer_i,i)
        transformers.append(transformer_i)
  
    x = layers.concatenate(transformers, axis=-1)
    #x = tf.expand_dims(x, -1) #-1 denotes the last dimension
    x = layers.Reshape((x.shape[1], 1))(x)
    x = stack_block_transformer_spatial(num_transformer_blocks,x)
    x = Dropout(0.2)(x)
    x = layers.Dense(100, activation='selu')(x)
    x = Dropout(0.2)(x)
    x = layers.Dense(20, activation='selu')(x)
    x = Dropout(0.2)(x)
    answer = layers.Dense(4, activation='softmax')(x)
    print("answser", answer.shape)
    model = Model(inputs, answer)
    
    opt = optimizers.Nadam(learning_rate=0.0001)
    #opt = optimizers.Nadam(lr=0.001)
    model.compile(loss= 'categorical_crossentropy', optimizer=opt, metrics=['accuracy']) #,experimental_run_tf_function=False)
    return model

"""
def add_fourier_features(data, num_features=8):
    n_timesteps = data.shape[0]
    time_indices = np.arange(n_timesteps) / (n_timesteps - 1)  # Normalize time between 0 and 1

    fourier_features = np.empty((n_timesteps, num_features))
    for i in range(num_features):
        harmonic = 1 + i * 2  # Adjust harmonic based on your needs
        fourier_features[:, i] = np.sin(2 * np.pi * harmonic * time_indices)

    return np.concatenate((data, fourier_features), axis=1)

def time_warping(data, max_shift=0.2):
    warping_factor = tf.random.uniform([], minval=1 - max_shift, maxval=1 + max_shift)

      # Calculate the number of samples for the resampled time series.
    original_length = tf.shape(data)[0]
    new_length = tf.cast(tf.math.floor(original_length * warping_factor), dtype=tf.int32)

      # Create a time axis with evenly spaced samples.
    time_axis = tf.linspace(0.0, 1.0, original_length)

      # Resample the data using linear interpolation.
    warped_data = tf.cast(tf.image.resize(tf.expand_dims(data, axis=-1), [new_length, 1]), dtype=data.dtype)

      # Optionally, handle edge cases (e.g., rounding errors) to ensure valid indices.
    warped_data = warped_data[:, 0]  # Extract the resampled data from the single-channel image

    return warped_data
def amplitude_scaling(data, minval=0.8, maxval=1.2):
 

    scale_factor = tf.random.uniform([], minval=minval, maxval=maxval)
    scaled_data = data * scale_factor
    return scaled_data
def add_noise(data, mean=0.0, stddev=0.1):
 

    noise = tf.random.normal(tf.shape(data), mean=mean, stddev=stddev)
    noisy_data = data + noise
    return noisy_data
def segment_dropout(data, segment_length, drop_probability=0.2):
    segments = tf.split(data, data.shape[0] // segment_length)
    random_mask = tf.random.uniform((len(segments),), minval=0.0, maxval=1.0)
    filtered_segments = [segment for segment, mask in zip(segments, random_mask) if mask > drop_probability]
    return tf.concat(filtered_segments, axis=0)
"""
# -*- coding: utf-8 -*-
"""
"""
import glob
import os
import argparse
import random
import sys
from tensorflow import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from collections import Counter

#from imbalanced_learn import over_sampling
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
class Data:

    def __init__(self,  input_data,  deep, gait_cycle, step=50, features=np.arange(1, 19), pk_level = True):
        '''
        :param load_or_get:  1: load data , 0: load preloaded datas ( npy)
        :param deep:  data in the format for deep learning algorithms
        :param gait_cycle: number of gait cycle per signal
        :param step: overlap between gait signals
        :param features: signals to be loaded ( coming from sensors)
        :param pk_level: if true , y is the parkinson level according
        '''

        self.deep = deep
        self.step = step
        self.nb_gait_cycle = gait_cycle


        self.features_to_load = features
        self.nb_features = self.features_to_load.shape[0]
        ###############
        self.X_data = np.array([])  # np.ones((self.nb_gait_cycle,self.nb_features))
        self.y_data = np.array([])
        self.nb_data_per_person = np.array([0])


        files = sorted(glob.glob(os.path.join(input_data, '*txt')))
        self.ctrl_list = []
        self.pk_list = []
        for file in files:

            if file.find(".txt") != -1:  # if control ("01.txt")
                if file.find("Co") != -1:  # if control
                    if file.find("Ga") != -1:
                        self.ctrl_list.append(file)
                    else:
                        pass
                elif file.find("Pt") != -1:  # if control
                    self.pk_list.append(file)

        random.shuffle(self.ctrl_list)
        random.shuffle(self.pk_list)
        self.pk_level = pk_level
        if pk_level == True:
            self.levels = pd.read_csv("data/demographics.csv")
            self.levels.set_index('ID', inplace=True)
        self.load(norm=None)
     
    def add_pos(self,input):
       # Positional encoding
        input_pos_encoding = tf.range(input.shape[1])/input.shape[1]
        input_pos_encoding = tf.expand_dims(input_pos_encoding, -1)
        input_pos_encoding= tf.cast(tf.tile(input_pos_encoding, [1,input.shape[2]]),tf.float32)
        # Add the positional encoding
        input = input + input_pos_encoding
        return input

    def separate_fold(self, fold_number, total_fold=10):
        proportion = 1 / total_fold  # .10 for 10 folds
        X = [self.X_ctrl, self.X_park]
        y = [self.y_ctrl, self.y_park]
        patients = [self.nb_data_per_person[:self.last_ctrl_patient],         self.nb_data_per_person[self.last_ctrl_patient:]] # counts separated by classe
        patients[1]= patients[1] - patients[1][0]
        diff_count = np.diff(self.nb_data_per_person)
        diff_count = [diff_count[:self.last_ctrl_patient], diff_count[self.last_ctrl_patient:]]
        self.count_val = np.array([0])
        self.count_train = np.array([0])
        classes = []
        class_counts = []

        for i in range(len(X)):
    # Use np.unique() to get the unique class labels and their counts
            unique_classes, class_counts_temp = np.unique(y[i], return_counts=True)
            classes.append(list(unique_classes))
            class_counts.append(list(class_counts_temp))
        for fold, (unique_class_labels, class_counts_fold) in enumerate(zip(classes, class_counts)):
            plt.bar(unique_class_labels, class_counts_fold, label=f"Fold {fold+1}")
        plt.xlabel("Class Label")
        plt.ylabel("Count")
        plt.title("Class Distribution in Folds")
        plt.legend()
        #plt.xticks(X, list(range(len(class_counts))))
        #plt.tight_layout() 
        plt.show()
        
        for i in range(len(X)):
            nbr_patients =  int(len(patients[i]) *proportion)
            start_patient = int(fold_number*nbr_patients )
            end_patient = (fold_number+1)*nbr_patients
            id_start = patients[i][start_patient]  # segment start
            id_end = patients[i][end_patient]  # end segment
            if i ==0 :
                self.X_val = X[i][id_start:id_end,:,:]
                self.X_train = np.delete(X[i], np.arange(id_start,id_end) , 0)

                self.y_val = y[i][id_start:id_end]
                self.y_train = np.delete(y[i], np.arange(id_start,id_end) , 0)


                self.count_val = np.append(self.count_val, diff_count[i][start_patient: end_patient])
                self.count_train = np.append(self.count_train, np.delete(diff_count[i], np.arange(start_patient, end_patient)))



            else:
                start_patient = start_patient #+ patients[0].shape[0]  # patients0.shape 0 is the number of patients in the first class
                end_patient =  end_patient# +patients[0].shape[0]
                self.X_val = np.vstack((self.X_val, X[i][id_start:id_end,:,:]))
                self.X_train = np.vstack((self.X_train, np.delete(X[i], np.arange(id_start,id_end) , 0) ))

                self.y_val = np.vstack((self.y_val, y[i][id_start:id_end] ))
                self.y_train = np.vstack((self.y_train, np.delete(y[i], np.arange(id_start,id_end) , 0) ))

                self.count_val = np.append( self.count_val , diff_count[i][start_patient: end_patient])
                self.count_train = np.append(self.count_train,np.delete(diff_count[i], np.arange(start_patient, end_patient)) )
 
        unique_classess1, class_countss1 = np.unique(self.y_train, return_counts=True)
        #print("Classes bef SMOTE:", unique_classess1)
        y_train_discrete = np.where(self.y_train == 0, 0, 
                             np.where(self.y_train == 2, 1, 
                             np.where(self.y_train == 2.5, 2, 
                             np.where(self.y_train == 3, 3, -1))))

# Check if y_train_discrete contains valid class labels
        if np.any(y_train_discrete == -1):
            raise ValueError("y_train contains invalid continuous values.")
        X_train_reshaped = self.X_train.reshape(self.X_train.shape[0], -1)
        smote = SMOTE(random_state=42, sampling_strategy='all')
        unique_classes_trainn, class_counts_trainn = np.unique(self.y_train, return_counts=True)
        unique_classes_vall, class_counts_vall = np.unique(self.y_val, return_counts=True)
        
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_reshaped, y_train_discrete)
        #print("resampled y ",y_train_resampled.shape)
        unique_classess, class_countss = np.unique(y_train_resampled, return_counts=True)
        #print("Classes after SMOTE:", unique_classess)
        
    
    # Reshape back to original 3D form after SMOTE
        self.X_train = X_train_resampled.reshape(-1, self.X_train.shape[1], self.X_train.shape[2])
        self.y_train = y_train_resampled
        #print("Before one-hot encoding:", np.unique(self.y_train))
        #print("Before one-hot encoding val:", np.unique(self.y_val))
       # self.y_train = self.one_hot_encoding(self.y_train)
        #self.y_val = self.one_hot_encoding(self.y_val)
        
        y_val_discrete = np.where(self.y_val == 0, 0, 
                             np.where(self.y_val == 2, 1, 
                             np.where(self.y_val == 2.5, 2, 
                             np.where(self.y_val == 3, 3, -1))))
        #print("After one-hot encoding shape val:", np.unique(y_val_discrete[:, 0]))
        self.y_val = y_val_discrete
        #self.y_train = self.one_hot_encoding(self.y_train)
        
        
       
        self.count_val = np.cumsum(self.count_val)
        self.count_train = np.cumsum(self.count_train )
        self.X_val = layers.LayerNormalization(epsilon=1e-6)(self.X_val)
        self.X_train = layers.LayerNormalization(epsilon=1e-6)(self.X_train)
        self.X_val = self.add_pos(self.X_val)
        self.X_train = self.add_pos(self.X_train)
       
        return self.X_train, self.X_val, self.y_train, self.y_val

    def load_data(self, liste, y):
        '''
        :param liste: list of patients filepaths
        :param y: 0 for control, 1 for parkinson
        :return:
        '''
        print(liste)
        for i in range(0, len(liste)):
            datas = np.loadtxt(liste[i])  # num cycle, n features
            datas = datas[:, self.features_to_load]
            #print(datas.shape[0])
            
            if  self.pk_level :
                #print("1")
                y =self.find_level(liste[i])
    
            if self.deep == 1:
                #print("2")
                X_data, y_data , self.nb_data_per_person = self.generate_datas(datas, y, self.nb_data_per_person)
              
            else:
                #print("3")
                X_data, y_data = self.generate_datas_ml(datas, y)
            if (self.X_data).size == 0:
                #print("4")
                self.X_data = X_data
                self.y_data = y_data
            else:
                #print("5")
                if self.deep == 1:
                    #print("6")
                    self.X_data = np.dstack((self.X_data, X_data))
                else:
                    #print("7")
                    self.X_data = np.vstack((self.X_data, X_data))  # shape nb data --- vector size
                self.y_data = np.vstack((self.y_data, y_data))
               # print(self.y_data)
                #print(X_data.shape, self.X_data.shape,flush=True)
    def load(self, norm = 'std'):
        print("load training control ")
        self.load_data(self.ctrl_list, 0)
        if self.deep == 1:
            self.last_ctrl= self.X_data.shape[2]
            self.last_ctrl_patient = len(self.nb_data_per_person)
        print("load training parkinson ")


        self.load_data(self.pk_list, 1)  # ncycle, nfeature, nombre de data


        ## all datas are loaded at this point, preprocessing now
        if self.deep == 1:
            self.X_data = self.X_data.transpose(2,0 , 1)  #0, 1

            if norm == 'std ':
                self.normalize()
            elif norm == 'l2':
                self.X_data = self.normalize_l2(self.X_data)

        if self.pk_level:
            self.one_hot_encoding(self.y_data)

        if self.deep == 1:
            self.X_ctrl = self.X_data[:self.last_ctrl]
            self.y_ctrl =  self.y_data[:self.last_ctrl]
            self.X_park = self.X_data[self.last_ctrl:]
            self.y_park = self.y_data[self.last_ctrl:]

        #print("saving training ")
        np.save(os.path.join(args.output, "Xdata_eff"), self.X_data)
        np.save(os.path.join(args.output, "ydata_eff"),self.y_data)
        np.save(os.path.join(args.output, "data_person_eff"),self.nb_data_per_person)
        np.save(os.path.join(args.output, "ctrl_list_eff"), self.ctrl_list)
        np.save(os.path.join(args.output, "pk_list_eff"), self.pk_list)

    def normalize(self):
        '''
        :return: Normalize to have a mean =  and std =1
        '''
        mean_train = np.mean(self.X_data,(0,1))
        std_train = np.std(self.X_data,(0,1))
        self.X_data= abs((self.X_data - mean_train) / std_train)
    def one_hot_encoding(self, y):

        y = np.where(y == 0, 0, y)      # Class 0
        y = np.where(y == 2, 1, y)      # Class 1
        y = np.where(y == 2.5, 2, y)    # Class 2
        y = np.where(y == 3, 3, y)      # Class 3
        y = np.nan_to_num(y)

        if np.any(y < 0) or np.any(y > 3):
            raise ValueError("y contains invalid class indices.")

        y_one_hot = to_categorical(y, num_classes=4)
        return y_one_hot

    def normalize_l2(self, data):
        '''
        :param data:  Function to perform L2 normalization
        :return:
        '''
        data = keras.backend.l2_normalize(data, axis=(1, 2))
        data = tf.keras.backend.get_value(data)
        return data


  
           


    def find_level(self,file):
        '''
        :param file: Dataframe
        :return:
        '''
        start = 'data/'
        end = '_'
        id = (file.split(start))[1].split(end)[0]
        print(id)
        y = self.levels.loc[id,'HoehnYahr']
        return y


    def generate_datas(self, datas, y, data_list):
        '''
        :param datas:  datas loaded for 1 patient
        :param y: label of the patient
        :param data_list: list containing the number of segments per patients
        :return:
        '''
        count = 0
        X_data = np.array([])
        y_data = np.array([])
        nb_datas = int(datas.shape[0] - self.nb_gait_cycle)
        for start in range(0, nb_datas, self.step):
            end = start + self.nb_gait_cycle
            data = datas[start:end, :]
            
            if X_data.size == 0:
                X_data = data
                y_data = y
            else:
                if (self.deep == 1):
                    X_data = np.dstack((X_data, data))
                else:
                    X_data = np.vstack((X_data, data))
                y_data = np.vstack((y_data, y))
            count = count + 1
        data_list = np.append(data_list, count+ data_list[-1])
        return X_data, y_data, data_list


    def get_datas(self):
        return self.X_data, self.y_data, self.X_test, self.y_test, self.X_val, self.y_val


import os
import numpy as np
from sklearn.metrics import confusion_matrix,  classification_report, accuracy_score
from scipy import stats
import pandas as pd
class Results:
    def __init__(self, filename_seg, filename_patient):
        '''
        :param filename_seg:  Filename  (.csv) where to save results at the segment levels
        :param filename_patient: Filename  (.csv) where to save results at the patient levels
        '''
        self.results_patients = np.zeros(3)
        self.results_segments = np.zeros(3)
        self.filename_seg = filename_seg
        self.filename_patient = filename_patient
    def add_result( self,res, accuracy,  segments = True  ):
        '''
        :param res: result of classification report (sklearn )
        :param accuracy:
        :param segments: 1 to add results at the segment level
        :return:
        '''
        if segments:
            specificity = res['0.0']['recall']
            sensitivy =  res['1.0']['recall']
        else:
            specificity = res['0']['recall']
            sensitivy =  res['1']['recall']
        all = np.array([specificity, sensitivy, accuracy])

        if segments:
            self.results_segments = np.vstack((self.results_segments, all))
        else:
            self.results_patients = np.vstack((self.results_patients, all ))

    def validate_patient(self, model, x_val, y_val, count):
        #shape=22
        '''
        :param model: trained model after 1 fold of cross validation
        :param x_val: x_Val for 1 forld of cross validation
        :param y_val: y_Val for 1 forld of cross validation
        :param count: vector containing the number of segments per patient
        :return:  save the results of the fold
        '''
        ## per segments
        pred_seg = model.predict(np.split(x_val, x_val.shape[2], axis=2))#(x_val, x_val.shape[2], axis=2)
        res = classification_report(np.rint(y_val), np.rint(pred_seg), output_dict = True )
        acc = accuracy_score(np.rint(y_val), np.rint(pred_seg))
        self.add_result(res, acc,True)

        eval = []
        y = []
        pred = []
        #shape=22
        for m in range(1, len(count)):
            i = count[m]
            j = count[m - 1]
            score = model.evaluate(np.split(x_val[j:i, :, :], x_val.shape[2], axis=2), y_val[j:i]) 
            eval.append(score)
            y.append(np.int(np.mean(y_val[j:i])))
            p = np.rint(model.predict(np.split(x_val[j:i, :, :], x_val.shape[2], axis=2)))
            pred.append(np.mean(p))

        res = classification_report(y, np.rint(pred), output_dict = True )
        print(classification_report(y, np.rint(pred)))

        acc = accuracy_score(np.rint(y), np.rint(pred))
        self.add_result(res, acc, False )
        res_segments_dict = {'Specificity': self.results_segments[1:,0],'Sensitivity': self.results_segments[1:,1],'Accuracy': self.results_segments[1:,2]  }
        df = pd.DataFrame.from_dict(res_segments_dict)
        df.to_csv(self.filename_seg)
        res_patients_dict =  {'Specificity': self.results_patients[1:,0],'Sensitivity': self.results_patients[1:,1],'Accuracy': self.results_patients[1:,2]  }
        df = pd.DataFrame.from_dict(res_patients_dict)
        df.to_csv(self.filename_patient)



class Results_level:
    '''
    Class to save results for severity prediction
    '''
    def __init__(self, filename_seg, filename_patient, dir):
        '''
        :param filename_seg: filename (csv) where to save the results
        :param filename_patient:
        :param dir: directory where results files are saved
        '''
        self.results_patients = np.zeros(1)
        self.results_segments = np.zeros(1)
        self.filename_seg = filename_seg
        self.filename_patient = filename_patient
        self.gt = np.array([])
        self.pred = np.array([])
        self.dir = dir
    def add_result( self,res, accuracy,  segments = True  ):

        all = np.array([ accuracy])

        if segments:
            self.results_segments = np.vstack((self.results_segments, all))
        else:
            self.results_patients = np.vstack((self.results_patients, all ))



    def validate_patient(self, model, x_val, y_val, count):
        shape=100
        '''
        :param model: trained model after 1 fold of cross validation
        :param x_val: x_Val for 1 forld of cross validation
        :param y_val: y_Val for 1 forld of cross validation
        :param count: vector containing the number of segments per patient
        :return:  save the results of the fold
        '''
        ## per segments
        y_val_one_hot = to_categorical(y_val, num_classes=4)
        pred_seg = model.predict(np.split(x_val, x_val.shape[2], axis=2))
        pred_seg=np.argmax(pred_seg, axis=1)
        print(" shape val:", np.unique(y_val[:, 0]))
        res = classification_report(np.rint(y_val), np.rint(pred_seg), output_dict = True ) #np.rint(np.argmax(pred_seg, axis=1))
        acc = accuracy_score(np.rint(y_val), np.rint(pred_seg)) #np.rint(np.argmax(pred_seg, axis=1))
        self.add_result(res, acc,True  )
        print('result', res)
        print('acc', acc)
        eval = []
        y = []
        pred = []
        shape=100
        for m in range(1, len(count)):
            i = count[m]
            j = count[m - 1]
            score = model.evaluate(np.split(x_val[j:i, :, :], x_val.shape[2] , axis=2), y_val_one_hot[j:i])#x_val.shape[2]
            eval.append(score)
            y_gt = np.argmax(y_val_one_hot[j:i],1)
            y_gt , _ = stats.mode(y_gt, axis = None)
            y.append(y_gt)
            p = np.rint(model.predict(np.split(x_val[j:i, :, :], x_val.shape[2], axis=2))) #x_val.shape[2]
            p = np.argmax(p, 1 )
            p, _ = stats.mode(p, axis=None)
            pred.append(p)

        res = classification_report(y, np.rint(pred), output_dict = True )
        print(classification_report(y, np.rint(pred)))
        self.gt = np.append(self.gt, y)
        self.pred = np.append(self.pred, np.rint(pred))
        acc = accuracy_score(np.rint(y), np.rint(pred))
        self.add_result(res, acc, False )
        res_segments_dict = {'Accuracy': self.results_segments[1:,0]}
        df = pd.DataFrame.from_dict(res_segments_dict)
        df.to_csv(self.filename_seg)
        res_patients_dict =  {'Accuracy': self.results_patients[1:,0]}
        df = pd.DataFrame.from_dict(res_patients_dict)
        df.to_csv(self.filename_patient)
        print(res)


#severity
    def write_results(self,fold):
        '''
        Called at the end to write the final result files
        :return:
        '''
        res_segments_dict = {'Accuracy': self.results_segments[1:, 0]}
        df = pd.DataFrame.from_dict(res_segments_dict)
        df.to_csv(f"{self.dir}/res_seg_{fold}.csv")
        res_patients_dict = {'Accuracy': self.results_patients[1:, 0]}
        df = pd.DataFrame.from_dict(res_patients_dict)
        df.to_csv(f"{self.dir}/res_pat_{fold}.csv")
        file_pred = os.path.join(self.dir, 'pred.csv')
        file_gt = os.path.join(self.dir, 'gt.csv')
        np.savetxt(file_pred, self.pred, delimiter="," )
        np.savetxt(file_gt,self.gt, delimiter=",")
        res = classification_report(self.gt, self.pred)
        #print(res)
        self.cm = confusion_matrix(self.gt, self.pred)
        file_conf_matrx = os.path.join(self.dir, 'confusion_matrix.csv')
        np.savetxt(file_conf_matrx, self.cm, delimiter=",")
        
    

import numpy as np
import argparse
np.random.seed(2) #2
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
import datetime
import os
import time
import matplotlib.pyplot as plt
def lr_scheduler(epoch, lr, warmup_epochs=3, decay_epochs=30, initial_lr=1e-6, base_lr=1e-3, min_lr=5e-5):
    if epoch <= warmup_epochs:
        pct = epoch / warmup_epochs
        return ((base_lr - initial_lr) * pct) + initial_lr

    if epoch > warmup_epochs and epoch < warmup_epochs+decay_epochs:
        pct = 1 - ((epoch - warmup_epochs) / decay_epochs)
        return ((base_lr - min_lr) * pct) + min_lr

    return min_lr

def plot_acc_loss(history, PLOT_NAME):
    fig = plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    print('*******************************************************')
    print(f"model hist is : \n {history.history}")
    
    plt.plot(history.history['accuracy'])
    print(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    print(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
 
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig(str(PLOT_NAME))
    plt.show()
def train(model, X_train, y_train, X_val, y_val, learning_rate, log_filename, filename):
    """
    :param model: Initial untrained model (or loaded model for resuming)
    :param X_train: Training data
    :param y_train: Training labels
    :param X_val: Validation data
    :param y_val: Validation labels
    :param learning_rate: Learning rate
    :param log_filename: Filename where the training results will be saved
    :param filename: File where the weights will be saved
    :return: Trained model, training history
    """
    logger = CSVLogger(log_filename, separator=',', append=True)
    
    if os.path.exists(filename):
        # Load the existing weights
        model.load_weights(filename)
        print("Loaded weights from previous training.")

        # Get the last epoch from the log file
        history = pd.read_csv(log_filename)
        start_epoch = len(history) 
        print(f"Resuming training from epoch {start_epoch}.")
    else:
        start_epoch = 0

    checkpointer = ModelCheckpoint(filepath=filename,save_weights_only=True,  monitor='val_accuracy', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10, verbose=1, mode='min')
    callbacks_list = [checkpointer, early_stopping, logger]
    print("log",log_filename)
    start_time = time.time()
    y_train_one_hot = to_categorical(y_train, num_classes=4)
    y_val_one_hot = to_categorical(y_val, num_classes=4)
    history = model.fit(np.split(X_train, X_train.shape[2], axis=2),
                       y_train_one_hot,
                       verbose=1,
                       shuffle=True,
                       epochs=30,
                       initial_epoch=start_epoch,
                       batch_size=64,
                       validation_data=(np.split(X_val, X_val.shape[2], axis=2), y_val_one_hot),
                       callbacks=callbacks_list)
    print(history.history.keys())
    #model.save_weights(filename)
    #model.load_weights(filename)
    learning_rate =  learning_rate / 2
    rms = optimizers.Nadam(learning_rate=learning_rate)
        
    model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
    end_time = time.time()
    duration = end_time - start_time
    print("Total training time = " + str(duration))
    print("hello",history)  # Print training history (loss, accuracy etc.)
    return model, history 

def train_classifier(args):
    '''
    Function that performs the detection of Parkinson
    :param args: Input arguments
    :return:
    '''
    exp_name = args.exp_name
    subfolder = os.path.join(args.output, exp_name +'_' + datetime.datetime.now().strftime("%m_%d"), datetime.datetime.now().strftime(
        "%H_%M"))
    file_result_patients = os.path.join(subfolder,'res_pat.csv')
    file_result_segments = os.path.join(subfolder,'res_seg.csv')
    model_file = os.path.join(subfolder, "model.json")
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)

    val_results = Results(file_result_segments, file_result_patients)
    datas = Data(args.input_data, 1, 100, pk_level= False )  #100 default in yosra's code
    
    for i in range(0,2):
       
        print ("---------------------------------------------------------------------------")
        print ("- Training Folder Number: "+str(i))
        print ("---------------------------------------------------------------------------")
        PLOT_NAME = "fold"+str(i)+".png"
        learning_rate = 0.0001
        model = multiple_transformer(datas.X_data.shape[2])
        model_json = model.to_json()
        with open(model_file, "w") as json_file:
            json_file.write(model_json)

        print('fold', str(i))
        X_train, X_val, y_train, y_val = datas.separate_fold(i)
        log_filename = os.path.join( subfolder ,"training_" + str(i) + ".csv")
        w_filename = os.path.join(subfolder,"w_"+str(i)+".weights.h5")
        model, history  = train(model,X_train, y_train, X_val, y_val, learning_rate, log_filename, w_filename)
        print('Validation !!')
        
        y_pred = model.predict_classes(X_val)  # Assuming predict_classes for classification
        cm = confusion_matrix(y_val, y_pred)
        model_name_txt = "train_classifier"  # Replace with actual model name
        FOLD_NAME = "fold_" + str(i) 
        results = str(model_name_txt) + "/" + FOLD_NAME
        f = open(results + "_confusion_matrix.csv", "w")  # Create a separate file for the confusion matrix
        f.write("{}\n".format(cm))
        f.close()
        print("Confusion Matrix:")
        print(cm)

 
        val_results.validate_patient(model, datas.X_val, datas.y_val, datas.count_val)
        plot_acc_loss(history, PLOT_NAME)

def train_severity(args):
  
    '''
    :param args: Input arguments
    :return:
    '''
    features = np.arange(1, 19)


    exp_name = args.exp_name

    ubfolder = os.path.join(args.output, exp_name + '_' + datetime.datetime.now().strftime("%m_%d"), datetime.datetime.now().strftime(
        "%H_%M"))
   
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)
    file_result_patients = os.path.join(subfolder ,'res_pat.csv')
    file_result_segments = os.path.join(subfolder ,'res_seg.csv')

    model_file = os.path.join(subfolder, "model.json")
    val_results = Results_level(file_result_segments, file_result_patients, subfolder )
    datas = Data(args.input_data, 1, 100, pk_level= True)
    learning_rate = 0.0002
    #X_train, X_val_list, y_train, y_val_list = datas.separate_fold(range(0, 10))

    for i in range(0,10):
        print ("---------------------------------------------------------------------------")
        print ("- Training Folder Number: "+str(i))
        print ("---------------------------------------------------------------------------")
        X_train, X_val, y_train, y_val = datas.separate_fold(i)
        
        PLOT_NAME = "fold"+str(i)+".png"
       
        model = multiple_transformer_5_level(datas.X_data.shape[2])
        model_json = model.to_json()
        with open(model_file, "w") as json_file:
            json_file.write(model_json)
        print('fold', str(i))
       # datas.separate_fold(i)
        log_filename = os.path.join(subfolder, "training" + str(i) + ".csv")
        w_filename = os.path.join(subfolder,"w_"+str(i)+".weights.h5")
        model, history  = train(model,X_train, y_train, X_val, y_val, learning_rate, log_filename,  w_filename )
 
        print('Validation !!', datas.count_val)
        val_results.validate_patient(model, datas.X_val, datas.y_val, datas.count_val)
        print("history", history)
        val_results.write_results(i)
        #plot_acc_loss(history, PLOT_NAME)   
        ''' 
        y_pred = model.predict(datas.X_val)  
        y_pred = np.argmax(y_pred, axis=1) 

        print('*****************')
        print('Confusion Matrix')
        cm = confusion_matrix(y_val, y_pred)
        #print(cm)

        print('Classification Report')
        target_names = ['Healthy (Severity 0)', 'Mild (Severity 2)', 'Medium (Severity 2.5)', 'High (Severity 3)']
        print(classification_report(y_val, y_pred, target_names=target_names))

        # **Save Confusion Matrix (optional):**
        model_name_txt = "model_convLSTM_Transf"  # Replace with actual model name
        FOLD_NAME = "fold_" + str(i) 
        results = str(model_name_txt) + "/" + FOLD_NAME
        f = open(results + "_confusion_matrix.txt", "w")  # Create a separate file for the confusion matrix
        f.write("{}\n".format(cm))
        f.close()
        '''


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_data", default='data/', type=str)
    #' 
    parser.add_argument("-exp_name", default='train_severity', type=str, help = 'train_classifier ; train_severity')
    parser.add_argument("-output", default='output_incep', type=str)
    args = parser.parse_args(args=[])
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    if args.exp_name == 'train_classifier' :
        train_classifier(args)
    if args.exp_name == 'train_severity':
        train_severity(args)
