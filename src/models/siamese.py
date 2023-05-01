import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


class SiameseNeuralNet():
    def __init__(self, inputShape, nHiddenLayers, embeddingDim, 
                 activation=None, dropout=None, continuous=False):
        # Create dict to call build funct
        self.buildParams = {'inputShape': inputShape,
                            'nHiddenLayers': nHiddenLayers,
                            'embeddingDim': embeddingDim}
        if activation != None:
            self.buildParams['activation'] = activation
        if dropout != None:
            self.buildParams['dropout'] = dropout
       
        self.continuous = continuous

        print("[INFO] building feature extractor...")
        self.buildFeatureExtractor(**self.buildParams)
        
        print("[INFO] building siamese network...")
        self.buildSiameseNet()
        
    def buildFeatureExtractor(self, inputShape, nHiddenLayers, embeddingDim, activation='relu', dropout=0.2):
        # define the inputs
        inputs = Input(inputShape)

        # create hidden layers
        x = inputs
        for i in range(nHiddenLayers):
            x = Dense(embeddingDim, activation=activation)(x)
            if dropout > 0.:
                x = Dropout(dropout)(x)

        # define output layer
        outputs = Dense(embeddingDim, activation=activation)(x)
        # create model
        self.featureExtractor = Model(inputs, outputs)
        
    def buildSiameseNet(self):
        inputShape = self.buildParams['inputShape']
        
        d1 = Input(shape=inputShape)
        d2 = Input(shape=inputShape)
        
        ft1 = self.featureExtractor(d1)
        ft2 = self.featureExtractor(d2)
        
        def euclideanDistance(drugs):
            (v1, v2) = drugs
            sumSquared = tf.math.reduce_sum(tf.math.square(v1 - v2), axis=1, keepdims=True)
            return tf.math.sqrt(tf.math.maximum(sumSquared, 1e-07))

        distance = Lambda(euclideanDistance, name='distLambdaFunc')([ft1, ft2])
        outputs = Dense(1, activation='sigmoid')(distance)
        self.siamese = Model(inputs=[d1, d2], outputs=outputs)
    
    def fit(self, dataPath, batchSize=64, epochs=10, 
            learningRate=0.001, decayRate=0.99, decaySteps=1000,
            earlyStopping=True, patience=10, minDelta=0.0001,
            saveModel=False, modelPath=None, valPath=None): 
        
        print("[INFO] loading data...")
        fpLen = self.buildParams['inputShape']
        
        data = pd.read_csv(dataPath).to_numpy()
        np.random.shuffle(data)
        
        if valPath == None:
            valSize = floor(len(data)*0.05)

            d1Train = tf.convert_to_tensor(data[:-valSize, :fpLen].reshape(-1, fpLen), dtype=tf.float32)
            d2Train = tf.convert_to_tensor(data[:-valSize, fpLen:-1].reshape(-1, fpLen), dtype=tf.float32)
        
            d1Val = tf.convert_to_tensor(data[-valSize:, :fpLen].reshape(-1, fpLen), dtype=tf.float32)
            d2Val = tf.convert_to_tensor(data[-valSize:, fpLen:-1].reshape(-1, fpLen), dtype=tf.float32)
        
            trainLabels = tf.convert_to_tensor(data[:-valSize, -1].reshape(-1, 1), dtype=tf.float32)
            valLabels = tf.convert_to_tensor(data[-valSize:, -1].reshape(-1, 1), dtype=tf.float32)
            
        else:
            val = pd.read_csv(valPath).to_numpy()
            d1Train = tf.convert_to_tensor(data[:, :fpLen].reshape(-1, fpLen), dtype=tf.float32)
            d2Train = tf.convert_to_tensor(data[:, fpLen:-1].reshape(-1, fpLen), dtype=tf.float32)
        
            d1Val = tf.convert_to_tensor(val[:, :fpLen].reshape(-1, fpLen), dtype=tf.float32)
            d2Val = tf.convert_to_tensor(val[:, fpLen:-1].reshape(-1, fpLen), dtype=tf.float32)
        
            trainLabels = tf.convert_to_tensor(data[:, -1].reshape(-1, 1), dtype=tf.float32)
            valLabels = tf.convert_to_tensor(val[:, -1].reshape(-1, 1), dtype=tf.float32)     
            del val
        
        del data
        
        print("[INFO] compiling model...") 
        schedule = ExponentialDecay(initial_learning_rate=learningRate,
                                    decay_steps=decaySteps,
                                    decay_rate=decayRate)

        opt = Adam(learning_rate=schedule)

        if self.continuous:
            lossFunc = 'mean_squared_error'
        else:
            lossFunc = 'binary_crossentropy'
        self.siamese.compile(loss=lossFunc, optimizer=opt, run_eagerly=True)
        
        print("[INFO] training model...")
        callBacks = []
        if earlyStopping:
            stopEarly = EarlyStopping(monitor='val_loss',
                                      min_delta=minDelta,
                                      patience=patience,
                                      restore_best_weights=True)
            callBacks.append(stopEarly)
        if saveModel & (modelPath != None):
            bestOnly = ModelCheckpoint(filepath = modelPath,
                                       monitor = 'val_loss',
                                       save_best_only= True)
            callBacks.append(bestOnly)

        history = self.siamese.fit([d1Train, d2Train], trainLabels,
                                   validation_data=([d1Val, d2Val], valLabels),
                                   batch_size=batchSize, epochs=epochs, 
                                   callbacks=callBacks, verbose=2)
        return history.history
