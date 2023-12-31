from typing_extensions import Concatenate

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Concatenate, Dense, Dropout
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

class SiamCDR_DNN():
    def __init__(self, cellLineModelPath, drugModelPath,
                nodeList, activation='relu', dropout=None):
        # Create dict to call build funct
        self.buildParams = {'nodeList': nodeList,
                            'activation': activation}
        if dropout != None:
            self.buildParams['dropout'] = dropout

        # load pretrained cell line encoder
        self.cellLineInputDim = 463
        if cellLineModelPath == 'None':
            print("[INFO] cell line feature extractor not loaded. Using raw features...")
            self.cellLineExtractor = None
        else:
            print("[INFO] loading cell line feature extractor...")
            self.cellLineExtractor = self.loadFeatureExtractor(modelPath=cellLineModelPath)
            self.cellLineExtractor._name = 'cellLineExtractor'
            self.cellLineExtractor.trainable = False
        
        # load pretrained Drug encoder
        self.drugInputDim = 256
        if drugModelPath == 'None':
            print("[INFO] drug feature extractor not loaded. Using raw features...")
            self.drugExtractor = None
        else:
            print("[INFO] loading drug feature extractor...")
            self.drugExtractor = self.loadFeatureExtractor(modelPath=drugModelPath)
            self.drugExtractor._name = 'drugExtractor'
            self.drugExtractor.trainable = False

        print("[INFO] building CDR model...")
        self.buildModel(**self.buildParams)

    # Loads either drug or cell line encoder
    def loadFeatureExtractor(self, modelPath):
        return load_model(modelPath).get_layer("model")

    def buildModel(self, nodeList, activation='relu', dropout=0.0):
        # input
        drugInput = Input(self.drugInputDim)
        cellLineInput = Input(self.cellLineInputDim)

        # combine drug and cell line features
        if self.drugExtractor != None:
            drugEmbed = self.drugExtractor(drugInput)
            if self.cellLineExtractor == None:
                pairEmbed = Concatenate()([drugEmbed, cellLineInput])
            else:
                cellLineEmbed = self.cellLineExtractor(cellLineInput)
                pairEmbed = Concatenate()([drugEmbed, cellLineEmbed])
        else:
            if self.cellLineExtractor == None:
                pairEmbed = Concatenate()([drugInput, cellLineInput])
            else:
                cellLineEmbed = self.cellLineExtractor(cellLineInput)
                pairEmbed = Concatenate()([drugInput, cellLineEmbed])
        

        # full embeddings that will be used as input to DNN
        self.pairEncoder = Model(inputs=[drugInput, cellLineInput], 
                                   outputs=pairEmbed, 
                                   name='pairEncoder')


        # create hidden layers of DNN
        x = self.pairEncoder([drugInput, cellLineInput])
        if len(nodeList) > 0:
            for nodes in nodeList:
                x = Dense(nodes, 
                         activation=activation)(x)
                if dropout > 0.:
                    x = Dropout(dropout)(x)

        outputs = Dense(1, activation='sigmoid')(x)

        # create model
        self.model = Model(inputs=[drugInput, cellLineInput], 
                           outputs=outputs, name='SiamCDR-DNN')

    def fit(self, train, val,
            learningRate=0.001, decayRate=0.99, decaySteps=100,
            earlyStopping=True, patience=15, minDelta=0.0001,
            batchSize=512, epochs=250, saveModel=True, modelPath=None):

        trainDrugs, trainCells, trainLabels = train
        valDrugs, valCells, valLabels = val

        del train, val

        print("[INFO] compiling model...")
        schedule = ExponentialDecay(initial_learning_rate=learningRate,
                                    decay_steps=decaySteps,
                                    decay_rate=decayRate)
        opt = Adam(learning_rate=schedule)
        self.model.compile(loss='binary_crossentropy', optimizer=opt, run_eagerly=True)
        
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

        history = self.model.fit([trainDrugs, trainCells], trainLabels,
                                   validation_data=([valDrugs, valCells], valLabels),
                                   batch_size=batchSize, epochs=epochs,
                                   callbacks=callBacks, verbose=2)
        return history.history

    def predict(self, input_data):
        return self.model.predict(input_data)

