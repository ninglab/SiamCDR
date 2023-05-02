from typing_extensions import Concatenate
import numpy as np
import pandas as pd
import random

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Concatenate, Dense, Dropout 
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

#### Define custom loss function

#### Define classes
class FewShotFusion():
    def __init__(self, cellLineModelPath, drugModelPath, nodeList, 
                 activation='relu', dropout=0.0):
        if cellLineModelPath == 'None':
            print("[INFO] cell line feature extractor not loaded. Using raw features...")
            self.cellLineExtractor = None
            cellOutSize = 463
        else:
            print("[INFO] loading cell line feature extractor...")
            self.cellLineExtractor, cellOutSize = self._loadFeatureExtractor(modelPath=cellLineModelPath)
            self.cellLineExtractor._name = 'cellLineExtractor'

        if drugModelPath == 'None':
            print("[INFO] drug feature extractor not loaded. Using raw features...")
            self.drugExtractor = None
            drugOutSize = 256
        else:
            print("[INFO] loading drug feature extractor...")
            self.drugExtractor, drugOutSize = self._loadFeatureExtractor(modelPath=drugModelPath)
            self.drugExtractor._name = 'drugExtractor' 

        # Create dict to call build funct
        self.nodeList = nodeList
        self.actFunc = activation
        self.dropout = dropout 
       
        self.inSize = cellOutSize + drugOutSize
        print("[INFO] Model input Dim:", self.inSize)
        print("[INFO] Model latent Dim:", self.nodeList[-1])

        print("[INFO] building feature extractor...")
        self._buildFeatureExtractor()

        print("[INFO] building siamese network...")
        self._buildSiameseNet()
 
    def triplet_loss(self, y_true, y_pred, alpha=1.0):
        anchor, positive, negative = y_pred[:, :self.nodeList[-1]],\
                                    y_pred[:, self.nodeList[-1]:2*self.nodeList[-1]],\
                                    y_pred[:, 2*self.nodeList[-1]:]
        
        posDist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
        negDist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
        return tf.maximum(posDist - negDist + alpha, 0.)

    def _loadFeatureExtractor(self, modelPath):
        model = load_model(modelPath)
        featureExtractor = model.get_layer("model")
        outputSize = featureExtractor.outputs[0].shape[-1]
        featureExtractor.trainable = False
        return featureExtractor, outputSize

    def _buildFeatureExtractor(self):
        # Define input layer
        inputs = Input(self.inSize)

        # create hidden layers of encoder
        x = inputs
        for nodes in self.nodeList[:-1]:
            x = Dense(nodes, activation=self.actFunc)(x)
            if self.dropout > 0.:
                x = Dropout(self.dropout)(x)

        # define output layer
        outputs = Dense(self.nodeList[-1], activation=self.actFunc)(x)
        # create model
        self.featureExtractor = Model(inputs, outputs)

    def _buildSiameseNet(self):
        inAnchor = Input(self.inSize)
        inPos = Input(self.inSize)
        inNeg = Input(self.inSize)

        embAnchor = self.featureExtractor(inAnchor)
        embPos = self.featureExtractor(inPos)
        embNeg = self.featureExtractor(inNeg)

        output = Concatenate()([embAnchor, embPos, embNeg])
        self.siamese = Model(inputs=[inAnchor, inPos, inNeg], outputs=output)

    def createBatch(self, cdr, rna, drugs, by='rna', batchSize=256):
        x_anchors = np.zeros((batchSize, self.inSize))
        x_positives = np.zeros((batchSize, self.inSize))
        x_negatives = np.zeros((batchSize, self.inSize))

        posPairs = cdr[cdr.effective == 1.]
        negPairs = cdr[cdr.effective == 0.]

        for i in range(batchSize):
            n = 0
            while n != 1:
                randPair = posPairs.sample(n=1)
                drug = randPair.name.values[0]
                cell = randPair.DepMap_ID.values[0]
                if by == 'rna':
                    posCell = cell; negCell = cell
                    try:
                        posDrug = posPairs[(posPairs.DepMap_ID == cell) & (posPairs.name != drug)].sample(n=1).name.values[0]
                        negDrug = negPairs[(negPairs.DepMap_ID == cell) & (negPairs.name != drug)].sample(n=1).name.values[0]
                    except ValueError:
                        continue
                else:
                    posDrug = drug; negDrug = drug
                    try:
                        posCell = posPairs[(posPairs.DepMap_ID != cell) & (posPairs.name == drug)].sample(n=1).DepMap_ID.values[0]
                        negCell = negPairs[(negPairs.DepMap_ID != cell) & (negPairs.name == drug)].sample(n=1).DepMap_ID.values[0]
                    except ValueError:
                        continue
                n = 1

            x_anchors[i] = np.array(drugs[drug] + rna[cell]) 
            x_positives[i] = np.array(drugs[posDrug] + rna[posCell])
            x_negatives[i] = np.array(drugs[negDrug] + rna[negCell])

        return [x_anchors, x_positives, x_negatives]

    def dataGenerator(self, cdr, rna, drugs, by='rna', batchSize=256):
        while True:
            x = self.createBatch(cdr, rna, drugs, batchSize=batchSize, by=by)
            y = np.zeros((batchSize, 3*self.nodeList[-1]))
            yield x, y

    def fit(self, rnaPath, drugPath, cdrPath, by='rna',
            learningRate=0.001, decayRate=0.99, decaySteps=80,
            batchSize=512, stepsPerEpoch=16, epochs=500, saveModel=False, modelPath=None):

        # Load RNA and Drugs and get embeddings if indicated
        fps = pd.read_csv(drugPath, index_col=0)
        rna = pd.read_csv(rnaPath, index_col=0)

        if self.cellLineExtractor != None:
            rna = pd.DataFrame(self.cellLineExtractor(rna.to_numpy()), index=rna.index)
        if self.drugExtractor != None:
            fps = pd.DataFrame(self.drugExtractor(fps.to_numpy()), index=fps.index)

        # Load drug cell line pairs and separate into positive and negative combos
        cdr = pd.read_csv(cdrPath).loc[:, ['DepMap_ID', 'name',  'effective']]
        cdr = cdr[cdr.DepMap_ID.isin(list(rna.index))]

        # Convert DFs to dicts for quicker access than loc
        rna = rna.T.to_dict('list')
        fps = fps.T.to_dict('list')

        print("[INFO] compiling model...")
        schedule = ExponentialDecay(initial_learning_rate=learningRate,
                                    decay_steps=decaySteps,
                                    decay_rate=decayRate)

        opt = Adam(learning_rate=schedule)
        self.siamese.compile(loss=self.triplet_loss, 
                             optimizer=opt,
                             run_eagerly=True)

        print("[INFO] training model...")
        callBacks = []
        stopEarly = EarlyStopping(monitor = 'loss',
                                  min_delta = 0.0001,
                                  patience = 10,
                                  restore_best_weights = True)
        callBacks.append(stopEarly)

        if saveModel & (modelPath != None):
            bestOnly = ModelCheckpoint(filepath = modelPath,
                                       monitor = 'loss',
                                       save_best_only= True)
            callBacks.append(bestOnly)

        random.seed(34)
        history = self.siamese.fit(self.dataGenerator(cdr, rna, fps, by=by, batchSize=batchSize),
                                   batch_size=batchSize, epochs=epochs,
                                   steps_per_epoch=stepsPerEpoch,
                                   callbacks=callBacks, verbose=2)
        return history.history

