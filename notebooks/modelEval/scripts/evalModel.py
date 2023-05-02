#### LOAD PACKAGES
import os
from re import S
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Concatenate, Input

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from .utils import Predictions, Compiler

#### Define Functions
def triplet_loss(y_true, y_pred, alpha=0.1):
    anchor, positive, negative = y_pred[:, :8],\
                                 y_pred[:, 8:2*8],\
                                 y_pred[:, 2*8:]
    posDist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
    negDist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
    return tf.maximum(posDist - negDist + alpha, 0.)

def countDrugsK(df, k=3):
    drugCount = {1:{}, 2:{}, 3:{}}
    wrong = []
    for cell, subdf in df.groupby(by='cell_line'):
        sortDF = subdf.sort_values(by='pred', ascending=False).reset_index(drop=True)
        drugs = sortDF.loc[:k-1, 'drug']
        n = 0
        for drug in drugs:
            n += 1
            if drug in drugCount[n].keys():
                drugCount[n][drug] += 1
            else:
                drugCount[n][drug] = 1
                for i in drugCount.keys():
                    if i != n:
                        drugCount[i][drug] = 0
        drug = drugs[0]

        if sortDF.iloc[:k, :].true.sum() == 0:
            wrong.append(cell)
            print(f"No true effective drugs identified in top {k} for {cell} (top drug: {drug})")
#             print(f"\tCell line: {sortDF.loc[0, 'cell_line']}; \n")
#         else:
#             print(f"\tCell line: {sortDF.loc[0, 'cell_line']}; Top drug: {drug}")
            
    drugCount = pd.DataFrame(drugCount)
    drugCount['total'] = drugCount[1] + drugCount[2] + drugCount[3]
    return drugCount, wrong


def clPrecision(preds, modelName=None, thresh=0.5, at=5, getResults=False, verbose=True):
    p1 = []
    p2 = []
    p3 = []
    p4 = []
    p5 = []
    p0 = []
    cellLines = []
    for cell, subdf in preds.groupby(by='cell_line'):
        nEff = subdf.true.sum()
        if nEff < 5:
            continue
            
        cellLines.append(cell)
        sortDF = subdf.sort_values(by='pred', ascending=False)
        p1.append(sortDF.iloc[:1, :].true.sum() / 1)
        p2.append(sortDF.iloc[:2, :].true.sum() / 2)
        p3.append(sortDF.iloc[:3, :].true.sum() / 3)
        p4.append(sortDF.iloc[:4, :].true.sum() / 4)
        p5.append(sortDF.iloc[:5, :].true.sum() / 5)
        if nEff >= 10:
            p0.append(sortDF.iloc[:10, :].true.sum() / 10)
            
    if at == 5:
        current = np.mean(p5)
    elif at == 1:
        current = np.mean(p1)
        
    if current >= thresh:
        if verbose:
            thresh = current
            if modelName != None:
                print(f"Model: {modelName}")
            print(f"\tPrecision@1: {round(np.mean(p1), 4)}")
            print(f"\tPrecision@2: {round(np.mean(p2), 4)}")
            print(f"\tPrecision@3: {round(np.mean(p3), 4)}")
            print(f"\tPrecision@4: {round(np.mean(p4), 4)}")
            print(f"\tPrecision@5: {round(np.mean(p5), 4)}")
            print(f"\tPrecision@10: {round(np.mean(p0), 4)}\n\n")
        
    if getResults:
        return [np.mean(p1), np.mean(p2), np.mean(p3), np.mean(p4), np.mean(p5)]
    
    if verbose:
        return thresh

def precision(preds, thresh=0.5, modelName=None, by='cellLine'):
    if by == 'cellLine':
        return clPrecision(preds, modelName, thresh=thresh)
    else:
        cancers = {}
        for ct, subdf in preds.groupby(by = 'cancer_type'):
            cancers[ct] = clPrecision(subdf, verbose=False, getResults=True)
        return pd.DataFrame(cancers, index=['p1', 'p2', 'p3', 'p4', 'p5']).T
    

#### Define Classes
class evalFullModel():
    def __init__(self, basePath, modelType):
        self.basePath = basePath
        self.modelType = modelType
                 
    def _getResults(self, metrics=['precision', 'recall', 'f1', 'auc']):
        preds = pd.read_csv(predPath, index_col=0)
        Evaluator = Predictions(None)
        cellLineResults = Evaluator._cellLinePerformance(preds, metrics)
        cancerResults = Evaluator._cancerTypePerformance(cellLineResults) 
        return cellLineResults, cancerResults
    
    def iterateModelResults(self, dataset='test', metrics=['precision', 'recall', 'f1', 'auc']):
        predDir = os.path.join(self.basePath, f"{dataset}_preds")
        resultDir = os.path.join(self.basePath, f"{dataset}_res")
        files = os.listdir(predDir)
        for f in files:
            predPath = os.path.join(predDir, f)
            modelName = "_".join(f.split('_')[:-1])
            cellLinePath = os.path.join(resultDir, f"{modelName}_CLresults.csv")
            cancerPath = os.path.join(resultDir, f"{modelName}_CancerResults.csv")
            cellLineResults, cancerResults = self._getResults(predPath, metrics)
            cellLineResults.to_csv(cellLinePath)
            cancerResults.to_csv(cancerPath)
    
    def compileResults(self, subdir, prefix=None, suffix='CancerResults.csv'):
        if prefix == None:
            prefix = self.modelType
        compiler = Compiler(selfbasePath, self.modelType)
        compiler.compileResults(subdir, prefix, suffix)
    
    def getResultsAndComile(self, prefix=None, datasets=['test', 'newcancer'], 
                       metrics=['precision', 'recall', 'f1', 'auc']):
        for d in datasets:
            iterateModelResults(d, metrics)
            subdir = f"{d}_res"
            compileResults(subdir, prefix)
            
    def iterateModels(self, modelName, dataset, thresh=0.5, k=1, by='cellLine', 
                      drug='embed', rna='embed', fusion=True):
        if (modelName != None):
            # get preds
            modelPredPath = os.path.join(self.basePath, f"{dataset}_preds", f"{modelName}_preds.csv")
            predDF = pd.read_csv(modelPredPath)

            predDF.sort_values(by='pred', ascending=False, inplace=True)
            if by == 'cellLine':
                print('Average Cell Line precision @ k')
                precision(predDF, modelName, thresh, by)

                print("Top ranked drug for each cell line:")
                counts, wrong = countDrugsK(predDF, k)

                print(f"\n# cell lines without highly effective drug among top-{k} predictions: {len(wrong)}")
#                 for w in wrong:
#                     print(f"\t{w}")

                counts.sort_values(by=[1, 2, 3], ascending=False, inplace=True)
                return predDF, wrong, counts

            else:
                df = precision(predDF, modelName, thresh, by)
                df.sort_values(by=['p1','p1','p3','p4','p5'], ascending=False, inplace=True)
                return df

        else:
            for f in files:
                modelPredPath = os.path.join(self.basePath, f"{dataset}_preds", f"{modelName}_preds.csv")
                predDF = pd.read_csv(modelPredPath)
                thresh = precision(predDF.copy(), f, thresh, by)

            print(thresh)
            
            
class evalLogisticModels():
    def __init__(self, train, trainEff, test, new=None,
                 fusionPath=None, drugPath=None, rnaPath=None):
        
        self.drugInputDim = 256
        self.rnaInputDim = 463
        
        if drugPath != None:
            self.drugEncoder = self.loadEncoder(drugPath, 'drug')
        else:
            self.drugEncoder = None
            
        if rnaPath != None:
            self.rnaEncoder = self.loadEncoder(rnaPath, 'rna')
        else:
            self.rnaEncoder = None
            
        if fusionPath != None:
            self.fusionEncoder = self.getEncoder(fusionPath)
         
        self.train = self.fusionEncoder(train)
        self.test = self.fusionEncoder(test)
        if new != None:
            self.new = self.fusionEncoder(new)
        self.lm = LogisticRegression().fit(self.train, trainEff)
    
    @staticmethod
    def loadEncoder(path, which='rna'):
        snn = load_model(path)
        encoder = snn.get_layer('model')
        encoder._name = f'{which}Encoder'
        return encoder
                                          
    def getEncoder(self, fusionPath):
        # Define encoded drug input
        drugInput = Input(self.drugInputDim)
        rnaInput = Input(self.rnaInputDim)
                                             
        if self.drugEncoder == None:
            if self.rnaEncoder == None:         
                pairEmbed = Concatenate()([drugInput, rnaInput])
            else: 
                rnaEmbed = self.rnaEncoder(rnaInput)
                pairEmbed = Concatenate()([drugInput, rnaEmbed])
        else:
            drugEmbed = self.drugEncoder(drugInput)
            if self.rnaEncoder == None:         
                pairEmbed = Concatenate()([drugEmbed, rnaInput])
            else: 
                rnaEmbed = self.rnaEncoder(rnaInput)
                pairEmbed = Concatenate()([drugEmbed, rnaEmbed])

        if fusionPath != None:
            fusionEncoder = load_model(fusionPath, custom_objects={'triplet_loss':triplet_loss})
            fusionEncoder = fusionEncoder.get_layer('model')
            pairEmbed = fusionEncoder(pairEmbed)

        return Model(inputs=[drugInput, rnaInput], outputs=pairEmbed)
                                             

    def evaluate(self, testDF, newDF=None, thresh=0.5, k=3, returnThresh=False):
        testDF['pred'] = [p[1] for p in self.lm.predict_proba(self.test)]
        testDF.sort_values(by='pred', ascending=False, inplace=True)

        if returnThresh:
            thresh = precision(testDF, thresh=thresh, by='cellLine')
            return thresh
        else:
            print('Average Cell Line precision @ k on test set')
            _ = precision(testDF, thresh=thresh, by='cellLine')

        print('Average Cell Line precision @ k on newcancer set')
        newDF['pred'] = [p[1] for p in self.lm.predict_proba(self.new)]
        _ = precision(newDF, thresh=thresh, by='cellLine')

#         print("Top ranked drug for each cell line:")
        counts, wrong = countDrugsK(testDF, k)
        counts.sort_values(by=[1, 2, 3], ascending=False, inplace=True)

        print(f"\n# of unique drugs among top-{k} predictions: {len(counts)}")
        return testDF, newDF, wrong, counts

    def getCancerPerformance(self, testDF, newDF):
        df = precision(testDF.copy(), thresh=0.5, by='cancer')
        df.sort_values(by=['p1','p1','p3','p4','p5'], ascending=False, inplace=True)

        df2 = precision(newDF.copy(), thresh=0.5, by='cancer')
        df2.sort_values(by=['p1','p1','p3','p4','p5'], ascending=False, inplace=True)
        return df, df2
                
                