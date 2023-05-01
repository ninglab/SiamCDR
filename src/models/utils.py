#### LOAD PACKAGES
import os
from re import S
import numpy as np
import pandas as pd

from sklearn.metrics import precision_score, roc_auc_score, precision_recall_curve, r2_score,\
                            mean_squared_error, precision_score, recall_score
from sklearn.preprocessing import binarize

#### DEFINE CONSTANTS
file_dir = os.path.dirname(__file__)

#### Define classes
class createDirs( object ):
    """
    Create the subdirectories given the path to a parent directory
    """
    def __init__(self, parent, kind='models'):
        self.parent = parent
        self.paths = self.define_subdirs()
        self.mkdir()

    def define_subdirs(self):
        paths = {}
        paths['models'] = os.path.join(self.parent, 'models')
        
        subResDirs = ['fit', 'train', 'val', 'test', 'newcancer']
        for d in subResDirs:
            if d != 'fit':
                key = f"{d}_res"
            else:
                key = d
            paths[key] = os.path.join(self.parent, f"{d}_results")

        subPredDirs = ['train', 'val', 'test', 'newcancer']
        for d in subPredDirs:
            paths[f"{d}_res"] = os.path.join(self.parent, f"{d}_preds")

        return paths
    
    def mkdir(self):
        for p in [self.parent] + list(self.paths.values()):
            if not os.path.exists(p):
                os.mkdir(p)

# Evaluation
class Statistics( object ):
    """
    Used to evaluate performance. Takes model predictions and ground truth labels.
    Can calculate precision, precision@k, recall, recall@k, f1-score, accuracy, and AUROC.
    """
    def __init__(self, true, pred, threshold=None):
        dataDict = {'pred':pred, 'true':true}
        self.data = pd.DataFrame(dataDict)
        if threshold != None:
            self.threshold = threshold
        else:
            self.threshold = self.getThreshold()

    def getThreshold(self):
        precision, recall, thresholds = precision_recall_curve(self.data.true, self.data.pred)
        precision += 0.0000001
        recall += 0.0000001
        fscore = (2 * precision * recall) / (precision + recall)

        ix = np.argmax(fscore)
        #  print(precision[ix], recall[ix], fscore[ix], thresholds[ix])
        return thresholds[ix]

    def getBinary(self):
        self.data['bin'] = binarize(self.data.pred.to_numpy().reshape(-1, 1), threshold=self.threshold)

    def evaluate(self, metric):
        if metric == 'precision':
            return self.precision()
        elif metric == 'recall':
            return self.recall()
        elif metric == 'f1':
            return self.f1()
        elif metric == 'auc':
            return self.auc()
        elif metric == 'acc':
            return self.accuracy()
        elif metric == 'mse':
            return self.mean_squared_error()
        elif metric == 'r2':
            return self.r2()

    def precision(self, k=None):
        df = self.data
        if k != None:
            df = df.sort_values(by='pred', ascending=False).reset_index().iloc[:k, :]
            tp = df.true.sum()
            total = len(df)
            try: return tp / total
            except ZeroDivisionError: return 0

        else:
            if self.threshold != None:
                df.pred = np.where(df.pred > self.threshold, 1, 0)
            return precision_score(y_true=df.true, y_pred=df.pred)
            
    def recall(self, k=None):
        df = self.data
        total = df.true.sum()
        if k != None:
            dfK = df.sort_values(by='pred', ascending=False).reset_index().iloc[:k, :]
            topK = dfK.true.sum()
            try: return topK / total
            except ZeroDivisionError: return 0
        
        else:
            if self.threshold != None:
                df.pred = np.where(df.pred > self.threshold, 1, 0)
            return recall_score(y_true=df.true, y_pred=df.pred)
    
    def f1(self):
        p = self.precision()
        r = self.recall()
        if (p == 0.) & (r == 0.):
            return 0
        else:
            return (2 * p * r) / (p + r)
    
    def accuracy(self):
        df = self.data
        ncorrect = df[df.pred == df.true].shape[0]
        return ncorrect / len(df)
    
    def auc(self):
        df = self.data
        if df.pred.dtype == float:
           self.getBinary()
           return roc_auc_score(df.true, df.bin)
        else:
            return roc_auc_score(df.true, df.pred)

    def mean_squared_error(self):
        df = self.data
        return mean_squared_error(df.true, df.pred)

    def r2(self):
        df = self.data
        return r2_score(df.true, df.pred)


class Predictions( object ):
    def __init__(self, model):
        self.predict = lambda x: model.predict(x)
    
    def _initPredDF(self, cdr, cancer_type):
        pred_dict = {'cell_line': cdr.DepMap_ID.values,
                     'cancer_type': cancer_type,
                     'drug': cdr.name.values,
                     'true': cdr.effective.values}

        return pd.DataFrame(pred_dict)

    def _cellLinePerformance(self, preds, metrics, threshold=None):
        cols = ['cell_line', 'cancer_type']
        cols.extend(metrics)
        CLresults = pd.DataFrame(columns=cols)
        CLresults['cell_line'] = preds.cell_line.unique()
        CLresults.set_index('cell_line', inplace=True)
        
        for cell_line, subdf in preds.groupby('cell_line'):
            cancer = subdf.cancer_type.values[0]
            if threshold != None:
                stats = Statistics(true=subdf.true.values, pred=subdf.pred.values, threshold=threshold)
            else:
                stats = Statistics(true=subdf.true.values, pred=subdf.pred.values)
            row = [cancer]
            for m in metrics:
                row.append(stats.evaluate(m))
            
            CLresults.loc[cell_line, :] = row
        CLresults = CLresults.infer_objects()
        #  CLresults.sort_values(by='cancer_type', inplace=True)
        return CLresults

    def _cancerTypePerformance(self, CLresults):
        cancerResults = CLresults.groupby(by='cancer_type').mean()
        cancerResults.loc['cancer_avg', :] = cancerResults.mean()
        cancerResults.loc['cell_line_avg', :] = CLresults.mean()
        return cancerResults

    def evalPerformance(self, input_data, cdr, cancer_type, metrics, res_out, pred_out='', save_preds=False):
        pred_df = self._initPredDF(cdr, cancer_type)
        pred_df['pred'] = np.array(self.predict(input_data))
        if save_preds:
            pred_df.to_csv(f'{pred_out}_preds.csv')
        CLresults = self._cellLinePerformance(pred_df, metrics)
        #  CLresults.to_csv(f"{res_out}_Threshold{str(round(threshold, 4)).split('.')[1]}_CLresults.csv")
        CLresults.to_csv(f"{res_out}_CLresults.csv")

        cancerResults = self._cancerTypePerformance(CLresults)
        cancerResults.to_csv(f"{res_out}_CancerResults.csv")

class Compiler( object ):
    def __init__(self, parent, model_type):
        hyper_lists = {'rnaRBM': [['mean', 'cov', 'fac', 'LR', 'DecayRate',
                                   'DecayStep', 'L1', 'MCsampleSteps'],
                                  [3, 4, 6]],
                       'combRBM': [['rnaRBM', 'mean', 'cov', 'fac', 'LR', 
                                    'DecayRate', 'DecayStep', 'L1', 'MCsampleSteps'],
                                    [4, 5, 7]],
                       'CDRsmcRBM': [['smcRBM', 'layers', 'nodes', 'dropout', 
                                     'activation', 'LR', 'DecayRate', 'DecayStep', 'L1'],#, 'Threshold'],
                                     [3, 5, 6, 8, 9]],
                       'DeepDSC': [['Encoder', 'Hidden', 'dropout', 'activation',
                                   'LR', 'DecayRate', 'DecayStep'],
                                   [0, 2, 3]],
                       'DeepDR': [],
                       'DeepCDR': [],
                       'RefDNN': [],
                       'fsCDR': [['nodeList', 'dropout', 'activation',
                                'learningRate', 'decayRate', 'decaySteps'],
                                [1, 3, 4]]}
        self.model_type = model_type
        self.formatting = hyper_lists[model_type]
        self.paths = createDirs(parent)

    def _listFiles(self, subdir, prefix='', suffix=''):
        path = self.paths.paths[subdir]
        files = os.listdir(path)
        f_list = [os.path.join(path, i) for i in files if (i.startswith(prefix)) & (i.endswith(suffix))]
        return f_list

    @staticmethod
    def _stripVal(paramVal):
        param = paramVal.rstrip('0123456789-')
        if paramVal.startswith('L1'):
            val = paramVal[len(param)+1:]
        else:
            val = paramVal[len(param):]
        return val

    def getDSCencoder(self, fname):
        encoder = fname.split('_DNN_')[0]
        encoder = encoder.split('Encoder_')[-1]
        params = fname.split('_DNN_')[-1]
        hidden = params.split('_DO')[0]
        hidden = hidden.split('Hidden_')[-1]
        params = f"DO{params.split('DO')[-1]}".split('_')[:-1]
        return encoder, hidden, params

    def getParams(self, file_path):
        start = 1
        if self.model_type in ['rnaRBM', 'combRBM']:
            start += 1
        
        vals = []
        float_idx = self.formatting[1]
        fname = file_path.split('/')[-1]
        if self.model_type == 'DeepDSC': 
            encoder, hidden, params = self.getDSCencoder(fname)
            vals.extend([encoder, hidden])
        elif self.model_type == 'fsCDR':
            fname = "_".join(fname.split('_')[start:-1])
            hidden, params = fname.split('_DO')
            vals.append(hidden[2:])
            params = f"DO{params}".split('_')
        else:
            params = fname.split('_')[start:-1]
        formatVal = lambda x: x if 'e' in x else x.replace('-', '.')
        for idx, param in enumerate(params):
            if param.startswith('AF'):
                val = param[2:]
            else:
                val = self._stripVal(param)
                val = formatVal(val)
                if self.model_type == 'DeepDSC':
                    if idx in float_idx:
                        val = f'0.{val}'
                        val = float(val)
                else:
                    if ('-' in val):
                        val = float(val)
                    elif set(val.split('.')[-1]) == {'0'}:
                        val = int(float(val))
                    else:
                        val = float(val)
            vals.append(val)
        return vals
        
    def compileResults(self, subdir, prefix='', suffix='', save=True):
        file_list = self._listFiles(subdir, prefix, suffix)
        cols = self.formatting[0].copy()
        results = []
        for n, file_path in enumerate(file_list):
            row = self.getParams(file_path)
            df = pd.read_csv(file_path, index_col='cancer_type')
            for i in df.index.values:
                if n == 0:
                    if i == 'cancer_avg':
                        name = 'allCancer'
                    elif i == 'cell_line_avg':
                        name = 'CellLine'
                    else:
                        name = i.split(' ')[0]
                        name = name.split('/')[0]
                for j in df.columns.values:
                    if n == 0:
                        cols.append(f'{name}_{j}')
                    row.append(df.loc[i, j])
            results.append(row)

        results = pd.DataFrame(results, columns=cols)
        if save:
            sub = subdir.split('_')[0]
            outname = f'{prefix}_{sub}_compiled_results.csv' 
            outpath = os.path.join(self.paths.parent, outname)
            results.to_csv(outpath, index=False)
        else:
            return results
