#### LOAD PACKAGES
import os
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import float32 as tf32
from tensorflow.keras import backend
from tensorflow.keras.losses import Loss

#### DEFINE CONSTANTS
file_dir = os.path.dirname(__file__)

#### Define classes
# Load data
class DataLoader( object ):
    """
    Splits supplied data into training, validation, testing subsets for n_folds.
    """
    def __init__(self, 
                drug_path=f'{file_dir}/../../data/drug_response/prism_drugs.csv', 
                cdr_path=f'{file_dir}/../../data/drug_response/binaryCDR.csv', 
                info_path=f'{file_dir}/../../data/cell_lines/CCLE_INFO.csv',
                drugsProcessed=False):
        self.drugs = self._load_drugs(drug_path, drugsProcessed)
        self.cdr = self._load_cdr(cdr_path)
        self.cancer_lookup = self._get_cancer_lookup(info_path)

    def _load_drugs(self, drug_path, drugsProcessed):
        if drugsProcessed:
            drug_fps = pd.read_csv(drug_path, index_col=0)
        else:
            drug_info = pd.read_csv(drug_path, index_col='name')

            def to_list(x):
                processed_str = x.replace('[', '').replace(']', '').replace('\n', '')
                array = np.asarray(processed_str.split(' ')).astype(int)
                return array.tolist()

            fps = drug_info.morgan_fp.apply(to_list).T
            drug_fps = pd.DataFrame(index=fps.index.values, columns=[*range(len(fps[0]))])
            drug_fps.loc[:, :] = fps.values.tolist()
        return drug_fps

    def _load_cdr(self, cdr_path):
        cdr = pd.read_csv(cdr_path)
        drug_names = self.drugs.index.values
        return cdr[cdr.name.isin(drug_names)]

    def _get_cancer_lookup(self, info_path):
        info = pd.read_csv(info_path, index_col='DepMap_ID')
        cell_lines = self.cdr.DepMap_ID.values.tolist()
        info = info[info.index.isin(cell_lines)]
        return lambda x: info.loc[x, 'primary_disease']

    def get_split(self, rna_path):
        # Load RNA data
        rna = pd.read_csv(rna_path, index_col='DepMap_ID')
        self.rna_dim = rna.shape[1]
        # pull the cell lines in the split
        cell_lines = rna.index.values.tolist() 
        # get the drug combinations for the defined cell lines
        combos = self.cdr[self.cdr.DepMap_ID.isin(cell_lines)]
        # get the labels and convert to TF tensor
        GT = tf.convert_to_tensor(combos.effective, dtype=tf32)
        # get the drugs and RNA and conver each to TF tensor
        Drugs = tf.convert_to_tensor(self.drugs.loc[combos.name.values, :], dtype=tf32)
        self.drug_dim = Drugs.shape[1]
        rna = tf.convert_to_tensor(rna, dtype=tf32)
        RNA = tf.gather_nd(rna, indices=[[cell_lines.index(x)] for x in combos.DepMap_ID])
        return RNA, Drugs, GT, combos

def get_median(v):
    v = tf.reshape(v, [-1])
    mid = v.get_shape()[0]//2 + 1
    return tf.nn.top_k(v, mid).values[-1]

def denseMSE(y_true, y_pred, weights):
    """Computes the mean squared error between labels and predictions.
    After computing the squared distance between the inputs, the mean value over
    the last dimension is returned.
    Args:
    y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.

    Returns:
    Mean squared error values. shape = `[batch_size, d0, .. dN-1]`.
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    #  return backend.mean(tf.math.multiply(weights, tf.math.squared_difference(y_pred, y_true)), axis=-1)
    return tf.math.multiply(weights, tf.reshape(tf.math.squared_difference(y_pred, y_true), (-1)))

class weightWrapper():
    def __init__(self, weightFn, threshold):
        self.weightFn = weightFn
        self.threshold = threshold
    
    def __call__(self, vals):
        pwVals = tf.where(vals < self.threshold, self.threshold, vals)
        return self.weightFn(pwVals)

class DenseLossWrapper(Loss):
    """Wraps a loss function in the `Loss` class."""

    def __init__(
        self, lossFn, weightFn, name=None, **kwargs
    ):
        """Initializes `LossFunctionWrapper` class.
        Args:
          fn: The loss function to wrap, with signature `fn(y_true, y_pred,
            **kwargs)`.
          reduction: Type of `tf.keras.losses.Reduction` to apply to
            loss. Default value is `AUTO`. `AUTO` indicates that the reduction
            option will be determined by the usage context. For almost all cases
            this defaults to `SUM_OVER_BATCH_SIZE`. When used with
            `tf.distribute.Strategy`, outside of built-in training loops such as
            `tf.keras` `compile` and `fit`, using `AUTO` or
            `SUM_OVER_BATCH_SIZE` will raise an error. Please see this custom
            training [tutorial](
            https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
          name: Optional name for the instance.
          **kwargs: The keyword arguments that are passed on to `fn`.
        """
        super().__init__(name=name)
        self.lossFn = lossFn
        self.weightFn = weightFn
        self._fn_kwargs = kwargs

    def call(self, y_true, y_pred):
        """Invokes the `LossFunctionWrapper` instance.
        Args:
          y_true: Ground truth values.
          y_pred: The predicted values.
        Returns:
          Loss values per sample.
        """
        #  if tf.is_tensor(y_pred) and tf.is_tensor(y_true):
            #  y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(
                #  y_pred, y_true
            #  )

        ag_fn = tf.__internal__.autograph.tf_convert(
            self.lossFn, tf.__internal__.autograph.control_status_ctx()
        )
        weights = self.weightFn(y_true)

        return ag_fn(y_true, y_pred, weights, **self._fn_kwargs)

class DenseLossMSE(DenseLossWrapper):
    def __init__(self,
                 weightFn,
                 flatThresh=None,
                 name='DenseMSE'):
        if flatThresh != None:
            weightFn = weightWrapper(weightFn, flatThresh)
        super().__init__(denseMSE, weightFn, name=name)

