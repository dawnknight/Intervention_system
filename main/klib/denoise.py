# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.externals import joblib
import h5py

class Denoise(object):
    """ Use Gaussian process regression to denoise
    """
    def __init__(self):
        self._done = True
        self.gp = joblib.load('data/GPR_cluster_800_meter_fix_ex4.pkl')
        self.parameter = self.gp.kernel_.get_params(deep=True)
        self.limbidx = np.array([4, 5, 6, 8, 9, 10, 20])
        [self.min, self.max] = h5py.File('data/model_CNN_0521_K2M_rel.h5', 'r')['minmax'][:]

    def cov(self, sita0, sita1, W1, W2, noise_level, x1, x2):
        """ calculate the covariance btw input data and the
            training data
        """
        dists1 = cdist(x1/W1, x2/W1, metric='sqeuclidean')
        dists2 = cdist(x1/W2, x2/W2, metric='sqeuclidean')
        k1 = np.exp(-.5*dists1)
        k2 = np.exp(-.5*dists2)
        k_return = sita0*k1+sita1*k2
        if np.array_equal(x1, x2):
            k_return = k_return+noise_level
        return k_return

    def gp_pred(self, testdata):
        """ predict the output according to the input data
            and the pre-train gpr model
        """
        # === extract parameter from pre-train model===
        sita0 = self.parameter["k1__k1__k1__constant_value"]
        w1    = self.parameter["k1__k1__k2__length_scale"]
        sita1 = self.parameter["k1__k2__k1__constant_value"]
        w2    = self.parameter["k1__k2__k2__length_scale"]
        noise_level = self.parameter["k2__noise_level"]
        traindata = self.gp.X_train_
        alpha_ = self.gp.alpha_
        y_train_mean = self.gp.y_train_mean
        # === Prediction ===
        k_trans = self.cov(sita0, sita1, w1, w2, noise_level, testdata, traindata)
        y_mean  = k_trans.dot(alpha_)
        return y_train_mean + y_mean

    def run(self, modjary, relary, threshold=0.6, onlyunrel=True):
        """ according the the relablity array reconstruct the 3D jpints
            position, only substitute the unreliable joints, if onlyunrel
            is True
        """
        mask = np.zeros([7, 3])
        modjary_norm = (modjary-self.min)/(self.max-self.min)
        reconj = (self.gp_pred(modjary_norm)*(self.max-self.min)+self.min)  # reconJ is 1*21 array
        if onlyunrel:
            unrelidx = np.where(relary[self.limbidx] < threshold)[0]
            mask[unrelidx, :] = np.array([1, 1, 1])
            # use unrelidx and reconJ to replace unreliable joints in modJary
            modjary[:, mask.flatten() == 1] = reconj[:, mask.flatten() == 1]
            return modjary, unrelidx
        else:
            return reconj, unrelidx
