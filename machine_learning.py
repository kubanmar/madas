from sklearn.base import BaseEstimator, RegressorMixin
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from functools import partial

import numpy as np

import multiprocessing, copy

def identity_function(x):
    return x

def distance_function(x):
    return 1-x

def averaged_distances(fingerprint_pairs):
    aver_dist = 0
    count = 0
    for pair in fingerprint_pairs:
        count += 1
        sim = pair[0].get_similarity(pair[1])
        aver_dist += 1 - sim
    aver_dist /= count
    return aver_dist

def lin_comb_distances(fingerprint_pairs, weights = None):
    distance = 0
    if weights == None:
        weights = np.ones(len(fingerprint_pairs[0][0]))
    for idx, pair in enumerate(fingerprint_pairs):
        distance += weights[idx] * (1 - pair[0].get_similarity(pair[1]))
    distance /= sum(weights)
    return distance

class SimiliarityKernelRegression(BaseEstimator, RegressorMixin):

    def __init__(self, alpha = 0, multiprocess = False, kernel_function = identity_function, use_y_kernel = True):
        self.alpha = alpha
        self.x_kernel = None
        self.y_kernel = None
        self.multiprocess = multiprocess
        self.kernel_function = kernel_function
        self.use_y_kernel = use_y_kernel
        self._intercept = 0
        self.cv_data = None

    def set_kernel(self, x_kernel, y_kernel):
        self.x_kernel = x_kernel
        self.y_kernel = y_kernel

    def set_get_alpha(self, alpha):
        if alpha != None:
            self.alpha = alpha
        else:
            alpha = self.alpha
        return alpha

    def fit(self, x_train, y_train, alpha = None, multiprocess = False):
        if self.x_kernel == None or self.y_kernel == None:
            self.set_kernel(x_train, y_train)
        if not len(x_train) == len(y_train):
            raise ValueError("Wrong shape of training data: len(x) != len(y)")

        alpha = self.set_get_alpha(alpha)
        applied_kernel_matrix = self._get_applied_kernel_matrix(x_train, multiprocess = multiprocess)

        regressor = linear_model.Ridge(alpha = alpha, fit_intercept = False, normalize = False)
        regressor.fit(applied_kernel_matrix, y_train)
        self.gammas = regressor.coef_

    def fitCV(self, x_train, y_train, alphas = None, multiprocess = False, cv_multiprocess = False):
        cv_data = []
        if self.x_kernel == None or self.y_kernel == None:
            self.set_kernel(x_train, y_train)
        if not len(x_train) == len(y_train):
            raise ValueError("Wrong shape of training data: len(x) != len(y)")
        best = None
        for idx, alpha in enumerate(alphas):
            rmse = self._cv_fit(x_train, y_train, alpha = alpha, multiprocess = cv_multiprocess)
            cv_data.append([alpha, rmse])
            print(alpha, rmse, best)
            if best == None:
                best = [rmse, idx]
                continue
            if rmse < best[0]:
                best = [rmse, idx]
        self.cv_data = cv_data
        self.alpha = alphas[best[1]]
        self.fit(x_train, y_train)


    def _cv_fit(self, x_train, y_train, alpha = 0, multiprocess = False, n_splittings = 4):
        if multiprocess:
            data = [(train_test_split(x_train, y_train, train_size = 0.9, test_size = 0.1), alpha) for idx in range(n_splittings)]
            with multiprocessing.Pool() as p:
                results = p.map(self._multi_cv_fit, data)
        else:
            results = []
            regressor = linear_model.Ridge(alpha = alpha, fit_intercept = False, normalize = False)
            for iteration in range(n_splittings):
                train_x, test_x, train_y, test_y = train_test_split(x_train, y_train, train_size = 0.9, test_size = 0.1)
                applied_kernel_matrix = self._get_applied_kernel_matrix(train_x, multiprocess = self.multiprocess)
                regressor.fit(applied_kernel_matrix, train_y)
                test_matrix = self._get_applied_kernel_matrix(test_x, multiprocess = self.multiprocess)
                score = mean_squared_error(test_y, regressor.predict(test_matrix))#regressor.score(test_matrix, test_y)
                results.append(score)
        averaged_rmse = sum(results) / len(results)
        return averaged_rmse

    def _multi_cv_fit(self, train_test_alpha):
        data, alpha = train_test_alpha
        train_x, test_x, train_y, test_y = data
        regressor = linear_model.Ridge(alpha = alpha, fit_intercept = False, normalize = False)
        applied_kernel_matrix = self._get_applied_kernel_matrix(train_x, multiprocess = self.multiprocess)
        regressor.fit(applied_kernel_matrix, train_y)
        test_matrix = self._get_applied_kernel_matrix(test_x, multiprocess = self.multiprocess)
        return mean_squared_error(test_y, regressor.predict(test_matrix))#regressor.score(test_matrix, test_y)

    def _get_applied_kernel_matrix(self, x_train, multiprocess = False):
        if multiprocess:
            try: # If Fingerprint() objects are not parallelizable, multiprocessing will fail with TypeError
                applied_kernel_matrix = self._get_applied_kernel_matrix_parallel(x_train)
            except TypeError:
                applied_kernel_matrix = self._get_applied_kernel_matrix_serial(x_train)
        else:
            applied_kernel_matrix = self._get_applied_kernel_matrix_serial(x_train)
        return applied_kernel_matrix

    def _get_applied_kernel_matrix_serial(self, x_train):
        applied_kernel_matrix = []
        for x in x_train:
            material = []
            for x_kernel, y_kernel in zip(self.x_kernel, self.y_kernel):
                if self.use_y_kernel:
                    material.append(self.kernel_function(x_kernel.get_similarity(x)) * y_kernel)
                else:
                    material.append(self.kernel_function(x_kernel.get_similarity(x)))
            applied_kernel_matrix.append(material)
        return applied_kernel_matrix

    def _get_applied_kernel_matrix_parallel(self, x_train):
        kernel_matrix = []
        for x_kernel in self.x_kernel:
            with multiprocessing.Pool() as p:
                material = p.map(x_kernel.get_similarity, x_train)
            kernel_matrix.append([self.kernel_function(mat) for mat in material])
        if self.use_y_kernel:
            applied_kernel_matrix = np.transpose(kernel_matrix) * self.y_kernel
        else:
            applied_kernel_matrix = np.transpose(kernel_matrix)
        return applied_kernel_matrix

    def predict(self, x):
        if not hasattr(self, 'gammas') and False:
            raise AssertionError("Need to fit the data first!")
        if not isinstance(x, (list, np.ndarray)):
            x = [x]
        predictions = []
        for xi in x:
            if self.use_y_kernel:
                similarity = np.array([self.kernel_function(fp.get_similarity(xi)) * gamma for gamma, fp in zip(self.gammas,self.x_kernel)])
                pred = np.dot(similarity, self.y_kernel)
            else:
                similarity = np.array([self.kernel_function(fp.get_similarity(xi)) for fp in self.x_kernel])
                pred = np.dot(similarity, self.gammas)
            predictions.append(pred)
        return predictions

    def get_random_kernel_train_test_data(self, fingerprints, target_properties, rel_kernel_size = 0.1, train_size = 0.75):
        x_data, x_kernel, y_data, y_kernel = train_test_split(fingerprints, target_properties, train_size = 1 - rel_kernel_size, test_size = rel_kernel_size)
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size = train_size, test_size = 1 - train_size)
        return x_kernel, x_train, x_test, y_kernel, y_train, y_test

class MultiKernelRegression(BaseEstimator, RegressorMixin):
    """
    Learn a target property from different fingerprints, i.e. as linear combination.

    """

    def __init__(self, regressor = linear_model.Ridge, regressor_parameters = {'alpha': 0, 'fit_intercept':False, 'normalize':False}, mixing_kernel = averaged_distances, kernel_parameters = {}, use_y_kernel = False):
        self.x_kernel = []
        self.y_kernel = []
        self.gammas = []
        self._intercept = 0
        self.use_y_kernel = use_y_kernel
        try:
            self.regressor = regressor(**regressor_parameters)
        except TypeError:
            raise AssertionError('Wrong keyword for regressor in regressor_parameters.')
        self.mixing_kernel = partial(mixing_kernel, **kernel_parameters) #Fails with TypeError upon call if mixing_kernel does not have the kwargs mentioned in kernel_parameters

    def set_kernel(self, x_kernel, y_kernel):
        self.x_kernel = x_kernel
        self.y_kernel = y_kernel

    def fit(self, fingerprints, target, multiprocess = True):
        """
        Fit MultiKernelRegression to data.
        Args:
            * fingerprints: list; list of fingerprints to describe each material
            * target: list; target values to fit
        Kwargs:
            * multiprocess: bool; default: True; try fitting in parallel
        """
        self.set_kernel(fingerprints, target) #for now, the training set is always the kernel set
        applied_kernel_matrix = self._get_applied_kernel_matrix(fingerprints, multiprocess = multiprocess)
        self.regressor.fit(applied_kernel_matrix, target)
        self.gammas = self.regressor.coef_
        self._intercept = self.regressor.intercept_

    def grid_CV_fit(self, fingerprints, target, regressor_parameters = {}, mixing_parameters = {}):
        """
        Calculate prediction errors for a set of regressor and mixing parameters.
        """
        raise NotImplementedError('Function not implemented (yet).')

    def _get_applied_kernel_matrix(self, x_train, multiprocess = True):
        if multiprocess:
            try: # If Fingerprint() objects are not parallelizable, multiprocessing will fail with TypeError
                applied_kernel_matrix = self._get_applied_kernel_matrix_parallel(x_train)
            except TypeError as err:
                print("Kernel matrix can not be calculated in parallel because of:", str(err),'\nContinuing with serial calculation.\n')
                applied_kernel_matrix = self._get_applied_kernel_matrix_serial(x_train)
        else:
            applied_kernel_matrix = self._get_applied_kernel_matrix_serial(x_train)
        return applied_kernel_matrix

    def _get_applied_kernel_matrix_serial(self, x_train):
        applied_kernel_matrix = []
        for x in x_train:
            material = []
            for x_kernel, y_kernel in zip(self.x_kernel, self.y_kernel):
                if self.use_y_kernel:
                    material.append(self.mixing_kernel(zip(x,x_kernel)) * y_kernel)
                else:
                    material.append(self.mixing_kernel(zip(x,x_kernel)))
            applied_kernel_matrix.append(material)
        return applied_kernel_matrix

    def _get_applied_kernel_matrix_parallel(self, x_train):
        with multiprocessing.Pool() as p:
            applied_kernel_matrix = p.map(self._multiprocess_similarity_list, [(x,self.x_kernel) for x in x_train])
        return applied_kernel_matrix


    def _multiprocess_similarity_list(self, reference__list):
        reference, fp_list = reference__list
        similarities = []
        for fp in fp_list:
            similarities.append(self.mixing_kernel(zip(reference, fp)))
        return similarities

    def predict(self, x):
        if not isinstance(x, (list, np.ndarray)):
            x = [x]
        predictions = []
        for xi in x:
            if self.use_y_kernel:
                similarity = np.array([self.mixing_kernel(zip(xi, kernel_x)) * gamma for gamma, kernel_x in zip(self.gammas,self.x_kernel)])
                pred = self._intercept + np.dot(similarity, self.y_kernel)
            else:
                similarity = np.array([self.mixing_kernel(zip(xi, kernel_x)) for kernel_x in self.x_kernel])
                pred = self._intercept + np.dot(similarity, self.gammas)
            predictions.append(pred)
        return predictions
