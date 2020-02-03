from machine_learning import MatrixMultiKernelLearning, linear_comb_dist_mat, linear_comb_sim_mat
from utils import rmsle
import copy
from sklearn import linear_model
import numpy as np
from similarity import SimilarityMatrix
from data_framework import MaterialsDatabase
from apis.local_data_API import API
from functools import partial
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import tensorflow as tf

from sklearn.decomposition import PCA
from scipy.optimize import minimize, NonlinearConstraint
from functools import partial
import random

class ProximityAveragePrediction():

    def __init__(self,
                 kernel_matrices,
                 prediction_matrices,
                 target_train,
                 target_test,
                 k = 3,
                 NN_model = None,
                 use_PCA = True,
                 n_PCA_components = 500):
        self.kernel_matrices = kernel_matrices
        self.prediction_matrices = prediction_matrices
        self.target_train = target_train
        self.target_test = target_test
        self.pca_generator = PCA(n_PCA_components)
        self.best_weigths = None
        self.k = k
        if use_PCA:
            self.training_data = self.pca_generator.fit_transform(self._get_training_classifier_input())
            self.test_data = self.pca_generator.transform(self._get_test_classifier_input())
        else:
            self.training_data = self._get_training_classifier_input()
            self.test_data = self._get_test_classifier_input()
        self.model = NN_model if NN_model != None else self._make_default_NN_model(len(self.training_data[0]))


    def approximate_best_weights(self, show_progress = True):
        weight_list = [] #make a grid of weights
        for a in range(11):
            ai = a/10
            for b in range(11-a):
                bi = b/10
                for c in range(11-a-b):
                    ci = c/10
                    di = abs(round(1-ai-bi-ci,10))
                    weight_list.append([ai,bi,ci,di])
        all_best = []
        for idx, target in enumerate(self.target_train):
            mids = self.kernel_matrices[0].mids
            best_weigths = None
            for weights, simat in zip(weight_list, self.trial_simats):
                neighbours_dict = list(simat.get_k_nearest_neighbors(mids[idx], k = self.k).values())[0]
                neighbour_mids = list(neighbours_dict.keys())
                neighbour_sims = list(neighbours_dict.values())
                neighbour_targets = [self.target_train[mids.index(mid)] for mid in neighbour_mids]
                pred = self._mean_similarity_prediction(neighbour_sims, neighbour_targets)
                error = rmsle(pred, self.target_train[idx])
                if best_weigths == None:
                    best_weigths = [[weights, weights], error, pred]
                if best_weigths[1][0] > error[0]:
                    best_weigths[0][0] = weights
                    best_weigths[1][0] = error[0]
                    best_weigths[2][0] = pred[0]
                if best_weigths[1][1] > error[1]:
                    best_weigths[0][1] = weights
                    best_weigths[1][1] = error[1]
                    best_weigths[2][1] = pred[1]
            all_best.append(best_weigths)
            if show_progress:
                print(idx/len(self.target_train)*100, end = '\r')
        return all_best

    def set_training_weights(self, best_weights):
        self.best_weights = best_weights

    def predict_training(self, use_weights = None):
        predicted_weights= self.model.predict(self.training_data) if use_weights == None else use_weights
        predictions_train = []
        for idx, weights in enumerate(predicted_weights):
            rows = [np.array(kernel[idx]) for kernel in self.kernel_matrices]
            pred = self._predict_nn(self._mean_similarity_prediction, self._weighted_kernel_row(rows, weights), target_train, exclude_index = idx)
            predictions_train.append(pred)
        return predictions_train

    def evaluate_training(self, return_values = False, weights = None):
        preds = self.predict_training(use_weights = weights)
        rmsle_results = rmsle(self.target_train, preds)
        mae_results = [0,0]
        mae_results[0] = mean_absolute_error([x[0] for x in preds], [x[0] for x in self.target_train])
        mae_results[1] = mean_absolute_error([x[1] for x in preds], [x[1] for x in self.target_train])
        print('\t', 'egap\t', 'eform')
        print('rmsle', round(rmsle_results[0],5), round(rmsle_results[1],5))
        print('mae', round(mae_results[0],5), round(mae_results[1],5))
        if return_values:
            return rmsle_results, mae_results

    def predict_test(self, use_weights = None):
        predicted_test_weights= self.model.predict(self.test_data) if use_weights == None else use_weights
        predictions_test = []
        for idx, weights in enumerate(predicted_test_weights):
            rows = [kernel[idx] for kernel in self.prediction_matrices]
            pred = self._predict_nn(self._mean_similarity_prediction, self._weighted_kernel_row(rows, weights), target_train)
            predictions_test.append(pred)
        return predictions_test

    def evaluate_test(self, return_values = False, weights = None):
        preds = self.predict_test(use_weights = weights)
        rmsle_results = rmsle(self.target_test, preds)
        mae_results = [0,0]
        mae_results[0] = mean_absolute_error([x[0] for x in preds], [x[0] for x in self.target_test])
        mae_results[1] = mean_absolute_error([x[1] for x in preds], [x[1] for x in self.target_test])
        print('\t', 'egap\t', 'eform')
        print('rmsle', round(rmsle_results[0],5), round(rmsle_results[1],5))
        print('mae', round(mae_results[0],5), round(mae_results[1],5))
        if return_values:
            return rmsle_results, mae_results

    def train_NN(self, epochs = 1000, batch_size = 32):
        train_data = self.training_data.tolist() if isinstance(self.training_data, np.ndarray) else self.training_data
        history = self.model.fit(train_data, self.best_weights, epochs = epochs, batch_size = batch_size)
        return history

    def train_evaluate_NN(self, max_epochs = 5000, n_test_evaluations = 5, batch_size = 32):
        training_histories = []
        training_errors = []
        test_errors = []
        epochs_step = int(max_epochs / n_test_evaluations)
        for training in range(n_test_evaluations):
            training_histories.append(self.train_NN(epochs = epochs_step, batch_size = batch_size))
            training_errors.append(self.evaluate_training(return_values=True))
            test_errors.append(self.evaluate_test(return_values=True))
            print((training+1)/n_test_evaluations)
        return training_histories, training_errors, test_errors

    def _make_default_NN_model(self, n_inputs):
        weight_model = tf.keras.Sequential([
            tf.keras.layers.Dense(200,input_shape = (n_inputs,), activation = 'relu'),
            tf.keras.layers.Dense(400,activation='tanh'),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Dense(200,activation='sigmoid'),
            tf.keras.layers.Dense(4, activation='softmax')
        ])
        weight_opt = tf.keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-9)
        weight_model.compile(optimizer = weight_opt, loss = tf.keras.losses.MSE, metrics = ['accuracy'])
        return weight_model

    def _get_training_classifier_input(self):
        classifier_matrix = []
        for midx, matrix in enumerate(self.kernel_matrices):
            if midx == 0:
                for row in matrix.get_square_matrix():
                    classifier_matrix.append(row.tolist())
            else:
                for index, row in enumerate(matrix.get_square_matrix()):
                    for item in row:
                        classifier_matrix[index].append(item)
        return classifier_matrix

    def _get_test_classifier_input(self):
        classifier_prediction_matrix = []
        for midx, matrix in enumerate(self.prediction_matrices):
            if midx == 0:
                for row in matrix.matrix:
                    classifier_prediction_matrix.append(row.tolist())
            else:
                for index, row in enumerate(matrix.matrix):
                    for item in row:
                        classifier_prediction_matrix[index].append(item)
        return classifier_prediction_matrix

    def optimize_all_weights(self, show_progress = True, error_function = rmsle, prediction_function = None, use_error_index = 0):
        prediction_function = self._mean_similarity_prediction if prediction_function == None else prediction_function
        weights = []
        for idx in range(len(kernel_matrices[0])):
            rows = [np.array(kernel[idx]) for kernel in self.kernel_matrices]
            fun_to_min = partial(self._prediction_error_weights,
                                 rows = rows, target = self.target_train[idx],
                                 prediction_function = prediction_function,
                                 use_error_index = use_error_index,
                                 exclude_index = idx)
            init_simplex = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1], [0.25,0.25,0.25,0.25]] #random.choices(weight_list, k=5)#
            minimization_result = minimize(fun_to_min, [0.25,0.25,0.25,0.25], method='nelder-mead', options={'initial_simplex':init_simplex})#, constraints=constraint)#, options={'gtol':1e-15})
            weights.append(minimization_result.x)
            if show_progress:
                print(round(idx/2400*100,2), end = '\r')
        return weights

    def _prediction_error_weights(self, weights, use_error_index = 0, rows = None, error_function = rmsle, target = 0, prediction_function = None, exclude_index = 0,**kwargs):
        row = self._weighted_kernel_row(rows, weights)
        prediction = self._predict_nn(prediction_function, row, self.target_train, exclude_index = exclude_index, **kwargs)
        return error_function(prediction, target)[use_error_index] + abs(1 - sum([abs(weight) for weight in weights])) + 10 * abs(sum([abs(weight) - weight for weight in weights]))

    def _predict_nn(self, pred_function, sim_row, target, exclude_index = None, **kwargs):
        most_similar = list(zip(target, sim_row))
        if exclude_index != None:
            most_similar = [value for idx, value in enumerate(most_similar) if idx != exclude_index]
        most_similar.sort(key = lambda x: x[1], reverse = True)
        neighbour_sims = [value[1] for value in most_similar[:self.k]]
        neighbour_targets = [value[0] for value in most_similar[:self.k]]
        pred = pred_function(neighbour_sims, neighbour_targets, **kwargs)
        return pred

    @staticmethod
    def _weighted_kernel_row(rows, weights):
        return sum([weight * row for weight, row in zip(weights, rows)])

    @staticmethod
    def _mean_similarity_prediction(similarities, targets):
        if isinstance(targets[0], (int, float)):
            mean = sum([sim*tar for sim, tar in zip(similarities, targets)]) / sum(similarities)
        else:
            mean = []
            for idx in range(len(targets[0])):
                mean.append(sum([sim*tar[idx] for sim, tar in zip(similarities, targets)]) / sum(similarities))
        return mean

class TargetValueModel(tf.keras.Model):

    def __init__(self, target, direct = False, deep = False):
        super(OneLayerModel, self).__init__()
        self.direct = direct
        self.deep = deep
        self.target = tf.convert_to_tensor(target)
        self.d1 = tf.keras.layers.Dense(2400,input_shape = (2400,), use_bias = False, activation = 'linear', name = 'd11')
        self.deep1 = tf.keras.layers.Dense(1000,input_shape = (2400,), use_bias = False, activation = 'sigmoid', name = 'deep11')
        self.deep2 = tf.keras.layers.Dense(100,input_shape = (1000,), use_bias = False, activation = 'sigmoid', name = 'deep21')
        self.deep3 = tf.keras.layers.Dense(1000,input_shape = (100,), use_bias = False, activation = 'sigmoid', name = 'deep31')
        self.out = tf.keras.layers.Dense(2400, activation='softmax', use_bias = False, input_shape = (1000,), name = 'out1', dtype = np.float32)


    def call(self, inputs, training = False):
        if not self.direct:
            x = self.d1(inputs)
        else:
            x = inputs
        if self.deep:
            x = self.deep1(x)
            x = self.deep2(x)
            x = self.deep3(x)
        out = self.out(x)
        return tf.tensordot(out, self.target,[1,1])


class TwoTargetValueModel(tf.keras.Model):

    def __init__(self, target, direct = False, deep = False):
        super(OneLayerModel, self).__init__()
        self.direct = direct
        self.deep = deep
        self.target = [tf.convert_to_tensor([values]) for values in target]
        self.d11 = tf.keras.layers.Dense(2400,input_shape = (2400,), use_bias = False, activation = 'linear', name = 'd11')
        self.d12 = tf.keras.layers.Dense(2400,input_shape = (2400,), use_bias = False, activation = 'linear', name = 'd12')
        self.deep11 = tf.keras.layers.Dense(1000,input_shape = (2400,), use_bias = False, activation = 'sigmoid', name = 'deep11')
        self.deep12 = tf.keras.layers.Dense(1000,input_shape = (2400,), use_bias = False, activation = 'sigmoid', name = 'deep12')
        self.deep21 = tf.keras.layers.Dense(100,input_shape = (1000,), use_bias = False, activation = 'sigmoid', name = 'deep21')
        self.deep22 = tf.keras.layers.Dense(100,input_shape = (1000,), use_bias = False, activation = 'sigmoid', name = 'deep22')
        self.deep31 = tf.keras.layers.Dense(1000,input_shape = (100,), use_bias = False, activation = 'sigmoid', name = 'deep31')
        self.deep32 = tf.keras.layers.Dense(1000,input_shape = (100,), use_bias = False, activation = 'sigmoid', name = 'deep32')
        self.out1 = tf.keras.layers.Dense(2400, activation='softmax', use_bias = False, input_shape = (1000,), name = 'out1', dtype = np.float32)
        self.out2 = tf.keras.layers.Dense(2400, activation='softmax', use_bias = False, input_shape = (1000,), name = 'out2')


    def call(self, inputs, training = False):
        if not self.direct:
            x = self.d11(inputs)
            y = self.d12(inputs)
        else:
            x = inputs
            y = inputs
        if self.deep:
            x = self.deep11(x)
            y = self.deep12(y)
            x = self.deep21(x)
            y = self.deep22(y)
            x = self.deep31(x)
            y = self.deep32(y)
        out1 = self.out1(x)
        out2 = self.out2(y)
        return tf.concat([tf.tensordot(out1, self.target[0],[1,1]),tf.tensordot(out2, self.target[1],[1,1])],1)
