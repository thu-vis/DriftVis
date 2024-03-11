import numpy as np
import os
# import pulp
from sklearn import mixture
from scipy.spatial import distance_matrix
import time

from .util import normalize, MD, KL, SysKL, move_avg, move_avg_prev, gaussian_true_log_prob, gaussian_log_prob

model = mixture.BayesianGaussianMixture(n_components=5, weight_concentration_prior=1e-2, mean_precision_prior=1e-2,
                                        max_iter=500)


class Detector(object):
    def __init__(self, attr_name, data, idx):
        self.cache = {}
        self.attr_name = attr_name
        self.data = self.get_data(data, idx)
        self.n = self.data.shape[0]
        self.dim = len(attr_name)

    def get_data(self, data, idx):
        return np.array(data[self.attr_name].iloc[idx])

    def add_data(self, data, idx):
        self.data = np.concatenate((self.data, self.get_data(data, idx)))
        self.n = self.data.shape[0]

    def set_data(self, data, idx):
        self.data = self.get_data(data, idx)
        self.n = self.data.shape[0]

    def set_cache(self, key, value):
        self.cache[key] = value

    def detect(self, models):
        raise NotImplementedError("Please define detect method")

class ED_Detector(Detector):
    def __init__(self, attr_name, data, idx, win_size):
        super().__init__(attr_name, data, idx)
        self.prev_n = 0 # in fact it records how many points have calculated the distance
        self.win_size = win_size
        self.method_name = "ED"
        self.name = f'{self.attr_name}_{self.method_name}'
        self.search_nn_cnt = 5
        self.cache_size = len(data)
        self.cache['result'] = None
        self.cache['result_n'] = 0
        self.cache['result_real'] = None
        self.cache['result_real_n'] = 0
        self.cache['dist'] = np.zeros((self.cache_size, self.cache_size), dtype=np.float)
        self.cache['dist_n'] = 0
        self.load_cache(data)
        # self.cache['As'] = np.zeros(self.cache_size)
        # self.cache['BCs'] = np.zeros(self.cache_size)
        # self.cache['A_n'] = 0
        # self.cache['B_n'] = 0
        # self.cache['knn'] = dict()
        # self.get_distance_thres()

    def add_data(self, data, idx):
        super().add_data(data, idx)
        self.update_cache_value()

    def get_distance_thres(self):
        self.update_A()
        self.update_BC()
        tmp = self.detect_real()[2*self.win_size-1:]
        mu = np.mean(tmp)
        sigma = np.std(tmp)
        print(f'{self.name}: mu:{mu}, sigma:{sigma}, thres:{mu+3*sigma}')


    def update_cache_value(self):
        self.update_dist()
        #self.update_A()
        #self.update_BC()

    def hardcode_cache(self, data):
        X = np.array(data[self.attr_name])
        self.cache['dist_n'] = len(data)
        if len(X.shape) == 1:
            X = X.reshape((-1,1))
        self.cache['dist'] = distance_matrix(X, X)
        self.save_cache()
        
    def save_cache(self):
        np.save('cache/' + self.name, self.cache['dist'])

    def load_cache(self, data):
        self.cache['dist_n'] = self.cache_size
        t=time.time()
        filename = f'cache/{self.name}.npy'
        if os.path.exists(filename):
            self.cache['dist'] = np.load(filename)
            print('cache: ' + self.name, time.time() - t)
        else:
            self.hardcode_cache(data)
            print('cal: ' + self.name, time.time() - t)

            

    def update_dist(self):
        # return
        if self.cache['dist_n'] >= self.n:
            return
        print('update dist')
        if self.dim == 1:
            for i in range(self.cache['dist_n'], self.n):
                self.cache['dist'][i, 0:self.n] = np.abs(self.data[i] - self.data).reshape(-1)
        else:
            for i in range(self.cache['dist_n'], self.n):
                self.cache['dist'][i, 0:self.n] = np.linalg.norm(self.data[i] - self.data, axis=1)
        self.cache['dist'][0:self.cache['dist_n'], self.cache['dist_n']:self.n] = self.cache['dist'][self.cache['dist_n']:self.n, 0:self.cache['dist_n']].T
        # self.update_nn_threshold()
        self.cache['dist_n'] = self.n

    def update_nn_threshold(self):
        for i in range(0, self.prev_n):
            new_arg_partition = np.argpartition(self.cache['dist'][i, self.prev_n:self.n], self.cache_knn_k)[:self.cache_knn_k]
            all_arg_partition = self.cache['knn'][i] + [idx + self.prev_n for idx in new_arg_partition]
            all_dist = np.array([self.cache['dist'][i, idx] for idx in all_arg_partition])
            self.cache['nn_dist'][i] = np.partition(all_dist, self.thres_k)[:self.thres_k]
            self.cache['knn'][i] = [all_arg_partition[_idx] for _idx in np.argpartition(all_dist, self.cache_knn_k)]
        for i in range(self.prev_n, self.n):
            self.cache['nn_dist'][i] = np.partition(self.cache['dist'][i,:self.n], self.thres_k)[:self.thres_k]
            self.cache['knn'][i] = np.argpartition(self.cache['dist'][i,:self.n], self.cache_knn_k)[:self.cache_knn_k].tolist()
        self.search_nn_thres = self.cache['nn_dist'][:self.n].mean()

    def update_BC(self, prev_n=-1):
        return
        if prev_n == -1:
            prev_n = self.cache['B_n']
        if self.n < self.win_size:
            print(f'win_size {self.win_size} is too large')
            return
        if prev_n < self.win_size:
            self.cache['BCs'] = np.zeros(self.cache_size)
            self.cache['BCs'][self.win_size-1] = self.BC_raw(np.arange(0, self.win_size)) / (self.win_size * self.win_size)
            self.update_BC(self.win_size)
            return
        self.cache['BCs'] = self.cache['BCs'] * (self.win_size * self.win_size)
        for i in range(prev_n, self.n):
            if i % 1000 == 0:
                print(f'{self.name} update_BC: Finshed {i} points')
            old_i = i - self.win_size
            BC_decrease = np.sum(self.cache['dist'][old_i, np.arange(old_i, i)]) * 2
            BC_increase = np.sum(self.cache['dist'][i, np.arange(old_i + 1, i + 1)]) * 2
            self.cache['BCs'][i] += self.cache['BCs'][i-1] + (BC_increase - BC_decrease)
        self.cache['BCs'] = self.cache['BCs'] / (self.win_size * self.win_size)
        self.cache['B_n'] = self.n

    def update_A(self, prev_n=-1):
        return
        if prev_n == -1:
            prev_n = self.cache['A_n']
        if self.n < 2 * self.win_size:
            print(f'win_size {self.win_size} is too large')
            return
        if prev_n < self.win_size:
            self.cache['As'] = np.zeros(self.cache_size)
            self.cache['As'][self.win_size-1] = self.A_raw(np.arange(0, self.win_size), np.arange(self.win_size, self.win_size * 2)) / (self.win_size * self.win_size)
            self.update_A(self.win_size)
            return
        self.cache['As'] = self.cache['As'] * (self.win_size * self.win_size)
        for i in range(prev_n, self.n - self.win_size):
            if i % 1000 == 0:
                print(f'{self.name} update_A: Finshed {i} points')
            old_i = i - self.win_size
            A_decrease1 = np.sum(self.cache['dist'][old_i, np.arange(i, i + self.win_size)])
            A_decrease2 = np.sum(self.cache['dist'][i, np.arange(old_i + 1, i)])
            A_increase1 = np.sum(self.cache['dist'][i, np.arange(i + 1, i + 1 + self.win_size)])
            A_increase2 = np.sum(self.cache['dist'][i + self.win_size, np.arange(old_i + 1, i)])
            self.cache['As'][i] += self.cache['As'][i-1] + (A_increase1 + A_increase2 - A_decrease1 - A_decrease2)
        self.cache['As'] = self.cache['As'] / (self.win_size * self.win_size)
        self.cache['A_n'] = self.n - self.win_size


    def A_raw(self, idx1, idx2):
        return np.sum(self.cache['dist'][idx1][:, idx2])

    def BC_raw(self, idx):
        return np.sum(self.cache['dist'][idx][:, idx])

    def distance_raw(self, idx1, idx2):
        A = self.A_raw(idx1, idx2) / len(idx1) / len(idx2)
        B = self.BC_raw(idx1) / len(idx1) / len(idx1)
        C = self.BC_raw(idx2) / len(idx2) / len(idx2)
        return 1 - (B + C) / (2 * A)

    def get_nn(self, idx, idxes):
        ret = []
        nn_index = np.argpartition(self.cache['dist'][idx, idxes], self.search_nn_cnt)[:self.search_nn_cnt]
        for _i in nn_index:
            ret.append(idxes[_i])
        return ret

    def get_nn_2(self, idx, idxes, k):
        if k <= self.search_nn_cnt:
            return idxes
        ret = []
        nn_index = np.argpartition(self.cache['dist'][idx, idxes], self.search_nn_cnt)[:self.search_nn_cnt]
        for _i in nn_index:
            ret.append(idxes[_i])
        return ret

    def get_nn_thres(self, idx, idxes):
        return self.get_nn(idx, idxes)
        ret = []
        nn_index = np.argpartition(self.cache['dist'][idx, idxes], self.search_nn_cnt)[:self.search_nn_cnt]
        for _i in nn_index:
            if self.cache['dist'][idx, idxes[_i]] < self.search_nn_thres:
                ret.append(idxes[_i])
        return ret

    def get_nn_thres_cache(self, idx, idxes):
        ret = []
        nn_index = self.cache['knn'][idx]
        for i in nn_index:
            if i in idxes and self.cache['dist'][idx, i] < self.search_nn_thres:
                ret.append(i)
        return ret

    def detect(self, models, labels, prev_n=-1,gmm_labels=[]):
        #return np.zeros(self.n).tolist()
        self.update_cache_value()
        if prev_n == -1:
            prev_n = self.cache['result_n']
        if prev_n == self.n:
            return self.cache['result'].tolist()
        return self._detect_gmm_nn(models, gmm_labels, prev_n)

    def _detect_gmm_nn(self, models, gmm_labels, prev_n):
        # pooling all training data together and use nn and gmm label to match
        training_idx = list(set([idx for model in models for idx in model['idx']]))
        unique_label = list(np.unique(gmm_labels))
        label2i = {label:i for i,label in enumerate(unique_label)}
        gmm_labels = [label2i[label] for label in gmm_labels]
        n_label = len(unique_label)
        _training_idx_set = [[] for label in unique_label]
        for i in training_idx:
            _training_idx_set[gmm_labels[i]].append(i)
        _k = [len(x) for x in _training_idx_set]

        nn_dict = {}
        nn_count = {}
        result = np.zeros(self.n)
        if prev_n > 0:
            result[:prev_n] = self.cache['result']
        lens = np.zeros(n_label, dtype='int')
        lens_model = np.zeros(n_label, dtype='int')
        drift_vals = np.zeros(n_label)
        As = np.zeros(n_label)
        Bs = np.zeros(n_label)
        Cs = np.zeros(n_label)
        _idx = [[] for label in unique_label]
        _training_idx = [[] for label in unique_label]

        if prev_n < self.win_size:
            prev_n = self.win_size
        for i in range(prev_n - self.win_size, prev_n):
            label = gmm_labels[i]
            _idx[label].append(i)
            nn_dict[i] = self.get_nn_2(i, _training_idx_set[label], _k[label])
            _training_idx[label] += set_plus(nn_count, nn_dict[i])

        lens = [len(x) for x in _idx]
        lens_model = [len(x) for x in _training_idx]
        for k in range(n_label):
            if lens[k] == 0 or lens_model[k] == 0:
                drift_vals[k] = lens[k]
            else:
                As[k] = self.A_raw(_idx[k], _training_idx[k])
                Bs[k] = self.BC_raw(_idx[k])
                Cs[k] = self.BC_raw(_training_idx[k])
                if As[k] < 1e-4:
                    drift_vals[k] = 0
                else:
                    drift_vals[k] = (1 - (Bs[k] / lens[k] / lens[k] + Cs[k] / lens_model[k] / lens_model[k]) / (2 * As[k] / lens[k] / lens_model[k])) * lens[k]

        if prev_n == self.win_size:
            result[prev_n - 1] = drift_vals.sum() / self.win_size

        for i in range(prev_n, self.n):
            if i % 1000 == 0:
                print(f'{self.name} detect: Finshed {i} points')
            label = gmm_labels[i]
            old_i = i - self.win_size
            old_label = gmm_labels[old_i]
            _idx[old_label].pop(0)
            lens[old_label] -= 1
            old_model_index = set_minus(nn_count, nn_dict[old_i])
            for tmp in old_model_index:
                del _training_idx[old_label][_training_idx[old_label].index(tmp)]
            lens_model[old_label] -= len(old_model_index)

            if lens_model[old_label] == 0:
                As[old_label] = 0
                Bs[old_label] = 0
                Cs[old_label] = 0
                drift_vals[old_label] = lens[old_label]
            else:
                A_decrease = np.sum(self.cache['dist'][old_i, _training_idx[old_label]]) + \
                         np.sum(self.cache['dist'][old_i, old_model_index])
                for old_idx in old_model_index:
                    A_decrease += np.sum(self.cache['dist'][_idx[old_label], old_idx])
                B_decrease = np.sum(self.cache['dist'][old_i, _idx[old_label]]) * 2
                C_decrease = 0
                for old_idx in old_model_index:
                    C_decrease += np.sum(self.cache['dist'][old_idx, _training_idx[old_label]])
                C_decrease *= 2
                for old_idx_1 in old_model_index:
                    for old_idx_2 in old_model_index:
                        C_decrease += self.cache['dist'][old_idx_1, old_idx_2]
                As[old_label] -= A_decrease
                Bs[old_label] -= B_decrease
                Cs[old_label] -= C_decrease
                if As[old_label] < 1e-4:
                    drift_vals[old_label] = 0
                else:
                    drift_vals[old_label] = (1 - (Bs[old_label] / lens[old_label] / lens[old_label] + Cs[old_label] / lens_model[old_label] / lens_model[old_label]) /
                                                (2 * As[old_label] / lens[old_label] / lens_model[old_label])) * lens[old_label]

            nn_dict[i] = self.get_nn_2(i, _training_idx_set[label], _k[label])
            new_model_index = set_plus(nn_count, nn_dict[i])
            lens[label] += 1
            lens_model[label] += len(new_model_index)
            if lens_model[label] == 0:
                drift_vals[label] = lens[label]
                As[label] = 0
                Bs[label] = 0
                Cs[label] = 0
            else:
                A_increase = np.sum(self.cache['dist'][i, _training_idx[label]]) + \
                         np.sum(self.cache['dist'][i, new_model_index])
                for new_idx in new_model_index:
                    A_increase += np.sum(self.cache['dist'][_idx[label], new_idx])
                B_increase = np.sum(self.cache['dist'][i, _idx[label]]) * 2
                C_increase = 0
                for new_idx in new_model_index:
                    C_increase += np.sum(self.cache['dist'][new_idx, _training_idx[label]])
                C_increase *= 2
                for new_idx_1 in new_model_index:
                    for new_idx_2 in new_model_index:
                        C_increase += self.cache['dist'][new_idx_1, new_idx_2]
                As[label] += A_increase
                Bs[label] += B_increase
                Cs[label] += C_increase
                if As[label] < 1e-4:
                    drift_vals[label] = 0
                else:
                    drift_vals[label] = (1 - (Bs[label] / lens[label] / lens[label] + Cs[label] / lens_model[label] / lens_model[label]) / (
                                2 * As[label] / lens[label] / lens_model[label])) * lens[label]
            _idx[label].append(i)
            _training_idx[label] += new_model_index
            result[i] = drift_vals.sum() / self.win_size
        self.cache['result'] = result
        return result.tolist()


    def detect_real(self):
        prev_n = self.cache['result_real_n']
        #min_thres = self.win_size
        if self.n < self.win_size * 2:
            return np.zeros(self.n).tolist()
        print(f'{self.name}: n:{self.n}')
        result = np.zeros(self.n)
        if prev_n > 0:
            result[:prev_n] = self.cache['result_real']
        if prev_n < 2 * self.win_size:
            prev_n = 2 * self.win_size
        result[prev_n:self.n] = 1 - (self.cache['BCs'][prev_n-self.win_size:self.n-self.win_size] +
                                                self.cache['BCs'][prev_n:self.n]) / \
                                               (self.cache['As'][prev_n-self.win_size:self.n-self.win_size] * 2)

        # result[2*self.win_size-1:self.n] = 1 - (self.cache['BCs'][self.win_size-1:self.n-self.win_size] +
        #                                         self.cache['BCs'][2*self.win_size-1:self.n]) / \
        #                                        (self.cache['As'][self.win_size-1:self.n-self.win_size] * 2)

        #result[self.win_size-1:self.n-self.win_size] = 1 - (self.cache['BCs'][self.win_size-1:self.n-self.win_size] + self.cache['BCs'][2 * self.win_size-1:self.n]) / (self.cache['As'][self.win_size-1:self.n-self.win_size] * 2)
        #for i in range(min_thres, self.win_size):
        #    result[i-1] = self.distance_raw(np.arange(i), np.arange(i, i + self.win_size))
        #for i in range(self.n - self.win_size, self.n - min_thres):
        #    result[i] = self.distance_raw(np.arange(i-self.win_size, i), np.arange(i+1, self.n))
        self.cache['result_real'] = result
        self.cache['result_real_n'] = self.n
        return result.tolist()

    def change_win_size(self, new_win_size):
        self.win_size = new_win_size
        self.update_A(0)
        self.update_BC(0)

    def get_drift_value(self, models, idx, gmm_labels):
        self.update_cache_value()
        # pooling all training data together and use gmm label to match
        training_idx = list(set([idx for model in models for idx in model['idx']]))
        unique_label = list(np.unique(gmm_labels))
        label2i = {label:i for i,label in enumerate(unique_label)}
        gmm_labels = [label2i[label] for label in gmm_labels]
        n_label = len(unique_label)
        _training_idx_set = [[] for label in unique_label]
        for i in training_idx:
            _training_idx_set[gmm_labels[i]].append(i)
        _k = [len(x) for x in _training_idx_set]

        _idx = [[] for label in unique_label]
        _training_idx = [[] for label in unique_label]
        for i in idx:
            label = gmm_labels[i]
            _idx[label].append(i)
            _training_idx[label] += self.get_nn_2(i, _training_idx_set[label], _k[label])

        drift_vals = np.zeros(len(unique_label))
        for i in range(n_label):
            _training_idx[i] = list(set(_training_idx[i]))
            if len(_training_idx[i]) == 0:
                drift_vals[i] = len(_idx[i])
            else:
                A = self.A_raw(_idx[i], _training_idx[i]) / len(_idx[i]) / len(_training_idx[i])
                B = self.BC_raw(_idx[i]) / len(_idx[i]) / len(_idx[i])
                C = self.BC_raw(_training_idx[i]) / len(_training_idx[i]) / len(_training_idx[i])
                if A < 1e-4:
                    drift_vals[i] = 0
                else:
                    drift_vals[i] = (1 - (B + C) / (2 * A)) * len(_idx[i])
        return drift_vals.sum() / len(idx)
        # return drift_vals.max() / len(_idx[drift_vals.argmax()])


def sumup(lists, weights):
    ret = np.zeros(len(lists[0]))
    for i in range(len(lists)):
        ret += np.array(lists[i]) * weights[i]
    return (ret / np.sum(weights)).tolist()


def CUSUM(x, tolerance, threshold):
    s_increase = np.zeros(x.size)
    s_decrease = np.zeros(x.size)
    alarm_time = np.array([], dtype=int)
    for i in range(1, x.size):
        s = x[i] - x[i - 1]
        s_increase[i] = np.max([0, s_increase[i - 1] + s - threshold])
        s_decrease[i] = np.max([0, s_decrease[i - 1] - s - threshold])
        if s_increase[i] > tolerance or s_decrease[i] > tolerance:
            alarm_time = np.append(alarm_time, i)
            s_increase[i], s_decrease[i] = 0, 0
    # return alarm_time, s_increase, s_decrease
    return (s_increase / tolerance).tolist()


def TVD(x, b, k, tmp):
    n = len(x)
    print("Processing a total of {} points with window size {}...".format(n, k))
    result = np.zeros(n)
    r = [-1, 6]
    for i in range(len(tmp), n - k + 1):
        tmp.append(np.histogram(x[i:i + k], bins=b, range=r)[0] / k)
    for i in range(k, n - k + 1):
        if i % 1000 == 0:
            print("Finshed {} points".format(i))
        diff = tmp[i - k] - tmp[i]
        result[i] = np.sum(diff[diff > 0])
    return result.tolist(), tmp


def old_TVD(x, b, k, tmp):
    n = len(x)
    print("Processing a total of {} points with window size {}...".format(n, k))
    result = np.zeros(n)
    r = [-1, 6]
    for i in range(len(tmp), n - k + 1):
        tmp.append(np.histogram(x[i:i + k], bins=b, range=r)[0] / k)
    for i in range(k, n - k + 1):
        if i % 1000 == 0:
            print("Finshed {} points".format(i))
        diff = tmp[i - k] - tmp[i]
        result[i] = np.sum(diff[diff > 0])
    return result.tolist(), tmp


def GMM(X, k, step, tmp, pre_result):
    n = len(X)
    for i in range(len(tmp) * step, n - k + 1, step):
        if i % (100 * step) == 0:
            print("Finshed {} points".format(i))
        tmp.append(get_model(X[i:i + k]))
    prev_n = len(pre_result)
    result = np.zeros(n)
    new_start = (prev_n - k - k) // step * step + step
    if new_start < k:
        new_start = k
    for i in range(0, prev_n):
        result[i] = pre_result[i] * 5
    for i in range(new_start, n - k + 1, step):
        m_l, s_l, w_l = tmp[(i - k) // step]
        m_r, c_r, w_r = tmp[i // step]
        result[i:i+step] = distance(m_l, s_l, w_l, m_r, c_r, w_r, MD)
    return move_avg(result / 5, step // 2).tolist(), tmp
    # return normalize(result), tmp


def get_model(X):
    model.fit(X)
    w = model.weights_ > 0.001
    return model.means_[w], model.covariances_[w], model.weights_[w] / np.sum(model.weights_[w])


def distance(m0, s0, w0, m1, s1, w1, method):
    n0 = w0.shape[0]
    n1 = w1.shape[0]
    cost = np.zeros(n0 * n1).reshape((n0, n1))
    for i in range(n0):
        for j in range(n1):
            cost[i, j] = method(m0[i], s0[i], m1[j], s1[j])
    return transportation_problem(cost, w0, w1)


# def transportation_problem(costs, x_max, y_max):
#     row = len(costs)
#     col = len(costs[0])
#     prob = pulp.LpProblem('Transportation Problem', sense=pulp.LpMinimize)
#     var = [[pulp.LpVariable(f'x{i}{j}', lowBound=0, cat=pulp.LpContinuous) for j in range(col)] for i in range(row)]
#
#     def flatten(x): return [y for l in x for y in flatten(l)] if type(x) is list else [x]
#     prob += pulp.lpDot(flatten(var), costs.flatten())
#     for i in range(row):
#         prob += (pulp.lpSum(var[i]) == x_max[i])
#     for j in range(col):
#         prob += (pulp.lpSum([var[i][j] for i in range(row)]) == y_max[j])
#     prob.solve()
#     return pulp.value(prob.objective)


def cache_gmm_model(X, k, step):
    ret = []
    for i in range(0, len(X) - k + 1, step):
        if i % (100 * step) == 0:
            print("Finshed {} points".format(i))
        ret.append(get_model(X[i:i + k]))
    return ret


def drift_test(models, train_X, test_X, labels):
  len_train_X = len(train_X)
  X = np.concatenate((np.array(train_X), np.array(test_X)))
  dist = distance_matrix(X, X)
  lens = np.zeros(len(models), dtype='int')
  lens_model = np.array([len(model['idx']) for model in models])
  As = np.zeros(len(models))
  Bs = np.zeros(len(models))
  Cs = np.zeros(len(models))
  ranges = [[] for x in models]
  for k, model in enumerate(models):
      Cs[k] = np.sum(dist[model['idx']][:, model['idx']]) / lens_model[k] / lens_model[k]
  for i in range(len(test_X)):
      ranges[labels[i]].append(i + len_train_X)
      lens[labels[i]] += 1
  lens_square = np.array([x * x for x in lens])
  for k, model in enumerate(models):
      As[k] = np.sum(dist[ranges[k]][:, model['idx']])
      Bs[k] = np.sum(dist[ranges[k]][:, ranges[k]])
  f = [lens[k] * (1 - (Bs[k] / lens_square[k] + Cs[k]) / (2 * As[k] / lens[k] / lens_model[k])) for k in range(len(models)) if lens[k] != 0]
  return sum(f) / len(test_X)


def set_plus(s, arr):
    ret = []
    for x in arr:
        if x not in s:
            s[x] = 1
            ret.append(x)
        else:
            s[x] += 1
    return ret

def set_minus(s, arr):
    ret = []
    for x in arr:
        if s[x] == 1:
            del s[x]
            ret.append(x)
        else:
            s[x] -= 1
    return ret