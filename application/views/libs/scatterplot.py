import numpy as np
import os
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from .IncrementalTSNE import IncrementalTSNE
import copy

class Layout(object):
    def __init__(self, attr_name, data, idx, normalize=False):
        self.normalize = normalize
        self.scaler = StandardScaler().fit(np.array(data[attr_name])) if normalize else None
        self.attr_name = attr_name
        self.data = self.get_data(data, idx)
        self.n = self.data.shape[0]
        self.dim = len(attr_name)
        self.result = None
        self.prev_n = 0
        self.rnd = np.random.RandomState(42)

    def get_data(self, data, idx):
        if not self.normalize:
            return np.array(data[self.attr_name].iloc[idx])
        else:
            return self.scaler.transform(data[self.attr_name].iloc[idx])

    def layout(self):
        pass

    def add_data(self, data, idx):
        self.data = np.concatenate((self.data, self.get_data(data, idx)))
        self.n = self.data.shape[0]
        self.layout()

    def set_data(self, data, idx, result):
        self.data = self.get_data(data, idx)
        self.n = self.data.shape[0]
        self.result = result
        self.prev_n = self.n


class PCA_layout(Layout):
    def __init__(self, attr_name, data, idx):
        super().__init__(attr_name, data, idx)
        self.pca = PCA(n_components=2)
        self.layout()

    def layout(self):
        if self.prev_n != self.n:
            self.pca.fit(self.data)
            self.result = self.pca.transform(self.data)
            self.prev_n = self.n
        return {
            'x': self.result[:, 0].tolist(),
            'y': self.result[:, 1].tolist(),
        }


class TSNE(Layout):
    def __init__(self, attr_name, data, idx):
        super().__init__(attr_name, data, idx)
        self.active_point = 1500
        self.constraint_cnt = 500
        self.skip_points = 0
        self.attr_idxes = []
        self.stacked_layout = []
        self.label_result = None

    def add_data(self, data, idx, gmm_result):
        self.gmm_result = gmm_result
        super().add_data(data, idx)


    def set_data(self, data, idx, result, label_result, gmm_result):
        self.gmm_result = gmm_result
        super().set_data(data, idx, result)
        self.label_result = label_result
        self.skip_points = max(self.n - self.active_point, 0)

    def sample(self):
        idx = np.arange(max(self.prev_n - self.active_point, 0), self.prev_n)
        idx = self.rnd.choice(idx, min(len(idx) // 2, self.constraint_cnt), replace=False)
        if len(idx) == 0:
            return None, None, None
        classes = self.gmm_result['classes']
        return self.data[idx], self.result[idx], classes[idx].astype(int)

    def search_nn(self):
        result = np.zeros((self.n - self.prev_n, 2))
        for i in range(self.prev_n, self.n):
            distance = np.linalg.norm(self.data[i] - self.data[self.skip_points:self.prev_n, ], axis=1)
            result[i - self.prev_n] = self.result[distance.argmin() + self.skip_points]
        self.result = np.concatenate((self.result, result))

    def layout(self):
        if self.prev_n != self.n:
            filename = self.hash()
            if os.path.exists(filename):
                self.load_cache(filename)
            else:
                if self.result is not None:
                    constraint_X, constraint_Y, constraint_classes = self.sample()
                    #print(len(constraint_X), len(constraint_Y), len(constraint_classes))
                    #print(constraint_X, constraint_Y, constraint_classes )
                    self.search_nn()
                    classes = self.gmm_result['classes']
                    n_labels = len(self.label_result[2])
                    constraint_weight = np.ones((len(constraint_X), ), dtype=float).astype(dtype=np.float64) 
                    if len(constraint_X) < self.constraint_cnt:
                        T = (self.constraint_cnt / (len(constraint_X) / 2)) ** 0.13# + 0.25
                        constraint_weight *= T
                    [con_X, con_Y, con_w] = self.label_result
                    if len(con_X) > 1:
                        con_X = np.array(con_X)
                        con_Y = np.array(con_Y)
                        con_w = np.array(con_w)
                        '''
                        for iter in range(100):
                            grad = []
                            for i in range(n_labels):
                                f = np.array([0, 0])
                                for j in range(n_labels):
                                    if i != j:
                                        t = con_Y[i] - con_Y[j]
                                        r = np.dot(t, t)
                                        f = 0.01 * t / r
                                grad.append(f)
                            con_Y = con_Y + grad
                        '''
                        constraint_X = np.concatenate((constraint_X, con_X))
                        constraint_Y = np.concatenate((constraint_Y, con_Y))
                        constraint_weight = np.concatenate((constraint_weight, con_w))
                        constraint_classes = np.concatenate((constraint_classes, np.array([i for i in range(n_labels)])))
                    tsne = IncrementalTSNE(n_components=2, init=self.result, perplexity=30, angle=0.3, n_jobs=8, n_iter=500, random_state=42)
                    n_labels = len(self.gmm_result['distribution'])
                    print(n_labels)
                    beta = 0.1 if n_labels < 5 else 0.5
                    alpha = 0.8 if n_labels < 5 else 0.5
                    self.result, self.label_result = tsne.fit_transform(self.data, skip_num_points=self.skip_points,
                                                    constraint_X=constraint_X, constraint_Y=constraint_Y,
                                                    gmm = self.gmm_result, label_alpha=alpha, label_beta=beta,
                                                    labels = classes, constraint_labels=constraint_classes, constraint_weight=constraint_weight)
                else:
                    classes = self.gmm_result['classes']
                    tsne = IncrementalTSNE(n_components=2, init='pca', method='barnes_hut', perplexity=30, angle=0.3, n_jobs=8, n_iter=1000, random_state=42)
                    n_labels = len(np.unique(classes))
                    beta = 0.1 if n_labels < 5 else 0.25
                    alpha = 0.9 if n_labels < 5 else 0.8
                    self.result, self.label_result = tsne.fit_transform(self.data, labels=classes, gmm=self.gmm_result, label_alpha=alpha, label_beta=beta)
            self.save_cache(filename)
            self.prev_n = self.n
            self.skip_points = max(self.n - self.active_point, 0)
        if len(self.attr_idxes) > 0:
            tmp_X = np.zeros((self.n, 2 + len(self.attr_idxes)))
            for i, idx in enumerate(self.attr_idxes):
                tmp_X[:, i] = self.data[:, idx]
            tmp_X[:, len(self.attr_idxes)] = self.result[:, 0].tolist()
            tmp_X[:, len(self.attr_idxes) + 1] = self.result[:, 1].tolist()
            result = PCA(n_components=2).fit_transform(tmp_X)
            self.current_layout =  {
                'x': result[:, 0].tolist(),
                'y': result[:, 1].tolist(),
            }
        else:
            self.current_layout = {
                'x': self.result[:, 0].tolist(),
                'y': self.result[:, 1].tolist(),
            }
        self.stacked_layout.append(self.current_layout)
    
    def get_layout(self):
        data = copy.copy(self.current_layout)
        trace = []
        if len(self.stacked_layout) > 1:
            trace = [[] for i in range(self.n)]
            for layout in self.stacked_layout:
                for i in range(len(layout['x'])):
                    trace[i].append({ 'x' : round(layout['x'][i], 4), 'y': round(layout['y'][i], 4)})
        data['layout'] = trace
        self.stacked_layout = []
        return data

    def set_attr_idxes(self, attr_idxes):
        self.attr_idxes = attr_idxes

    def get_attr_idxes(self):
        return self.attr_idxes

    def hash(self):
        keys = [self.n, self.prev_n, round(self.result[0,0],3), round(self.result[0,1],3)]
        s = 'cache/tsne_' + ','.join([str(x) for x in keys])
        return s
    
    def save_cache(self, filename):
        cache = dict()
        cache['n'] = self.n
        cache['label_result'] = self.label_result
        cache['result'] = self.result.copy()
        cache['stacked_layout'] = self.stacked_layout.copy()
        cache['tsne_rnd_state'] = self.rnd.get_state()
        with open(filename, 'wb') as fout:
            pickle.dump(cache, fout)

    def load_cache(self, filename):
        with open(filename, 'rb') as fout:
            cache = pickle.load(fout)
        print('use cache tsne')
        self.prev_n = self.n = cache['n']
        self.skip_points = max(self.n - self.active_point, 0)
        self.label_result = cache['label_result']
        self.result = cache['result'].copy()
        self.current_layout = {
            'x': self.result[:, 0].tolist(),
            'y': self.result[:, 1].tolist(),
        }
        self.stacked_layout = cache['stacked_layout'].copy()
        self.rnd.set_state(cache['tsne_rnd_state'])
