'''
'id': hash(tuple(idx)) % 100000007,
'idx': idx,
'use': True,
'model': lin_svc,
'mu': np.mean(X, axis=0),
's': sigma,
'log_det': np.linalg.slogdet(sigma)[1],
'sinv': np.linalg.pinv(sigma),
'weight': 1,
'log_weight': 0,
'count': len(y),
**stat(predict, y, ""),
'X': X,
'y': y,
'''
import numpy as np
from .util import distance_between_gmm_and_model, gaussian_log_prob


class Ensemble(object):
    def __init__(self):
        self.models = []

    def add_model(self, model):
        if isinstance(model, list):
            self.models = self.models + model
        else:
            self.models.append(model)

    def get_all_model(self):
        return self.models

    def get_all_used_model(self):
        return [model for model in self.models if model['use']]

    @staticmethod
    def extract_model_info(model):
        # keys = ['id', 'count', 'TP', 'TN', 'FP', 'FN', 'acc', 'use', ]
        keys = ['id', 'count', 'use', ]
        keys_tolist = ['idx', 's', 'sinv', 'mu', ]
        return {**{key: model[key] for key in keys}, **{key: model[key].tolist() for key in keys_tolist}}

    def to_json(self):
        return [Ensemble.extract_model_info(model) for model in self.models]

    def search_model(self, model_id):
        for model in self.models:
            if model['id'] == model_id:
                return model
        return None

    def toggle(self, model_id):
        model = self.search_model(model_id)
        model['use'] = not model['use']

    def set_use(self, model_id, use_flag):
        model = self.search_model(model_id)
        model['use'] = use_flag

    def use_all(self):
        for model in self.models:
            model['use'] = True

    def unuse_all(self):
        for model in self.models:
            model['use'] = False

    def generate_distance_for_points(self, X, idx=None):
        models = self.get_all_used_model()
        if idx:
            X = X[idx, :]
        result = np.zeros((X.shape[0], len(models)))
        for k, model in enumerate(models):
            for i, x in enumerate(X):
                result[i][k] = gaussian_log_prob(x, model['mu'], model['sinv'], model['log_det'])
        return result

