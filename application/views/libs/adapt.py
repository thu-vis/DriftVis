import numpy as np
from sklearn import svm, mixture, ensemble
from sklearn import metrics as sk_metrics 
from sklearn.model_selection import StratifiedKFold
from .util import MD, stat, move_avg, distance, gaussian_log_prob, generate_squares, softmax
import math
#import lightgbm

TRAIN_MODEL = True

n_components = 10
gmm_max_iter = 10000
svm_max_iter = 10000
gmm = mixture.BayesianGaussianMixture(n_components=n_components, weight_concentration_prior=1e-2,
                                      mean_precision_prior=1e-2, max_iter=gmm_max_iter)

class WrapModel:
    def __init__(self, labels, model, model_para, k_fold_list, fit_by_cv=None):
        self.labels = labels.tolist()
        #self.model = model(**model_para)
        #if model is lightgbm.LGBMClassifier:
        #    model_para['n_estimators'] = round(math.sqrt(len(k_fold_list[0])/0.7)*5)
        self.model_list = [model(**model_para) for i in range(len(k_fold_list))]
        self.k_fold_list = k_fold_list
        #self.seen_label = None
        self.seen_label = {}
        self.fit_by_cv = fit_by_cv

    def fit(self, X, y): 
        realy = y
        #y = np.array([self.labels.index(l) for l in y])
        for i, fold_idxes in enumerate(self.k_fold_list):
            self.seen_label[i] = np.unique(realy[fold_idxes])
            if len(self.seen_label[i]) > 1:
                if self.model_list[i] is None:
                    continue
                if self.fit_by_cv:
                    x_train = X[fold_idxes]
                    y_train = y[fold_idxes]
                    skf = StratifiedKFold(n_splits=5)
                    splitResult = [(x,y) for x,y in skf.split(x_train, y_train)]
                    collec = [self._fit_fold(self.model_list[i], x_train, y_train, train_idxs, valid_idxs)
                        for train_idxs, valid_idxs in splitResult]
                    self._fit_fold(self.model_list[i], x_train, y_train,
                                   *splitResult[np.array(collec).argmax()], only_fit=True) 
                else:
                    try:
                        self.model_list[i].fit(X[fold_idxes], y[fold_idxes])
                    except Exception:
                        self.model_list[i] = None
        # Original process:
        #self.seen_label = np.unique(y)
        #if len(self.seen_label) > 1:
        #    self.model.fit(X, y) 

    def _fit_fold(self, model, x_all, y_all, train_idxs, valid_idxs, only_fit=False):
        model.fit(x_all[train_idxs], y_all[train_idxs])
        if only_fit:
            return
        y_predict = y_all[valid_idxs]
        y_produce = model.predict_proba(x_all[valid_idxs])
        y_produce = y_produce.argmax(axis=1)
        #if len(y_produce.shape) == 2 and y_produce.shape[1] == 2:
        #    y_produce = y_produce[:, 0] < y_produce[:, 1]
        #return accuracy_score(y_predict, y_produce)
        return sk_metrics.balanced_accuracy_score(y_predict, y_produce)
        ## using auc
        #temp = sorted([(2 if yt else 1, y_produce[i][1]) for i,yt in enumerate(y_predict)], key=lambda x:x[0])
        #fpr, tpr, thresholds = sk_metrics.roc_curve([x[0] for x in temp], [x[1] for x in temp], pos_label=2)
        #return sk_metrics.auc(fpr, tpr)

    def predict_proba(self, X, fold_no):
        result = np.zeros((len(X), len(self.labels)))
        if len(self.seen_label[fold_no]) == 1:
            result[:, self.labels.index(self.seen_label[fold_no][0])] = 1
            return result
        else:
            if self.model_list[fold_no] is None:
                return None
            proba = self.model_list[fold_no].predict_proba(X)
            for i, c in enumerate(self.model_list[fold_no].classes_):
                result[:, self.labels.index(c)] = proba[:, i]
            return result

    def predict(self, X, fold_no):
        if len(self.seen_label[fold_no]) == 1:
            return np.full(len(X), self.seen_label[fold_no][0])
        #return self.model.predict(X)
        return self.model_list[fold_no].predict(X)

def train_single_model(X, y, idx, classify_labels, train_test_list, _model, model_para, fit_by_cv=None): 
    #train_idx = [i for i in idx if train_test[i]]
    #train_X, train_y = X[train_idx], y[train_idx]
    train_X, train_y = np.array(X), np.array(y)
    X, y, idx = np.array(X)[idx], np.array(y)[idx], np.array(idx)
    print(f"all data for model: {len(idx)}, all data for train: {len([i for i in idx if train_test_list[0][i]] )}")
    model = None
    if TRAIN_MODEL:
        model = WrapModel(classify_labels, _model, model_para,
                          [[i for i in idx if train_test[i]] for train_test in train_test_list],
                          fit_by_cv=fit_by_cv)
        # model.fit(X, y)
        #train_y = np.array([('not' not in l) for l in train_y])
        model.fit(train_X, train_y)
    # predict = lin_svc.predict(X)
    sigma = np.cov(X.transpose())
    return {
        'id': hash(tuple(idx)) % 100000007,
        'idx': idx,
        'use': False,
        'model': model,
        'mu': np.mean(X, axis=0),
        's': sigma,
        'log_det': np.linalg.slogdet(sigma)[1],
        'sinv': np.linalg.pinv(sigma),
        'weight': 1,
        'log_weight': 0,
        'count': len(y),
        # **stat(predict, y, ""),
        'X': X,
        'y': y,
    }

def pointwise_test_ensemble_model(ensemble, X, y, idx, train_test_list):
    fold_list = [[i for i in idx if not train_test[i]]
                for train_test in train_test_list]
    X, y = np.array(X), np.array(y)
    pointwise_result = [pointwise_predict(ensemble, X[idxes], fold_no)
                        for fold_no,idxes in enumerate(fold_list)]
    y_pred = [x.argmax(axis=1) for x in pointwise_result]
    y = np.array([ensemble[0]['model'].labels.index(l) for l in y])
    fold_acc = [(fold_no, sk_metrics.accuracy_score(y[idxes], y_pred[fold_no]).item())
                    for fold_no,idxes in enumerate(fold_list)] 
    fold_acc = sorted(fold_acc, key = lambda x:x[1])

    myf = lambda x: format(x, '.4f')
    info = {}
    for acc in fold_acc:
        sacc = myf(acc[1])
        if info.get(sacc) is None:
            info[sacc] = 0    
        info[sacc] += 1
    print(' ------------ kelei current accuracy: ', info)

    selected_index = len(fold_acc)//2
    used_fold_no = fold_acc[selected_index][0]
    used_fold_acc = fold_acc[selected_index][1] 
    idx = np.array(fold_list[used_fold_no])
    y = y[idx]
    y_pred = y_pred[used_fold_no]
    pointwise_result = pointwise_result[used_fold_no]
    return {
        'acc': used_fold_acc,
        **generate_squares(y, y_pred, pointwise_result, ensemble[0]['model'].labels, idx),
    }

def add_proportion_for_test_result(test_result, new_idx):
    for detail in test_result['details']:
        detail['proportion'] = len([i for i in detail['idx'] if i in new_idx]) / len(detail['idx'])
    return test_result


def pointwise_predict(ensemble, X, fold_no=0):
    return pointwise_classify(ensemble, X, fold_no)


def pointwise_classify(ensemble, X, fold_no=0):
    ret = np.zeros(X.shape[0])
    #predict_proba = np.array(list(map(lambda m: m['model'].predict_proba(X, fold_no), ensemble)))
    predict_proba = list(map(lambda m: m['model'].predict_proba(X, fold_no), ensemble))
    distances = np.array(list(map(lambda m: np.array([gaussian_log_prob(x, m['mu'], m['sinv'], m['log_det'])
                                                      for x in X]), ensemble))).transpose()
    distances = np.exp(-distances)
    pooled = np.zeros(predict_proba[0].shape)
    n_models = len(ensemble)
    for i in range(X.shape[0]):
        distances[i] = softmax(distances[i])
        for j in range(n_models):
            #pooled[i] += distances[i, j] * predict_proba[j, i]
            if predict_proba[j] is None:
                continue
            pooled[i] += distances[i, j] * predict_proba[j][i]
    return pooled
