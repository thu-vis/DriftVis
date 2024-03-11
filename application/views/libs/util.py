import numpy as np
from numpy.linalg import inv, slogdet, norm
from scipy.linalg import sqrtm
from sklearn.metrics import r2_score
from sklearn.metrics import precision_recall_fscore_support

def Origin(arr, k, inf=None, sup=None):
    return move_avg(normalize(arr, inf, sup), k).tolist()


def normalize(arr, inf, sup):
    if not sup:
        sup = np.max(arr)
    if not inf:
        inf = np.min(arr)
    return ((arr - inf) / (sup - inf)).tolist()


def move_avg(arr, k=1):
    n = len(arr)
    win = 2 * k + 1
    cumsum = np.cumsum(arr, dtype=float)
    ret = np.zeros(n)
    for i in range(0, k + 1):
        ret[i] = cumsum[i + k] / (i + k + 1)
    for i in range(n - k, n):
        ret[i] = (cumsum[-1] - cumsum[i - k - 1]) / (n - i + k)
    ret[k + 1:-k] = (cumsum[win:] - cumsum[:-win]) / win
    return ret


def move_avg_prev(arr, k=0):
    n = len(arr)
    cumsum = np.cumsum(arr, dtype=float)
    for i in range(n-1, k, -1):
        cumsum[i] = (cumsum[i] - cumsum[i-k-1]) / (k + 1)
    for i in range(min(n, k+1)):
        cumsum[i] /= (i + 1)
    return cumsum


'''
distances for two multivariate gaussian distribution
'''


def KL(m0, s0, m1, s1):
    return 0.5 * (np.trace(inv(s1).dot(s0)) + (m1 - m0).dot(inv(s1)).dot(m1 - m0) - m0.shape + slogdet(s1)[1] -
                  slogdet(s1)[0])


def SysKL(m0, s0, m1, s1):
    return (KL(m0, s0, m1, s1) + KL(m1, s1, m0, s0)) / 2


def MD(m0, s0, m1, s1):
    return norm(m0 - m1) + np.trace(s0 + s1 - 2 * sqrtm(sqrtm(s1).dot(s0).dot(sqrtm(s1))))


def stat(predict, y, prefix):
    if y.dtype == 'bool':
        return {
            f'{prefix}acc': np.mean(predict == y).item(),
            f'{prefix}TP': np.sum(predict & y).item(),
            f'{prefix}TN': np.sum((~predict) & (~y)).item(),
            f'{prefix}FP': np.sum(predict & (~y)).item(),
            f'{prefix}FN': np.sum((~predict) & y).item(),
        }
    else:
        return {
            f'{prefix}r2': r2_score(y, predict).item(),
        }


def softmax(a):
    tmp = np.exp(a - np.max(a))
    return tmp / np.sum(tmp)


def distance(X, mu, sinv):
    tmp = X - mu
    if len(tmp.shape) == 1:
        return tmp.dot(sinv).dot(tmp)
    return np.array([x.dot(sinv).dot(x) for x in tmp])


def gaussian_log_prob(x, mu, sinv, log_det):
    tmp = x - mu
    # return (-(tmp.dot(sinv).dot(tmp)) - mu.shape[0] * np.log(2 * np.pi) - log_det) / 2
    return (tmp.dot(sinv).dot(tmp)) + log_det


def gaussian_true_log_prob(x, mu, sinv, log_det):
    tmp = x - mu
    return (-(tmp.dot(sinv).dot(tmp)) - mu.shape[0] * np.log(2 * np.pi) - log_det) / 2


def distance_between_gmm_and_model(gmm, models):
    ret = 0.0
    for component in gmm:
        dis = np.min([MD(gmm[component]['mu'], gmm[component]['s'], model['mu'], model['s']) for model in models])
        ret += dis * gmm[component]['weight']
    return ret


def pos_average(x):
    return np.average(x[x > 0])


def get_percentage(idx, idxes):
    ret = np.zeros(len(idxes) + 1)
    if len(idx) == 0:
        ret[len(idxes)] = 1
        return ret.tolist()
    for i in idx:
        used = False
        for key, val in enumerate(idxes):
            if i in val:
                ret[key] += 1
                used = True
        if not used:
            ret[len(idxes)] += 1
    return (ret / np.sum(ret)).tolist()


def basic_stat(names, X, y, idx):
    X, y, idx = np.array(X), np.array(y), np.array(idx)
    X, y = X[idx], y[idx]
    ret = {'y_true': np.sum(y).item()}
    for i, x in enumerate(names):
      ret[f'{x}_mu'] = np.mean(X[:, i]).item()
      ret[f'{x}_sd'] = np.std(X[:, i]).item()
    return ret


def cross(a):
    return np.outer(a, a)


def update_component(component, x, p):
    component['n'] += 1
    component['sp'] += p
    component['weight'] = component['sp'] / component['n']
    component['log_weight'] = np.log(component['weight'])
    mu_new = component['mu'] + (x - component['mu']) * p / component['sp']
    component['s'] += (cross(x - mu_new) - component['s']) * p / component['sp'] - cross(mu_new - component['mu'])
    component['sinv'] = np.linalg.pinv(component['s'])
    component['log_det'] = np.linalg.slogdet(component['s'])[1]
    component['mu'] = mu_new

def shift_component(component, x, p):
    mu_new = component['mu'] + (x - component['mu']) * p / component['sp']
    # component['s'] += (cross(x - mu_new) - component['s']) * p / component['sp'] - cross(mu_new - component['mu'])
    # component['sinv'] = np.linalg.pinv(component['s'])
    # component['log_det'] = np.linalg.slogdet(component['s'])[1]
    component['mu'] = mu_new



def merge_component(comp1, comp2):
    component = {}
    assert comp1['n'] == comp2['n']
    component['n'] = comp1['n']
    component['sp'] = comp1['sp'] + comp2['sp']
    component['weight'] = component['sp'] / component['n']
    component['log_weight'] = np.log(component['weight'])
    component['mu'] = (comp1['mu'] * comp1['sp'] + comp2['mu'] * comp2['sp']) / component['sp']
    component['s'] = (comp1['s'] + cross(comp1['mu'])) * comp1['sp'] / component['sp'] + \
                     (comp2['s'] + cross(comp2['mu'])) * comp2['sp'] / component['sp'] - \
                     np.cross(component['mu'])
    component['sinv'] = np.linalg.pinv(component['s'])
    component['log_det'] = np.linalg.slogdet(component['s'])[1]


def component_remove(comp1, comp2):
    component = {}
    assert comp1['n'] == comp2['n']
    component['n'] = comp1['n']
    component['sp'] = comp1['sp'] - comp2['sp']
    component['weight'] = component['sp'] / component['n']
    component['log_weight'] = np.log(component['weight'])
    component['mu'] = (comp1['mu'] * comp1['sp'] - comp2['mu'] * comp2['sp']) / component['sp']
    component['s'] = ((comp1['s'] + cross(comp1['mu'])) * comp1['sp'] - \
                     (comp2['s'] + cross(comp2['mu'])) * comp2['sp']) / component['sp'] - \
                     cross(component['mu'])
    component['sinv'] = np.linalg.pinv(component['s'])
    component['log_det'] = np.linalg.slogdet(component['s'])[1]


def generate_squares(y_true, y_pred, scores, class_name, idx):
    '''
    y_true: n * 1
    y_pred: n * 1
    scores are n * n_class matrix
    ret are {
        'class1': {
            'bin0': { # '0-0.1'
                    'TP': 100,
                    'FP': [0, 2, 1, ...] # class others but predicted as class 1
                    'FN': [0, 1, 0, ...] # class1 but predicted as others
                }
        }
    }
    '''
    ret = {
        _class: {
          f'bin{i}': {
            'TP': 0,
            'FP': [0 for _ in class_name],
            'FN': [0 for _ in class_name],
            'TP_idx': [],
            'FP_idx': [[] for _ in class_name],
            'FN_idx': [[] for _ in class_name],
          } for i in range(10)
        } for _class in class_name
    }
    idx = idx.tolist()
    for i in range(len(y_true)):
        max_score = scores[i, y_pred[i]]
        try:
            bin_id = int(round(max_score * 10-0.5))
        except:
            bin_id = 9
        if bin_id == 10:
            bin_id = 9
        bin_id = f'bin{bin_id}'
        if y_true[i] == y_pred[i]:
            ret[class_name[y_pred[i]]][bin_id]['TP'] += 1
            ret[class_name[y_pred[i]]][bin_id]['TP_idx'].append(idx[i])
        else:
            ret[class_name[y_true[i]]][bin_id]['FN'][y_pred[i]] += 1
            ret[class_name[y_pred[i]]][bin_id]['FP'][y_true[i]] += 1
            ret[class_name[y_true[i]]][bin_id]['FN_idx'][y_pred[i]].append(idx[i])
            ret[class_name[y_pred[i]]][bin_id]['FP_idx'][y_true[i]].append(idx[i])
    #print(precision_recall_fscore_support(y_true, y_pred))
    return {
        'squares': ret,
        'classes': class_name,
    }
