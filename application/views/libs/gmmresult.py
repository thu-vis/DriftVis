import numpy as np
from sklearn import svm, mixture
from .util import gaussian_true_log_prob, gaussian_log_prob, softmax, update_component, gaussian_true_log_prob, shift_component
from .hypothesis import Hypothesis
import logging
logging.basicConfig(level = logging.DEBUG,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


gmm_max_iter = 10000
svm_max_iter = 10000


class GMMResult(object):

    def __init__(self, attr_name, X, idx, gmm_thres):
        self.component_relax_thres = dict()
        self.component_strict_thres = dict()
        self.attr_name = attr_name
        self.gmm_thres = gmm_thres
        self.X = self.get_data(X, idx)
        self.n = self.X.shape[0]
        self.idx = np.arange(self.n)
        self.prev_n = 0
        self.labels = []
        self.rnd = np.random.RandomState(42)


    def build_initial_gmm(self):
        self.gmm_result = self.build_gmm(self.X)
        self.rnd.randn(120)
        self.calculate_threshold()
        self.update_idx()
        self.prev_n = self.n



    def set_data(self, X, idx, gmm_result, threshold):
        self.X = self.get_data(X, idx)
        self.n = self.X.shape[0]
        self.idx = np.arange(self.n)
        self.prev_n = 0
        self.gmm_result = gmm_result
        self.component_relax_thres, self.component_strict_thres = threshold
        self.update_idx()
        self.prev_n = self.n
        self.labels = []

    def recover_labels(self, old_max=-1):
        self.labels = [self.labels[i] for i in self.gmm_result['preserve_labels']]
        if old_max != -1:
            old = [i for i in self.labels if i <= old_max]
            new = [old_max + 1 + i for i in range(len(self.labels)-len(old))]
            self.labels = old + new
        for i, label in enumerate(self.labels):
            self.gmm_result['classes'][self.gmm_result['classes'] == i] = -label
        self.gmm_result['classes'] = np.abs(self.gmm_result['classes'])
        self.gmm_result['distribution'] = {
            label: self.gmm_result['distribution'][i] for i, label in enumerate(self.labels)
        }

    def calculate_threshold(self):
        self.component_relax_thres = dict()
        self.component_strict_thres = dict()
        bidx = np.full(self.X.shape[0], True, dtype=bool)
        for i in self.gmm_result['uncertain_ids']:
            bidx[i] = False
        labels = list(self.gmm_result['distribution'].keys())
        points_component_distance = np.zeros((self.n, len(labels)))
        for i in range(self.X.shape[0]):
            points_component_distance[i] = [gaussian_log_prob(self.X[i], self.gmm_result['distribution'][label]['mu'],
                                                                     self.gmm_result['distribution'][label]['sinv'],
                                                                     self.gmm_result['distribution'][label]['log_det'])
                                                   - self.gmm_result['distribution'][label]['log_weight'] for label in labels]
        tmp = np.array([])
        for k, _ in enumerate(labels):
            k_distance = points_component_distance[np.logical_and(bidx, points_component_distance.argmin(axis=1)==k)][:,k]
            if k_distance.size == 0:
                k_distance = np.array([1e-5])
            tmp = np.concatenate([tmp, k_distance])
            if len(labels) > 5: # many components, make it hard to build more
                self.component_relax_thres[k] = np.max(k_distance) + (np.max(k_distance) - np.min(k_distance)) / 100
                self.component_strict_thres[k] = np.quantile(k_distance, 0.9)
            else: # few components, courage to build more
                self.component_relax_thres[k] = np.quantile(k_distance, 0.95)
                self.component_strict_thres[k] = np.quantile(k_distance, 0.9)


    def update_idx(self):
        for component in self.gmm_result['distribution']:
            self.gmm_result['distribution'][component]['idx'] = self.idx[self.gmm_result['classes'] == component]

    def get_gmm_result(self):
        return self.gmm_result

    def get_data(self, data, idx):
        return np.array(data[self.attr_name].iloc[idx])

    def add_data(self, data, idx):
        self.X = np.concatenate((self.X, self.get_data(data, idx)))
        self.n = self.X.shape[0]
        self.idx = np.arange(self.n)
        self.incremental_gmm()
        self.prev_n = self.n
        self.update_idx()

    def M_step(self):
        for component in self.gmm_result['distribution']:
            distribution = self.gmm_result['distribution'][component]
            bidx = self.gmm_result['classes'] == component
            X = self.X[bidx]
            sigma = np.cov(X.transpose())
            distribution['mu'] = np.mean(X, axis=0)
            distribution['s'] = sigma
            distribution['log_det'] = np.linalg.slogdet(sigma)[1]
            distribution['sinv'] = np.linalg.pinv(sigma)


    def build_gmm(self, X, weights_init=None, means_init=None, precisions_init=None, n_keep=-1, n_components=10):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if weights_init is not None:
            logger.debug("BUILDGMM: with init")
            n_comp = len(weights_init)
            gmm = mixture.GaussianMixture(n_components=n_comp,  max_iter=gmm_max_iter, n_init=5,
                                          weights_init=weights_init, means_init=means_init, precisions_init=precisions_init)
            gmm.fit(X)
        else:
            logger.debug("BUILDGMM: without init")
            gmm = self._build_gmm(X, 'BIC', n_components=n_components)
        predict = gmm.predict(X)

        # remove small groups
        labels = np.unique(predict)
        freq = np.array([np.sum(predict == label) for label in labels])
        sinvs = {label: np.linalg.pinv(gmm.covariances_[label]) for label in labels}
        log_dets = {label: np.linalg.slogdet(gmm.covariances_[label])[1] for label in labels}
        logger.debug(f"BUILDGMM: points in each label:{freq}")

        if len(labels) >= 5:
            threshold = predict.shape[0] // len(labels) * 0.75
        else:
            threshold = predict.shape[0] // len(labels) * 0.5
        threshold = min(100, threshold)
        if weights_init is not None:
            threshold = 0
        preserve_labels = []
        discard_labels = []
        for i, label in enumerate(labels):
            if threshold <= freq[i]:
                preserve_labels.append(label)
            else:
                discard_labels.append(label)

        if n_keep != -1 and len(preserve_labels) > n_keep:
            n_keep = min(len(freq), n_keep)
            thres = freq[freq.argsort()[-n_keep]]
            preserve_labels = [l for i, l in enumerate(labels) if freq[i] >= thres]
            discard_labels = [l for i, l in enumerate(labels) if freq[i] < thres]

        logger.debug(f"BUILDGMM: discard_labels:{discard_labels}")

        preserve_weights = np.sum([gmm.weights_[label] for label in preserve_labels])
        uncertain_ids = []
        threshold = {label: [] for label in preserve_labels}  # key are labels, may be 0,1,3,4,6
        for i in range(predict.shape[0]):
            if predict[i] in preserve_labels:
                threshold[predict[i]].append(gaussian_log_prob(X[i], gmm.means_[predict[i]], sinvs[predict[i]], log_dets[predict[i]])
                                                - np.log(gmm.weights_[predict[i]] / preserve_weights))
        threshold = [np.quantile(threshold[x], 0.95) for x in threshold]  # now "key" are index, must be 0,1,2,3,4
        for i in range(predict.shape[0]):
            if predict[i] in discard_labels:
                distances = np.array([gaussian_log_prob(X[i], gmm.means_[label], sinvs[label], log_dets[label])
                            - np.log(gmm.weights_[label] / preserve_weights) for label in preserve_labels])
                min_idx = distances.argmin()
                predict[i] = preserve_labels[min_idx]
                if distances[min_idx] > threshold[min_idx]:
                    uncertain_ids.append(i)
        logger.debug(f"BUILDGMM: uncertain_ids:{len(uncertain_ids)}")
        classes = np.zeros(predict.shape[0], dtype=np.int)
        for i, label in enumerate(preserve_labels):
            classes[predict == label] = i
        return {
            'classes': classes,
            'distribution': {
                i: {
                    'mu': gmm.means_[label],
                    's': gmm.covariances_[label],
                    'log_det': log_dets[label],
                    'sinv': sinvs[label],
                    'sp': gmm.weights_[label] / preserve_weights * len(X),
                    'n': len(X),
                    'weight': gmm.weights_[label] / preserve_weights,
                    'log_weight': np.log(gmm.weights_[label] / preserve_weights),
                } for i, label in enumerate(preserve_labels)
            },
            'uncertain_ids': uncertain_ids,
            'preserve_labels': preserve_labels,
        }

    def incremental_gmm(self):
        self.incremental_gmm_relax()


    def incremental_gmm_relax(self):
        old_distribution = self.gmm_result['distribution']
        old_labels = list(old_distribution.keys())
        component_start_idx = np.max(old_labels) + 1
        old_bidx = np.array([i < self.prev_n for i in range(self.n)])
        new_bidx = ~old_bidx
        old_X = self.X[old_bidx]
        new_X = self.X[new_bidx]

        assign = np.full(new_X.shape[0], -1, dtype=np.int)
        anyway_assign = np.full(new_X.shape[0], -1, dtype=np.int)
        for i in range(new_X.shape[0]):
            distances = np.array([gaussian_log_prob(new_X[i], old_distribution[label]['mu'],
                                  old_distribution[label]['sinv'], old_distribution[label]['log_det']) - old_distribution[label]['log_weight']
                                  for label in old_labels])
            min_idx = distances.argmin()
            anyway_assign[i] = old_labels[min_idx]
            if distances[min_idx] < self.component_relax_thres[min_idx]:
                assign[i] = anyway_assign[i]
                if distances[min_idx] < self.component_strict_thres[min_idx]:
                    log_ps = [old_distribution[label]['log_weight'] +
                              gaussian_true_log_prob(new_X[i], old_distribution[label]['mu'], old_distribution[label]['sinv'], old_distribution[label]['log_det'])
                              for label in old_labels]
                    ps = softmax(log_ps)
                    for k, label in enumerate(old_labels):
                        update_component(old_distribution[label], new_X[i], ps[k])

        for i in range(new_X.shape[0]):
            if assign[i] < 0:
                self.gmm_result['uncertain_ids'].append(i + self.prev_n)
        logger.debug(f"Incremental: unmatch:{np.sum(assign < 0)}, new_data:{(self.n - self.prev_n)}, uncertainids: {len(self.gmm_result['uncertain_ids'])}")
        build_new_gmm_thres = max(self.gmm_thres, self.n // 12)
        if len(self.gmm_result['uncertain_ids']) > build_new_gmm_thres:
            logger.debug(f"Incremental: build new gmm for new data: {len(self.gmm_result['uncertain_ids'])}")
            n_max_new_comp = min(7, 1+round(len(self.gmm_result['uncertain_ids'])/50))
            new_gmm = self.build_gmm(self.X[self.gmm_result['uncertain_ids']], n_components=n_max_new_comp)
            new_distribution = new_gmm['distribution']
            if len(new_distribution) == 0:
                self.gmm_result['classes'] = np.concatenate((self.gmm_result['classes'], anyway_assign))
                return
            if len(old_distribution) + len(new_distribution) < 5:
                self.gmm_result = self.direct_merge(self.gmm_result, new_gmm)
                self.calculate_threshold()
                return
            weights_init = [old_distribution[key]['weight'] * (self.n - len(self.gmm_result['uncertain_ids'])) for key in old_distribution] +\
                           [new_distribution[key]['weight'] * len(self.gmm_result['uncertain_ids']) for key in new_distribution]
            weights_init = np.array(weights_init) / np.sum(weights_init)
            means_init = [old_distribution[key]['mu'] for key in old_distribution] +\
                         [new_distribution[key]['mu'] for key in new_distribution]
            precision_init = [old_distribution[key]['sinv'] for key in old_distribution] +\
                             [new_distribution[key]['sinv'] for key in new_distribution]
            old_max_label = int(np.max(list(old_distribution.keys()))) + 1
            self.labels = list(old_distribution.keys()) + [l + old_max_label for l in new_distribution.keys()]
            self.gmm_result = self.build_gmm(self.X, weights_init, means_init, precision_init, n_keep=len(old_labels)+4)
            self.recover_labels(old_max=max(old_distribution.keys()))
            self.calculate_threshold()
        else:
            self.gmm_result['classes'] = np.concatenate((self.gmm_result['classes'], anyway_assign))


    def direct_merge(self, old_gmm, new_gmm):
        old_distribution = old_gmm['distribution']
        new_distribution = new_gmm['distribution']
        old_labels = list(old_distribution.keys())
        preserve_labels = list(old_distribution.keys()) + [max(old_labels) + 1 + x for x in new_distribution]
        distribution = {**old_distribution,**{max(old_labels) + 1 + x:new_distribution[x] for x in new_distribution}}
        uncertain_ids = new_gmm['uncertain_ids']

        thres = dict()
        bidx = np.full(self.X.shape[0], True, dtype=bool)
        for i in uncertain_ids:
            bidx[i] = False
        points_component_distance = np.zeros((self.n, len(preserve_labels)))
        for i in range(self.X.shape[0]):
            points_component_distance[i] = [gaussian_log_prob(self.X[i], distribution[label]['mu'],
                                                              distribution[label]['sinv'],
                                                              distribution[label]['log_det']) for label in preserve_labels]
        classes = points_component_distance.argmin(axis=1)
        for k, _ in enumerate(preserve_labels):
            k_distance = points_component_distance[np.logical_and(bidx, classes == k)][:, k]
            if k_distance.size == 0:
                k_distance = np.array([1e-5])
            thres[k] = np.quantile(k_distance, 0.99)
        uncertain_ids = [i for i in range(self.X.shape[0]) if points_component_distance[i, classes[i]] > thres[classes[i]]]
        print(len(uncertain_ids))
        for k, label in enumerate(preserve_labels):
            distribution[label]['weight'] = (classes == k).sum() / (self.n - len(uncertain_ids)),
            distribution[label]['log_weight'] = np.log((classes == k).sum() / (self.n - len(uncertain_ids))),
            distribution[label]['n'] = (self.n - len(uncertain_ids))
            distribution[label]['sp'] = (classes == k).sum()
        return {
            'classes': np.array([preserve_labels[x] for x in classes],dtype=int),
            'distribution': distribution,
            'uncertain_ids': uncertain_ids,
            'preserve_labels': preserve_labels,
        }


    def analysis_component(self, idxes):
        selected_classes = self.gmm_result['classes'][idxes]
        labels = np.unique(selected_classes)
        freq = np.array([np.sum(selected_classes == label) for label in labels])
        percentage_1 = freq / np.sum(freq)
        percentage_2 = [freq[i] / np.sum(self.gmm_result['classes'] == label) for i, label in enumerate(labels)]
        logger.debug(labels, percentage_1, percentage_2)
        threshold = 1 / len(labels) / 2
        return [label for i, label in enumerate(labels) if percentage_1[i] > threshold and percentage_2[i] > 0.5]


    def adjust_component_merge(self, idxes):
        selected_classes = self.gmm_result['classes'][idxes]
        labels = list(self.gmm_result['distribution'].keys())
        freq = np.array([np.sum(selected_classes == label) for label in labels])
        alls = np.array([len(self.gmm_result['distribution'][label]['idx']) for label in labels])
        percentage = [freq[i] / alls[i] for i, label in enumerate(labels)]
        discard_labels = [label for i, label in enumerate(labels) if percentage[i] > 0.6]
        remain_labels = [label for label in labels if label not in discard_labels]

        if len(discard_labels) > 0:
            insert_label = int(np.min(discard_labels))
            insert_loc = labels.index(insert_label)
        else:
            insert_label = int(np.max(labels)) + 1
            insert_loc = len(labels)

        final_idxes = idxes

        self.gmm_result['classes'][final_idxes] = -1
        X = self.X[final_idxes]
        weights_init = [self.gmm_result['distribution'][label]['weight'] for label in remain_labels]
        weights_init.insert(insert_loc, len(final_idxes) / len(selected_classes))
        weights_init = np.array(weights_init) / np.sum(weights_init)
        means_init = [self.gmm_result['distribution'][label]['mu'] for label in remain_labels]
        means_init.insert(insert_loc, np.mean(X, axis=0))
        precision_init = [self.gmm_result['distribution'][label]['sinv'] for label in remain_labels]
        precision_init.insert(insert_loc, np.linalg.inv(np.cov(X.transpose())))
        self.labels = list(remain_labels)
        self.labels.insert(insert_loc, insert_label)

        self.gmm_result = self.build_gmm(self.X, weights_init, means_init, precision_init)
        self.recover_labels()

        # self.M_step()
        self.update_idx()

    def adjust_component(self, idxes):
        old_classes = self.gmm_result['classes'].copy()
        selected_classes = old_classes[idxes]
        labels = list(self.gmm_result['distribution'].keys())
        freq = np.array([np.sum(selected_classes == label) for label in labels])
        alls = np.array([len(self.gmm_result['distribution'][label]['idx']) for label in labels])
        percentage = [freq[i] / alls[i] for i, label in enumerate(labels)]
        discard_labels = [label for i, label in enumerate(labels) if percentage[i] > 0.8]
        remain_labels = [label for label in labels if label not in discard_labels]

        final_idxes = set(idxes)
        for label in discard_labels:
            final_idxes = final_idxes.union(set(np.argwhere(old_classes == label).reshape(-1)))
        final_idxes = list(final_idxes)

        logger.debug(f"Incremental: build new gmm for new data: {len(final_idxes)}")
        old_distribution = {key:self.gmm_result['distribution'][key] for key in remain_labels}
        new_distribution = self.build_gmm(self.X[final_idxes])['distribution']
        weights_init = [old_distribution[key]['weight'] * (self.n - len(final_idxes)) for key in old_distribution] +\
                        [new_distribution[key]['weight'] * len(final_idxes) for key in new_distribution]
        weights_init = np.array(weights_init) / np.sum(weights_init)
        means_init = [old_distribution[key]['mu'] for key in old_distribution] +\
                        [new_distribution[key]['mu'] for key in new_distribution]
        precision_init = [old_distribution[key]['sinv'] for key in old_distribution] +\
                            [new_distribution[key]['sinv'] for key in new_distribution]
        old_max_label = int(np.max(list(old_distribution.keys()))) + 1

        new_key_map = dict()
        percentage_matrix=np.zeros((len(new_distribution.keys()), len(discard_labels)))
        if len(new_distribution.keys()) > 0 and len(discard_labels) > 0:
            for key in new_distribution:
                freq = np.array([np.sum(old_classes == label) for label in discard_labels])
                if freq.sum() > 0:
                    percentage_matrix[key] = freq.sum()
            while percentage_matrix.max() > 0:
                loc = percentage_matrix.argmax()
                row = loc // len(discard_labels)
                col = loc % len(discard_labels)
                new_key_map[row] = discard_labels[col]
                percentage_matrix[:,col] = 0
                percentage_matrix[row, :] = 0
        for key in new_distribution:
            if key not in new_key_map:
                new_key_map[key] = old_max_label
                old_max_label += 1

        self.labels = list(old_distribution.keys()) + [new_key_map[l] for l in new_distribution.keys()]
        self.gmm_result = self.build_gmm(self.X, weights_init, means_init, precision_init)
        self.recover_labels(old_max=max(old_distribution.keys()))
        self.calculate_threshold()
        self.update_idx()

    def slight_adjust_component(self, idxes):
        selected_classes = self.gmm_result['classes'][idxes]
        labels = list(self.gmm_result['distribution'].keys())
        freq = np.array([np.sum(selected_classes == label) for label in labels])
        major_label = labels[freq.argmax()]

        for idx in idxes:
            if idx in self.gmm_result['uncertain_ids']:
                shift_component(self.gmm_result['distribution'][major_label], self.X[idx], 3)
                del self.gmm_result['uncertain_ids'][self.gmm_result['uncertain_ids'].index(idx)]
            else:
                if self.gmm_result['classes'][idx] != major_label:
                    shift_component(self.gmm_result['distribution'][major_label], self.X[idx], 3)
                    shift_component(self.gmm_result['distribution'][self.gmm_result['classes'][idx]], self.X[idx], -3)

        self.gmm_result['classes'][idxes] = major_label


        labels = list(self.gmm_result['distribution'].keys())
        freq = np.array([np.sum(self.gmm_result['classes'] == label) for label in labels])
        for i, label in enumerate(labels):
            if freq[i] / freq.sum() < 0.01 or freq[i] < 10:
                self.delete_component(label)
        labels = list(self.gmm_result['distribution'].keys())
        for i in range(self.X.shape[0]):
            distances = np.array([gaussian_log_prob(self.X[i], self.gmm_result['distribution'][label]['mu'],
                                                    self.gmm_result['distribution'][label]['sinv'],
                                                    self.gmm_result['distribution'][label]['log_det'])
                                  - self.gmm_result['distribution'][label]['log_weight']
                                  for label in labels])
            min_idx = distances.argmin()
            self.gmm_result['classes'][i] = labels[min_idx]
        self.gmm_result['classes'][idxes] = major_label
        self.update_idx()
        self.calculate_threshold()

    def delete_component(self, d_label):
        idxes = np.argwhere(self.gmm_result['classes'] == d_label).flatten()
        distribution = self.gmm_result['distribution']
        del distribution[d_label]
        preserve_labels = [label for label in distribution]
        for i in idxes:
            log_ps = [distribution[label]['log_weight'] +
                      gaussian_true_log_prob(self.X[i], distribution[label]['mu'], distribution[label]['sinv'],
                                             distribution[label]['log_det'])
                      for label in preserve_labels]
            ps = softmax(log_ps)
            max_idx = ps.argmax()
            for k, label in enumerate(preserve_labels):
                shift_component(distribution[label], self.X[i], ps[k])
            self.gmm_result['classes'][i] = preserve_labels[max_idx]
        self.gmm_result['distribution'] = distribution
        self.gmm_result['preserve_labels'] = preserve_labels

    @staticmethod
    def extract_gmm_info(model):
        keys = []
        keys_tolist = ['idx', ]
        return {**{key: model[key] for key in keys}, **{key: model[key].tolist() for key in keys_tolist}}

    def to_json(self):
        keys = list(self.gmm_result['distribution'].keys())
        return {
            'keys': keys,
            'components': {key: GMMResult.extract_gmm_info(self.gmm_result['distribution'][key]) for key in keys}
        }

    def get_idxes(self):
        return [self.gmm_result['distribution'][key]['idx'] for key in self.gmm_result['distribution']]

    def get_classes(self):
        return self.gmm_result['classes']

    def _build_gmm(self, X, method, n_components=10):
        if method not in ['bayesian', 'BIC']:
            raise NotImplementedError(f"Please implement the method {method}")

        if method == 'bayesian':
            gmm = mixture.BayesianGaussianMixture(n_components=n_components, weight_concentration_prior=1e-2,
                                                    mean_precision_prior=1e-2, max_iter=gmm_max_iter, n_init=5)
            gmm.fit(X)
            return gmm

        if method == 'BIC':
            models = []
            max_k = min(n_components, len(X) // 25 + 1)
            for k in range(1, max_k+1):
                model = mixture.GaussianMixture(n_components=k, max_iter=gmm_max_iter, n_init=5, random_state=self.rnd)
                model.fit(X)
                models.append(model)
            BICs = [model.bic(X) for model in models]
            return models[np.argmin(BICs)]