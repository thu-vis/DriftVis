import json
import os
import re
import uuid
import pickle
from joblib import dump, load
import numpy as np
import pandas as pd
import math
from flask import Blueprint, request
from flask import jsonify
from scipy.sparse import load_npz

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
#import lightgbm
from functools import reduce

from .libs.detect import ED_Detector, cache_gmm_model, sumup, drift_test
from .libs.scatterplot import TSNE
from .libs.util import Origin, pos_average, get_percentage, move_avg_prev, basic_stat
from .libs.ensemble import Ensemble
from .libs.gmmresult import GMMResult
from .libs.adapt import train_single_model, pointwise_test_ensemble_model, pointwise_classify
from .libs.history import History

data = Blueprint("data", __name__)
data_dir = "data/"

SVD_per_chunk = 500

DataConfig = {
    'synthetic.csv': {
        'attributes': ['x1', 'x2'],
        'default_method': 'ED',
        'label': 'y',
        'current_len': 500,
        'next_step': 500,
        'skip_step': 1,
        'normalized': False,
        'k-fold': 200,
        'gmm_thres': 200,
        'test_size': 500,
    },
    'weather.csv': {
        'attributes': ['temp', 'seaLevelPressure', 'visibility', 'avgWind',
                         'dewPoint', 'maxWind', 'maxTemp', 'minTemp'],
        'model_attributes': reduce(lambda x,y:x+y,
        [[x+(str(y) if y<0 else '') for x in ['temp', 'seaLevelPressure', 'visibility', 'avgWind',
                         'dewPoint', 'maxWind', 'maxTemp', 'minTemp']] for y in range(-2, 1)]),
        #'attributes': ['temp', 'seaLevelPressure', 'visibility', 'avgWind'],
        #'model_attributes': reduce(lambda x,y:x+y,
        #    [[x+(str(y) if y<0 else '') for x in ['temp', 'seaLevelPressure', 'visibility', 'avgWind',
        #                         'avgWind', 'maxWind', 'maxTemp', 'minTemp']] for y in range(-2, 1)]),
        'default_method': 'ED',
        'label': 'rain',
        'skip': #5628,
            #5985 + 6,
            #10005,
            #14015,
            150,
        #'current_len': 365,
        'current_len': 90,
        #'next_step': [[1]*15,[1]*15,[1]*22,[1], [1]*10, [1]*17, [1]*21,[1], [1]*15, [1]*13, [1]*24, [1]*80, [1]*41, [1]*30, [1]*75] + [[1]*365]*(365),
        'next_step': [[1]*15,[1]*15,[1]*23, [1]*10, [1]*17, [1]*22, [1]*15, [1]*13, [1]*24, [1]*80, [1]*41, [1]*30, [1]*75, [1]*260, [1]*365, [1]*40, [1]*105, [1]*220, [1]*365] + [[1]*365]*6,
        #'next_step': [[365]]*5,
        'skip_step': 1,
        'win_size': 30,
        'win_size_min': 7,
        'win_size_max': 365,
        'origin_win_size': 7,
        'normalize': False,
        #'model': lightgbm.LGBMClassifier,
        #'model_para': { 'boosting_type': 'gbdt',
        #               'max_depth': 6, 'num_leaves': 25, 'n_estimators': 50,
        #               'min_child_samples': 20, 'colsample_bytree': 0.1, 'class_weight':'balanced',
        #               'objective': 'binary', 'metric': 'binary_logloss', 'silent': True},
        'model': #LogisticRegression,
            LogisticRegressionCV,
        'model_para': {
            'cv': 5,
            'class_weight': 'balanced',
            'solver': 'liblinear',
            'max_iter': 50 },
        'k-fold': 200,
        'normalized': True,
        'train_test_sample_rate': 0.7,
        'fit_by_cv': False,
        'gmm_thres': 35,
        #'gmm_thres': 300,
        'test_size': 60,
        'all_size': 3000,
    },
    'paper.csv': {
        'attributes': [f'dim{i}' for i in range(10)],
        'default_method': 'ED',
        'label': 'label',
        'win_size': 500,
        'current_len': 152 + 197 + 207 + 198,
        #'next_step': [207 + 207 + 204 + 217, 250 + 262 + 292 + 306, 368 + 360 + 411 + 403, 569 + 679 + 1009 + 1428],
        'next_step': [[207, 207, 204, 217], [250, 262, 292, 306], [368, 360, 411, 403], [569, 679, 1009], [1428]],
        #'next_step': [[207], [207], [204], [217], [250], [262], [292], [306], [368], [360], [411], [403], [569], [679], [1009], [1428]],
        'skip_step': 1,
        'origin': 'text',
        'mul': 10,
        'keymap': {
            'dim0': 'distribution,feature',
            'dim1': 'image,object',
            'dim2': 'policy,agent',
            'dim3': 'inference,bayesian',
            'dim4': 'node,tree',
            'dim5': 'kernel,neuron',
            'dim6': 'graph,node',
            'dim7': 'regret,bound',
            'dim8': 'matrix,rank',
            'dim9': 'regret,bandit',
        },
        'tfidf': 'tfidf.npz',
        'tfidf_name': 'fname.txt',
        'model': SVC,
        'model_para': {'C': 4, 'kernel': 'rbf', 'gamma': 2**-3, 'probability': True, 'random_state': 42},
        'normalized': False,
        'gmm_thres': 300,
        'test_size': 500,
    },
}

session = {}

@data.route('/setDataset', methods=['GET', 'POST'])
def set_dataset():
    global session
    session = {}
    np.random.seed(2020)
    post = json.loads(request.get_data())
    # load basic settings
    session['dataset'] = post['dataset'] + '.csv'
    tempConfig = DataConfig[session['dataset']]
    session['model'] = tempConfig['model']
    session['model_para'] = tempConfig['model_para']
    session['label'] = tempConfig['label']
    session['GMM_attribute'] = tempConfig['attributes'].copy()
    session['scatter_attribute'] = tempConfig['attributes'].copy()
    session['attributes'] = tempConfig['attributes'].copy()
    if 'model_attributes' in tempConfig:
        session['model_attribute'] = tempConfig['model_attributes'].copy()
    else:
        session['model_attribute'] = tempConfig['attributes'].copy()
    if 'keymap' in tempConfig:
        keymap = tempConfig['keymap']
        session['GMM_attribute'] = list(map(lambda x:keymap[x], session['GMM_attribute']))
        session['scatter_attribute'] = list(map(lambda x:keymap[x], session['scatter_attribute']))
        session['attributes'] = list(map(lambda x:keymap[x], session['attributes']))
        session['model_attribute'] = list(map(lambda x:keymap[x], session['model_attribute']))

    session['line_plot_attribute'] = []
    session['method_name'] = 'ED'
    session['test_size'] = tempConfig['test_size']
    session['win_size'] = tempConfig['win_size'] if 'win_size' in tempConfig else 100
    session['win_size_min'] = tempConfig['win_size_min'] if 'win_size_min' in DataConfig[
        session['dataset']] else 0
    session['win_size_max'] = tempConfig['win_size_max'] if 'win_size_max' in DataConfig[
        session['dataset']] else 300
    session['origin_win_size'] = tempConfig['origin_win_size'] if 'origin_win_size' in tempConfig else 25
    for attr in session['attributes']:
        session['line_plot_attribute'].append(f'{attr}_Origin')
        session['line_plot_attribute'].append(f'{attr}_{session["method_name"]}')
        #session['line_plot_attribute'].append(f'{attr}_{session["method_name"]}2')
    session['line_plot_attribute'].append(session["method_name"])
    #session['line_plot_attribute'].append(session["method_name"] + '2')
    # init cache
    session['cache'] = dict()
    session['cache']['line_data'] = dict()
    session['cache_result'] = dict()
    session['detector'] = dict()
    # load data
    session['data'] = pd.read_csv(data_dir + session['dataset'])

    if 'tfidf' in tempConfig:
        session['tfidf'] = np.array(load_npz(data_dir + tempConfig['tfidf']).todense())
        session['tfidf'] = session['tfidf'][session['data']['idx'],]
        with open(data_dir + tempConfig['tfidf_name']) as f:
            session['tfidf_name'] = json.load(f)
        if 'skip' in tempConfig:
            session['tfidf'] = session['tfidf'][tempConfig['skip']:]
        if 'shuffle_idx' in tempConfig:
            session['tfidf'] = session['tfidf'][tempConfig['shuffle_idx'],:]

    if 'visibility' in session['data'].columns:
        session['data']['visibility'] += np.random.normal(0,1e-4,len(session['data']))
    if 'normalize' in tempConfig and tempConfig['normalize']:
        for attr in session['attributes']:
            session['data'][attr] = (session['data'][attr] - session['data'][attr].min()) / (session['data'][attr].max() - session['data'][attr].min())
    if 'keymap' in tempConfig:
        keymap = tempConfig['keymap']
        for key in keymap:
            session['data'][keymap[key]] = session['data'][key]
            session['data'].drop(columns=key, inplace=True)

    if 'mul' in tempConfig:
        for attr in session['GMM_attribute']:
            session['data'][attr] *= tempConfig['mul']

    if 'skip' in tempConfig:
        session['data'] = session['data'].iloc[range(tempConfig['skip'], len(session['data']))].reset_index(drop=True)
    all_size = tempConfig['all_size'] if 'all_size' in tempConfig else len(session['data'])
    session['data'] = session['data'].iloc[range(0, all_size, tempConfig['skip_step'])].reset_index(drop=True)

    if 'shuffle_idx' in tempConfig:
        session['data'] = session['data'].iloc[tempConfig['shuffle_idx']].reset_index(drop=True)

    session['data'] = session['data'].fillna(method='ffill')


    session['train_test_sample_rate'] = 0.6 if tempConfig.get('train_test_sample_rate') is None \
        else tempConfig['train_test_sample_rate']
    if tempConfig.get('k-fold') is None:
        np.random.seed(100)
        session['k-fold'] = 3
        for i in range(session['k-fold']):
            session[f'k-fold-{i}'] = np.random.choice([True, False], size=len(session['data']),
                                                      p=[session['train_test_sample_rate'],
                                                         1 - session['train_test_sample_rate']])
        #session['k-fold'] = 1
        #session[f'k-fold-0'] = np.random.choice([True, False], size=len(session['data']),
        #                                        p=[session['train_test_sample_rate'], 1-session['train_test_sample_rate']])
    else:
        session['k-fold'] = tempConfig['k-fold']
        tmp = np.load('./data/flag.npy')
        for i in range(session['k-fold']):
             session[f'k-fold-{i}'] = tmp[i, :]
        #session['k-fold'] = tempConfig['k-fold']
        #tempSkip = 0
        #idxs = [tempConfig['current_len']]
        #nextStep = tempConfig['next_step']
        #idxs.extend([90 for i in range(tempSkip + tempConfig['current_len'], len(session['data']), 90)])
        ##idxs.extend(list(map(lambda x:sum(x), nextStep)))
        #startIdxs = [sum(idxs[:i]) for i in range(len(idxs))]
        #def _temp_gen_selected_straified(data):
        #    train_idxs, test_idxs, _, __ = train_test_split(
        #        [x for x in range(len(data['rain']))], data['rain'], test_size=(1-session['train_test_sample_rate']),
        #        stratify=data['rain'])
        #    dic = {x:True for x in train_idxs}
        #    dic.update({x:False for x in test_idxs})
        #    return np.array([dic[x] for x in range(len(data))])
        ## generate train_test to keep rate in each batch
        #for i in range(session['k-fold']):
        #    tempSelectedFlag = np.zeros(len(session['data']), dtype=bool)
        #    tempFlag = np.concatenate([_temp_gen_selected_straified(session['data'][st: st+idxs[j]])
        #            for j, st in enumerate(startIdxs)], axis=0)
        #    tempSelectedFlag[tempSkip:tempSkip+tempFlag.shape[0]] = tempFlag
        #    session[f'k-fold-{i}'] = tempSelectedFlag
        #tmp = np.zeros((session['k-fold'], len(session['data'])), dtype=bool)
        #for i in range(session['k-fold']):
        #    tmp[i,:] = session[f'k-fold-{i}']
        #np.save('./data/flag.npy',tmp)
        # Original method
        #for i in range(session['k-fold']):
        #    session[f'k-fold-{i}'] = np.random.choice([True, False], size=len(session['data']),
        #                                            p=[session['train_test_sample_rate'], 1-session['train_test_sample_rate']])

    if session['label'] == None:
        session['label'] = "NoLabel"
        session['data'][session['label']] = 0
    if 'timestamp' not in session['data'].columns.tolist():
        session['data']['timestamp'] = pd.date_range(start='1/1/2000', periods=len(session['data']), freq='h').astype(str)
    session['total_count'] = len(session['data'].index)
    session['init_len'] = tempConfig['current_len']
    session['current_len'] = tempConfig['current_len']
    session['range'] = [i for i in range(session['current_len'])]
    session['new_idx'] = session['range']
    session['X'] = np.array(session['data'][session['GMM_attribute']])
    session['trainX'] = np.array(session['data'][session['model_attribute']])
    session['y'] = np.array(session['data'][session['label']])
    if tempConfig['normalized']:
        print('Normalize trainX.')
        session['trainX'] = preprocessing.scale(session['trainX'])
    if session['y'].dtype == 'bool':
        session['y'] = np.array([session['label'] if x else 'not '+session['label'] for x in session['y']])
    session['display_attr'] = [l for l in session['GMM_attribute']]
    if len(np.unique(session['y'])) < len(session['y']) / 2:
        session['classes'] = np.unique(session['y'])
    else:
        session['classes'] = None
    if 'origin' in tempConfig:
        attr = tempConfig['origin']
        # session['display_attr'].append(attr)
        session['data'][attr] = [l.replace('"', "'") for l in session['data'][attr]]
    # init data chunk
    session['data_chunk'] = [{
        'id': 0,
        'drift_value': 0.0,
        'latest': False,
        'idx': session['new_idx'].copy(),
        'count': session['current_len'],
        # 'gmm': GMMResult.build_gmm(session['X'].iloc[session['new_idx']])
    }, {
        'id': 1,
        'idx': session['new_idx'].copy(),
        'count': session['current_len'],
        'drift_value': 0.0,
        'latest': False,
        # 'gmm': GMMResult.build_gmm(session['X'].iloc[session['new_idx']])
    }]
    session['data_chunk_id'] = 2
    # init gmm
    session['gmm_result'] = GMMResult(session['GMM_attribute'], session['data'], session['range'], gmm_thres=tempConfig['gmm_thres'])

    # init model
    session['ensemble'] = Ensemble()
    #session['ensemble'].add_model(train_single_model(session['trainX'], session['y'], session['range'], session['classes'],
    #                                                session[f'k-fold-0'], session['model'], session['model_para']))
    # init detector
    session['detector']['ED'] = ED_Detector(session['GMM_attribute'], session['data'], session['range'], session['win_size'])
    for tmp in session['line_plot_attribute']:
        if '_' in tmp:
            attr, method = tmp.split('_')
            if method == 'ED':
                session['detector'][tmp] = ED_Detector([attr], session['data'], session['range'], session['win_size'])

    session['scatter'] = TSNE(session['scatter_attribute'], session['data'], session['range'])
    session['history'] = dict()

    QUICK_MODE = False
    if QUICK_MODE:
        tempConfig['next_step'] = list(map(lambda x:[sum(x)], tempConfig['next_step']))

    USE_CACHE_LOAD = True
    cache_file = './data/weather_init' if session['dataset'] == 'weather.csv' else './data/paper_init'
    if USE_CACHE_LOAD and cache_file:
        with open(cache_file, 'rb') as fin:
            history = pickle.load(fin)
            history.apply(session)
            adapt(0)
    else:
        session['ensemble'].add_model(get_single_model(session['range']))
        session['ensemble'].use_all()
        session['gmm_result'].build_initial_gmm()
        session['gmm_label'] = session['gmm_result'].get_gmm_result()['classes'].tolist()
        session['scatter'].gmm_result = session['gmm_result'].get_gmm_result()
        session['scatter'].layout()
        session['stored_ensemble'] = [None, [model['id'] for model in session['ensemble'].get_all_used_model()]]
        adapt(0)

    return jsonify({
        'win_size': session['win_size'],
        'win_size_min': session['win_size_min'],
        'win_size_max': session['win_size_max'],
        'timestamp': session['data']['timestamp'].to_json(orient='records'),
        'method_name': session['method_name'],
        'attributes': session['attributes'],
        'drift_keys': session['line_plot_attribute'],
        'total_count': session['total_count'],
        'current_len': session['current_len'],
        'default_method': tempConfig['default_method'],
        'models': get_models(),
        'chunks': get_chunks(),
        'gmm_label': session['gmm_label'],
        'distributions': session['gmm_result'].to_json()
    })


@data.route('/linePlotData', methods=['GET', 'POST'])
def line_plot_data():
    global session
    return jsonify(session['cache']['line_data'])


@data.route('/scatterPlotData', methods=['GET', 'POST'])
def scatter_plot_data():
    global session
    return jsonify(session['scatter'].get_layout())


@data.route('/getLabel', methods=['GET', 'POST'])
def get_label():
    global session
    post = json.loads(request.get_data())
    use_gmm_label = post['use_gmm_label']
    if use_gmm_label:
        return jsonify({'label': session['gmm_label']})
    else:
        return jsonify({'label': session['y'][session['range']].tolist(),
                        'unique_label': session['classes'].tolist()})
        # return jsonify({'label': [1 if x else 0 for x in session['y'][session['range']]]})


@data.route('/addModel', methods=['GET', 'POST'])
def add_model():
    global session
    post = json.loads(request.get_data())
    idx = post['idx']
    hash_id = hash(tuple(idx)) % 100000007
    #np.save(f'cache/training{len(idx)}_{hash_id}.npy', np.array(idx))
    if session['ensemble'].search_model(hash_id):
        return '', 204
    #model = train_single_model(session['trainX'],  session['y'], idx, session['classes'],
    #                           session[f'k-fold-0'], session['model'], session['model_para'])
    model = get_single_model(idx)
    session['ensemble'].add_model(model)
    model_info = Ensemble.extract_model_info(model)
    model_info['gaussian_percentage'] = get_percentage(model_info['idx'], session['gmm_result'].get_idxes())
    return jsonify({'model': model_info})


def set_first_chunk(idx):
    global session
    chunk = {
        'id': 0,
        'idx': idx,
        'count': len(idx),
        'gaussian_percentage': get_percentage(idx, session['gmm_result'].get_idxes()),
        'model_percentage': get_model_percentage(idx),
        'latest': False,
    }
    chunk['drift'] = get_drift_score(chunk)
    session['data_chunk'][0] = chunk


@data.route('/singleChunkInfo', methods=['GET', 'POST'])
def single_chunk_info():
    global session
    post = json.loads(request.get_data())
    idx = post['idx']
    set_first_chunk(idx)
    return jsonify(session['data_chunk'][0])


def get_chunks():
    global session
    # drift_vals = np.array(session['cache']['line_data'][session["method_name"]])
    return [{
        'id': chunk['id'],
        'idx': chunk['idx'],
        'count': chunk['count'],
        'gaussian_percentage': get_percentage(chunk['idx'], session['gmm_result'].get_idxes()),
        # 'model_percentage': get_percentage(chunk['idx'], [model['idx'] for model in session['ensemble'].get_all_used_model()]),
        'model_percentage': get_model_percentage(chunk['idx']),
        # 'drift': pos_average(drift_vals[chunk['idx']]),
        'drift': get_drift_score(chunk),
    } for i, chunk in enumerate(session['data_chunk'])]


def get_models():
    global session
    ret = session['ensemble'].to_json()
    for model_info in ret:
        model_info['gaussian_percentage'] = get_percentage(model_info['idx'], session['gmm_result'].get_idxes())
    return ret

def get_single_model(idxes):
    return train_single_model(
            session['trainX'], session['y'], idxes, session['classes'],
            [session[f'k-fold-{i}'] for i in range(session['k-fold'])], session['model'],
            session['model_para'], DataConfig[session['dataset']].get('fit_by_cv'))

@data.route('/updateTable', methods=['GET', 'POST'])
def update_table():
    return jsonify({'chunks': get_chunks(), 'models': get_models()})


@data.route('/getDistributions', methods=['GET', 'POST'])
def get_distribution():
    global session
    return jsonify({'distributions': session['gmm_result'].to_json()})


@data.route('/nextData', methods=['GET', 'POST'])
def next_data():
    global session
    new_data_len = sum(DataConfig[session['dataset']]['next_step'][session['data_chunk_id']-2])
    session['range'] = [i for i in range(session['current_len'] + new_data_len)]
    session['new_idx'] = [i for i in range(session['current_len'], session['current_len'] + new_data_len)]
    session['data_chunk'].append({
        'id': session['data_chunk_id'],
        'idx': session['new_idx'].copy(),
        'count': new_data_len,
        'drift_value': 0.0,
        'latest': False,
        # 'gmm': GMMResult.build_gmm(session['X'].iloc[session['new_idx']])
    })

    for min_batch in DataConfig[session['dataset']]['next_step'][session['data_chunk_id']-2]:
        new_idx = [i for i in range(session['current_len'], session['current_len'] + min_batch)]
        session['current_len'] += min_batch
        session['gmm_result'].add_data(session['data'], new_idx)
        session['gmm_label'] = session['gmm_result'].get_gmm_result()['classes'].tolist()
        for detector in session['detector']:
            session['detector'][detector].add_data(session['data'], new_idx)
        adapt(-1)
        session['scatter'].add_data(session['data'], new_idx, session['gmm_result'].get_gmm_result())


    test_size = max(session['test_size'], sum(DataConfig[session['dataset']]['next_step'][session['data_chunk_id']-2]))
    set_first_chunk(list(range(session['current_len'] - test_size, session['current_len'])))

    session['data_chunk_id'] += 1
    distribution = session['gmm_result'].get_gmm_result()['distribution']
    #save_history()
    return jsonify({
        'current_len': session['current_len'],
        'gmm_label': session['gmm_label'],
    })


@data.route('/updateModel', methods=['GET', 'POST'])
def update_model():
    global session
    post = json.loads(request.get_data())
    model_id = post['id']
    all_use = post['all_use']
    #session['ensemble'].toggle(model_id)
    for model in session['ensemble'].get_all_model():
        model['use'] = all_use[str(model['id'])]
    return '', 200


@data.route('/cache', methods=['GET', 'POST'])
def cache():
    global session
    # session['cache']['GMM'] = []
    # session['cache_result']['GMM'] = []
    # session['cache_result']['GMM'], session['cache']['GMM'] = GMM(session['data'].iloc[[i for i in range(5600)]][session['GMM_attribute']],
    #                                                               100, 9, session['cache']['GMM'], session['cache_result']['GMM'])
    # dump(session['cache']['GMM'], 'cache/cache_GMM_100_9')
    # dump(session['cache_result']['GMM'], 'cache/cache_result_GMM')
    # dump(cache_gmm_model(session['X'], 100, 5), 'cache/cache_GMM_100_5')
    # dump(cache_gmm_model(session['X'], 200, 5), 'cache/cache_GMM_200_5')
    return '', 200


@data.route('/precompute', methods=['GET', 'POST'])
def precompute():
    # session['cache']['GMM'] = load('cache/cache_GMM_100_9')
    # session['cache_result']['GMM'] = load('cache/cache_result_GMM')
    # session['cache']['GMM'] = []
    # session['cache_result']['GMM'] = []
    # session['cache_result']['GMM'], session['cache']['GMM'] = GMM(session['data'].iloc[session['range']][DataConfig[session['dataset']]['GMM_attribute']],
    #                                                               100, 9, session['cache']['GMM'], session['cache_result']['GMM'])
    return '', 200


@data.route('/adjustComponent', methods=['GET', 'POST'])
def adjust_component():
    global session
    post = json.loads(request.get_data())
    idxes = post['idxes']
    #session['gmm_result'].adjust_component_merge(idxes)
    session['gmm_result'].slight_adjust_component(idxes)
    session['gmm_label'] = session['gmm_result'].get_gmm_result()['classes'].tolist()
    adapt(0)
    return jsonify({
        'chunks': get_chunks(),
        'models': get_models(),
        'gmm_label': session['gmm_label'],
        'line_data': session['cache']['line_data'],
    })


@data.route('/adapt', methods=['GET', 'POST'])
def api_adapt():
    current_model_ids = [model['id'] for model in session['ensemble'].get_all_used_model()]
    if current_model_ids == session['stored_ensemble'][1] or len(current_model_ids) == 0:
        return '', 200
    session['stored_ensemble'].pop(0)
    session['stored_ensemble'].append(current_model_ids)
    adapt(0)
    for chunk in session['data_chunk']:
        chunk['latest'] = False
    return '', 200


def adapt(prev_n):
    '''
    :param prev_n: will not update first n points. n = -1 means keep all.
    '''
    global session
    session['point_model_distance'] = session['ensemble'].generate_distance_for_points(session['trainX'], list(range(session['current_len'])))
    session['model_labels'] = np.argmin(session['point_model_distance'], axis=1)
    ret = dict()
    ret['keys'] = []
    for tmp in session['line_plot_attribute']:
        ret['keys'].append(tmp)
        if '_' in tmp:
            attr, method = tmp.split('_')
            if method == 'ED':
                if session['dataset'] == 'paper.csv':  # calculate drift for batch instead of sliding window
                    ret[tmp] = np.zeros(session['current_len'])
                    if prev_n == -1:
                        previous_len = len(session['cache']['line_data'][tmp])
                        idx = list(range(previous_len))
                        ret[tmp][idx] = np.array(session['cache']['line_data'][tmp])[idx]
                    else:
                        previous_len = 0
                    chunks = [session['init_len']]
                    for chunk in DataConfig[session['dataset']]['next_step']:
                        chunks += chunk
                    i = 0
                    while i < previous_len:
                        i += chunks.pop(0)
                    while i < session['current_len']:
                        idx = list(range(i, i + chunks[0]))
                        ret[tmp][idx] = session['detector'][tmp].get_drift_value(session['ensemble'].get_all_used_model(), idx, session['gmm_label'])
                        i += chunks.pop(0)
                    ret[tmp] = ret[tmp].tolist()
                else:
                    ret[tmp] = session['detector'][tmp].detect(session['ensemble'].get_all_used_model(), session['model_labels'], prev_n,gmm_labels=session['gmm_label'])
            if method == 'Origin':
                ret[tmp] = Origin(session['data'][attr][session['range']], session['origin_win_size'], session['data'][attr].min(),
                                  session['data'][attr].max())
        else:
            method = tmp
            if method == 'ED':
                if session['dataset'] == 'paper.csv':  # calculate drift for batch instead of sliding window
                    ret[tmp] = np.zeros(session['current_len'])
                    if prev_n == -1:
                        previous_len = len(session['cache']['line_data'][tmp])
                        idx = list(range(previous_len))
                        ret[tmp][idx] = np.array(session['cache']['line_data'][tmp])[idx]
                    else:
                        previous_len = 0
                    chunks = [session['init_len']]
                    for chunk in DataConfig[session['dataset']]['next_step']:
                        chunks += chunk
                    i = 0
                    while i < previous_len:
                        i += chunks.pop(0)
                    while i < session['current_len']:
                        idx = list(range(i, i + chunks[0]))
                        ret[tmp][idx] = session['detector'][tmp].get_drift_value(session['ensemble'].get_all_used_model(), idx, session['gmm_label'])
                        i += chunks.pop(0)
                    ret[tmp] = ret[tmp].tolist()
                else:
                    ret[tmp] = session['detector']['ED'].detect(session['ensemble'].get_all_used_model(), session['model_labels'], prev_n,gmm_labels=session['gmm_label'])
    ret['ED'] = np.concatenate([ret['ED']]+[ret[f'{attr}_ED'] for attr in session['attributes']]).reshape(len(session['attributes'])+1,-1).max(axis=0).tolist()
    _, idx = np.unique(np.array(ret['ED']), return_index=True)
    print("drift:", np.array(ret['ED'])[np.sort(idx)])
    session['cache']['line_data'] = ret


@data.route('/changeWinSize', methods=['GET', 'POST'])
def change_win_size():
    global session
    post = json.loads(request.get_data())
    session['win_size'] = post['win_size']
    for key in session['detector']:
        session['detector'][key].change_win_size(session['win_size'])
    adapt(0)
    return '', 200


@data.route('/getPredictStat', methods=['POST'])
def get_predict_stat():
    global session
    post = json.loads(request.get_data())
    idx = post['idx']
    if len(idx) > 0:
        if session['ensemble'].get_all_used_model()[0]['model']:
            result1 = pointwise_test_ensemble_model(session['ensemble'].get_all_used_model(), session['trainX'], session['y'], idx)
        else:
            result1 = dict()
        result2 = basic_stat(session['GMM_attribute'], session['X'], session['y'], idx)
        return jsonify({#'drift': get_drift_score(idx),
          **result1, **result2 })
    else:
        return jsonify(test())

@data.route('/getGridOrigin', methods=['POST'])
def get_grid_origin():
    global session
    post = json.loads(request.get_data())
    idxes = post['idxes']
    values = []
    ret = {'idxes': idxes}
    for idx in idxes:
        valueDic = {}
        for attr in session['display_attr']:
            valueDic[attr] = session['data'][attr][idx]
        values.append(valueDic)
    ret['values'] = values
    return jsonify(ret)


@data.route('/getOrigin', methods=['POST'])
def get_origin():
    global session
    post = json.loads(request.get_data())
    idx = post['idx']
    idxes = post['idxes']
    ret = {'idx': idx}
    if len(idxes) > 0:
        m = session['data'].iloc[post['idxes']][session['GMM_attribute']].mean()
        minV = session['data'].iloc[post['idxes']][session['GMM_attribute']].min()
        maxV = session['data'].iloc[post['idxes']][session['GMM_attribute']].max()
        meanDic = {}
        minDic = {}
        maxDic = {}
        valueDic = {}
        for attr in session['display_attr']:
            #if attr in m:
            #    ret[f'{attr}_diff'] = session['data'][attr][idx] - m[attr]
            #else:
            #    ret[attr] = session['data'][attr][idx]
            if attr in m:
                valueDic[attr] = session['data'][attr][idx]
                if attr in m:
                    meanDic[attr] = m[attr]
                    maxDic[attr] = maxV[attr]
                    minDic[attr] = minV[attr]
        ret['min'] = minDic
        ret['max'] = maxDic
        ret['mean'] = meanDic
        ret['value'] = valueDic
        if 'tfidf' in session:
            ret['current_batch_tfidf'] = calculate_tfidf(idxes)
            top_words = [pair[0] for pair in ret['current_batch_tfidf']][:10]
            top_words_idx = [session['tfidf_name'].index(word) for word in top_words]
            titles = []
            for i in idxes:
                score = session['tfidf'][i, top_words_idx].sum()
                titles.append([session['data']['text'][i], score])
            ret['titles'] = [[k, v] for k, v in sorted(titles, key=lambda item: item[1])][::-1][:3]
    else:
        valueDic = {}
        for attr in session['display_attr']:
            #ret[attr] = session['data'][attr][idx]
            valueDic[attr] = session['data'][attr][idx]
        if 'tfidf' in DataConfig[session['dataset']]:
            valueDic['text'] = session['data']['text'][idx] + session['data']['abstract'][idx]
        ret['value'] = valueDic
    return jsonify(ret)


def test():
    global session
    if 'test_data' not in session:
        return '', 200
    return '',200
    if session['ensemble'].get_all_used_model()[0]['model']:
        result1 = pointwise_test_ensemble_model(session['ensemble'].get_all_used_model(),
                                                session['test_data'][session['GMM_attribute']],
                                                session['test_data'][session['label']],
                                                [i for i in range(len(session['test_data']))])
    else:
        result1 = dict()
    point_model_distance = session['ensemble'].generate_distance_for_points(np.array(session['test_data'][session['GMM_attribute']]))
    return {
            **result1,
            'drift': drift_test(session['ensemble'].get_all_used_model(), session['X'][session['range']],
                                session['test_data'][session['GMM_attribute']], np.argmin(point_model_distance, axis=1))
            }


@data.route('/setTsneAttr', methods=['POST'])
def set_tsne_attr():
    global session
    post = json.loads(request.get_data())
    attrs = post['attrs']
    attr_idxes = [session['scatter_attribute'].index(x) for x in attrs]
    session['scatter'].set_attr_idxes(attr_idxes)
    return '', 200


@data.route('/saveAsSVG', methods=['POST'])
def save_as_svg():
    info = json.loads(request.get_data())
    pattern = 'data-content=".*?"'
    info['svg'] = re.sub(pattern, '', info['svg'])
    with open('application/templates/svg.txt') as fin:
        xml_temp = fin.read()

    area = info['area']
    dic = {
        'ViewBox': ' '.join([area['x'], area['y'],
                             area['w'], area['h']]),
        'Content': info['svg']
    }

    xml = ''
    pp = xml_temp.split('$')
    for i, p in enumerate(pp):
        if (i & 1) == 1:
            xml += dic[p]
        else:
            xml += p

    cache_dir = 'application/static/cache'
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    path = 'cache/{}.svg'.format(info['no'])
    with open('application/static/'+path, 'w') as fout:
        fout.write(xml)
    return jsonify({'static_svg_path': path})


def get_model_percentage(idx, method='min'):
    global session
    if len(idx) == 0:
        return [1]
    a = session['point_model_distance'][idx]
    if method == 'min':
        tmp = np.bincount(np.argmin(a, axis=1))
        while len(tmp) < len(session['ensemble'].get_all_used_model()):
            tmp = np.append(tmp, 0)
        return (tmp / np.sum(tmp)).tolist()
    if method == 'linear':
        tmp = np.sum(np.exp(-a),axis=0)
        return (tmp / np.sum(tmp)).tolist()
    if method == 'softmax':
        tmp = np.exp(-a)
        tmp = tmp.T / (np.sum(tmp, axis=1)+0.0001)
        tmp = np.sum(tmp, axis=1)
        return (tmp / np.sum(tmp)).tolist()


def get_drift_score(chunk):
    global session
    if chunk['latest']:
        return chunk['drift_score']
    idx = chunk['idx']
    if len(idx) == 0:
        return 0
    scores = []
    # scores.append(session['detector']['ED'].get_drift_value(session['ensemble'].get_all_used_model(), idx, session['model_labels']))
    # for attr in session['attributes']:
    #     scores.append(session['detector'][f'{attr}_ED'].get_drift_value(session['ensemble'].get_all_used_model(), idx, session['model_labels']))
    scores.append(session['detector']['ED'].get_drift_value(session['ensemble'].get_all_used_model(), idx, session['gmm_label']))
    for attr in session['attributes']:
        scores.append(session['detector'][f'{attr}_ED'].get_drift_value(session['ensemble'].get_all_used_model(), idx, session['gmm_label']))
    chunk['drift_score'] = np.array(scores).max()
    chunk['latest'] = True
    return np.array(scores).max()


@data.route('/saveHistory', methods=['POST'])
def save_history():
    # return '',200
    global session
    post = json.loads(request.get_data())
    uuid_str = str(uuid.uuid1())
    name = post['name'] if 'name' in post else uuid_str
    session['history'][uuid_str] = History(session)
    with open('cache/'+name, 'wb') as fout:
        pickle.dump(session['history'][uuid_str], fout)
    return jsonify({'uuid': uuid_str})

@data.route('/loadHistory', methods=['POST'])
def load_history():
    global session
    post = json.loads(request.get_data())
    uuid_str = post['uuid']
    session['history'][uuid_str].apply(session)
    return jsonify({
        'chunks': get_chunks(),
        'models': get_models(),
        'gmm_label': session['gmm_label'],
    })

@data.route('/getSquareData', methods=['POST'])
def get_square_data():
    global session
    post = json.loads(request.get_data())
    if 'idxes' not in post:
        idxes = list(range(session['current_len'] - session['test_size'], session['current_len']))
    else:
        idxes = post['idxes']

    ret = {}
    if session['stored_ensemble'][0] != None:
        session['ensemble'].unuse_all()
        for model_id in session['stored_ensemble'][0]:
            session['ensemble'].toggle(model_id)
        ret['previous_result'] = pointwise_test_ensemble_model(
            session['ensemble'].get_all_used_model(), session['trainX'], session['y'], idxes,
            #session[f'k-fold-0'])
            [session[f'k-fold-{i}'] for i in range(session['k-fold'])])
    if session['stored_ensemble'][1] != None:
        session['ensemble'].unuse_all()
        for model_id in session['stored_ensemble'][1]:
            session['ensemble'].toggle(model_id)
        ret['current_result'] = pointwise_test_ensemble_model(
            session['ensemble'].get_all_used_model(), session['trainX'], session['y'], idxes,
            #session[f'k-fold-0'])
            [session[f'k-fold-{i}'] for i in range(session['k-fold'])])

    return jsonify(ret)


def calculate_tfidf(idx):
    idx = [i for i in idx if session['data']['abstract'][i] != 'Abstract Missing']
    tfidf = session['tfidf'][idx,:].sum(axis=0)
    pairs = [[session['tfidf_name'][i], tfidf[i]] for i in range(tfidf.shape[0]) if tfidf[i] > 0]
    return [[k, v] for k, v in sorted(pairs, key=lambda item: item[1])][::-1]


@data.route('/getPredictVector', methods=['POST'])
def get_predict_vector():
    global session
    ret = {}
    post = json.loads(request.get_data())
    if 'end_idx' in post:
        idx = post['end_idx']
        idxes = list(range(idx-365,idx))
        tmp = pointwise_test_ensemble_model(
                session['ensemble'].get_all_used_model(), session['trainX'], session['y'], idxes,
                [session[f'k-fold-{i}'] for i in range(session['k-fold'])])
        ret['result'] = tmp['acc']
        ret['detailed'] = tmp['detailed']
        ret['median'] = tmp['median']
    else:
        for n in [30,45,60,90]:
            idxes = list(range(session['current_len'] - n, session['current_len']))
            tmp = pointwise_test_ensemble_model(
                session['ensemble'].get_all_used_model(), session['trainX'], session['y'], idxes,
                [session[f'k-fold-{i}'] for i in range(session['k-fold'])])
            ret[n] = tmp['acc']
    return jsonify(ret)