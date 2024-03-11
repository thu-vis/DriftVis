import copy
class History(object):
    def __init__(self, session):
        self.win_size = session['win_size']
        self.line_data = session['cache']['line_data']
        self.current_len = session['current_len']
        self.range = session['range'].copy()
        self.new_idx = session['new_idx'].copy()
        self.data_chunk = copy.deepcopy(session['data_chunk'])
        self.data_chunk_id = session['data_chunk_id']
        self.gmm_result_distribution = copy.deepcopy(session['gmm_result'].get_gmm_result())
        self.gmm_result_thres = (session['gmm_result'].component_relax_thres, session['gmm_result'].component_strict_thres)
        self.models = copy.deepcopy(session['ensemble'].get_all_model())
        self.point_model_distance = session['point_model_distance'].copy()
        self.scatter_result = session['scatter'].result.copy()
        self.stacked_layout = session['scatter'].stacked_layout.copy()
        self.current_layout = session['scatter'].current_layout.copy()
        self.scatter_label_result = session['scatter'].label_result
        self.scatterplot_attr_idxes = session['scatter'].get_attr_idxes().copy()
        self.stored_ensemble = session['stored_ensemble'].copy()
        self.gmm_rnd_state = session['gmm_result'].rnd.get_state()
        self.tsne_rnd_state = session['scatter'].rnd.get_state()


    def apply(self, session):
        win_size_changed = session['win_size'] != self.win_size
        session['win_size'] = self.win_size
        session['cache']['line_data'] = self.line_data
        session['current_len'] = self.current_len
        session['range'] = self.range.copy()
        session['new_idx'] = self.new_idx.copy()
        session['data_chunk'] = copy.deepcopy(self.data_chunk)
        session['data_chunk_id'] = self.data_chunk_id
        for detector in session['detector']:
            session['detector'][detector].set_data(session['data'], session['range'])
            if win_size_changed:
                session['detector'][detector].change_win_size(session['win_size'])
        session['gmm_result'].set_data(session['data'], session['range'], copy.deepcopy(self.gmm_result_distribution), self.gmm_result_thres)
        session['gmm_label'] = session['gmm_result'].get_gmm_result()['classes'].tolist()
        session['ensemble'].models = []
        session['ensemble'].add_model(copy.deepcopy(self.models))
        session['point_model_distance'] = self.point_model_distance.copy()
        session['scatter'].set_attr_idxes(self.scatterplot_attr_idxes.copy())
        session['scatter'].set_data(session['data'], session['range'], self.scatter_result.copy(), self.scatter_label_result, session['gmm_result'].get_gmm_result())
        session['scatter'].stacked_layout = self.stacked_layout.copy()
        session['scatter'].current_layout = self.current_layout.copy()
        session['stored_ensemble'] = self.stored_ensemble.copy()
        session['gmm_result'].rnd.set_state(self.gmm_rnd_state)
        session['scatter'].rnd.set_state(self.tsne_rnd_state)