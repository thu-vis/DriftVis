/*
 *  tsne.h
 *  Header file for incremental t-SNE.
 *
 *  Created by Shouxing Xiang.
 */

#pragma once

class TSNE
{
public:
	TSNE(int _knn_tree, int _verbose);
	TSNE();
	void run_bhtsne(double* X, int N, int D, double* Y, int no_dim, double perplexity, double angle, int n_jobs,
		int n_iter, int random_state, int forest_size, double accuracy, double early_exaggeration, double learning_rate,
		int skip_num_points, int exploration_n_iter,
		int n_neighbors, int* neighbors_nn, double* distances_nn);
	void run_bhtsne(double* X, int N, int D, double* Y, int no_dim, double* constraint_X, double* constraint_Y,
		int constraint_N, double alpha, double perplexity, double angle, int n_jobs,
		int n_iter, int random_state, int forest_size, double accuracy, double early_exaggeration, double learning_rate,
		int skip_num_points, int exploration_n_iter,
		int n_neighbors, int* neighbors_nn, int* constraint_neighbors_nn, double* distances_nn, double* constraint_distances_nn, double* constraint_weight);
	void multi_run_bhtsne(double* X, int N, int D, double* Y, int no_dim, double perplexity, double angle, int n_jobs,
		int n_iter, int random_state, int forest_size, double accuracy, double early_exaggeration, double learning_rate,
		int skip_num_points, int exploration_n_iter);
	void multi_run_bhtsne(double* X, int N, int D, double* Y, int no_dim, double* constraint_X, double* constraint_Y,
		int constraint_N, double alpha, double perplexity, double angle, int n_jobs,
		int n_iter, int random_state, int forest_size, double accuracy, double early_exaggeration, double learning_rate,
		int skip_num_points, int exploration_n_iter);
	void binary_search_perplexity(double* distances_nn, int* neighbors_nn, int N, int n_neighbors,
		double perplexity, double* conditional_P);
	void constraint_binary_search_perplexity(double* distances_nn, int* neighbors_nn, int N, int n_neighbors, int number,
		double perplexity, double* conditional_P);
	void k_neighbors(double* X1, int N1, double* X2, int N2, int D, int n_neighbors, int* neighbors_nn, double* distances_nn, int forest_size,
		int subdivide_variance_size, int leaf_number);
	void k_neighbors(double* X, int N, int D, int n_neighbors, int* neighbors_nn, double* distances_nn, int forest_size,
		int subdivide_variance_size, int leaf_number);


//private:
	int knn_tree, verbose;

	void gradient(double* val_P, int* neighbors, int length, int* indptr, int ind_len, double* pos_output,
		double* forces, int N, int n_dimensions, double* error, double theta,
		int skip_num_points, bool need_eval_error);
	void constraint_gradient(double* val_P, int* neighbors, int length, int* indptr, int ind_len, double* pos_output,
		double* forces, int N, int n_dimensions, double* constraint_pos, int constraint_N, double* error, double theta,
		int skip_num_points, bool need_eval_error, double* constraint_weight);
	double compute_gradient(double* val_P, int* neighbors, int length, int* indptr, int ind_len, double* pos_reference,
		double* tot_force, int N, int n_dimensions, QuadTree* qt, double theta, int start,
		int stop, bool need_eval_error);
	double constraint_compute_gradient(double* val_P, int* neighbors, int length, int* indptr, int ind_len, double* pos_reference,
		double* tot_force, int N, int n_dimensions, double* constraint_pos, int constraint_N, QuadTree* qt, double theta, int start,
		int stop, bool need_eval_error, double* constraint_weight);
	void compute_gradient_negative(double* neg_f, int N, int n_dimensions, QuadTree* qt,
		double* sum_Q, double theta, int start, int stop, bool need_eval_error);
	void constraint_compute_gradient_negative(double* pos_reference, double* neg_f, int N, int n_dimensions, int constraint_N,
		QuadTree* qt, double* sum_Q, double theta, int start, int stop, bool need_eval_error);
	double compute_gradient_positive(double* val_P, int* neighbors, int* indptr, int ind_len,
		double* pos_reference, double* pos_f, int n_dimensions, double sQ, int start, bool need_eval_error);
	double constraint_compute_gradient_positive(double* val_P, int* neighbors, int* indptr, int ind_len,
		double* pos_reference, double* pos_f, int n_dimensions, double* constraint_pos, int constraint_N, double sQ, int start, bool need_eval_error, double* constraint_weight);
	void symmetrize_matrix(int** _rowP, int** _colP, double** _valP, int N);
	void compute_distance(double* X, int N, int D, double* dist);
	void compute_sum(double* X, int N, int D, int axis, double* res);
	double compute_sum(double* X, int N);
	double squared_euclidean_distance(double* a, double* b, int length);
	double compute_maximum_p(double *P, int length);
	double dot(double* x, double* y, int length);
	bool any_not_zero(double* X, int length);
	void matrix_operation(double* X1, int N1, int D1, double* X2, int N2, int D2, int oper, double* res);
	void zero_mean(double* X, int N, int D);
	void compute_gaussian_perplexity(double* X, int N, int D, int** _row_P, int** _col_P, 
		double** _val_P, double perplexity, int K, int verbose, int forest_size, 
		double accuracy);
	void constraint_compute_gaussian_perplexity(double* X, int N, double* constraint_X, int constraint_N, int D, int** _row_P, int** _col_P,
		double** _val_P, double perplexity, int K, int verbose, int forest_size,
		double accuracy);
	void multi_symmetrize_matrix(int** _row_P, int** _col_P, double** _val_P, int N);
	double multi_compute_gradient(int* inp_row_P, int* inp_col_P, double* inp_val_P, double* Y, int N, int no_dims, double* dC, double theta, bool eval_error);
	double constraint_multi_compute_gradient(int* inp_row_P, int* inp_col_P, double* inp_val_P, double* Y, int N, double* constraint_Y, 
		int constraint_N, int no_dims, double* dC, double theta, bool eval_error);
	double evaluate_error(int* row_P, int* col_P, double* val_P, double* Y, int N, int no_dims, double theta);
};