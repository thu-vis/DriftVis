/*
 *  tsne.cpp
 *  Implementation of tsne.
 *
 *  Created by Shouxing Xiang.
 */

#include <cmath>
#include <cfloat>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <float.h>
#include <iostream>
#include <fstream>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "quad_tree.h"
#include "vp_tree.h"
#include "tsne.h"
#include "kd_tree.h"
#include "forest.h"
#include "parameter_selection.h"
using namespace std;


#ifdef _OPENMP
    #define NUM_THREADS(N) ((N) >= 0 ? (N) : omp_get_num_procs() + (N) + 1)
#else
    #define NUM_THREADS(N) (1)
#endif

TSNE::TSNE(int _knn_tree, int _verbose) {
	knn_tree = _knn_tree;
	verbose = _verbose;
}

TSNE::TSNE() {
	knn_tree = 0;
	verbose = 0;
}

void TSNE::k_neighbors(double* X1, int N1, double* X2, int N2, int D, int n_neighbors, int* neighbors_nn, double* distances_nn, int forest_size,
	int subdivide_variance_size, int leaf_number) {
	clock_t t = 0;
	if (verbose > 0) {
		t = clock();
	}

	if (knn_tree == 0) {
		std::vector<Node> obj_X(N2, Node());

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < N2; i++) {
			obj_X[i] = Node(i, -1, X2 + i * D, D, -1, -1);
		}
		KdTreeForest *forest = new KdTreeForest(obj_X, N2, D, forest_size, subdivide_variance_size, leaf_number);
		double dup_ratio = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:dup_ratio)
#endif
		for (int i = 0; i < N1; i++) {
			//printf("[t-SNE] Computing %d neighbors of point %d.\n", n_neighbors, i);
			std::vector<int> indices;
			std::vector<double> distances;
			indices.clear();
			distances.clear();
			// Find nearest neighbors
			Node target = Node(X1 + i * D, D);
			double _dup_ratio = forest->priority_search(target, n_neighbors, &indices, &distances);

			dup_ratio += _dup_ratio;

			for (int j = 0; j < n_neighbors; j++) {
				neighbors_nn[i * n_neighbors + j] = indices[j];
				distances_nn[i * n_neighbors + j] = distances[j];
			}
		}
		//printf("duplicate-ratio= %f\n", dup_ratio / double(N1));

		delete forest;
	}
	else if (knn_tree == 1) {
		std::vector<DataPoint> obj_X(N2, DataPoint(D, -1, X2));

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int n = 0; n < N2; n++) {
			obj_X[n] = DataPoint(D, n, X2 + n * D);
		}
		VpTreeForest *forest = new VpTreeForest(obj_X, forest_size, leaf_number);
		double dup_ratio = 0.0;


#ifdef _OPENMP
#pragma omp parallel for reduction(+:dup_ratio)
#endif
		for (int i = 0; i < N1; i++) {
			//printf("[t-SNE] Computing %d neighbors of point %d.\n", n_neighbors, i);
			std::vector<int> indices;
			std::vector<double> distances;
			indices.clear();
			distances.clear();
			// Find nearest neighbors
			DataPoint target = DataPoint(D, -1, X1 + i * D);
			double _dup_ratio = forest->priority_search(target, n_neighbors, &indices, &distances);

			dup_ratio += _dup_ratio;

			for (int j = 0; j < n_neighbors; j++) {
				neighbors_nn[i * n_neighbors + j] = indices[j];
				distances_nn[i * n_neighbors + j] = distances[j];
			}
		}
		//printf("duplicate-ratio= %f\n", dup_ratio / double(N1));

		delete forest;
	}
	else if (knn_tree == 2) {
		std::vector<Node> obj_X(N2, Node());

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < N2; i++) {
			obj_X[i] = Node(i, -1, X2 + i * D, D, -1, -1);
		}
		KdTree *tree = new KdTree(obj_X, N2, D, subdivide_variance_size);

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < N1; i++) {
			//printf("[t-SNE] Computing %d neighbors of point %d.\n", n_neighbors, i);
			std::vector<int> indices;
			std::vector<double> distances;
			indices.clear();
			distances.clear();
			// Find nearest neighbors
			Node target = Node(X1 + i * D, D);
			tree->search(target, n_neighbors, &indices, &distances, leaf_number);
			for (int j = 0; j < n_neighbors; j++) {
				neighbors_nn[i * n_neighbors + j] = indices[j];
				distances_nn[i * n_neighbors + j] = distances[j];
			}
		}
		delete tree;
	}
	else if (knn_tree == 3) {
		std::vector<Node> obj_X(N2, Node());

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < N2; i++) {
			obj_X[i] = Node(i, -1, X2 + i * D, D, -1, -1);
		}
		KdTree *tree = new KdTree(obj_X, N2, D, subdivide_variance_size);

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < N1; i++) {
			//printf("[t-SNE] Computing %d neighbors of point %d.\n", n_neighbors, i);
			std::vector<int> indices;
			std::vector<double> distances;
			indices.clear();
			distances.clear();
			// Find nearest neighbors
			Node target = Node(X1 + i * D, D);
			tree->priority_search(target, n_neighbors, &indices, &distances, leaf_number);

			for (int j = 0; j < n_neighbors; j++) {
				neighbors_nn[i * n_neighbors + j] = indices[j];
				distances_nn[i * n_neighbors + j] = distances[j];
			}
		}
		delete tree;
	}
	else if (knn_tree == 4) {
		std::vector<DataPoint> obj_X(N2, DataPoint(D, -1, X2));

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int n = 0; n < N2; n++) {
			obj_X[n] = DataPoint(D, n, X2 + n * D);
		}
		VpTree *tree = new VpTree(obj_X);

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < N1; i++) {
			//printf("[t-SNE] Computing %d neighbors of point %d.\n", n_neighbors, i);
			std::vector<int> indices;
			std::vector<double> distances;
			indices.clear();
			distances.clear();
			// Find nearest neighbors
			DataPoint target = DataPoint(D, -1, X1 + i * D);
			tree->search(target, n_neighbors, &indices, &distances, leaf_number);

			for (int j = 0; j < n_neighbors; j++) {
				neighbors_nn[i * n_neighbors + j] = indices[j];
				distances_nn[i * n_neighbors + j] = distances[j];
			}
		}
		delete tree;
	}
	else if (knn_tree == 5) {
		std::vector<DataPoint> obj_X(N2, DataPoint(D, -1, X2));

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int n = 0; n < N2; n++) {
			obj_X[n] = DataPoint(D, n, X2 + n * D);
		}
		VpTree *tree = new VpTree(obj_X);

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < N1; i++) {
			//printf("[t-SNE] Computing %d neighbors of point %d.\n", n_neighbors, i);
			std::vector<int> indices;
			std::vector<double> distances;
			indices.clear();
			distances.clear();
			// Find nearest neighbors
			DataPoint target = DataPoint(D, -1, X1 + i * D);
			tree->priority_search(target, n_neighbors, &indices, &distances, leaf_number);

			for (int j = 0; j < n_neighbors; j++) {
				neighbors_nn[i * n_neighbors + j] = indices[j];
				distances_nn[i * n_neighbors + j] = distances[j];
			}
		}
		delete tree;
	}

	//    std::vector<int> indices;
	if (verbose > 0) {
		t = clock() - t;
		printf("[t-SNE] Computed %d neighbors of %d points in %f seconds.\n", n_neighbors, N1, double(t) / CLOCKS_PER_SEC);
	}
	return;
}

void TSNE::k_neighbors(double* X, int N, int D, int n_neighbors, int* neighbors_nn, double* distances_nn, int forest_size,
	int subdivide_variance_size, int leaf_number) {
	clock_t t = 0;
	if (verbose > 0) {
		t = clock();
	}

	if (knn_tree == 0) {
		std::vector<Node> obj_X(N, Node());

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < N; i++) {
			obj_X[i] = Node(i, -1, X + i * D, D, -1, -1);
		}
		printf("Before build KdTree.\n");
		KdTreeForest *forest = new KdTreeForest(obj_X, N, D, forest_size, subdivide_variance_size, leaf_number);
		double dup_ratio = 0.0;

#ifdef _OPENMP
#pragma omp parallel for reduction(+:dup_ratio)
#endif
		for (int i = 0; i < N; i++) {
			if (i % 100 == 0) {
				printf("%d neighbors have been found.\n", i);
			}
			//printf("[t-SNE] Computing %d neighbors of point %d.\n", n_neighbors, i);
			std::vector<int> indices;
			std::vector<double> distances;
			indices.clear();
			distances.clear();
			// Find nearest neighbors
			Node target = Node(X + i * D, D);
			double _dup_ratio = forest->priority_search(target, n_neighbors + 1, &indices, &distances);

			dup_ratio += _dup_ratio;

			for (int j = 0; j < n_neighbors; j++) {
				neighbors_nn[i * n_neighbors + j] = indices[j + 1];
				distances_nn[i * n_neighbors + j] = distances[j + 1];
			}
		}
		//printf("duplicate-ratio= %f\n", dup_ratio / double(N));

		delete forest;
	}
	else if (knn_tree == 1) {
		std::vector<DataPoint> obj_X(N, DataPoint(D, -1, X));

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int n = 0; n < N; n++) {
			obj_X[n] = DataPoint(D, n, X + n * D);
		}
		VpTreeForest *forest = new VpTreeForest(obj_X, forest_size, leaf_number);
		double dup_ratio = 0.0;


#ifdef _OPENMP
#pragma omp parallel for reduction(+:dup_ratio)
#endif
		for (int i = 0; i < N; i++) {
			//printf("[t-SNE] Computing %d neighbors of point %d.\n", n_neighbors, i);
			std::vector<int> indices;
			std::vector<double> distances;
			indices.clear();
			distances.clear();
			// Find nearest neighbors
			double _dup_ratio = forest->priority_search(obj_X[i], n_neighbors + 1, &indices, &distances);
			dup_ratio += _dup_ratio;

			for (int j = 0; j < n_neighbors; j++) {
				//            neighbors_nn[i * n_neighbors + j] = indices[j + 1];
				neighbors_nn[i * n_neighbors + j] = indices[j + 1];// .index();
				distances_nn[i * n_neighbors + j] = distances[j + 1];
				//            printf("%f\t", distances[j + 1]);
			}
			//        printf("\n");
		}
		//printf("duplicate-ratio= %f\n", dup_ratio / double(N));

		delete forest;
	}
	else if (knn_tree == 2) {
		std::vector<Node> obj_X(N, Node());

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < N; i++) {
			obj_X[i] = Node(i, -1, X + i * D, D, -1, -1);
		}
		KdTree *tree = new KdTree(obj_X, N, D, subdivide_variance_size);

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < N; i++) {
			//printf("[t-SNE] Computing %d neighbors of point %d.\n", n_neighbors, i);
			std::vector<int> indices;
			std::vector<double> distances;
			indices.clear();
			distances.clear();
			// Find nearest neighbors
			Node target = Node(X + i * D, D);
			tree->search(target, n_neighbors + 1, &indices, &distances, leaf_number);
			for (int j = 0; j < n_neighbors; j++) {
				neighbors_nn[i * n_neighbors + j] = indices[j + 1];
				distances_nn[i * n_neighbors + j] = distances[j + 1];
			}
		}
		delete tree;
	}
	else if (knn_tree == 3) {
		std::vector<Node> obj_X(N, Node());

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < N; i++) {
			obj_X[i] = Node(i, -1, X + i * D, D, -1, -1);
		}
		KdTree *tree = new KdTree(obj_X, N, D, subdivide_variance_size);

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < N; i++) {
			//printf("[t-SNE] Computing %d neighbors of point %d.\n", n_neighbors, i);
			std::vector<int> indices;
			std::vector<double> distances;
			indices.clear();
			distances.clear();
			// Find nearest neighbors
			Node target = Node(X + i * D, D);
			tree->priority_search(target, n_neighbors + 1, &indices, &distances, leaf_number);

			for (int j = 0; j < n_neighbors; j++) {
				neighbors_nn[i * n_neighbors + j] = indices[j + 1];
				distances_nn[i * n_neighbors + j] = distances[j + 1];
			}
		}
		delete tree;
	}
	else if (knn_tree == 4) {
		std::vector<DataPoint> obj_X(N, DataPoint(D, -1, X));

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int n = 0; n < N; n++) {
			obj_X[n] = DataPoint(D, n, X + n * D);
		}
		VpTree *tree = new VpTree(obj_X);

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < N; i++) {
			//printf("[t-SNE] Computing %d neighbors of point %d.\n", n_neighbors, i);
			std::vector<int> indices;
			std::vector<double> distances;
			indices.clear();
			distances.clear();
			// Find nearest neighbors
			tree->search(obj_X[i], n_neighbors + 1, &indices, &distances, leaf_number);

			for (int j = 0; j < n_neighbors; j++) {
				//            neighbors_nn[i * n_neighbors + j] = indices[j + 1];
				neighbors_nn[i * n_neighbors + j] = indices[j + 1];// .index();
				distances_nn[i * n_neighbors + j] = distances[j + 1];
				//            printf("%f\t", distances[j + 1]);
			}
			//        printf("\n");
		}
		delete tree;
	}
	else if (knn_tree == 5) {
		std::vector<DataPoint> obj_X(N, DataPoint(D, -1, X));

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int n = 0; n < N; n++) {
			obj_X[n] = DataPoint(D, n, X + n * D);
		}
		VpTree *tree = new VpTree(obj_X);

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < N; i++) {
			//printf("[t-SNE] Computing %d neighbors of point %d.\n", n_neighbors, i);
			std::vector<int> indices;
			std::vector<double> distances;
			indices.clear();
			distances.clear();
			// Find nearest neighbors
			tree->priority_search(obj_X[i], n_neighbors + 1, &indices, &distances, leaf_number);

			for (int j = 0; j < n_neighbors; j++) {
				neighbors_nn[i * n_neighbors + j] = indices[j + 1];
				distances_nn[i * n_neighbors + j] = distances[j + 1];
			}
		}
		delete tree;
	}

	//    std::vector<int> indices;
	if (verbose > 0) {
		t = clock() - t;
		printf("[t-SNE] Computed %d neighbors of %d points in %f seconds.\n", n_neighbors, N, double(t) / CLOCKS_PER_SEC);
	}
	return;
	}


void TSNE::constraint_binary_search_perplexity(double* distances_nn, int* neighbors_nn, int N, int n_neighbors, int number,
	double perplexity, double* conditional_P) {
	int *indptr = new int[number + 1], 
		*indexes = new int[N * n_neighbors], 
		*order = new int[N * n_neighbors];
	int *offset = new int[number];

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int i = 0; i < number; i++) {
		offset[i] = 0;
	}

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int i = 0; i <= number; i++) {
		indptr[i] = 0;
	}
	for (int i = 0; i < N * n_neighbors; i++) {
		indptr[neighbors_nn[i] + 1]++;
	}
	for (int i = 1; i <= number; i++) {
		indptr[i] += indptr[i - 1];
	}
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < n_neighbors; j++) {
			int index = neighbors_nn[i * n_neighbors + j];
			indexes[indptr[index] + offset[index]] = i;
			order[indptr[index] + offset[index]] = j;
			offset[index]++;
		}
	}

	/*for (int i = 0; i < number; i++) {
		for (int j = indptr[i]; j < indptr[i + 1]; j++) {
			printf("%f ", indexes[j]);
		}
		printf("\n");
	}
	system("pause");*/

	delete[] offset;
	int step_completed = 0;
	double beta_sum = 0.0;
	int *fl = new int[N * n_neighbors];
	for (int i = 0; i < N * n_neighbors; i++) {
		fl[i] = 0;
	}

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int i = 0; i < number; i++) {
		for (int j = indptr[i]; j < indptr[i + 1]; j++) {
			fl[indexes[j] * n_neighbors + order[j]] = 1;
		}
	}
	delete[] fl;

#ifdef _OPENMP
#pragma omp parallel for reduction(+:step_completed, beta_sum)
#endif
	for (int i = 0; i < number; i++) {
		if (indptr[i] == indptr[i + 1]) {
			continue;
		}
		// Initialize some variables for binary search
		double beta = 1.0,
			min_beta = -DBL_MAX,
			max_beta = DBL_MAX,
			sum_P = 0.0,
			desired_entropy = log(perplexity),
			tol = 1e-5,
			entropy_diff, entropy, sum_dist_P;

		// Iterate until we found a good perplexity
		for (int iter = 0; iter < 100; iter++) {
			sum_P = 0.0;
			for (int j = indptr[i]; j < indptr[i + 1]; j++) {
				conditional_P[indexes[j] * n_neighbors + order[j]] = exp(-distances_nn[indexes[j] * n_neighbors + order[j]] * beta);
				sum_P += conditional_P[indexes[j] * n_neighbors + order[j]];
			}
			if (sum_P < DBL_EPSILON) {
				sum_P = DBL_EPSILON;
			}
			sum_dist_P = 0.0;
			for (int j = indptr[i]; j < indptr[i + 1]; j++) {
				conditional_P[indexes[j] * n_neighbors + order[j]] /= sum_P;
				sum_dist_P += distances_nn[indexes[j] * n_neighbors + order[j]] * conditional_P[indexes[j] * n_neighbors + order[j]];
			}
			entropy = log(sum_P) + beta * sum_dist_P;
			entropy_diff = entropy - desired_entropy;

			if (abs_d(entropy_diff) <= tol) {
				break;
			}
			if (entropy_diff > 0.0) {
				min_beta = beta;
				if (max_beta == DBL_MAX) {
					beta *= 2.0;
				}
				else {
					beta = (beta + max_beta) / 2.0;
				}
			}
			else {
				max_beta = beta;
				if (min_beta == -DBL_MAX) {
					beta /= 2.0;
				}
				else {
					beta = (beta + min_beta) / 2.0;
				}
			}
		}

		// Print progress
		step_completed += 1;
		beta_sum += beta;


		if (verbose > 0 && (step_completed % 1000 == 0 || step_completed == number))
		{
#ifdef _OPENMP
#pragma omp critical
#endif
			printf("[t-SNE]  - point %d of %d\n", step_completed, number);
		}
	}
	if (verbose > 0) {
#ifdef _OPENMP
#pragma omp critical
#endif
		printf("[t-SNE] Mean sigma: %f\n", sqrt((double)beta_sum / number));
	}
	delete[] indptr;
	delete[] indexes;
	delete[] order;
}

void TSNE::binary_search_perplexity(double* distances_nn, int* neighbors_nn, int N, int D,
                                    double perplexity, double* conditional_P) {
	int step_completed = 0;
	double beta_sum = 0.0;

#ifdef _OPENMP
#pragma omp parallel for reduction(+:step_completed, beta_sum)
#endif
	for (int i = 0; i < N; i++) {
		// Initialize some variables for binary search
		bool using_neighbors = (neighbors_nn != NULL);
		double beta = 1.0, 
				min_beta = -DBL_MAX,
				max_beta = DBL_MAX,
				sum_P = 0.0,
				desired_entropy = log(perplexity),
				tol = 1e-5,
				entropy_diff, entropy, sum_dist_P;

		// Iterate until we found a good perplexity
		for (int iter = 0; iter < 100; iter++) {
			sum_P = 0.0;
			for (int j = 0; j < D; j++) {
				if (j != i || using_neighbors) {
					conditional_P[i * D + j] = exp(-distances_nn[i * D + j] * beta);
					sum_P += conditional_P[i * D + j];
				}
			}
			if (sum_P < DBL_EPSILON) {
				sum_P = DBL_EPSILON;
			}
			sum_dist_P = 0.0;
			for (int j = 0; j < D; j++) {
				conditional_P[i * D + j] /= sum_P;
				sum_dist_P += distances_nn[i * D + j] * conditional_P[i * D + j];
			}
			entropy = log(sum_P) + beta * sum_dist_P;
			entropy_diff = entropy - desired_entropy;

			if (abs_d(entropy_diff) <= tol) {
				break;
			}
			if (entropy_diff > 0.0) {
				min_beta = beta;
				if (max_beta == DBL_MAX) {
					beta *= 2.0;
				}
				else {
					beta = (beta + max_beta) / 2.0;
				}
			}
			else {
				max_beta = beta;
				if (min_beta == -DBL_MAX) {
					beta /= 2.0;
				}
				else {
					beta = (beta + min_beta) / 2.0;
				}
			}
		}

		// Print progress
		step_completed += 1;
		beta_sum += beta;
		

		if (verbose > 0 && (step_completed % 1000 == 0 || step_completed == N))
		{
			#ifdef _OPENMP
				#pragma omp critical
			#endif
			printf("[t-SNE]  - point %d of %d\n", step_completed, N);
		}
	}
	if (verbose > 0) {
#ifdef _OPENMP
#pragma omp critical
#endif
		printf("[t-SNE] Mean sigma: %f\n", sqrt((double)beta_sum / N));
	}
}

void TSNE::constraint_gradient(double* val_P, int* neighbors, int length, int* indptr, int ind_len, double* pos_output,
	double* forces, int N, int n_dimensions, double* constraint_pos, int constraint_N, double* error, double theta,
	int skip_num_points, bool need_eval_error, double* constraint_weight) {
	// This function is designed to be called from external Python
	// it passes the 'forces' array by reference and fills thats array
	// up in - place
	if (constraint_N != ind_len - 1) {
		printf("[t-SNE] Pij and pos_output shapes are incompatible.\n");
		return;
	}
	if (verbose > 10) {
		printf("[t-SNE] Initializing tree of n_dimensions %d, consist of %d points.\n", n_dimensions, N);
	}

	QuadTree* tree = new QuadTree(constraint_pos, constraint_N, n_dimensions, constraint_weight);
	if (verbose > 10) {
		printf("[t-SNE] Computing gradient of %d points.\n", N);
	}

	double res_error = constraint_compute_gradient(val_P, neighbors, length, indptr, ind_len, pos_output, forces, N, n_dimensions,
		constraint_pos, constraint_N, tree, theta, skip_num_points, -1, need_eval_error, constraint_weight);
	error[0] = need_eval_error ? res_error : error[0];
	delete tree;
	return;
}

void TSNE::gradient(double* val_P, int* neighbors, int length, int* indptr, int ind_len, double* pos_output,
                                    double* forces, int N, int n_dimensions, double* error, double theta,
                                    int skip_num_points, bool need_eval_error) {
	// This function is designed to be called from external Python
	// it passes the 'forces' array by reference and fills thats array
	// up in - place
	if (N != ind_len - 1) {
		printf("[t-SNE] Pij and pos_output shapes are incompatible.\n");
		return;
	}
	if (verbose > 10) {
		printf("[t-SNE] Initializing tree of n_dimensions %d, consist of %d points.\n", n_dimensions, N);
	}

    //printf("point15\n");
	QuadTree* tree = new QuadTree(pos_output, N, n_dimensions);
	if (verbose > 10) {
		printf("[t-SNE] Computing gradient of %d points.\n", N);
	}
    //printf("point9\n");
	double res_error = compute_gradient(val_P, neighbors, length, indptr, ind_len, pos_output, forces, N, n_dimensions,
		tree, theta, skip_num_points, -1, need_eval_error);
	error[0] = need_eval_error ? res_error : error[0];
    //printf("point10\n");
	delete tree;
    return;
}

double TSNE::constraint_compute_gradient(double* val_P, int* neighbors, int length, int* indptr, int ind_len, double* pos_reference,
	double* tot_force, int N, int n_dimensions, double* constraint_pos, int constraint_N, QuadTree* qt, double theta, int start,
	int stop, bool need_eval_error, double* constraint_weight) {
	// Having created the tree, calculate the gradient
	// in two components, the positive and negative forces
	double sum_Q = 0.0;
	clock_t t1 = 0, t2 = 0;
	double sQ, error;
	double* pos_f = new double[N * n_dimensions]();
	double* neg_f = new double[N * n_dimensions]();

	if (pos_f == NULL || neg_f == NULL) {
		printf("[t-SNE] Memory allocation failed!\n");
		exit(1);
	}

	if (verbose > 10) {
		t1 = clock();
	}


#ifdef _OPENMP
#pragma omp parallel for reduction(+:sum_Q)
#endif
	for (int n = 0; n < N; n++) {
		double this_Q = .0;
		qt->constraint_compute_non_edge_forces(pos_reference + n * n_dimensions, theta, neg_f + n * n_dimensions, &this_Q);
		sum_Q += this_Q;
	}
	/*constraint_compute_gradient_negative(pos_reference, neg_f, N, n_dimensions, constraint_N, qt, &sum_Q, theta, start,
		stop);*/
	if (verbose > 10) {
		t2 = clock();
		printf("[t-SNE] Computing negative gradient: %4.4f seconds\n", ((double)(t2 - t1) / CLOCKS_PER_SEC));
	}
	sQ = sum_Q;

	if (verbose > 10) {
		t1 = clock();
	}
	error = constraint_compute_gradient_positive(val_P, neighbors, indptr, ind_len, pos_reference, pos_f, 
		n_dimensions, constraint_pos, constraint_N, sQ, start, need_eval_error, constraint_weight);

	if (verbose > 10) {
		t2 = clock();
		printf("[t-SNE] Computing positive gradient: %4.4f seconds\n", ((double)(t2 - t1) / CLOCKS_PER_SEC));
	}

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int i = start; i < N; i++) {
		for (int j = 0; j < n_dimensions; j++) {
			int coord = i * n_dimensions + j;
			tot_force[coord] = pos_f[coord] - (neg_f[coord] / sQ);
		}
	}

	delete[] neg_f;
	delete[] pos_f;
	return error;
	
}

double TSNE::compute_gradient(double* val_P, int* neighbors, int length, int* indptr, int ind_len, double* pos_reference,
	double* tot_force, int N, int n_dimensions, QuadTree* qt, double theta, int start,
	int stop, bool need_eval_error) {
	// Having created the tree, calculate the gradient
	// in two components, the positive and negative forces
	double sum_Q = 0.0;
	clock_t t1 = 0, t2 = 0;
	double sQ, error;
	double* pos_f = new double[N * n_dimensions]();
	double* neg_f = new double[N * n_dimensions]();

	if (pos_f == NULL || neg_f == NULL) {
		printf("[t-SNE] Memory allocation failed!\n");
		exit(1);
	}

	if (verbose > 10) {
		t1 = clock();
	}
    //printf("point11\n");


	// NoneEdge forces

#ifdef _OPENMP
#pragma omp parallel for reduction(+:sum_Q)
#endif
	for (int n = 0; n < N; n++) {
		double this_Q = .0;
		qt->constraint_compute_non_edge_forces(pos_reference + n * n_dimensions, theta, neg_f + n * n_dimensions, &this_Q);
		sum_Q += this_Q;
	}

	/*compute_gradient_negative(neg_f, N, n_dimensions, qt, &sum_Q, theta, start, 
								stop);*/
    //printf("point12\n");
	if (verbose > 10) {
		t2 = clock();
		printf("[t-SNE] Computing negative gradient: %4.4f seconds\n", ((double)(t2 - t1) / CLOCKS_PER_SEC));
	}
	sQ = sum_Q;

	if (verbose > 10) {
		t1 = clock();
	}
    //printf("point13\n");
	error = compute_gradient_positive(val_P, neighbors, indptr, ind_len, pos_reference, pos_f,
										n_dimensions, sQ, start, need_eval_error);
    //printf("point14\n");

	if (verbose > 10) {
		t2 = clock();
		printf("[t-SNE] Computing positive gradient: %4.4f seconds\n", ((double)(t2 - t1) / CLOCKS_PER_SEC));
	}
	/*double grad_norm1 = 0.0, grad_norm2 = 0.0;
	for (int i = 0; i < N * n_dimensions; i++) {
		grad_norm1 += pos_f[i] * pos_f[i];
		grad_norm2 += neg_f[i] / sQ * neg_f[i] / sQ;
	}
	printf("grad_norm_pos:%f\ngrad_norm_neg:%f\n\n", grad_norm1, grad_norm2);*/


#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int i = start; i < N; i++) {
		for (int j = 0; j < n_dimensions; j++) {
			int coord = i * n_dimensions + j;
			tot_force[coord] = pos_f[coord] - (neg_f[coord] / sQ);
		}
	}

	delete[] neg_f;
	delete[] pos_f;
	return error;
}

void TSNE::constraint_compute_gradient_negative(double* pos_reference, double* neg_f, int N, int n_dimensions, int constraint_N,
	QuadTree* qt, double* sum_Q, double theta, int start, int stop, bool need_eval_error) {
	if (stop == -1) {
		stop = N;
	}

	double dta = 0.0, dtb = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:dta, dtb)
#endif
	for (int i = start; i < stop; i++) {
		double s_Q = 0.0;
		int offset = n_dimensions + 2;
		clock_t t1 = 0, t2 = 0, t3 = 0;
		double neg_force[3];
		double iQ;
		double* summary = new double[constraint_N * offset]();
		if (summary == NULL) {
				printf("[t-SNE] Memory allocation failed!\n");
				exit(1);
		}
		// Clear the arrays
		for (int j = 0; j < n_dimensions; j++) {
			neg_force[j] = 0.0;
		}
		iQ = 0.0;
		// Find which nodes are summarizing and collect their centers of mass
		// deltas, and sizes, into vectorized arrays
		if (verbose > 10) {
			t1 = clock();
		}
		int idx = qt->constraint_summarize(pos_reference + start * n_dimensions, theta, summary, 0);

		if (verbose > 10) {
			t2 = clock();
		}
		// Compute the t - SNE negative force
		// for the digits dataset, walking the tree
		// is about 10 - 15x more expensive than the
		// following for loop
		double dist2s, size, qijZ, mult;
		for (int j = 0; j < idx / offset; j++) {
			dist2s = summary[j * offset + n_dimensions];
			size = summary[j * offset + n_dimensions + 1];
			qijZ = 1.0 / (1.0 + dist2s); // 1 / (1 + dist)

			s_Q += size * qijZ;   // size of the node * q
			mult = size * qijZ * qijZ;

#ifdef _OPENMP
#pragma omp parallel for
#endif
			for (int k = 0; k < n_dimensions; k++) {
				neg_force[k] += mult * summary[j * offset + k];
			}
		}

#ifdef _OPENMP
#pragma omp critical
#endif
		*sum_Q += s_Q;

		if (verbose > 10) {
			t3 = clock();
		}

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int j = 0; j < n_dimensions; j++) {
			neg_f[i * n_dimensions + j] = neg_force[j];
		}
		if (verbose > 10) {
			dta += double(t2 - t1) / CLOCKS_PER_SEC;
			dtb += double(t3 - t2) / CLOCKS_PER_SEC;
		}
		delete[] summary;
	}
	if (verbose > 10) {
		printf("[t-SNE] Tree: %4.4f seconds | Force computation: %4.4f seconds\n", dta, dtb);
	}

	// Put sum_Q to machine EPSILON to avoid divisions by 0
	*sum_Q = max(*sum_Q, DBL_MIN);
	return;
}

void TSNE::compute_gradient_negative(double* neg_f, int N, int n_dimensions, QuadTree* qt,
	double* sum_Q, double theta, int start, int stop, bool need_eval_error) {
	if (stop == -1) {
		stop = N;
	}
	double dta = 0.0, dtb = 0.0;


#ifdef _OPENMP
#pragma omp parallel for reduction(+:dta, dtb)
#endif
	for (int i = start; i < stop; i++) {
		double s_Q = 0.0;
		int offset = n_dimensions + 2;
		clock_t t1 = 0, t2 = 0, t3 = 0;
		double neg_force[3];
		double iQ;
		double* summary = new double[N * offset]();
		if (summary == NULL) {
			printf("[t-SNE] Memory allocation failed!\n");
			exit(1);
	}
		// Clear the arrays

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int j = 0; j < n_dimensions; j++) {
			neg_force[j] = 0.0;
		}
		iQ = 0.0;
		// Find which nodes are summarizing and collect their centers of mass
		// deltas, and sizes, into vectorized arrays
		if (verbose > 10) {
			t1 = clock();
		}
		int idx = qt->summarize(i, theta, summary, 0);
		
		if (verbose > 10) {
			t2 = clock();
		}
		// Compute the t - SNE negative force
		// for the digits dataset, walking the tree
		// is about 10 - 15x more expensive than the
		// following for loop
		double dist2s, size, qijZ, mult;
		for (int j = 0; j < idx / offset; j++) {
			dist2s = summary[j * offset + n_dimensions];
			size = summary[j * offset + n_dimensions + 1];
			qijZ = 1.0 / (1.0 + dist2s); // 1 / (1 + dist)
			s_Q += size * qijZ;   // size of the node * q
			mult = size * qijZ * qijZ;

#ifdef _OPENMP
#pragma omp parallel for 
#endif
			for (int k = 0; k < n_dimensions; k++) {
				neg_force[k] += mult * summary[j * offset + k];
			}
		}

#ifdef _OPENMP
#pragma omp critical
#endif
		*sum_Q += s_Q;

		if (verbose > 10) {
			t3 = clock();
		}

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int j = 0; j < n_dimensions; j++) {
			neg_f[i * n_dimensions + j] = neg_force[j];
		}
		if (verbose > 10) {
			dta += double(t2 - t1) / CLOCKS_PER_SEC;
			dtb += double(t3 - t2) / CLOCKS_PER_SEC;
		}
		delete[] summary;
	}
	if (verbose > 10) {
		printf("[t-SNE] Tree: %4.4f seconds | Force computation: %4.4f seconds\n", dta, dtb);
	}

	// Put sum_Q to machine EPSILON to avoid divisions by 0
	*sum_Q = max(*sum_Q, DBL_MIN);
	return;
}

double TSNE::constraint_compute_gradient_positive(double* val_P, int* neighbors, int* indptr, int ind_len,
	double* pos_reference, double* pos_f, int n_dimensions, double* constraint_pos, int constraint_N, double sum_Q, int start, bool need_eval_error, double* constraint_weight) {
	// Sum over the following expression for i not equal to j
	// grad_i = p_ij(1 + || y_i - y_j || ^ 2) ^ -1 (y_i - y_j)
	// This is equivalent to compute_edge_forces in the authors' code
	// It just goes over the nearest neighbors instead of all the data points
	// (unlike the non-nearest neighbors version of `compute_gradient_positive')
	clock_t t1 = 0, t2 = 0;
	if (verbose > 10) {
		t1 = clock();
	}
	int n_samples = ind_len - 1;
	double C = 0.0;

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int i = start; i < n_samples; i++) {
		// Init the gradient vector
		for (int j = 0; j < n_dimensions; j++) {
			pos_f[i * n_dimensions + j] = 0.0;
		}
	}

	for (int j = 0; j < constraint_N; j++) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int k = indptr[j]; k < indptr[j + 1]; k++) {
			int i = neighbors[k];
			double dij = 0.0, pij = val_P[k];

			double buff[3];
			for (int ax = 0; ax < n_dimensions; ax++) {
				buff[ax] = pos_reference[i * n_dimensions + ax] - constraint_pos[j * n_dimensions + ax];
				dij += buff[ax] * buff[ax];
			}
			double qij = 1.0 / (1.0 + dij);
			dij = pij * qij;
			qij /= sum_Q;
			if (need_eval_error) {
				C += pij * log((pij + DBL_MIN) / (qij + DBL_MIN));
			}

			for (int ax = 0; ax < n_dimensions; ax++) {
				pos_f[i * n_dimensions + ax] += dij * buff[ax] * constraint_weight[j];
			}
		}
	}

	if (verbose > 10) {
		t2 = clock();
		printf("[t-SNE] Computed error=%4.4f in %4.4f seconds\n", C, (double)(t2 - t1) / CLOCKS_PER_SEC);
	}
	return C;
}

double TSNE::compute_gradient_positive(double* val_P, int* neighbors, int* indptr, int ind_len,
	double* pos_reference, double* pos_f, int n_dimensions, double sum_Q, int start, bool need_eval_error) {
	// Sum over the following expression for i not equal to j
	// grad_i = p_ij(1 + || y_i - y_j || ^ 2) ^ -1 (y_i - y_j)
	// This is equivalent to compute_edge_forces in the authors' code
	// It just goes over the nearest neighbors instead of all the data points
	// (unlike the non-nearest neighbors version of `compute_gradient_positive')
	clock_t t1 = 0, t2 = 0;
	if (verbose > 10) {
		t1 = clock();
	}
	int n_samples = ind_len - 1;
	double C = 0.0;

#ifdef _OPENMP
#pragma omp parallel for reduction(+:C)
#endif
	for (int i = start; i < n_samples; i++) {
		int ind1 = i * n_dimensions;
		// Compute the positive interaction for the nearest neighbors
		
		for (int k = indptr[i]; k < indptr[i + 1]; k++) {
			double D = .0;
			int ind2 = neighbors[k] * n_dimensions;
			for (int d = 0; d < n_dimensions; d++) {
				double t = pos_reference[ind1 + d] - pos_reference[ind2 + d];
				D += t * t;
			}
			double dij = 0.0, pij = val_P[k];
			double qij = 1.0 / (1.0 + D);
			dij = pij * qij;
			qij /= sum_Q;
			if (need_eval_error) {
				C += pij * log((pij + FLT_MIN) / (qij + FLT_MIN));
			}

			for (int ax = 0; ax < n_dimensions; ax++) {
				pos_f[ind1 + ax] += dij * (pos_reference[ind1 + ax] - pos_reference[ind2 + ax]);
			}
		}
	}
	if (verbose > 10) {
		t2 = clock();
		printf("[t-SNE] Computed error=%4.4f in %4.4f seconds\n", C, (double)(t2 - t1) / CLOCKS_PER_SEC);
	}
	return C;
}

void TSNE::symmetrize_matrix(int** _rowP, int** _colP, double** _valP, int N) {
	// Get sparse matrix
	int* rowP = *_rowP;
	int* colP = *_colP;
	double* valP = *_valP;

	// Count number of elements and row counts of symmetric matrix
	int* rowCounts = new int[N];
	if (rowCounts == NULL) { 
		printf("Memory allocation failed!\n"); 
		exit(1); 
	}

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int i = 0; i < N; i++) {
		rowCounts[i] = 0;
	}

	for (int n = 0; n < N; n++) {
		for (int i = rowP[n]; i < rowP[n + 1]; i++) {
			int neighbor_index = colP[i];
			// Check whether element (colP[i], n) is present
			bool present = false;
			for (int m = rowP[neighbor_index]; m < rowP[neighbor_index + 1]; m++) {
				if (colP[m] == n) { 
					present = true; 
					if (valP[m] + valP[i] != 0.0) {
						rowCounts[n]++;
					}
					break;
				}
			}
			if (!present && valP[i] != 0) {
				rowCounts[n]++;
				rowCounts[neighbor_index]++;
			}
		}
	}
	int noElem = 0;
	for (int n = 0; n < N; n++) { 
		noElem += rowCounts[n]; 
	}

	// Allocate memory for symmetrized matrix
	int* symRowP = new int[N + 1];
	int* symColP = new int[noElem];
	double* symValP = new double[noElem];

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int i = 0; i < noElem; i++) {
		symColP[i] = -1;
		symValP[i] = -1.0;
	}
	if (symRowP == NULL || symColP == NULL || symValP == NULL) { 
		printf("Memory allocation failed!\n"); 
		exit(1); 
	}

	// Construct new row indices for symmetric matrix
	symRowP[0] = 0;
	for (int n = 0; n < N; n++) { 
		symRowP[n + 1] = symRowP[n] + rowCounts[n]; 
	}

	// Fill the result matrix
	int* offset = new int[N];
	if (offset == NULL) { 
		printf("Memory allocation failed!\n"); 
		exit(1); 
	}

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int i = 0; i < N; i++) {
		offset[i] = 0;
	}
	for (int n = 0; n < N; n++) {
		for (int i = rowP[n]; i < rowP[n + 1]; i++) {                                 // considering element(n, colP[i])
			// Check whether element (col_P[i], n) is present
			bool present = false;
			for (int m = rowP[colP[i]]; m < rowP[colP[i] + 1]; m++) {
				if (colP[m] == n) {
					present = true;
					if (n <= colP[i] && valP[i] + valP[m] != 0.0) {                                                // make sure we do not add elements twice
						symColP[symRowP[n] + offset[n]] = colP[i];
						symColP[symRowP[colP[i]] + offset[colP[i]]] = n;
						symValP[symRowP[n] + offset[n]] = valP[i] + valP[m];
						symValP[symRowP[colP[i]] + offset[colP[i]]] = valP[i] + valP[m];

						offset[n]++;
						if (colP[i] != n) {
							offset[colP[i]]++;
						}
					}
					break;
				}
			}

			// If (colP[i], n) is not present, there is no addition involved
			if (!present && valP[i] != 0.0) {
				symColP[symRowP[n] + offset[n]] = colP[i];
				symColP[symRowP[colP[i]] + offset[colP[i]]] = n;
				symValP[symRowP[n] + offset[n]] = valP[i];
				symValP[symRowP[colP[i]] + offset[colP[i]]] = valP[i];
				offset[n]++;
				if (colP[i] != n) {
					offset[colP[i]]++;
				}		
			}
		}
	}

	/*for (int n = 0; n < N; n++) {
		for (int i = symRowP[n]; i < symRowP[n + 1]; i++) {
			printf("(%d, %d)\t%f\n", n, symColP[i], symValP[i]);
		}
	}*/

	double sum = 0.0;

#ifdef _OPENMP
#pragma omp parallel for reduction(+:sum)
#endif
	for (int i = 0; i < noElem; i++) { 
		sum += symValP[i]; 
	}
	sum = max(sum, DBL_EPSILON);
	// Divide the result by sum
	bool flag = false;

	for (int i = 0; i < noElem; i++) {
		symValP[i] /= sum;
		flag = flag || (abs_d(symValP[i]) > 1.0);
	}
	if (flag) {
		printf("All value should in range [-1.0, 1.0]");
	}

	// Free up some memery
	delete[] offset;
	delete[] rowCounts;

	delete[] * _rowP;
	delete[] * _colP;
	delete[] * _valP;
	// Return symmetrized matrices
	*_rowP = symRowP;
	*_colP = symColP;
	*_valP = symValP;
}

void TSNE::multi_symmetrize_matrix(int** _row_P, int** _col_P, double** _val_P, int N) {
	// Get sparse matrix
	int* row_P = *_row_P;
	int* col_P = *_col_P;
	double* val_P = *_val_P;

	// Count number of elements and row counts of symmetric matrix
	int* row_counts = (int*)calloc(N, sizeof(int));
	if (row_counts == NULL) { 
		printf("[t-SNE] Memory allocation failed!\n"); 
		exit(1); 
	}
	for (int n = 0; n < N; n++) {
		for (int i = row_P[n]; i < row_P[n + 1]; i++) {

			// Check whether element (col_P[i], n) is present
			bool present = false;
			for (int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
				if (col_P[m] == n) {
					present = true;
					break;
				}
			}
			if (present) {
				row_counts[n]++;
			}
			else {
				row_counts[n]++;
				row_counts[col_P[i]]++;
			}
		}
	}
	int no_elem = 0;
	for (int n = 0; n < N; n++) {
		no_elem += row_counts[n];
	}
	// Allocate memory for symmetrized matrix
	int*    sym_row_P = (int*)malloc((N + 1) * sizeof(int));
	int*    sym_col_P = (int*)malloc(no_elem * sizeof(int));
	double* sym_val_P = (double*)malloc(no_elem * sizeof(double));
	if (sym_row_P == NULL || sym_col_P == NULL || sym_val_P == NULL) { 
		printf("[t-SNE] Memory allocation failed!\n"); 
		exit(1); 
	}

	// Construct new row indices for symmetric matrix
	sym_row_P[0] = 0;
	for (int n = 0; n < N; n++) sym_row_P[n + 1] = sym_row_P[n] + row_counts[n];

	// Fill the result matrix
	int* offset = (int*)calloc(N, sizeof(int));
	if (offset == NULL) { 
		printf("[t-SNE] Memory allocation failed!\n"); 
		exit(1); 
	}
	for (int n = 0; n < N; n++) {
		for (int i = row_P[n]; i < row_P[n + 1]; i++) {                                 // considering element(n, col_P[i])

																						// Check whether element (col_P[i], n) is present
			bool present = false;
			for (int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
				if (col_P[m] == n) {
					present = true;
					if (n <= col_P[i]) {                                                // make sure we do not add elements twice
						sym_col_P[sym_row_P[n] + offset[n]] = col_P[i];
						sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
						sym_val_P[sym_row_P[n] + offset[n]] = val_P[i] + val_P[m];
						sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i] + val_P[m];
					}
				}
			}

			// If (col_P[i], n) is not present, there is no addition involved
			if (!present) {
				sym_col_P[sym_row_P[n] + offset[n]] = col_P[i];
				sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
				sym_val_P[sym_row_P[n] + offset[n]] = val_P[i];
				sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i];
			}

			// Update offsets
			if (!present || (n <= col_P[i])) {
				offset[n]++;
				if (col_P[i] != n) {
					offset[col_P[i]]++;
				}
			}
		}
	}

	// Divide the result by two
	for (int i = 0; i < no_elem; i++) {
		sym_val_P[i] /= 2.0;
	}

	// Return symmetrized matrices
	free(*_row_P); *_row_P = sym_row_P;
	free(*_col_P); *_col_P = sym_col_P;
	free(*_val_P); *_val_P = sym_val_P;

	// Free up some memery
	free(offset); offset = NULL;
	free(row_counts); row_counts = NULL;
}

void TSNE::compute_sum(double* X, int N, int D, int axis, double* res) {
	int n;
	if (axis == 0) {
		n = D;

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < n; i++) {
			res[i] = 0.0;
		}
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < D; j++) {
				res[j] += X[i * D + j];
			}
		}
		return;
	}
	else if (axis == 1) {
		n = N;

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < n; i++) {
			res[i] = 0.0;
		}
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < D; j++) {
				res[i] += X[i * D + j];
			}
		}
		return;
	}
	return;
}

double TSNE::compute_sum(double* X, int N) {
	double sum = 0.0;

#ifdef _OPENMP
#pragma omp parallel for reduction(+:sum)
#endif
	for (int i = 0; i < N; i++) {
		sum += X[i];
	}
	return sum;
}

void TSNE::compute_distance(double* X, int N, int D, double* dist) {
	int count = 0;
	for (int i = 0; i < N; i++) {
		for (int j = i + 1; j < N; j++) {
			double temp = squared_euclidean_distance(X + i * D, X + j * D, D) + 1.0;
			temp = 1.0 / temp;
			dist[count] = temp;
			count++;
		}
	}
}

double TSNE::squared_euclidean_distance(double* a, double* b, int length) {
	double dist = 0.0;

#ifdef _OPENMP
#pragma omp parallel for reduction(+:dist)
#endif
	for (int i = 0; i < length; i++) {
		dist += (a[i] - b[i]) * (a[i] - b[i]);
	}
	return dist;
}

double TSNE::compute_maximum_p(double *P, int length) {
	double maximum = DBL_MIN;
	for (int i = 0; i < length; i++) {
		maximum = max(maximum, P[i]);
	}
	return maximum;
}

double TSNE::dot(double* x, double* y, int length) {
	double res = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:res)
#endif
	for (int i = 0; i < length; i++) {
		res += x[i] * y[i];
	}
	return res;
}

bool TSNE::any_not_zero(double* X, int length) {

	for (int i = 0; i < length; i++) {
		if (X[i] != 0.0) {
			return true;
		}
	}
	return false;
}

void TSNE::matrix_operation(double* X1, int N1, int D1, double* X2, int N2, int D2, int oper, double* res) {
	// oper 0: +, 1: -, 2: *
	int N, D;
	if (oper == 0) {
		if (N1 == N2 && D1 == D2) {
			N = N1;
			D = D1;

#ifdef _OPENMP
#pragma omp parallel for
#endif
			for (int i = 0; i < N; i++) {
				for (int j = 0; j < D; j++) {
					res[i * D + j] = X1[i * D + j] + X2[i * D + j];
				}
			}
			return;
		}
		else if (N1 == D2 && N2 == 1 && D1 == 1) {
			N = N1;
			D = D2;
#ifdef _OPENMP
#pragma omp parallel for
#endif
			for (int i = 0; i < N; i++) {
				for (int j = 0; j < D; j++) {
					res[i * D + j] = X1[i] + X2[j];
				}
			}
			return;
		}
		else if (N2 == D1 && N1 == 1 && D2 == 1) {
			N = N2;
			D = D1;

#ifdef _OPENMP
#pragma omp parallel for
#endif
			for (int i = 0; i < N; i++) {
				for (int j = 0; j < D; j++) {
					res[i * D + j] = X1[j] + X2[i];
				}
			}
			return;
		}
	}
	else if (oper == 1) {
		if (N1 == N2 && D1 == D2) {
			N = N1;
			D = D1;

#ifdef _OPENMP
#pragma omp parallel for
#endif
			for (int i = 0; i < N; i++) {
				for (int j = 0; j < D; j++) {
					res[i * D + j] = X1[i * D + j] - X2[i * D + j];
				}
			}
			return;
		}
		else if (N1 == D2 && N2 == 1 && D1 == 1) {
			N = N1;
			D = D2;
#ifdef _OPENMP
#pragma omp parallel for
#endif
			for (int i = 0; i < N; i++) {
				for (int j = 0; j < D; j++) {
					res[i * D + j] = X1[i] - X2[j];
				}
			}
			return;
		}
		else if (N2 == D1 && N1 == 1 && D2 == 1) {
			N = N2;
			D = D1;

#ifdef _OPENMP
#pragma omp parallel for 
#endif
			for (int i = 0; i < N; i++) {
				for (int j = 0; j < D; j++) {
					res[i * D + j] = X1[j] - X2[i];
				}
			}
			return;
		}
	}
	else if (oper == 2) {
		if (N1 == N2 && D1 == D2) {
			N = N1;
			D = D1;

#ifdef _OPENMP
#pragma omp parallel for
#endif
			for (int i = 0; i < N; i++) {
				for (int j = 0; j < D; j++) {
					res[i * D + j] = X1[i * D + j] * X2[i * D + j];
				}
			}
			return;
		}
		else if (N1 == D2 && N2 == 1 && D1 == 1) {
			N = N1;
			D = D2;

#ifdef _OPENMP
#pragma omp parallel for
#endif
			for (int i = 0; i < N; i++) {
				for (int j = 0; j < D; j++) {
					res[i * D + j] = X1[i] * X2[j];
				}
			}
			return;
		}
		else if (N2 == D1 && N1 == 1 && D2 == 1) {
			N = N2;
			D = D1;

#ifdef _OPENMP
#pragma omp parallel for
#endif
			for (int i = 0; i < N; i++) {
				for (int j = 0; j < D; j++) {
					res[i * D + j] = X1[j] * X2[i];
				}
			}
			return;
		}
	}
}

void TSNE::zero_mean(double* X, int N, int D) {

	// Compute data mean
	double* mean = (double*)calloc(D, sizeof(double));
	if (mean == NULL) {
		printf("Memory allocation failed!\n");
		exit(1);
	}
	for (int n = 0; n < N; n++) {
		for (int d = 0; d < D; d++) {
			mean[d] += X[n * D + d];
		}
	}
	for (int d = 0; d < D; d++) {
		mean[d] /= (double)N;
	}

	// Subtract data mean
	for (int n = 0; n < N; n++) {
		for (int d = 0; d < D; d++) {
			X[n * D + d] -= mean[d];
		}
	}
	free(mean); mean = NULL;
}

void TSNE::compute_gaussian_perplexity(double* X, int N, int D, int** _row_P, int** _col_P, double** _val_P, double perplexity, int K, int verbose, int forest_size, double accuracy) {

	if (perplexity > K) {
		printf("[t-SNE] Perplexity should be lower than K!\n");
	}

	// Allocate the memory we need
	*_row_P = (int*)malloc((N + 1) * sizeof(int));
	*_col_P = (int*)calloc(N * K, sizeof(int));
	*_val_P = (double*)calloc(N * K, sizeof(double));
	if (*_row_P == NULL || *_col_P == NULL || *_val_P == NULL) {
		printf("[t-SNE] Memory allocation failed!\n");
		exit(1);
	}

	/*
	row_P -- offsets for `col_P` (i)
	col_P -- K nearest neighbors indices (j)
	val_P -- p_{i | j}
	*/

	int* row_P = *_row_P;
	int* col_P = *_col_P;
	double* val_P = *_val_P;

	row_P[0] = 0;
	for (int n = 0; n < N; n++) {
		row_P[n + 1] = row_P[n] + K;
	}





	// Build forest on data set
	std::vector<Node> obj_X(N, Node());

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int i = 0; i < N; i++) {
		obj_X[i] = Node(i, -1, X + i * D, D, -1, -1);
	}
	// Loop over all points to find nearest neighbors
	if (verbose) {
		printf("[t-SNE] Building tree...\n");
	}
	int leaf_number = int(0.05 * N);
	if (accuracy > 0.3) {
		leaf_number += int(0.4 * (accuracy - 0.3) * N);
	}
	KdTreeForest *forest = new KdTreeForest(obj_X, N, D, forest_size, 5, leaf_number);


	int steps_completed = 0;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int n = 0; n < N; n++)
	{
		std::vector<double> cur_P(K);
		std::vector<int> indices;
		std::vector<double> distances;

		// Find nearest neighbors
		forest->priority_search(obj_X[n], K + 1, &indices, &distances);

		// Initialize some variables for binary search
		bool found = false;
		double beta = 1.0;
		double min_beta = -DBL_MAX;
		double max_beta = DBL_MAX;
		double tol = 1e-5;

		// Iterate until we found a good perplexity
		int iter = 0; double sum_P;
		while (!found && iter < 100) {

			// Compute Gaussian kernel row
			for (int m = 0; m < K; m++) {
				cur_P[m] = exp(-beta * distances[m + 1]);
			}

			// Compute entropy of current row
			sum_P = DBL_MIN;
			for (int m = 0; m < K; m++) {
				sum_P += cur_P[m];
			}
			double H = .0;
			for (int m = 0; m < K; m++) {
				H += beta * (distances[m + 1] * cur_P[m]);
			}
			H = (H / sum_P) + log(sum_P);

			// Evaluate whether the entropy is within the tolerance level
			double Hdiff = H - log(perplexity);
			if (Hdiff < tol && -Hdiff < tol) {
				found = true;
			}
			else {
				if (Hdiff > 0) {
					min_beta = beta;
					if (max_beta == DBL_MAX || max_beta == -DBL_MAX)
						beta *= 2.0;
					else
						beta = (beta + max_beta) / 2.0;
				}
				else {
					max_beta = beta;
					if (min_beta == -DBL_MAX || min_beta == DBL_MAX)
						beta /= 2.0;
					else
						beta = (beta + min_beta) / 2.0;
				}
			}

			// Update iteration counter
			iter++;
		}

		// Row-normalize current row of P and store in matrix
		for (int m = 0; m < K; m++) {
			cur_P[m] /= sum_P;
		}
		for (int m = 0; m < K; m++) {
			col_P[row_P[n] + m] = indices[m + 1];
			val_P[row_P[n] + m] = cur_P[m];
		}

		// Print progress
#ifdef _OPENMP
#pragma omp atomic
#endif
		++steps_completed;

		if (verbose && steps_completed % (N / 10) == 0)
		{
#ifdef _OPENMP
#pragma omp critical
#endif
			printf("[t-SNE]  - point %d of %d\n", steps_completed, N);
		}
	}

	// Clean up memory
	obj_X.clear();
	delete forest;
}

void TSNE::constraint_compute_gaussian_perplexity(double* X, int N, double* constraint_X, int constraint_N, int D, int** _row_P, int** _col_P, double** _val_P, double perplexity, int K, int verbose, int forest_size, double accuracy) {

	if (perplexity > K) {
		printf("[t-SNE] Perplexity should be lower than K!\n");
	}

	// Allocate the memory we need
	*_row_P = (int*)calloc(constraint_N + 1, sizeof(int));
	*_col_P = (int*)calloc(constraint_N * K, sizeof(int));
	*_val_P = (double*)calloc(constraint_N * K, sizeof(double));
	int *indptr = (int*)calloc(N + 1, sizeof(int)),
		*indexes = (int*)calloc(constraint_N * K, sizeof(int)),
		*offset = (int*)calloc(N, sizeof(int)), 
		*order = (int*)calloc(constraint_N * K, sizeof(int));
	if (*_row_P == NULL || *_col_P == NULL || *_val_P == NULL
		|| offset == NULL || order == NULL) {
		printf("[t-SNE] Memory allocation failed!\n");
		exit(1);
	}

	/*
	row_P -- offsets for `col_P` (i)
	col_P -- K nearest neighbors indices (j)
	val_P -- p_{i | j}
	*/

	int* row_P = *_row_P;
	int* col_P = *_col_P;
	double* val_P = *_val_P;

	for (int n = 0; n <= constraint_N; n++) {
		row_P[n] = n * K;
	}


	// Build forest on data set
	std::vector<Node> obj_X(N, Node());
	std::vector<Node> constraint_obj_X(constraint_N, Node());

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int i = 0; i < N; i++) {
		obj_X[i] = Node(i, -1, X + i * D, D, -1, -1);
	}

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int i = 0; i < constraint_N; i++) {
		constraint_obj_X[i] = Node(i, -1, constraint_X + i * D, D, -1, -1);
	}
	// Loop over all points to find nearest neighbors
	if (verbose) {
		printf("[t-SNE] Building tree...\n");
	}
	int leaf_number = int(0.05 * N);
	if (accuracy > 0.3) {
		leaf_number += int(0.4 * (accuracy - 0.3) * N);
	}
	KdTreeForest *forest = new KdTreeForest(obj_X, N, D, 1, 5, N);

	double *distance = (double*)calloc(constraint_N * K, sizeof(double));
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int n = 0; n < constraint_N; n++)
	{
		std::vector<double> cur_P(K);
		std::vector<int> indices;
		std::vector<double> distances;

		// Find nearest neighbors
		forest->priority_search(constraint_obj_X[n], K, &indices, &distances);
		for (int i = 0; i < K; i++) {
			col_P[n * K + i] = indices[i];
			distance[n * K + i] = distances[i];
		}
	}
	// Clean up memory
	delete forest;
	obj_X.clear();
	constraint_obj_X.clear();



	for (int n = 0; n < constraint_N * K; n++)
	{
		indptr[col_P[n] + 1]++;
	}
	for (int n = 1; n <= N; n++) {
		indptr[n] += indptr[n - 1];
	}

	for (int n = 0; n < constraint_N; n++)
	{
		for (int i = 0; i < K; i++) {
			int index = col_P[n * K + i];
			indexes[indptr[index] + offset[index]] = n;
			order[indptr[index] + offset[index]] = i;
			offset[index]++;
		}
	}
	delete[] offset;


	/*for (int i = 0; i < N; i++) {
		for (int j = row_P[i]; j < row_P[i + 1]; j++) {
			printf("%f ", col_P[j]);
		}
		printf("\n");
	}
	system("pause");*/

	int step_completed = 0;
	double beta_sum = 0.0;

#ifdef _OPENMP
#pragma omp parallel for reduction(+:step_completed, beta_sum)
#endif
	for (int i = 0; i < N; i++) {
		if (indptr[i] == indptr[i + 1]) {
			continue;
		}
		// Initialize some variables for binary search
		double beta = 1.0,
			min_beta = -DBL_MAX,
			max_beta = DBL_MAX,
			sum_P = 0.0,
			desired_entropy = log(perplexity),
			tol = 1e-5,
			entropy_diff, entropy, sum_dist_P;

		// Iterate until we found a good perplexity
		for (int iter = 0; iter < 100; iter++) {
			sum_P = 0.0;
			for (int j = indptr[i]; j < indptr[i + 1]; j++) {
				val_P[indexes[j] * K + order[j]] = exp(-distance[indexes[j] * K + order[j]] * beta);
				sum_P += val_P[indexes[j] * K + order[j]];
			}
			if (sum_P < DBL_EPSILON) {
				sum_P = DBL_EPSILON;
			}
			sum_dist_P = 0.0;
			for (int j = indptr[i]; j < indptr[i + 1]; j++) {
				val_P[indexes[j] * K + order[j]] /= sum_P;
				sum_dist_P += distance[indexes[j] * K + order[j]] * val_P[indexes[j] * K + order[j]];
			}
			entropy = log(sum_P) + beta * sum_dist_P;
			entropy_diff = entropy - desired_entropy;

			if (abs_d(entropy_diff) <= tol) {
				break;
			}
			if (entropy_diff > 0.0) {
				min_beta = beta;
				if (max_beta == DBL_MAX) {
					beta *= 2.0;
				}
				else {
					beta = (beta + max_beta) / 2.0;
				}
			}
			else {
				max_beta = beta;
				if (min_beta == -DBL_MAX) {
					beta /= 2.0;
				}
				else {
					beta = (beta + min_beta) / 2.0;
				}
			}
		}

		// Print progress
		step_completed += 1;
		beta_sum += beta;


		if (verbose > 0 && (step_completed % 1000 == 0 || step_completed == N))
		{
#ifdef _OPENMP
#pragma omp critical
#endif
			printf("[t-SNE]  - point %d of %d\n", step_completed, N);
		}
	}
	if (verbose > 0) {
#ifdef _OPENMP
#pragma omp critical
#endif
		printf("[t-SNE] Mean sigma: %f\n", sqrt((double)beta_sum / N));
	}
	delete[] indptr;
	delete[] indexes;
	delete[] order;
	delete[] distance;
}

double TSNE::multi_compute_gradient(int* inp_row_P, int* inp_col_P, double* inp_val_P, double* Y,
	int N, int no_dims, double* dC, double theta, bool eval_error) {
	// Construct quadtree on current map
	QuadTree* tree = new QuadTree(Y, N, no_dims);

	// Compute all terms required for t-SNE gradient
	double* Q = new double[N];
	double* pos_f = new double[N * no_dims]();
	double* neg_f = new double[N * no_dims]();

	double P_i_sum = 0.;
	double C = 0.;

	if (pos_f == NULL || neg_f == NULL) {
		printf("[t-SNE]  Memory allocation failed!\n"); 
		exit(1);
	}

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int n = 0; n < N; n++) {

		// NoneEdge forces
		double this_Q = .0;
		tree->compute_non_edge_forces(n, theta, neg_f + n * no_dims, &this_Q);
		Q[n] = this_Q;
	}
	double sum_Q = 0.0;
	for (int i = 0; i < N; i++) {
		sum_Q += Q[i];
	}

	clock_t t1, t2;
	if (verbose > 10) {
		t1 = clock();
	}

#ifdef _OPENMP
#pragma omp parallel for reduction(+:P_i_sum,C)
#endif
	for (int n = 0; n < N; n++) {
		// Edge forces
		int ind1 = n * no_dims;
		for (int i = inp_row_P[n]; i < inp_row_P[n + 1]; i++) {

			// Compute pairwise distance and Q-value
			double D = .0;
			int ind2 = inp_col_P[i] * no_dims;
			for (int d = 0; d < no_dims; d++) {
				double t = Y[ind1 + d] - Y[ind2 + d];
				D += t * t;
			}

			// Sometimes we want to compute error on the go
			if (eval_error) {
				P_i_sum += inp_val_P[i];
				C += inp_val_P[i] * log((inp_val_P[i] + FLT_MIN) / ((1.0 / (1.0 + D)) + FLT_MIN));
			}

			D = inp_val_P[i] / (1.0 + D);
			// Sum positive force
			for (int d = 0; d < no_dims; d++) {
				pos_f[ind1 + d] += D * (Y[ind1 + d] - Y[ind2 + d]);
			}
		}
	}
	if (verbose > 10) {
		t2 = clock();
		printf("[t-SNE] Computing positive gradient: %f seconds\n", double((double)(t2 - t1) / CLOCKS_PER_SEC));
	}


	// Compute final t-SNE gradient
	for (int i = 0; i < N * no_dims; i++) {
		dC[i] = pos_f[i] - (neg_f[i] / sum_Q);
	}

	delete tree;
	delete[] pos_f;
	delete[] neg_f;
	delete[] Q;

	C += P_i_sum * log(sum_Q);

	return C;
}

double TSNE::constraint_multi_compute_gradient(int* inp_row_P, int* inp_col_P, double* inp_val_P, double* Y,
	int N, double* constraint_Y, int constraint_N, int no_dims, double* dC, double theta, bool eval_error) {
	// Construct quadtree on current map
	QuadTree* tree = new QuadTree(constraint_Y, constraint_N, no_dims);

	// Compute all terms required for t-SNE gradient
	double* Q = new double[N];
	double* pos_f = new double[N * no_dims]();
	double* neg_f = new double[N * no_dims]();

	double P_i_sum = 0.;
	double C = 0.;

	if (pos_f == NULL || neg_f == NULL) {
		printf("[t-SNE]  Memory allocation failed!\n");
		exit(1);
	}

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int n = 0; n < N; n++) {
		// NoneEdge forces
		double this_Q = .0;
		tree->constraint_compute_non_edge_forces(Y + n * no_dims, theta, neg_f + n * no_dims, &this_Q);
		Q[n] = this_Q;
	}
	double sum_Q = 0.;
	for (int i = 0; i < N; i++) {
		sum_Q += Q[i];
	}

	for (int n = 0; n < constraint_N; n++) {
		// Edge forces
		int ind1 = n * no_dims;

#ifdef _OPENMP
#pragma omp parallel for reduction(+:P_i_sum,C)
#endif
		for (int i = inp_row_P[n]; i < inp_row_P[n + 1]; i++) {
			int ind2 = inp_col_P[i] * no_dims;

			// Compute pairwise distance and Q-value
			double D = .0;
			for (int d = 0; d < no_dims; d++) {
				double t = constraint_Y[ind1 + d] - Y[ind2 + d];
				D += t * t;
			}
			double qij = 1.0 / (1.0 + D);
			D = inp_val_P[i] * qij;
			qij /= sum_Q;
			// Sometimes we want to compute error on the go
			if (eval_error) {
				P_i_sum += inp_val_P[i];
				C += inp_val_P[i] * log(max(inp_val_P[i], DBL_EPSILON) / max(qij, DBL_EPSILON));
			}

			
			// Sum positive force
			for (int d = 0; d < no_dims; d++) {
				pos_f[ind2 + d] += D * (Y[ind2 + d] - constraint_Y[ind1 + d]);
			}
		}
	}

	// Compute final t-SNE gradient
	for (int i = 0; i < N * no_dims; i++) {
		dC[i] = pos_f[i] - (neg_f[i] / sum_Q);
	}


	delete tree;
	delete[] pos_f;
	delete[] neg_f;
	delete[] Q;

	C += P_i_sum * log(sum_Q);

	return C;
}

double TSNE::evaluate_error(int* row_P, int* col_P, double* val_P, double* Y, int N, int no_dims, double theta) {

	// Get estimate of normalization term
	QuadTree* tree = new QuadTree(Y, N, no_dims);

	double* buff = new double[no_dims]();
	double sum_Q = .0;
	for (int n = 0; n < N; n++) {
		tree->compute_non_edge_forces(n, theta, buff, &sum_Q);
}
	delete tree;
	delete[] buff;

	// Loop over all edges to compute t-SNE error
	double C = .0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:C)
#endif
	for (int n = 0; n < N; n++) {
		int ind1 = n * no_dims;
		for (int i = row_P[n]; i < row_P[n + 1]; i++) {
			double Q = .0;
			int ind2 = col_P[i] * no_dims;
			for (int d = 0; d < no_dims; d++) {
				double b = Y[ind1 + d] - Y[ind2 + d];
				Q += b * b;
			}
			Q = (1.0 / (1.0 + Q)) / sum_Q;
			C += val_P[i] * log((val_P[i] + FLT_MIN) / (Q + FLT_MIN));
		}
	}

	return C;
}

void TSNE::run_bhtsne(double* X, int N, int D, double* Y, int no_dim, double* constraint_X, double* constraint_Y,
	int constraint_N, double alpha, double perplexity, double angle, int n_jobs,
	int n_iter, int random_state, int forest_size, double accuracy, double early_exaggeration, double learning_rate,
	int skip_num_points, int exploration_n_iter, 
	int n_neighbors, int* neighbors_nn, int* constraint_neighbors_nn, double* distances_nn, double* constraint_distances_nn, double* constraint_weight) {
#ifdef _OPENMP
	omp_set_num_threads(NUM_THREADS(n_jobs));
#if _OPENMP >= 200805
	omp_set_schedule(omp_sched_guided, 0);
#endif
#endif
	if (verbose > 0) {
		printf("[t-SNE] Computing %d nearest neighbors...\n", n_neighbors);
	}
/*
	printf("X: ");
	for (int i = 0; i < 5; ++i) {
		printf("%lf ", X[i]);
	}
	for (int i = N * D - 5; i < N * D; ++i) {
		printf("%lf ", X[i]);
	}
	printf("\n");

	printf("Y: ");
	for (int i = 0; i < 5; ++i) {
		printf("%lf ", Y[i]);
	}
	for (int i = N * no_dim - 5; i < N * no_dim; ++i) {
		printf("%lf ", Y[i]);
	}
	printf("\n");

	printf("constraint_X: ");
	for (int i = 0; i < 5; ++i) {
		printf("%lf ", constraint_X[i]);
	}
	for (int i = constraint_N * D - 5; i < constraint_N * D; ++i) {
		printf("%lf ", constraint_X[i]);
	}
	printf("\n");

	printf("constraint_Y: ");
	for (int i = 0; i < 5; ++i) {
		printf("%lf ", constraint_Y[i]);
	}
	for (int i = constraint_N * no_dim - 5; i < constraint_N * no_dim; ++i) {
		printf("%lf ", constraint_Y[i]);
	}
	printf("\n");

	
	printf("constraint_weight: ");
	for (int i = 0; i < 5; ++i) {
		printf("%lf ", constraint_weight[i]);
	}
	for (int i = constraint_N - 5; i < constraint_N; ++i) {
		printf("%lf ", constraint_weight[i]);
	}
	printf("\n");

	
	printf("constraint_neighbors_nn: ");
	for (int i = 0; i < 5; ++i) {
		printf("%d ", constraint_neighbors_nn[i]);
	}
	for (int i = n_neighbors * constraint_N - 5; i < n_neighbors * constraint_N; ++i) {
		printf("%d ", constraint_neighbors_nn[i]);
	}
	printf("\n");

	
	printf("constraint_distances_nn: ");
	for (int i = 0; i < 5; ++i) {
		printf("%lf ", constraint_distances_nn[i]);
	}
	for (int i = n_neighbors * constraint_N - 5; i < n_neighbors * constraint_N; ++i) {
		printf("%lf ", constraint_distances_nn[i]);
	}
	printf("\n");

	
	printf("neighbors_nn: ");
	for (int i = 0; i < 5; ++i) {
		printf("%d ", neighbors_nn[i]);
	}
	for (int i = n_neighbors * N - 5; i < n_neighbors * N; ++i) {
		printf("%d ", neighbors_nn[i]);
	}
	printf("\n");

	
	printf("distances_nn: ");
	for (int i = 0; i < 5; ++i) {
		printf("%lf ", distances_nn[i]);
	}
	for (int i = n_neighbors * N - 5; i < n_neighbors * N; ++i) {
		printf("%lf ", distances_nn[i]);
	}
	printf("\n");
	*/

	// Find the nearest neighbors for every point
	clock_t t0 = clock(), t1;
	int temp_leaf_number = int(0.05 * N);
	if (accuracy > 0.3) {
		temp_leaf_number += int(0.4 * (accuracy - 0.3) * N);
	}

	if (accuracy == 1.0) {
		temp_leaf_number = forest_size * N;
	}
	//xsx_t1 = clock();
	//printf("[t-SNE] Computed neighbors for %d samples in %4.4fs...\n", N, (double)(xsx_t1 - xsx_t0) / CLOCKS_PER_SEC);

	/*if (verbose > 0) {
		t1 = clock();
		printf("[t-SNE] Computed neighbors for %d samples in %4.4fs...\n", N, (double)(t1 - t0) / CLOCKS_PER_SEC);
	}*/

	//xsx_t0 = clock();
	double *val_P = new double[N * n_neighbors],
		*constraint_val_P = new double[constraint_N * n_neighbors];
	binary_search_perplexity(distances_nn, neighbors_nn, N, n_neighbors, perplexity, val_P);
	constraint_binary_search_perplexity(constraint_distances_nn, constraint_neighbors_nn, constraint_N, n_neighbors, N, perplexity, constraint_val_P);
	int *indptr = new int[N + 1],
		*constraint_indptr = new int[constraint_N + 1];

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int i = 0; i < N + 1; i++) {
		indptr[i] = i * n_neighbors;
	}

#ifdef _OPENMP
#pragma omp parallel for 
#endif
	for (int i = 0; i < constraint_N + 1; i++) {
		constraint_indptr[i] = i * n_neighbors;
	}
	int *neighbors = new int[N * n_neighbors],
		*constraint_neighbors = new int[constraint_N * n_neighbors];

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int i = 0; i < N * n_neighbors; i++) {
		neighbors[i] = neighbors_nn[i];
	}

#ifdef _OPENMP
#pragma omp parallel for 
#endif
	for (int i = 0; i < constraint_N * n_neighbors; i++) {
		constraint_neighbors[i] = constraint_neighbors_nn[i];
	}
	symmetrize_matrix(&indptr, &neighbors, &val_P, N);
	double sum = 0.0;

#ifdef _OPENMP
#pragma omp parallel for reduction(+:sum)
#endif
	for (int i = 0; i < constraint_N * n_neighbors; i++) {
		sum += constraint_val_P[i];
		//std::cout << i << ": " << constraint_val_P[i] << ", sum = " << sum << '\n';
	}
	sum = max(sum, DBL_EPSILON);
	// Divide the result by sum
	bool flag = false;
	for (int i = 0; i < constraint_N; i++) {
		for (int j = 0; j < n_neighbors; j++) {
			constraint_val_P[i * n_neighbors + j] /= sum;
			//printf("%f ", constraint_val_P[i * n_neighbors + j]);
			flag = flag || (abs_d(constraint_val_P[i * n_neighbors + j]) > 1.0);
		}
		//printf("\n");
	}
	//system("pause");
	if (flag) {
		printf("All value should in range [-1.0, 1.0]");
	}
	//xsx_t1 = clock();
	//printf("[t-SNE] Computed joint_probabilities in %4.4fs...\n", (double)(xsx_t1 - xsx_t0) / CLOCKS_PER_SEC);

	// Learning schedule(part 1) : do 250 iteration with lower momentum but
	// higher learning rate controlled via the early exageration parameter

#ifdef _OPENMP
#pragma omp parallel for 
#endif
	for (int i = 0; i < indptr[N]; i++) {
		val_P[i] *= early_exaggeration;
	}

#ifdef _OPENMP
#pragma omp parallel for 
#endif
	for (int i = 0; i < constraint_indptr[constraint_N]; i++) {
		constraint_val_P[i] *= early_exaggeration;
	}
	//printf("point1\n");
	double *params = new double[N * no_dim],
		*update = new double[N * no_dim],
		*gains = new double[N * no_dim],
		*constraint_params = new double[constraint_N * no_dim];

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int i = 0; i < constraint_N * no_dim; i++) {
		constraint_params[i] = constraint_Y[i];
	}
	//printf("point2\n");
	double *grad = new double[N * no_dim],
		*constraint_grad = new double[N * no_dim];

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int i = 0; i < N * no_dim; i++) {
		params[i] = Y[i];
		update[i] = 0.0;
		gains[i] = 1.0;
		grad[i] = 0.0;
		constraint_grad[i] = 0.0;
		params[i] = floor(params[i] * 1e8) * 1e-8;
	}
	double best_error = DBL_MAX;

	//printf("point3\n");
	double *error = new double[1],
		*constraint_error = new double[1];
	error[0] = DBL_MAX;
	constraint_error[0] = DBL_MAX;
	int best_iter = 0;
	int n_iter_check = 10, n_iter_without_progress = 100;
	double min_gain = 0.01, momentum = 0.5, min_grad_norm = 1e-07;

	
	printf("constraint_val_P: ");
	for (int i = 0; i < 5; ++i) {
		printf("%lf ", constraint_val_P[i]);
	}
	for (int i = constraint_N * n_neighbors - 5; i < constraint_N * n_neighbors; ++i) {
		printf("%lf ", constraint_val_P[i]);
	}
	printf("\n");

	t0 = clock();
	//xsx_t0 = clock();
	// perform main loop
	int iter;
	for (iter = 0; iter < n_iter; iter++) {
		/*
		if (iter < 100) {
			printf("[iter #%d] ", iter);
			for (int i = 0; i < 20; ++i) {
				printf("%.7f ", params[i]);
			}
			printf("\n");
		}
		*/
		bool need_eval_error = iter % n_iter_check == 0 
			&& iter > 0 || iter == exploration_n_iter;
		//printf("point8\n");
		gradient(val_P, neighbors, indptr[N], indptr, N + 1, params, grad, N,
			no_dim, error, angle, skip_num_points, need_eval_error);

		//printf("point4\n");
		constraint_gradient(constraint_val_P, constraint_neighbors, constraint_indptr[constraint_N],
			constraint_indptr, constraint_N + 1, params, constraint_grad, N,
			no_dim, constraint_params, constraint_N, constraint_error,
			angle, skip_num_points, need_eval_error, constraint_weight);
/*
		if (iter < 100) {
			printf("[grad #%d] ", iter);
			for (int i = 0; i < 20; ++i) {
				printf("%.7f ", grad[i]);
			}
			printf("\n");
			printf("[grad #%d] ", iter);
			for (int i = 0; i < 10; ++i) {
				printf("%.7f ", constraint_grad[i]);
			}
			for (int i =  N * no_dim - 10; i < N * no_dim; ++i) {
				printf("%.7f ", constraint_grad[i]);
			}
			printf("\n");
		}
*/
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < N * no_dim; i++) {
			grad[i] = grad[i] * (1.0 - alpha) + constraint_grad[i] * alpha;
		}
		error[0] = error[0] * (1.0 - alpha) + constraint_error[0] * alpha;

		double grad_norm = 0.0;
		//printf("point5\n");

#ifdef _OPENMP
#pragma omp parallel for reduction(+:grad_norm)
#endif
		for (int i = 0; i < N * no_dim; i++) {
			if (isnan(grad[i]) || isinf(grad[i])) {
				printf("%d, %d, %f\n", i / no_dim, i % no_dim, grad[i]);
				system("pause");
			}
			grad[i] *= 4;
			grad_norm += grad[i] * grad[i];
			if (update[i] * grad[i] < 0.0) {
				if (DBL_MAX - gains[i] > 0.2) {
					gains[i] += 0.2;
				}
				else {
					gains[i] = DBL_MAX;
				}
			}
			else {
				gains[i] *= 0.8;
				if (gains[i] < min_gain) {
					gains[i] = min_gain;
				}
			}
			grad[i] *= gains[i];
			update[i] = momentum * update[i] - learning_rate * grad[i];
			params[i] += update[i];
			params[i] = floor(params[i] * 1e9) * 1e-9;
		}
		//printf("point6\n");
		grad_norm = sqrt(grad_norm);
		//printf("point7\n");
		if (iter % n_iter_check == 0 && iter > 0) {
			t1 = clock() - t0;
			t0 = clock();
			printf("[t-SNE] Iteration %d: error = %.7f, constraint=%.7f, gradient norm = %.7f (%d iterations in %0.3fs)\n",
				iter, error[0], constraint_error[0], grad_norm, n_iter_check, (double)(t1) / CLOCKS_PER_SEC);
			if (error[0] < best_error) {
				best_error = error[0];
				best_iter = iter;
			}
			else if (iter - best_iter > n_iter_without_progress && iter > exploration_n_iter) {
				printf("[t-SNE] Iteration %d: did not make any progress during the last %d episodes. Finished.\n",
					iter, n_iter_without_progress);
				break;
			}
			if (grad_norm <= min_grad_norm && iter > exploration_n_iter) {
				printf("[t-SNE] Iteration %d: gradient norm %f. Finished.\n", iter, grad_norm);
				break;
			}
		}
		if (iter == exploration_n_iter) {
			printf("[t-SNE] KL divergence after %d iterations with early exaggeration: %f\n", iter, error[0]);
			// Learning schedule(part 2) : disable early exaggeration and finish
			// optimization with a higher momentum at 0.8

#ifdef _OPENMP
#pragma omp parallel for
#endif
			for (int i = 0; i < indptr[N]; i++) {
				val_P[i] /= early_exaggeration;
			}

#ifdef _OPENMP
#pragma omp parallel for
#endif
			for (int i = 0; i < constraint_indptr[constraint_N]; i++) {
				constraint_val_P[i] /= early_exaggeration;
			}
			momentum = 0.8;
		}
	}

	// printf("max k is %d\n", max_k);
	/*
	if (max_k + 1 < n_records) {
		for (int i = 0; i < N * no_dim; i++) {
			layouts[(max_k + 1) * N * no_dim + i] = 1e10;
		}
	}
	*/

	//xsx_t1 = clock();
	//printf("[t-SNE] Computed main loop in %4.4fs...\n", (double)(xsx_t1 - xsx_t0) / CLOCKS_PER_SEC);
	if (verbose > 0) {
		printf("[t-SNE] Error after %d iterations: %f\n", iter, error[0]);
	}

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int i = 0; i < N * no_dim; i++) {
		Y[i] = params[i];
	}

	delete[] grad;
	delete[] error;
	delete[] gains;
	delete[] update;
	delete[] params;
	delete[] neighbors;
	delete[] indptr;
	delete[] val_P;
	delete[] constraint_val_P;
	delete[] constraint_indptr;
	delete[] constraint_neighbors;
	delete[] constraint_params;
	delete[] constraint_grad;
	delete[] constraint_error;
}

void TSNE::run_bhtsne(double* X, int N, int D, double* Y, int no_dim, double perplexity, double angle, int n_jobs,
	int n_iter, int random_state, int forest_size, double accuracy, double early_exaggeration, double learning_rate,
	int skip_num_points, int exploration_n_iter,
	int n_neighbors, int* neighbors_nn, double* distances_nn) {
#ifdef _OPENMP
	omp_set_num_threads(NUM_THREADS(n_jobs));
#if _OPENMP >= 200805
	omp_set_schedule(omp_sched_guided, 0);
#endif
#endif
	if (verbose > 0) {
		printf("[t-SNE] Computing %d nearest neighbors...\n", n_neighbors);
	}
	//clock_t xsx_t0, xsx_t1;

	// Find the nearest neighbors for every point
	clock_t t0 = clock(), t1;
	int temp_leaf_number = int(0.05 * N);
	if (accuracy > 0.3) {
		temp_leaf_number += int(0.4 * (accuracy - 0.3) * N);
	}
	if (accuracy == 1.0) {
		temp_leaf_number = forest_size * N;
	}
	//xsx_t0 = clock();
	//xsx_t1 = clock();
	//printf("[t-SNE] Computed neighbors for %d samples in %4.4fs...\n", N, (double)(xsx_t1 - xsx_t0) / CLOCKS_PER_SEC);

	if (verbose > 0) {
		t1 = clock();
		printf("[t-SNE] Computed neighbors for %d samples in %4.4fs...\n", N, (double)(t1 - t0) / CLOCKS_PER_SEC);
	}
	//xsx_t0 = clock();
	double* val_P = new double[N * n_neighbors];
	binary_search_perplexity(distances_nn, neighbors_nn, N, n_neighbors, perplexity, val_P);
	int* indptr = new int[N + 1];

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int i = 0; i < N + 1; i++) {
		indptr[i] = i * n_neighbors;
	}
	int* neighbors = new int[N * n_neighbors];

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int i = 0; i < N * n_neighbors; i++) {
		neighbors[i] = neighbors_nn[i];
	}
	/*for (int n = 0; n < 10; n++) {
	for (int i = indptr[n]; i < indptr[n + 1]; i++) {
	printf("(%d, %d)\t%f\n", n, neighbors[i], val_P[i]);
	}
	}*/
	symmetrize_matrix(&indptr, &neighbors, &val_P, N);
	//xsx_t1 = clock();
	//printf("[t-SNE] Computed joint_probabilities in %4.4fs...\n", (double)(xsx_t1 - xsx_t0) / CLOCKS_PER_SEC);

	// Learning schedule(part 1) : do 250 iteration with lower momentum but
	// higher learning rate controlled via the early exageration parameter

#ifdef _OPENMP
#pragma omp parallel for 
#endif
	for (int i = 0; i < indptr[N]; i++) {
		val_P[i] *= early_exaggeration;
	}
	double *params = new double[N * no_dim],
		*update = new double[N * no_dim],
		*gains = new double[N * no_dim];
	double* grad = new double[N * no_dim];

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int i = 0; i < N * no_dim; i++) {
		params[i] = Y[i];
		update[i] = 0.0;
		gains[i] = 1.0;
		grad[i] = 0;
	}
	double best_error = DBL_MAX;


	double* error = new double[1];
	error[0] = DBL_MAX;
	double g_dist_sum = 0.0, g_max_p = 0.0, g_min_p = 0.0;
	bool g_dist_sum_exist = false, g_max_p_exist = false, g_min_p_exist = false;
	int best_iter = 0;
	int n_iter_check = 10, n_iter_without_progress = 100;
	double min_gain = 0.01, momentum = 0.5, min_grad_norm = 1e-07;

	t0 = clock();
	//xsx_t0 = clock();
	// perform main loop
	int iter;
	for (iter = 0; iter < n_iter; iter++) {
		//printf("%d\n", iter);
		bool need_eval_error = iter % n_iter_check == 0
			&& iter > 0 || iter == exploration_n_iter;
		gradient(val_P, neighbors, indptr[N], indptr, N + 1, params, grad, N,
			no_dim, error, angle, skip_num_points, need_eval_error);
		double grad_norm = 0.0;

#ifdef _OPENMP
#pragma omp parallel for reduction(+:grad_norm)
#endif
		for (int i = 0; i < N * no_dim; i++) {
			grad[i] *= 4;
			grad_norm += grad[i] * grad[i];
			if (update[i] * grad[i] < 0.0) {
				if (DBL_MAX - gains[i] > 0.2) {
					gains[i] += 0.2;
				}
				else {
					gains[i] = DBL_MAX;
				}
			}
			else {
				gains[i] *= 0.8;
				if (gains[i] < min_gain) {
					gains[i] = min_gain;
				}
			}
			grad[i] *= gains[i];
			update[i] = momentum * update[i] - learning_rate * grad[i];
			params[i] += update[i];
			params[i] = floor(params[i] * 1e9) * 1e-9;
		}

		/*
		if (iter <= 100) {
			printf("it#%d: ", iter);
			for (int i = 0; i < 100; i++) {
				printf("%lf ", params[i]);
			}
			printf("\n");
		}
		*/
		grad_norm = sqrt(grad_norm);
		if (iter % n_iter_check == 0 && iter > 0) {
			t1 = clock();
			printf("[t-SNE] Iteration %d: error = %.7f, gradient norm = %.7f (%d iterations in %0.3fs)\n",
				iter, error[0], grad_norm, n_iter_check, (double)(t1 - t0) / CLOCKS_PER_SEC);
			t0 = clock();
			if (error[0] < best_error) {
				best_error = error[0];
				best_iter = iter;
			}
			else if (iter - best_iter > n_iter_without_progress && iter > exploration_n_iter) {
				printf("[t-SNE] Iteration %d: did not make any progress during the last %d episodes. Finished.\n",
					iter, n_iter_without_progress);
				break;
			}
			if (grad_norm <= min_grad_norm && iter > exploration_n_iter) {
				printf("[t-SNE] Iteration %d: gradient norm %f. Finished.\n", iter, grad_norm);
				break;
			}
		}
		if (iter == exploration_n_iter) {
			printf("[t-SNE] KL divergence after %d iterations with early exaggeration: %f\n", iter, error[0]);
			// Learning schedule(part 2) : disable early exaggeration and finish
			// optimization with a higher momentum at 0.8

#ifdef _OPENMP
#pragma omp parallel for
#endif
			for (int i = 0; i < indptr[N]; i++) {
				val_P[i] /= early_exaggeration;
			}
			momentum = 0.8;
		}
	}

	//xsx_t1 = clock();
	//printf("[t-SNE] Computed main loop in %4.4fs...\n", (double)(xsx_t1 - xsx_t0) / CLOCKS_PER_SEC);
	if (verbose > 0) {
		printf("[t-SNE] Error after %d iterations: %f\n", iter, error[0]);
	}

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int i = 0; i < N * no_dim; i++) {
		Y[i] = params[i];
	}

	delete[] grad;
	delete[] error;
	delete[] gains;
	delete[] update;
	delete[] params;
	delete[] neighbors;
	delete[] indptr;
	delete[] val_P;
}

void TSNE::multi_run_bhtsne(double* X, int N, int D, double* Y, int no_dim, double* constraint_X, double* constraint_Y,
	int constraint_N, double alpha, double perplexity, double angle, int n_jobs,
	int n_iter, int random_state, int forest_size, double accuracy, double early_exaggeration, double learning_rate,
	int skip_num_points, int exploration_n_iter) {
#ifdef _OPENMP
	omp_set_num_threads(NUM_THREADS(n_jobs));
#if _OPENMP >= 200805
	omp_set_schedule(omp_sched_guided, 0);
#endif
#endif

	if (N - 1 < 3 * perplexity) {
		perplexity = (N - 1) / 3;
		if (verbose) {
			printf("[t-SNE] Perplexity too large for the number of data points! Adjusting ...\n");
		}
	}

	if (verbose) {
		printf("[t-SNE] Using no_dims = %d, perplexity = %f, and theta = %f\n", no_dim, perplexity, angle);
	}

	/*int *neighbors_in_constraint = new int[N * 2];
	double *distances_in_constraint = new double[N * 2];
	k_neighbors(X, N, constraint_X, constraint_N, D, 2, neighbors_in_constraint, distances_in_constraint, 1, 5, constraint_N);

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int i = 0; i < N; i++) {
		double dis0 = distances_in_constraint[i * 2],
			dis1 = distances_in_constraint[i * 2 + 1];
		int index0 = neighbors_in_constraint[i * 2],
			index1 = neighbors_in_constraint[i * 2 + 1];
		if (dis0 + dis1 == 0.0) {
			for (int j = 0; j < no_dim; j++) {
				Y[i * no_dim + j] = (constraint_Y[index1 * no_dim + j]
					+ constraint_Y[index0 * no_dim + j]) / 2.0;
			}
		}
		else {
			for (int j = 0; j < no_dim; j++) {
				Y[i * no_dim + j] = (dis0 * constraint_Y[index1 * no_dim + j]
					+ dis1 * constraint_Y[index0 * no_dim + j]) / (dis0 + dis1);
			}
		}
	}
	delete[] neighbors_in_constraint;
	delete[] distances_in_constraint;*/


	// Set learning parameters
	double total_time = .0;
	clock_t start, end;
	int stop_lying_iter = exploration_n_iter, mom_switch_iter = exploration_n_iter;
	double momentum = .5, final_momentum = .8;
	double eta = learning_rate;
	//printf("[t-SNE] point1\n");

	// Allocate some memory
	double* dY = (double*)malloc(N * no_dim * sizeof(double)),
		*constraint_dY = (double*)malloc(N * no_dim * sizeof(double));
	double* uY = (double*)calloc(N * no_dim, sizeof(double));
	double* gains = (double*)malloc(N * no_dim * sizeof(double));
	//printf("[t-SNE] point2\n");
	if (dY == NULL || uY == NULL || gains == NULL || constraint_dY == NULL) {
		printf("[t-SNE] Memory allocation failed!\n");
		exit(1);
	}

	for (int i = 0; i < N * no_dim; i++) {
		gains[i] = 1.0;
	}
	//printf("[t-SNE] point3\n");

	// Normalize input data (to prevent numerical problems)
	if (verbose) {
		printf("[t-SNE] Computing input similarities...\n");
	}


	start = clock();
	//zero_mean(X, N, D);

	/*double max_X = .0;

	for (int i = 0; i < N * D; i++) {
		if (X[i] > max_X) {
			max_X = X[i];
		}
	}*/

	/*for (int i = 0; i < constraint_N * D; i++) {
		if (constraint_X[i] > max_X) {
			max_X = constraint_X[i];
		}
	}*/

	/*for (int i = 0; i < N * D; i++) {
		X[i] /= max_X;
	}

	for (int i = 0; i < constraint_N * D; i++) {
		constraint_X[i] /= max_X;
	}*/

	// Compute input similarities
	int* row_P; int* col_P; double* val_P;
	int* constraint_row_P; int* constraint_col_P; double* constraint_val_P;





	//==========================================================================
//	int n_neighbors = (int)(3 * perplexity);
//	int *constraint_neighbors_nn = new int[constraint_N * n_neighbors];
//	double *constraint_distances_nn = new double[constraint_N * n_neighbors];
//
//	int temp_leaf_number = int(0.05 * N);
//	if (accuracy > 0.3) {
//		temp_leaf_number += int(0.4 * (accuracy - 0.3) * N);
//	}
//	k_neighbors(constraint_X, constraint_N, X, N, D, n_neighbors, constraint_neighbors_nn, constraint_distances_nn, 1, 5, N);
//	double *constraint_val_P = new double[constraint_N * n_neighbors];
//	constraint_binary_search_perplexity(constraint_distances_nn, constraint_neighbors_nn, constraint_N, n_neighbors, N, perplexity, constraint_val_P);
//	int *constraint_row_P = new int[constraint_N + 1];
//
//#ifdef _OPENMP
//#pragma omp parallel for 
//#endif
//	for (int i = 0; i < constraint_N + 1; i++) {
//		constraint_row_P[i] = i * n_neighbors;
//	}
//	int *constraint_col_P = new int[constraint_N * n_neighbors];
//
//#ifdef _OPENMP
//#pragma omp parallel for 
//#endif
//	for (int i = 0; i < constraint_N * n_neighbors; i++) {
//		constraint_col_P[i] = constraint_neighbors_nn[i];
//	}
//	
//	double sum = 0.0;
//
//#ifdef _OPENMP
//#pragma omp parallel for reduction(+:sum)
//#endif
//	for (int i = 0; i < constraint_N * n_neighbors; i++) {
//		sum += constraint_val_P[i];
//		//std::cout << i << ": " << constraint_val_P[i] << ", sum = " << sum << '\n';
//	}
//	sum = max(sum, DBL_EPSILON);
//	// Divide the result by sum
//	for (int i = 0; i < constraint_N; i++) {
//		for (int j = 0; j < n_neighbors; j++) {
//			constraint_val_P[i * n_neighbors + j] /= sum;
//			//printf("%f ", constraint_val_P[i * n_neighbors + j]);
//		}
//		//printf("\n");
//	}
	//system("pause");
	//==========================================================================








	// Compute asymmetric pairwise input similarities
	constraint_compute_gaussian_perplexity(X, N, constraint_X, constraint_N, D, 
		&constraint_row_P, &constraint_col_P, &constraint_val_P, perplexity,
		(int)(3 * perplexity), verbose, forest_size, accuracy);

	// Compute asymmetric pairwise input similarities
	compute_gaussian_perplexity(X, N, D, &row_P, &col_P, &val_P, perplexity,
		(int)(3 * perplexity), verbose, forest_size, accuracy);

	// Symmetrize input similarities
	multi_symmetrize_matrix(&row_P, &col_P, &val_P, N);

	double sum_P = .0;

	for (int i = 0; i < row_P[N]; i++) {
		sum_P += val_P[i];
	}
	sum_P = max(sum_P, DBL_EPSILON);

	for (int i = 0; i < row_P[N]; i++) {
		val_P[i] /= sum_P;
	}

	//sum_P = .0;
	////printf("%d %d", constraint_N, constraint_row_P[constraint_N]);
	//for (int i = 0; i < constraint_row_P[constraint_N]; i++) {
	//	sum_P += constraint_val_P[i];
	//}
	//sum_P = max(sum_P, DBL_EPSILON);

	//for (int i = 0; i < constraint_row_P[constraint_N]; i++) {
	//	constraint_val_P[i] /= sum_P;
	//	//printf("%f ", constraint_val_P[i]);
	//}
	//system("pause");

	end = clock();
	if (verbose) {
		printf("[t-SNE] Done in %4.2f seconds (sparsity = %f)!\n[t-SNE] Learning embedding...\n",
			(double)(end - start) / CLOCKS_PER_SEC, (double)row_P[N] / ((double)N * (double)N));
	}

	// Lie about the P-values

	for (int i = 0; i < row_P[N]; i++) {
		val_P[i] *= early_exaggeration;
	}

	for (int i = 0; i < constraint_row_P[N]; i++) {
		constraint_val_P[i] *= early_exaggeration;
	}

	// Perform main training loop
	start = clock();
	int iter;
	for (iter = 0; iter < n_iter; iter++) {

		bool need_eval_error = (verbose && ((iter % 50 == 0 && iter > 0) || (iter == n_iter - 1)));

		// Compute approximate gradient
		double error = multi_compute_gradient(row_P, col_P, val_P, Y, N, no_dim, dY, angle, need_eval_error);
		double constraint_error = constraint_multi_compute_gradient(constraint_row_P, constraint_col_P, constraint_val_P, 
			Y, N, constraint_Y, constraint_N, no_dim, constraint_dY, angle, need_eval_error);
		error = error * (1.0 - alpha) + constraint_error * alpha;

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < N * no_dim; i++) {
			dY[i] = dY[i] * (1.0 - alpha) + constraint_dY[i] * alpha;
		}

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < N * no_dim; i++) {
			// Update gains
			gains[i] = (sign(dY[i]) != sign(uY[i])) ? (gains[i] + .2) : (gains[i] * .8 + .01);

			// Perform gradient update (with momentum and gains)
			uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
			Y[i] = Y[i] + uY[i];
		}

		// Make solution zero-mean
		//zero_mean(Y, N, no_dim);

		// Stop lying about the P-values after a while, and switch momentum
		if (iter == stop_lying_iter) {
			for (int i = 0; i < row_P[N]; i++) {
				val_P[i] /= early_exaggeration;
			}

			for (int i = 0; i < constraint_row_P[N]; i++) {
				constraint_val_P[i] /= early_exaggeration;
			}
		}
		if (iter == mom_switch_iter) {
			momentum = final_momentum;
		}

		// Print out progress
		if (need_eval_error) {
			end = clock();

			total_time += (double)(end - start) / CLOCKS_PER_SEC;
			printf("[t-SNE] Iteration %d: error is %f (50 iterations in %4.2f seconds)\n", iter, error, (double)(end - start) / CLOCKS_PER_SEC);
			start = clock();
		}

	}
	end = clock();
	total_time += (double)(end - start) / CLOCKS_PER_SEC;

	double final_error = evaluate_error(row_P, col_P, val_P, Y, N, no_dim, angle);

	// Clean up memory
	free(dY);
	free(uY);
	free(gains);

	free(row_P); row_P = NULL;
	free(col_P); col_P = NULL;
	free(val_P); val_P = NULL;

	if (verbose) {
		printf("[t-SNE] Fitting performed in %4.2f seconds.\n", total_time);
		printf("[t-SNE] Error after %d iterations: %f\n", iter, final_error);
	}
}

void TSNE::multi_run_bhtsne(double* X, int N, int D, double* Y, int no_dim, double perplexity, double angle, int n_jobs,
	int n_iter, int random_state, int forest_size, double accuracy, double early_exaggeration, double learning_rate,
	int skip_num_points, int exploration_n_iter) {
#ifdef _OPENMP
	omp_set_num_threads(NUM_THREADS(n_jobs));
#if _OPENMP >= 200805
	omp_set_schedule(omp_sched_guided, 0);
#endif
#endif

	if (N - 1 < 3 * perplexity) {
		perplexity = (N - 1) / 3;
		if (verbose) {
			printf("[t-SNE] Perplexity too large for the number of data points! Adjusting ...\n");
		}
	}

	if (verbose) {
		printf("[t-SNE] Using no_dims = %d, perplexity = %f, and theta = %f\n", no_dim, perplexity, angle);
		//printf("[t-SNE] point3\n");
	}
		
	// Set learning parameters
	double total_time = .0;
	clock_t start, end;
	int stop_lying_iter = exploration_n_iter, mom_switch_iter = exploration_n_iter;
	double momentum = .5, final_momentum = .8;
	double eta = learning_rate;
	//printf("[t-SNE] point1\n");

	// Allocate some memory
	double* dY = (double*)malloc(N * no_dim * sizeof(double));
	double* uY = (double*)calloc(N * no_dim, sizeof(double));
	double* gains = (double*)malloc(N * no_dim * sizeof(double));
	//printf("[t-SNE] point2\n");
	if (dY == NULL || uY == NULL || gains == NULL) { 
		printf("[t-SNE] Memory allocation failed!\n"); 
		exit(1); 
	}

	for (int i = 0; i < N * no_dim; i++) {
		gains[i] = 1.0;
	}
	//printf("[t-SNE] point3\n");

	// Normalize input data (to prevent numerical problems)
	if (verbose) {
		printf("[t-SNE] Computing input similarities...\n");
	}
		

	start = clock();
	zero_mean(X, N, D);
	double max_X = .0;

	for (int i = 0; i < N * D; i++) {
		if (X[i] > max_X) max_X = X[i];
	}

	for (int i = 0; i < N * D; i++) {
		X[i] /= max_X;
	}

	// Compute input similarities
	int* row_P; int* col_P; double* val_P;

	// Compute asymmetric pairwise input similarities
	compute_gaussian_perplexity(X, N, D, &row_P, &col_P, &val_P, perplexity, 
		(int)(3 * perplexity), verbose, forest_size, accuracy);

	// Symmetrize input similarities
	multi_symmetrize_matrix(&row_P, &col_P, &val_P, N);

	double sum_P = .0;

	for (int i = 0; i < row_P[N]; i++) {
		sum_P += val_P[i];
	}

	for (int i = 0; i < row_P[N]; i++) {
		val_P[i] /= sum_P;
	}
	end = clock();
	if (verbose) {
		printf("[t-SNE] Done in %4.2f seconds (sparsity = %f)!\n[t-SNE] Learning embedding...\n", 
			(double)(end - start) / CLOCKS_PER_SEC, (double)row_P[N] / ((double)N * (double)N));
	}

	// Lie about the P-values

	for (int i = 0; i < row_P[N]; i++) {
		val_P[i] *= early_exaggeration;
	}

	// Perform main training loop
	start = clock();
	int iter;
	for (iter = 0; iter < n_iter; iter++) {

		bool need_eval_error = (verbose && (iter % 50 == 0 && iter > 0) || (iter == n_iter - 1));

		// Compute approximate gradient
		double error = multi_compute_gradient(row_P, col_P, val_P, Y, N, no_dim, dY, angle, need_eval_error);
		

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < N * no_dim; i++) {
			// Update gains
			gains[i] = (sign(dY[i]) != sign(uY[i])) ? (gains[i] + .2) : (gains[i] * .8 + .01);

			// Perform gradient update (with momentum and gains)
			uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
			Y[i] = Y[i] + uY[i];
		}

		// Make solution zero-mean
		zero_mean(Y, N, no_dim);

		// Stop lying about the P-values after a while, and switch momentum
		if (iter == stop_lying_iter) {
			for (int i = 0; i < row_P[N]; i++) {
				val_P[i] /= early_exaggeration;
			}
		}
		if (iter == mom_switch_iter) {
			momentum = final_momentum;
		}

		// Print out progress
		if (need_eval_error) {
			end = clock();

			total_time += (double)(end - start) / CLOCKS_PER_SEC;
			printf("[t-SNE] Iteration %d: error is %f (50 iterations in %4.2f seconds)\n", iter, error, (double)(end - start) / CLOCKS_PER_SEC);
			start = clock();
		}

	}
	end = clock(); 
	total_time += (double)(end - start) / CLOCKS_PER_SEC;

	double final_error = evaluate_error(row_P, col_P, val_P, Y, N, no_dim, angle);

	// Clean up memory
	free(dY);
	free(uY);
	free(gains);

	free(row_P); row_P = NULL;
	free(col_P); col_P = NULL;
	free(val_P); val_P = NULL;

	if (verbose) {
		printf("[t-SNE] Fitting performed in %4.2f seconds.\n", total_time);
		printf("[t-SNE] Error after %d iterations: %f\n", iter, final_error);
	}
}


//int main() {
//	int N = 10000;
//	int D = 512;
//	//int constraint_N = 100;
//	//
//	/*int n_neighbors = 80;
//	int *neighbors_nn = new int[N * n_neighbors];
//	double *distances_nn = new double[N * n_neighbors];*/
//	////double *distances = new double[N * D];
//	///*int length = 90,
//	//	ind_len = 11;*/
//	int no_dim = 2;
//	//int constraint_number = 3;
//	double *X = new double[N * D],
//		*Y = new double[N * no_dim];//,
//		//*constraint_X = new double[constraint_N * D],
//		//*constraint_Y = new double[constraint_N * no_dim];
//	for (int i = 0; i < N * no_dim; i++) {
//		Y[i] = double(rand()) / RAND_MAX * 10.0;
//	}
//	std::ifstream fin("C:/Users/shouxing/PycharmProjects/DQAnalyzer/init.txt");
//	//std::cout << "===X===\n";
//	//for (int i = 0; i < N; i++) {
//	//	for (int j = 0; j < no_dim; j++) {
//	//		fin >> Y[i * no_dim + j];
//	//		//cout << i << '\t' << j << '\t' << X[i * D + j] << endl;
//	//	}
//	//}
//	fin.close();
//	ofstream out;
//	//out.open("C:/Users/shouxing/PycharmProjects/DQAnalyzer/old_Y.txt");
//	////std::cout << "===X===\n";
//	//for (int i = 0; i < N; i++) {
//	//	for (int j = 0; j < no_dim; j++) {
//	//		out << Y[i * no_dim + j];
//	//		if (j < no_dim - 1) {
//	//			out << ' ';
//	//		}
//	//	}
//	//	out << '\n';
//	//}
//	//out.close();
//	//	//*pos_param = new double[constraint_number * N],
//	//	//*constraint = new double[constraint_number * no_dim];
//	///*	*val_P = new double[length],
//	//	*grad = new double[N * D];
//	//int *neighbors = new int[length],
//	//	*indptr = new int[ind_len];*/
//
//
//	///*for (int i = 0; i < N * n_neighbors; i++) {
//	//	neighbors_nn[i] = 0;
//	//	distances_nn[i] = .0;
//	//}*/
//	fin.open("C:/Users/shouxing/PycharmProjects/DQAnalyzer/data/X_image.txt");
//	//std::cout << "===X===\n";
//	for (int i = 0; i < N; i++) {
//		for (int j = 0; j < D; j++) {
//			fin >> X[i * D + j];
//			//cout << i << '\t' << j << '\t' << X[i * D + j] << endl;
//		}
//	}
//	fin.close();
//	//fin.open("C:/Users/shouxing/PycharmProjects/Approximated-Guided-Incremental-tSNE/examples/Y.txt");
//	//std::cout << "===X===\n";
//	/*for (int i = 0; i < N; i++) {
//		for (int j = 0; j < no_dim; j++) {
//			Y[i * no_dim + j] = double(rand()) / RAND_MAX;
//		}
//	}*/
//	//fin.close();
//	//fin.open("C:/Users/shouxing/PycharmProjects/Approximated-Guided-Incremental-tSNE/examples/constraint_X.txt");
//	//std::cout << "===X===\n";
//	/*for (int i = 0; i < constraint_N; i++) {
//		for (int j = 0; j < D; j++) {
//			fin >> constraint_X[i * D + j];
//		}
//	}
//	fin.close();
//	fin.open("C:/Users/shouxing/PycharmProjects/Approximated-Guided-Incremental-tSNE/examples/constraint_Y.txt");*/
//	//std::cout << "===X===\n";
//	/*for (int i = 0; i < constraint_N; i++) {
//		for (int j = 0; j < no_dim; j++) {
//			fin >> constraint_Y[i * no_dim + j];
//		}
//	}
//	fin.close();*/
//	////fin.open("C:/Users/shouxing/PycharmProjects/Approximated-Guided-Incremental-tSNE/examples/Y.txt");
//	//////std::cout << "===Y===\n";
//	////for (int i = 0; i < N; i++) {
//	////	for (int j = 0; j < no_dim; j++) {
//	////		fin >> Y[i * no_dim + j];// double(rand()) / RAND_MAX * 10.0;
//	////		/*if (i < 10 && j < 10) {
//	////			std::cout << Y[i * no_dim + j] << '\t';
//	////		}*/
//	////	}
//	////	/*if (i < 10) {
//	////		std::cout << '\n';
//	////	}*/
//	////}
//	////fin.close();
//	////fin.open("C:/Users/shouxing/PycharmProjects/Approximated-Guided-Incremental-tSNE/examples/pos_param.txt");
//	//////std::cout << "===pos_param===\n";
//	////for (int i = 0; i < constraint_number; i++) {
//	////	for (int j = 0; j < N; j++) {
//	////		fin >> pos_param[i * N + j];// double(rand()) / RAND_MAX * 10.0;
//	////		if (i < 10 && j < 10) {
//	////			//std::cout << pos_param[i * N + j] << '\t';
//	////		}
//	////	}
//	////	if (i < 10) {
//	////		//std::cout << '\n';
//	////	}
//	////}
//	////fin.close();
//	////fin.open("C:/Users/shouxing/PycharmProjects/Approximated-Guided-Incremental-tSNE/examples/constraint.txt");
//	//////std::cout << "===constraint===\n";
//	////for (int i = 0; i < constraint_number; i++) {
//	////	for (int j = 0; j < no_dim; j++) {
//	////		fin >> constraint[i * no_dim + j];// double(rand()) / RAND_MAX * 10.0;
//	////		if (i < 10 && j < 10) {
//	////			//std::cout << constraint[i * no_dim + j] << '\t';
//	////		}
//	////	}
//	////	if (i < 10) {
//	////		//std::cout << '\n';
//	////	}
//	////}
//	////fin.close();
//	////fin.open("C:/Users/shouxing/PycharmProjects/Approximated-Guided-Incremental-tSNE/examples/grad.txt");
//	////for (int i = 0; i < N; i++) {
//	////	for (int j = 0; j < D; j++) {
//	////		fin >> grad[i * D + j];// double(rand()) / RAND_MAX * 10.0;
//	////		std::cout << grad[i * D + j] << '\t';
//	////	}
//	////	std::cout << '\n';
//	////}
//	////fin.close();
//	////fin.open("C:/Users/shouxing/PycharmProjects/Approximated-Guided-Incremental-tSNE/examples/neighbors.txt");
//	////for (int j = 0; j < length; j++) {
//	////	double temp;
//	////	fin >> temp;
//	////	neighbors[j] = int(temp);// double(rand()) / RAND_MAX * 10.0;
//	////	std::cout << neighbors[j] << '\n';
//	////}
//	////fin.close();
//	////fin.open("C:/Users/shouxing/PycharmProjects/Approximated-Guided-Incremental-tSNE/examples/indptr.txt");
//	////for (int j = 0; j < ind_len; j++) {
//	////	double temp;
//	////	fin >> temp;
//	////	indptr[j] = int(temp);// double(rand()) / RAND_MAX * 10.0;
//	////	std::cout << indptr[j] << '\n';
//	////}
//	////fin.close();
//	////double* error = new double[1];
//	////error[0] = 0.0;
//	TSNE tsne(0, 3);
//	/*double* condition_P = new double[N * D];
//	for (int i = 0; i < N * D; i++) {
//		condition_P[i] = 0.0;
//	}
//	tsne.binary_search_perplexity(distances, NULL, N, D, 30.0, 1, condition_P);*/
//	//tsne.run_bhtsne(X, N, D, Y, no_dim, constraint_X, constraint_Y, constraint_N, 0.05, 30.0, 0.5, 4, 1000, 1, 2, 0.3, 12.0, 200.0, 0, 250);
//	//tsne.run_bhtsne(X, N, D, Y, no_dim, 30.0, 0.5, 8, 1000, 1, 2, 0.3, 12.0, 200.0, 0, 250);
//	//tsne.run_bhtsne(X, N, D, Y, no_dim, constraint_X, constraint_Y, constraint_N, 0.8, 30.0, 0.5, 8, 1000, 1, 2, 0.3, 12.0, 200.0, 0, 250);
//	tsne.run_bhtsne(X, N, D, Y, no_dim, 30.0, 0.5, 4, 1000, 1, 3, 1.0, 12.0, 200.0, 0, 250);
//
//	std::cout << "finish\n";
//	//ofstream out;
//	out.open("C:/Users/shouxing/PycharmProjects/DQAnalyzer/Y.txt");
//	//std::cout << "===X===\n";
//	for (int i = 0; i < N; i++) {
//		for (int j = 0; j < no_dim; j++) {
//			out << Y[i * no_dim + j];
//			if (j < no_dim - 1) {
//				out << ' ';
//			}
//		}
//		out << '\n';
//	}
//	out.close();
//	delete[] X;
//	delete[] Y;
//	//delete[] constraint_X;
//	//delete[] constraint_Y;
//	/*for (int i = 0; i < N; i++) {
//		for (int j = 0; j < D; j++) {
//			std::cout << condition_P[i * D + j] << '\t';
//		}
//		std::cout << '\n';
//	}*/
//	/*tsne.gradient(val_P, neighbors, length, indptr, ind_len, X, grad, N, D, error, 0.5, 1, 1.0,
//	                0, NULL, 0, 0, NULL, 0, 0, 0);
//	QuadTree* tree = new QuadTree(X, N, D);
//	tree->summarize(0, 0.5, grad, 0);*/
//	//delete[] Y;
//	/*TSNE tsne(0, 0);
//	printf("==========begin=============\n");
//	int *neighbors = new int[N * 80];
//	double *distances = new double[N * 80];
//	tsne.k_neighbors(X, N, D, 80, neighbors, distances, 1, 1, N);
//	double* max_dis = new double[N];
//	for (int i = 0; i < N; i++) {
//		max_dis[i] = distances[i * 80 + 79];
//	}
//	delete[] neighbors;
//	delete[] distances;
//	clock_t t0, t1;
//	std::vector<Node> obj_X(N, Node());
//	for (int i = 0; i < N; i++) {
//		obj_X[i] = Node(i, -1, X + i * D, D, -1, -1);
//	}
//	t0 = clock();
//	KdTreeForest *forest = new KdTreeForest(obj_X, N, D, 4, 5, N * 4);
//	forest->accuracy_trend_test(obj_X, 80, max_dis);
//	delete forest;
//	t1 = clock();
//	printf("T=4 %f\n", (double)(t1 - t0) / CLOCKS_PER_SEC);
//	t0 = clock();
//	forest = new KdTreeForest(obj_X, N, D, 6, 5, N * 6);
//	forest->accuracy_trend_test(obj_X, 80, max_dis);
//	delete forest;
//	t1 = clock();
//	printf("T=6 %f\n", (double)(t1 - t0) / CLOCKS_PER_SEC);
//	delete[] max_dis;*/
//
//	//ParameterSelection ps(X, N, D, 80, false);
//	//int knn_tree, forest_size, leaf_number;
//	//double weights[8] = { 0.0, 0.0,
//	//						0.0, 1.0,
//	//						1.0, 0.0,
//	//						1.0, 1.0 };
//	//for (int i = 3; i < 4; i++) {
//	//	ps.set_weight(weights[i * 2], weights[i * 2 + 1]);
//	//	std::cout << "build_time_weight: " << weights[i * 2] << ", memory_weight: " << weights[i * 2 + 1] << "\n";
//	//	for (int j = 3; j > 0; j -= 2) {
//	//		double acc = j / 10.0;
//	//		ps.parameter_selection(acc, knn_tree, forest_size, leaf_number);
//	//		std::cout << "accuracy: " << acc << ", knn_tree: " << knn_tree << ", forest_size: " << forest_size << ", leaf_number: " << leaf_number << "\n";
//	//	}
//	//	//system("pause");
//	//}
//	/*TSNE tsne(0, 0);
//	tsne.k_neighbors(X, N, D, n_neighbors, neighbors_nn, distances_nn, 5, 5, 150);
//	TSNE tsne1(1, 1);
//	tsne1.k_neighbors(X, N, D, n_neighbors, neighbors_nn, distances_nn, 2, 5, 20);*/
//	/*TSNE tsne2(2, 1);
//	tsne2.k_neighbors(X, N, D, n_neighbors, neighbors_nn, distances_nn, 5, 5, 20);*/
//	/*TSNE tsne3(3, 1);
//	tsne3.k_neighbors(X, N, D, n_neighbors, neighbors_nn, distances_nn, 5, 5, 20);
//	TSNE tsne4(4, 1);
//	tsne4.k_neighbors(X, N, D, n_neighbors, neighbors_nn, distances_nn, 5, 5, 20);
//	TSNE tsne5(5, 1);
//	tsne5.k_neighbors(X, N, D, n_neighbors, neighbors_nn, distances_nn, 5, 5, 20);*/
//	system("pause");
//	//delete[] X;
//
//	//for (int i = 0; i < N; i++) {
//	//	std::cout << "point=" << i << '\n';
//	//	for (int j = 0; j < n_neighbors; j++) {
//	//		std::cout << neighbors_nn[i * n_neighbors + j] << '\t';
//	//	}
//	//	std::cout << '\n';
//	//	for (int j = 0; j < n_neighbors; j++) {
//	//		std::cout << distances_nn[i * n_neighbors + j] << '\t';
//	//	}
//	//	//std::cout << "\n\n";
//	//}
//
//	////brute distance
//	//double distances_brute[10][10];
//	//for (int i = 0; i < 10; i++) {
//	//	for (int j = 0; j < 10; j++) {
//	//		distances_brute[i][j] = 0.0;
//	//		for (int k = 0; k < D; k++) {
//	//			distances_brute[i][j] += (X[i * D + k] - X[j * D + k]) * (X[i * D + k] - X[j * D + k]);
//	//		}
//	//	}
//	//}
//
//	return 0;
//}

extern "C"
{
#ifdef _WIN64
	__declspec(dllexport)
#endif
		extern void binary_search_perplexity(double* distances_nn, int* neighbors_nn, int N, int D,
			double perplexity, int verbose, double* conditional_P) {
		srand(unsigned(time(0)));
		TSNE tsne(0, verbose);
		tsne.binary_search_perplexity(distances_nn, neighbors_nn, N, D, perplexity, conditional_P);
		printf("[t-SNE]  Finish perp for %d points.\n", N);
	}

#ifdef _WIN64
	__declspec(dllexport)
#endif
		extern void run_bhtsne(double* X, int N, int D, double* Y, int no_dim, double* constraint_X, double* constraint_Y,
			int constraint_N, double alpha, double perplexity, double angle, int n_jobs,
			int n_iter, int random_state, int verbose, double accuracy, double early_exaggeration, double learning_rate,
			int skip_num_points, int exploration_n_iter,
			int n_neighbors, int* neighbors_nn, int* constraint_neighbors_nn, double* distances_nn, double* constraint_distances_nn, double* constraint_weight) {
		//srand(unsigned(time(0)));
		srand(unsigned(random_state));
		int knn_tree = 0, forest_size = 3;
		TSNE tsne(knn_tree, verbose);
		if (constraint_N > 0) {
			tsne.run_bhtsne(X, N, D, Y, no_dim, constraint_X, constraint_Y, constraint_N, alpha, perplexity, angle, n_jobs,
				n_iter, random_state, forest_size, accuracy, early_exaggeration, learning_rate,
				skip_num_points, exploration_n_iter,
				n_neighbors, neighbors_nn, constraint_neighbors_nn, distances_nn, constraint_distances_nn, constraint_weight);
		}
		else {
			tsne.run_bhtsne(X, N, D, Y, no_dim, perplexity, angle, n_jobs,
				n_iter, random_state, forest_size, accuracy, early_exaggeration, learning_rate,
				skip_num_points, exploration_n_iter,
				n_neighbors, neighbors_nn, distances_nn);
		}
	}

#ifdef _WIN64
	__declspec(dllexport)
#endif
		extern void multi_run_bhtsne(double* X, int N, int D, double* Y, int no_dim, double* constraint_X, double* constraint_Y,
			int constraint_N, double alpha, double perplexity, double angle, int n_jobs,
			int n_iter, int random_state, int verbose, double accuracy, double early_exaggeration, double learning_rate,
			int skip_num_points, int exploration_n_iter) {
		//srand(unsigned(time(0)));
		srand(unsigned(random_state));
		int knn_tree = 0, forest_size = 3;
		TSNE tsne(knn_tree, verbose);
		if (constraint_N > 0) {
			tsne.multi_run_bhtsne(X, N, D, Y, no_dim, constraint_X, constraint_Y,
				constraint_N, alpha, perplexity, angle, n_jobs,
				n_iter, random_state, forest_size, accuracy, early_exaggeration, learning_rate,
				skip_num_points, exploration_n_iter);
		}
		else {
			tsne.multi_run_bhtsne(X, N, D, Y, no_dim, perplexity, angle, n_jobs,
				n_iter, random_state, forest_size, accuracy, early_exaggeration, learning_rate,
				skip_num_points, exploration_n_iter);
		}
	}

#ifdef _WIN64
	__declspec(dllexport)
#endif
		extern void k_neighbors(double* X1, int N1, double* X2, int N2, int D, int n_neighbors, int* neighbors_nn, double* distances_nn,
		int forest_size, int subdivide_variance_size, int leaf_number, int knn_tree, int verbose) {
		srand(unsigned(time(0)));
		TSNE tsne(knn_tree, verbose);
		tsne.k_neighbors(X1, N1, X2, N2, D, n_neighbors, neighbors_nn, distances_nn, forest_size, subdivide_variance_size, leaf_number);
	}
}
