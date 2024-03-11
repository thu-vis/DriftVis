/*
 *  parameter_selection.cpp
 *  Implementation of parameter_selection.
 *
 *  Created by Shouxing Xiang.
 */

#include <cmath>
#include <cfloat>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <iostream>
#include <fstream>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "vp_tree.h"
#include "kd_tree.h"
#include "forest.h"
#include "parameter_selection.h"


#ifdef _OPENMP
    #define NUM_THREADS(N) ((N) >= 0 ? (N) : omp_get_num_procs() + (N) + 1)
#else
    #define NUM_THREADS(N) (1)
#endif


ParameterSelection::ParameterSelection(double* _data, int _N, int _D, int _n_neighbor, double _build_time_weight, double _memory_weight, bool if_sample) {
	data = _data;
	N = _N;
	D = _D;
	build_time_weight = build_time_weight;
	if (build_time_weight < 0.0) {
		build_time_weight = 0.0;
	}
	memory_weight = _memory_weight;
	if (memory_weight < 0.0) {
		memory_weight = 0.0;
	}
	n_neighbor = _n_neighbor;
	sample_n_neighbor = n_neighbor;
	sample_N = N;
	if (N > 1000 && if_sample) {
		sample_N /= 10;
		sample_n_neighbor /= 10;
	}
	init_sampled_data();
	compute_true_knn_distance();
}

double ParameterSelection::compute_neighbor_accuracy(int* neighbors1, double* distances1, int* neighbors2, double* distances2, int N, int D) {
	int count = 0;
	for (int i = 0; i < N; i++) {
		for (int j = 0, k = 0; j < D && k < D; j++) {
			if (distances1[j] < distances2[k]) {
				continue;
			}
			if (neighbors1[j] == neighbors2[k]) {
				count++;
				k++;
				continue;
			}
			while (distances1[j] > distances2[k] && k < D) {
				k++;
			}
		}
	}
	return double(count) / double(N) / double(D);
}

ParameterSelection::ParameterSelection(double* _data, int _N, int _D, int _n_neighbor, bool if_sample) {
	data = _data;
	N = _N;
	D = _D;
	build_time_weight = 1.0;
	memory_weight = 0.0;
	n_neighbor = _n_neighbor;
	sample_n_neighbor = n_neighbor;
	sample_N = N;
	if (N > 1000 && if_sample) {
		sample_N /= 10; 
		sample_n_neighbor /= 10;
	}
	init_sampled_data();
	compute_true_knn_distance();
}

ParameterSelection::~ParameterSelection() {
	delete[] sample_indexes;
	delete[] true_knn_distance;
	delete[] true_distances;
	delete[] true_neighbors;
}

void ParameterSelection::set_weight(double _build_time_weight, double _memory_weight) {
	build_time_weight = _build_time_weight;
	memory_weight = _memory_weight;
}

void ParameterSelection::parameter_selection(double _accuracy, int& knn_tree, int& forest_size, int& leaf_number) {
	accuracy = _accuracy;
	int forest_size3 = 1,
		forest_size4 = 1,
		leaf_number1 = 1,
		leaf_number2 = 1,
		leaf_number3 = 1,
		leaf_number4 = 1;
	double cost4 = kd_forest_selection(forest_size4, leaf_number4), 
		cost1 = vp_tree_selection(leaf_number1),
		cost2 = kd_tree_selection(leaf_number2),
		cost3 = vp_forest_selection(forest_size3, leaf_number3);
	if (memory_weight == 0.0) {
		if (cost1 < cost2 && cost1 < cost3 && cost1 < cost4) {
			knn_tree = 5;
			forest_size = 1;
			leaf_number = leaf_number1;
		}
		else if (cost2 < cost3 && cost2 < cost4) {
			knn_tree = 3;
			forest_size = 1;
			leaf_number = leaf_number2;
		}
		else if (cost3 < cost4) {
			knn_tree = 1;
			forest_size = forest_size3;
			leaf_number = leaf_number3;
		}
		else {
			knn_tree = 0;
			forest_size = forest_size4;
			leaf_number = leaf_number4;
		}
		return;
	}
	double cost = fmin(fmin(cost1, cost2), fmin(cost3, cost4));
	cost1 = vp_tree_selection(leaf_number1, memory_weight, cost);
	cost2 = kd_tree_selection(leaf_number2, memory_weight, cost);
	cost3 = vp_forest_selection(forest_size3, leaf_number3, memory_weight, cost);
	cost4 = kd_forest_selection(forest_size4, leaf_number4, memory_weight, cost);
	if (cost1 < cost2 && cost1 < cost3 && cost1 < cost4) {
		knn_tree = 5;
		forest_size = 1;
		leaf_number = leaf_number1;
	}
	else if (cost2 < cost3 && cost2 < cost4) {
		knn_tree = 3;
		forest_size = 1;
		leaf_number = leaf_number2;
	}
	else if (cost3 < cost4) {
		knn_tree = 1;
		forest_size = forest_size3;
		leaf_number = leaf_number3;
	}
	else {
		knn_tree = 0;
		forest_size = forest_size4;
		leaf_number = leaf_number4;
	}
	if (N > sample_N) {
		
	}
	return;
}

double ParameterSelection::vp_tree_selection(int& leaf_number, double _memory_weight, double _opt_cost) {
	std::cout << "vp_tree_selection\n";
	double build_time, search_time, memory_ratio, min_cost;
	memory_ratio = 7.0 / double(D);
	std::cout << "memory_ratio = " << memory_ratio << "\n";
	clock_t t0, t1;
	t0 = clock();
	std::vector<DataPoint> obj_X(sample_N, DataPoint(D, -1, data));
	for (int n = 0; n < sample_N; n++) {
		obj_X[n] = DataPoint(D, n, data + sample_indexes[n] * D);
	}
	VpTree *tree = new VpTree(obj_X);
	t1 = clock();
	build_time = double(t1 - t0) / CLOCKS_PER_SEC;
	std::cout << "build_time = " << build_time << "\n";

	int* leaf_numbers = new int[sample_N];

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int i = 0; i < sample_N; i++) {
		std::vector<int> indices;
		std::vector<double> distances;
		indices.clear();
		distances.clear();
		// Find nearest neighbors
		leaf_numbers[i] = tree->least_leaf_number_search(obj_X[i], sample_n_neighbor + 1, true_knn_distance[i], accuracy);
	}
	int max_leaf_number = 0, min_leaf_number = sample_N;
	for (int i = 0; i < sample_N; i++) {
		max_leaf_number = max_leaf_number < leaf_numbers[i] ? leaf_numbers[i] : max_leaf_number;
		min_leaf_number = min_leaf_number > leaf_numbers[i] ? leaf_numbers[i] : min_leaf_number;
	}
	delete[] leaf_numbers;

	int* knn_neighbors = new int[sample_N * sample_n_neighbor];
	double* knn_distances = new double[sample_N * sample_n_neighbor];
	int temp_leaf_number = (max_leaf_number + min_leaf_number) / 2;
	while (min_leaf_number < max_leaf_number - sample_N / 20) {
		temp_leaf_number = (max_leaf_number + min_leaf_number) / 2;
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (int i = 0; i < sample_N; i++) {
			//printf("[t-SNE] Computing %d neighbors of point %d.\n", n_neighbors, i);
			std::vector<int> indices;
			std::vector<double> distances;
			indices.clear();
			distances.clear();
			// Find nearest neighbors
			tree->priority_search(obj_X[i], sample_n_neighbor + 1, &indices, &distances, temp_leaf_number);
			for (int j = 0; j < sample_n_neighbor; j++) {
				knn_neighbors[i * sample_n_neighbor + j] = indices[j + 1];
				knn_distances[i * sample_n_neighbor + j] = distances[j + 1];
			}
		}
		double acc = compute_neighbor_accuracy(true_neighbors, true_distances, knn_neighbors, knn_distances, sample_N, sample_n_neighbor);
		if (acc == accuracy) {
			break;
		}
		else if (acc < accuracy) {
			min_leaf_number = temp_leaf_number;
		}
		else {
			max_leaf_number = temp_leaf_number;
		}
	}


	delete[] knn_distances;
	delete[] knn_neighbors;


	

	std::cout << "leaf_number = " << temp_leaf_number << "\n";


	t0 = clock();
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int i = 0; i < sample_N; i++) {
		//printf("[t-SNE] Computing %d neighbors of point %d.\n", n_neighbors, i);
		std::vector<int> indices;
		std::vector<double> distances;
		indices.clear();
		distances.clear();
		// Find nearest neighbors
		tree->priority_search(obj_X[i], sample_n_neighbor + 1, &indices, &distances, temp_leaf_number);
	}
	t1 = clock();
	search_time = double(t1 - t0) / CLOCKS_PER_SEC;
	std::cout << "search_time = " << search_time << "\n";
	delete tree;
	min_cost = (search_time + build_time_weight * build_time) / _opt_cost + memory_weight * memory_ratio;
	std::cout << "cost = " << min_cost << "\n\n";
	leaf_number = temp_leaf_number;
	return min_cost;
}

double ParameterSelection::kd_tree_selection(int& leaf_number, double _memory_weight, double _opt_cost) {
	std::cout << "kd_tree_selection\n";
	double build_time, search_time, memory_ratio, min_cost;
	memory_ratio = 6.0 / double(D);
	std::cout << "memory_ratio = " << memory_ratio << "\n";
	clock_t t0, t1;
	t0 = clock();
	std::vector<Node> obj_X(sample_N, Node());
	for (int i = 0; i < sample_N; i++) {
		obj_X[i] = Node(i, -1, data + sample_indexes[i] * D, D, -1, -1);
	}
	KdTree *tree = new KdTree(obj_X, sample_N, D, 5);
	
	t1 = clock();
	build_time = double(t1 - t0) / CLOCKS_PER_SEC;
	std::cout << "build_time = " << build_time << "\n";
	int* leaf_numbers = new int[sample_N];

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int i = 0; i < sample_N; i++) {
		std::vector<int> indices;
		std::vector<double> distances;
		indices.clear();
		distances.clear();
		// Find nearest neighbors
		leaf_numbers[i] = tree->least_leaf_number_search(obj_X[i], sample_n_neighbor + 1, true_knn_distance[i], accuracy);
	}
	int max_leaf_number = 0, min_leaf_number = sample_N;
	for (int i = 0; i < sample_N; i++) {
		max_leaf_number = max_leaf_number < leaf_numbers[i] ? leaf_numbers[i] : max_leaf_number;
		min_leaf_number = min_leaf_number > leaf_numbers[i] ? leaf_numbers[i] : min_leaf_number;
	}
	delete[] leaf_numbers;

	int* knn_neighbors = new int[sample_N * sample_n_neighbor];
	double* knn_distances = new double[sample_N * sample_n_neighbor];
	int temp_leaf_number = (max_leaf_number + min_leaf_number) / 2;
	while (min_leaf_number < max_leaf_number - sample_N / 20) {
		temp_leaf_number = (max_leaf_number + min_leaf_number) / 2;
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < sample_N; i++) {
			//printf("[t-SNE] Computing %d neighbors of point %d.\n", n_neighbors, i);
			std::vector<int> indices;
			std::vector<double> distances;
			indices.clear();
			distances.clear();
			// Find nearest neighbors
			tree->priority_search(obj_X[i], sample_n_neighbor + 1, &indices, &distances, temp_leaf_number);
			for (int j = 0; j < sample_n_neighbor; j++) {
				knn_neighbors[i * sample_n_neighbor + j] = indices[j + 1];
				knn_distances[i * sample_n_neighbor + j] = distances[j + 1];
			}
		}
		double acc = compute_neighbor_accuracy(true_neighbors, true_distances, knn_neighbors, knn_distances, sample_N, sample_n_neighbor);
		if (acc == accuracy) {
			break;
		}
		else if (acc < accuracy) {
			min_leaf_number = temp_leaf_number;
		}
		else {
			max_leaf_number = temp_leaf_number;
		}
	}


	delete[] knn_distances;
	delete[] knn_neighbors;


	std::cout << "leaf_number = " << temp_leaf_number << "\n";

	t0 = clock();
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int i = 0; i < sample_N; i++) {
		//printf("[t-SNE] Computing %d neighbors of point %d.\n", n_neighbors, i);
		std::vector<int> indices;
		std::vector<double> distances;
		indices.clear();
		distances.clear();
		// Find nearest neighbors
		tree->priority_search(obj_X[i], sample_n_neighbor + 1, &indices, &distances, temp_leaf_number);
	}
	t1 = clock();
	search_time = double(t1 - t0) / CLOCKS_PER_SEC;
	std::cout << "search_time = " << search_time << "\n";
	delete tree;
	min_cost = (search_time + build_time_weight * build_time) / _opt_cost + memory_weight * memory_ratio;
	std::cout << "cost = " << min_cost << "\n\n";
	leaf_number = temp_leaf_number;
	return min_cost;
}

double ParameterSelection::vp_forest_selection(int& forest_size, int& leaf_number, double _memory_weight, double _opt_cost) {
	std::cout << "vp_forest_selection\n";
	double build_time, search_time, memory_ratio, min_cost;
	int example_forest_sizes[5] = { 2, 4, 8, 16, 32 };
	double _costs[2] = { DBL_MAX , DBL_MAX };
	int _forest_sizes[2] = { 0, 0 };
	int _leaf_numbers[2] = { -1, -1 };
	int* leaf_numbers = new int[sample_N];
	std::vector<DataPoint> obj_X(sample_N, DataPoint(D, -1, data));
	for (int n = 0; n < sample_N; n++) {
		obj_X[n] = DataPoint(D, n, data + sample_indexes[n] * D);
	}

	std::cout << "check example forest_size\n";
	for (int k = 0; k < 5; k++) {
		int forest_size = example_forest_sizes[k];
		std::cout << "forest_size = " << forest_size << "\n";
		memory_ratio = forest_size * 7.0 / double(D);
		std::cout << "memory_ratio = " << memory_ratio << "\n";
		clock_t t0, t1;
		t0 = clock();
		VpTreeForest *forest = new VpTreeForest(obj_X, forest_size, 0);
		t1 = clock();
		build_time = double(t1 - t0) / CLOCKS_PER_SEC;
		std::cout << "build_time = " << build_time << "\n";

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < sample_N; i++) {
			std::vector<int> indices;
			std::vector<double> distances;
			indices.clear();
			distances.clear();
			// Find nearest neighbors
			leaf_numbers[i] = forest->least_leaf_number_search(obj_X[i], sample_n_neighbor + 1, true_knn_distance[i], accuracy);
		}
		int max_leaf_number = 0, min_leaf_number = sample_N;
		for (int i = 0; i < sample_N; i++) {
			max_leaf_number = max_leaf_number < leaf_numbers[i] ? leaf_numbers[i] : max_leaf_number;
			min_leaf_number = min_leaf_number > leaf_numbers[i] ? leaf_numbers[i] : min_leaf_number;
		}



		int* knn_neighbors = new int[sample_N * sample_n_neighbor];
		double* knn_distances = new double[sample_N * sample_n_neighbor];
		int temp_leaf_number = (max_leaf_number + min_leaf_number) / 2;
		forest->leaf_number = temp_leaf_number;
		while (min_leaf_number < max_leaf_number - sample_N / 20) {
			temp_leaf_number = (max_leaf_number + min_leaf_number) / 2;
			forest->leaf_number = temp_leaf_number;
#ifdef _OPENMP
#pragma omp parallel for
#endif
			for (int i = 0; i < sample_N; i++) {
				std::vector<int> indices;
				std::vector<double> distances;
				indices.clear();
				distances.clear();
				forest->priority_search(obj_X[i], sample_n_neighbor + 1, &indices, &distances);
				for (int j = 0; j < sample_n_neighbor; j++) {
					knn_neighbors[i * sample_n_neighbor + j] = indices[j + 1];
					knn_distances[i * sample_n_neighbor + j] = distances[j + 1];
				}
			}
			double acc = compute_neighbor_accuracy(true_neighbors, true_distances, knn_neighbors, knn_distances, sample_N, sample_n_neighbor);
			if (acc == accuracy) {
				break;
			}
			else if (acc < accuracy) {
				min_leaf_number = temp_leaf_number;
			}
			else {
				max_leaf_number = temp_leaf_number;
			}
		}


		delete[] knn_distances;
		delete[] knn_neighbors;


		std::cout << "leaf_number = " << temp_leaf_number << "\n";
		t0 = clock();
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < sample_N; i++) {
			std::vector<int> indices;
			std::vector<double> distances;
			indices.clear();
			distances.clear();
			forest->priority_search(obj_X[i], sample_n_neighbor + 1, &indices, &distances);
		}
		t1 = clock();
		search_time = double(t1 - t0) / CLOCKS_PER_SEC;
		std::cout << "search_time = " << search_time << "\n";
		delete forest;
		min_cost = (search_time + build_time_weight * build_time) / _opt_cost + memory_weight * memory_ratio;
		std::cout << "cost = " << min_cost << "\n\n";
		if (min_cost < _costs[0]) {
			_costs[1] = _costs[0];
			_costs[0] = min_cost;
			_forest_sizes[1] = _forest_sizes[0];
			_forest_sizes[0] = forest_size;
			_leaf_numbers[1] = _leaf_numbers[0];
			_leaf_numbers[0] = temp_leaf_number;
		}
		else if (min_cost < _costs[1]) {
			_costs[1] = min_cost;
			_forest_sizes[1] = forest_size;
			_leaf_numbers[1] = temp_leaf_number;
		}
	}

	// downhill simplex
	int next_step = 0, //0: reflection, 1: expanision, 2: contraction
		forest_size_c, 
		leaf_number_c;
	double cost_c;

	double alpha = 0.9, beta = 1.0, gama = 0.5;
	std::cout << "downhill simplex\n";
	for (int k = 0; k < 10; k++) {
		int forest_size;
		double average = (_forest_sizes[0] + _forest_sizes[1]) * 0.5;
		if (next_step == 0) {
			forest_size = int(average + alpha * (average - _forest_sizes[1]));
		}
		else if (next_step == 1) {
			forest_size = int(forest_size_c + beta * (forest_size_c - average));
		}
		else {
			forest_size = int(forest_size_c + gama * (_forest_sizes[1] - average));
		}
		if (forest_size < 2) {
			break;
		}
		std::cout << "forest_size = " << forest_size << "\n";
		memory_ratio = forest_size * 7.0 / double(D);
		std::cout << "memory_ratio = " << memory_ratio << "\n";
		clock_t t0, t1;
		t0 = clock();
		VpTreeForest *forest = new VpTreeForest(obj_X, forest_size, 0);
		t1 = clock();
		build_time = double(t1 - t0) / CLOCKS_PER_SEC;
		std::cout << "build_time = " << build_time << "\n";

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < sample_N; i++) {
			std::vector<int> indices;
			std::vector<double> distances;
			indices.clear();
			distances.clear();
			// Find nearest neighbors
			leaf_numbers[i] = forest->least_leaf_number_search(obj_X[i], sample_n_neighbor + 1, true_knn_distance[i], accuracy);
		}
		int max_leaf_number = 0, min_leaf_number = sample_N;
		for (int i = 0; i < sample_N; i++) {
			max_leaf_number = max_leaf_number < leaf_numbers[i] ? leaf_numbers[i] : max_leaf_number;
			min_leaf_number = min_leaf_number > leaf_numbers[i] ? leaf_numbers[i] : min_leaf_number;
		}

		int* knn_neighbors = new int[sample_N * sample_n_neighbor];
		double* knn_distances = new double[sample_N * sample_n_neighbor];
		int temp_leaf_number = (max_leaf_number + min_leaf_number) / 2;
		forest->leaf_number = temp_leaf_number;
		while (min_leaf_number < max_leaf_number - sample_N / 20) {
			temp_leaf_number = (max_leaf_number + min_leaf_number) / 2;
			forest->leaf_number = temp_leaf_number;
#ifdef _OPENMP
#pragma omp parallel for
#endif
			for (int i = 0; i < sample_N; i++) {
				std::vector<int> indices;
				std::vector<double> distances;
				indices.clear();
				distances.clear();
				forest->priority_search(obj_X[i], sample_n_neighbor + 1, &indices, &distances);
				for (int j = 0; j < sample_n_neighbor; j++) {
					knn_neighbors[i * sample_n_neighbor + j] = indices[j + 1];
					knn_distances[i * sample_n_neighbor + j] = distances[j + 1];
				}
		}
			double acc = compute_neighbor_accuracy(true_neighbors, true_distances, knn_neighbors, knn_distances, sample_N, sample_n_neighbor);
			if (acc == accuracy) {
				break;
			}
			else if (acc < accuracy) {
				min_leaf_number = temp_leaf_number;
			}
			else {
				max_leaf_number = temp_leaf_number;
			}
	}


		delete[] knn_distances;
		delete[] knn_neighbors;


		std::cout << "leaf_number = " << temp_leaf_number << "\n";
		t0 = clock();
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < sample_N; i++) {
			std::vector<int> indices;
			std::vector<double> distances;
			indices.clear();
			distances.clear();
			forest->priority_search(obj_X[i], sample_n_neighbor + 1, &indices, &distances);
		}
		t1 = clock();
		search_time = double(t1 - t0) / CLOCKS_PER_SEC;
		std::cout << "search_time = " << search_time << "\n";
		delete forest;
		min_cost = (search_time + build_time_weight * build_time) / _opt_cost + memory_weight * memory_ratio;
		std::cout << "cost = " << min_cost << "\n\n";
		if (next_step == 0) {
			if (min_cost < _costs[0]) {
				cost_c = min_cost;
				forest_size_c = forest_size;
				leaf_number_c = temp_leaf_number;
				next_step = 1;
			}
			else if (min_cost < _costs[1]) {
				_costs[1] = _costs[0];
				_costs[0] = min_cost;
				_forest_sizes[1] = _forest_sizes[0];
				_forest_sizes[0] = forest_size;
				_leaf_numbers[1] = _leaf_numbers[0];
				_leaf_numbers[0] = temp_leaf_number;
			}
			else {
				forest_size_c = int(average);
				next_step = 2;
			}
		}
		else if (next_step == 1) {
			if (min_cost < cost_c) {
				_costs[1] = _costs[0];
				_costs[0] = min_cost;
				_forest_sizes[1] = _forest_sizes[0];
				_forest_sizes[0] = forest_size;
				_leaf_numbers[1] = _leaf_numbers[0];
				_leaf_numbers[0] = temp_leaf_number;
			}
			else {
				_costs[1] = _costs[0];
				_costs[0] = cost_c;
				_forest_sizes[1] = _forest_sizes[0];
				_forest_sizes[0] = forest_size_c;
				_leaf_numbers[1] = _leaf_numbers[0];
				_leaf_numbers[0] = leaf_number_c;
			}
			next_step = 0;
		}
		else {
			if (min_cost < _costs[0]) {
				_costs[1] = _costs[0];
				_costs[0] = min_cost;
				_forest_sizes[1] = _forest_sizes[0];
				_forest_sizes[0] = forest_size;
				_leaf_numbers[1] = _leaf_numbers[0];
				_leaf_numbers[0] = temp_leaf_number;
				next_step = 0;
			}
			else if (min_cost < _costs[1]) {
				_costs[1] = min_cost;
				_forest_sizes[1] = forest_size;
				_leaf_numbers[1] = temp_leaf_number;
				next_step = 0;
			}
			else {
				forest_size_c = forest_size;
			}
		}		
	}
	
	delete[] leaf_numbers;
	forest_size = _forest_sizes[0];
	leaf_number = _leaf_numbers[0];
	return _costs[0];
}

double ParameterSelection::kd_forest_selection(int& forest_size, int& leaf_number, double _memory_weight, double _opt_cost) {
	std::cout << "kd_forest_selection\n";
	double build_time, search_time, memory_ratio, min_cost;
	int example_forest_sizes[5] = { 2, 4, 8, 16, 32 };
	double _costs[2] = { DBL_MAX , DBL_MAX };
	int _forest_sizes[2] = { 0, 0 };
	int _leaf_numbers[2] = { -1, -1 };
	int* leaf_numbers = new int[sample_N];
	std::vector<Node> obj_X(sample_N, Node());
	for (int n = 0; n < sample_N; n++) {
		obj_X[n] = Node(n, -1, data + sample_indexes[n] * D, D, -1, -1);
	}

	std::cout << "check example forest_size\n";
	for (int k = 0; k < 5; k++) {
		int forest_size = example_forest_sizes[k];
		std::cout << "forest_size = " << forest_size << "\n";
		memory_ratio = forest_size * 7.0 / double(D);
		std::cout << "memory_ratio = " << memory_ratio << "\n";
		clock_t t0, t1;
		t0 = clock();
		KdTreeForest *forest = new KdTreeForest(obj_X, sample_N, D, forest_size, 5, 0);
		t1 = clock();
		build_time = double(t1 - t0) / CLOCKS_PER_SEC;
		std::cout << "build_time = " << build_time << "\n";

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < sample_N; i++) {
			std::vector<int> indices;
			std::vector<double> distances;
			indices.clear();
			distances.clear();
			// Find nearest neighbors
			leaf_numbers[i] = forest->least_leaf_number_search(obj_X[i], sample_n_neighbor + 1, true_knn_distance[i], accuracy);
		}
		int max_leaf_number = 0, min_leaf_number = sample_N;
		double avg_leaf_number = 0.0;
		for (int i = 0; i < sample_N; i++) {
			max_leaf_number = max_leaf_number < leaf_numbers[i] ? leaf_numbers[i] : max_leaf_number;
			min_leaf_number = min_leaf_number > leaf_numbers[i] ? leaf_numbers[i] : min_leaf_number;
			avg_leaf_number += double(leaf_numbers[i]) / sample_N;
		}
		max_leaf_number = int(avg_leaf_number);
		forest->leaf_number = max_leaf_number;
		std::cout << "leaf_number = " << max_leaf_number << "\n";
		t0 = clock();
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < sample_N; i++) {
			std::vector<int> indices;
			std::vector<double> distances;
			indices.clear();
			distances.clear();
			forest->priority_search(obj_X[i], sample_n_neighbor + 1, &indices, &distances);
		}
		t1 = clock();
		search_time = double(t1 - t0) / CLOCKS_PER_SEC;
		std::cout << "search_time = " << search_time << "\n";
		delete forest;
		min_cost = (search_time + build_time_weight * build_time) / _opt_cost + memory_weight * memory_ratio;
		std::cout << "cost = " << min_cost << "\n\n";
		if (min_cost < _costs[0]) {
			_costs[1] = _costs[0];
			_costs[0] = min_cost;
			_forest_sizes[1] = _forest_sizes[0];
			_forest_sizes[0] = forest_size;
			_leaf_numbers[1] = _leaf_numbers[0];
			_leaf_numbers[0] = max_leaf_number;
		}
		else if (min_cost < _costs[1]) {
			_costs[1] = min_cost;
			_forest_sizes[1] = forest_size;
			_leaf_numbers[1] = max_leaf_number;
		}
	}

	// downhill simplex
	int next_step = 0, //0: reflection, 1: expanision, 2: contraction
		forest_size_c,
		leaf_number_c;
	double cost_c;

	double alpha = 0.9, beta = 1.0, gama = 0.5;
	std::cout << "downhill simplex\n";
	for (int k = 0; k < 10; k++) {
		int forest_size;
		double average = (_forest_sizes[0] + _forest_sizes[1]) * 0.5;
		if (next_step == 0) {
			forest_size = int(average + alpha * (average - _forest_sizes[1]));
		}
		else if (next_step == 1) {
			forest_size = int(forest_size_c + beta * (forest_size_c - average));
		}
		else {
			forest_size = int(forest_size_c + gama * (_forest_sizes[1] - average));
		}
		if (forest_size < 2) {
			break;
		}
		std::cout << "forest_size = " << forest_size << "\n";
		memory_ratio = forest_size * 7.0 / double(D);
		std::cout << "memory_ratio = " << memory_ratio << "\n";
		clock_t t0, t1;
		t0 = clock();
		KdTreeForest *forest = new KdTreeForest(obj_X, sample_N, D, forest_size, 5, 0);
		t1 = clock();
		build_time = double(t1 - t0) / CLOCKS_PER_SEC;
		std::cout << "build_time = " << build_time << "\n";

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < sample_N; i++) {
			std::vector<int> indices;
			std::vector<double> distances;
			indices.clear();
			distances.clear();
			// Find nearest neighbors
			leaf_numbers[i] = forest->least_leaf_number_search(obj_X[i], sample_n_neighbor + 1, true_knn_distance[i], accuracy);
		}
		int max_leaf_number = 0, min_leaf_number = sample_N;
		for (int i = 0; i < sample_N; i++) {
			max_leaf_number = max_leaf_number < leaf_numbers[i] ? leaf_numbers[i] : max_leaf_number;
			min_leaf_number = min_leaf_number > leaf_numbers[i] ? leaf_numbers[i] : min_leaf_number;
		}

		int* knn_neighbors = new int[sample_N * sample_n_neighbor];
		double* knn_distances = new double[sample_N * sample_n_neighbor];
		int temp_leaf_number = (max_leaf_number + min_leaf_number) / 2;
		forest->leaf_number = temp_leaf_number;
		while (min_leaf_number < max_leaf_number - sample_N / 20) {
			temp_leaf_number = (max_leaf_number + min_leaf_number) / 2;
			forest->leaf_number = temp_leaf_number;
#ifdef _OPENMP
#pragma omp parallel for
#endif
			for (int i = 0; i < sample_N; i++) {
				std::vector<int> indices;
				std::vector<double> distances;
				indices.clear();
				distances.clear();
				forest->priority_search(obj_X[i], sample_n_neighbor + 1, &indices, &distances);
				for (int j = 0; j < sample_n_neighbor; j++) {
					knn_neighbors[i * sample_n_neighbor + j] = indices[j + 1];
					knn_distances[i * sample_n_neighbor + j] = distances[j + 1];
				}
			}
			double acc = compute_neighbor_accuracy(true_neighbors, true_distances, knn_neighbors, knn_distances, sample_N, sample_n_neighbor);
			if (acc == accuracy) {
				break;
			}
			else if (acc < accuracy) {
				min_leaf_number = temp_leaf_number;
			}
			else {
				max_leaf_number = temp_leaf_number;
			}
	}


		delete[] knn_distances;
		delete[] knn_neighbors;


		std::cout << "leaf_number = " << temp_leaf_number << "\n";
		t0 = clock();
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < sample_N; i++) {
			std::vector<int> indices;
			std::vector<double> distances;
			indices.clear();
			distances.clear();
			forest->priority_search(obj_X[i], sample_n_neighbor + 1, &indices, &distances);
		}
		t1 = clock();
		search_time = double(t1 - t0) / CLOCKS_PER_SEC;
		std::cout << "search_time = " << search_time << "\n";
		delete forest;
		min_cost = (search_time + build_time_weight * build_time) / _opt_cost + memory_weight * memory_ratio;
		std::cout << "cost = " << min_cost << "\n\n";
		if (next_step == 0) {
			if (min_cost < _costs[0]) {
				cost_c = min_cost;
				forest_size_c = forest_size;
				leaf_number_c = temp_leaf_number;
				next_step = 1;
			}
			else if (min_cost < _costs[1]) {
				_costs[1] = _costs[0];
				_costs[0] = min_cost;
				_forest_sizes[1] = _forest_sizes[0];
				_forest_sizes[0] = forest_size;
				_leaf_numbers[1] = _leaf_numbers[0];
				_leaf_numbers[0] = temp_leaf_number;
			}
			else {
				forest_size_c = int(average);
				next_step = 2;
			}
		}
		else if (next_step == 1) {
			if (min_cost < cost_c) {
				_costs[1] = _costs[0];
				_costs[0] = min_cost;
				_forest_sizes[1] = _forest_sizes[0];
				_forest_sizes[0] = forest_size;
				_leaf_numbers[1] = _leaf_numbers[0];
				_leaf_numbers[0] = temp_leaf_number;
			}
			else {
				_costs[1] = _costs[0];
				_costs[0] = cost_c;
				_forest_sizes[1] = _forest_sizes[0];
				_forest_sizes[0] = forest_size_c;
				_leaf_numbers[1] = _leaf_numbers[0];
				_leaf_numbers[0] = leaf_number_c;
			}
			next_step = 0;
		}
		else {
			if (min_cost < _costs[0]) {
				_costs[1] = _costs[0];
				_costs[0] = min_cost;
				_forest_sizes[1] = _forest_sizes[0];
				_forest_sizes[0] = forest_size;
				_leaf_numbers[1] = _leaf_numbers[0];
				_leaf_numbers[0] = temp_leaf_number;
				next_step = 0;
			}
			else if (min_cost < _costs[1]) {
				_costs[1] = min_cost;
				_forest_sizes[1] = forest_size;
				_leaf_numbers[1] = temp_leaf_number;
				next_step = 0;
			}
			else {
				forest_size_c = forest_size;
			}
		}
	}

	delete[] leaf_numbers;
	forest_size = _forest_sizes[0];
	leaf_number = _leaf_numbers[0];
	return _costs[0];
}

void ParameterSelection::init_sampled_data() {
	sample_indexes = new int[sample_N];
	bool *flag = new bool[N];
	for (int i = 0; i < N; i++) {
		flag[i] = false;
	}
	int index = int((double(rand()) / RAND_MAX) * N);
	if (index >= N) {
		index = N - 1;
	}
	for (int i = 0; i < sample_N; i++) {
		while (flag[index]) {
			index = int((double(rand()) / RAND_MAX) * N);
			if (index >= N) {
				index = N - 1;
			}
		}
		flag[index] = true;
		sample_indexes[i] = index;
	}
	delete[] flag;
}

void ParameterSelection::compute_true_knn_distance() {
	// compute true k nearest neighbors distance
	std::vector<DataPoint> obj_X(sample_N, DataPoint(D, -1, data));
	for (int n = 0; n < sample_N; n++) {
		obj_X[n] = DataPoint(D, n, data + sample_indexes[n] * D);
	}
	VpTree *tree = new VpTree(obj_X);
	true_knn_distance = new double[sample_N];
	true_neighbors = new int[sample_N * sample_n_neighbor];
	true_distances = new double[sample_N * sample_n_neighbor];

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int i = 0; i < sample_N; i++) {
		std::vector<int> indices;
		std::vector<double> distances;
		indices.clear();
		distances.clear();
		// Find nearest neighbors
		tree->search(obj_X[i], sample_n_neighbor + 1, &indices, &distances, sample_N + 1);
		for (int j = 0; j < sample_n_neighbor; j++) {
			true_neighbors[i * sample_n_neighbor + j] = indices[j + 1];
			true_distances[i * sample_n_neighbor + j] = distances[j + 1];
		}
		true_knn_distance[i] = distances[sample_n_neighbor];
	}
	delete tree;
}