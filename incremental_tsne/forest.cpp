/*
*  forest.cpp
*  Implementation of forest.
*
*  Created by Shouxing Xiang.
*/

#include <iostream>
#include <float.h>
#include "forest.h"

KdTreeForest::KdTreeForest(const std::vector<Node>& items, int _tree_size, int _data_dim, int _forest_size, int _subdivide_variance_size, int _leaf_number) {
	tree_size = _tree_size;
	data_dim = _data_dim;
	forest_size = _forest_size;
	subdivide_variance_size = _subdivide_variance_size;
	leaf_number = _leaf_number;

	for (int i = 0; i < forest_size; i++) {
		KdTree tree = KdTree(items, tree_size, data_dim, subdivide_variance_size);
		trees.push_back(tree);
	}
}

KdTreeForest::~KdTreeForest() {}


void KdTreeForest::accuracy_trend_test(const std::vector<Node>& targets, int n_neighbors, double* max_distance) {

	std::priority_queue<PriorityQueueItem> heap;
	std::set<int> indexes;

	// Use a priority queue to store node to be search
	std::priority_queue<PriorityQueueItem> node_queue;
	int length = targets.size();
	int* acc_count = new int[leaf_number];
	for (int i = 0; i < leaf_number; i++) {
		acc_count[i] = 0;
	}
	int full_count = 0;
	for (int i = 0; i < length; i++) {
		while (!heap.empty()) {
			heap.pop();
		}
		while (!node_queue.empty()) {
			node_queue.pop();
		}
		indexes.clear();
		for (int j = 0; j < forest_size; j++) {
			node_queue.push(PriorityQueueItem(0, -DBL_MAX, false, j));
		}

		// Variable that tracks the distance to the farthest point in our results
		double tau = DBL_MAX;
		//target, node_queue, n_neighbors, heap, indexes, tau, _leaf_number
		// Perform the search
		int _leaf_number = 0;
		int least_count = n_neighbors;
		int temp, temp1;
		while (!node_queue.empty() && (least_count > 0 || int(heap.size()) < n_neighbors)) {
			int tree_id = node_queue.top().tree_id;
			temp = _leaf_number;
			temp1 = heap.size();
			trees[tree_id].forest_least_leaf_number_search_helper(node_queue, targets[i], n_neighbors, heap, indexes, tau, max_distance[i], _leaf_number, least_count);
			if (temp != _leaf_number) {
				acc_count[_leaf_number - 1] += n_neighbors - least_count;
			}
			if (temp1 != heap.size() && temp1 == n_neighbors - 1) {
				full_count += _leaf_number - 1;
			}
		}
		for (int j = _leaf_number; j < leaf_number; j++) {
			acc_count[j] += n_neighbors - least_count;
		}
	}
	printf("full time: %d\n", full_count / length);
	for (int i = 0; i < leaf_number / forest_size; i++) {
		if (acc_count[i] < length * n_neighbors) {
			printf("%d %f\n", i, double(acc_count[i]) / double(length) * 100.0 / double(n_neighbors));
		}
	}
}

//Function that uses the tree and priority queue to find the k nearest neighbors of target
double KdTreeForest::priority_search(const Node& target, int n_neighbors, std::vector<int>* indices, std::vector<double>* distances) {
	// Use a priority queue to store intermediate results on
	std::priority_queue<PriorityQueueItem> heap;
	std::set<int> indexes;

	// Use a priority queue to store node to be search
	std::priority_queue<PriorityQueueItem> node_queue;
	for (int i = 0; i < forest_size; i++) {
		node_queue.push(PriorityQueueItem(0, -DBL_MAX, false, i));
	}

	// Variable that tracks the distance to the farthest point in our results
	double tau = DBL_MAX;

	// Perform the search
	int _leaf_number = leaf_number;
	int dup_count = 0;
	while (!node_queue.empty() && (_leaf_number > 0 || int(heap.size()) < n_neighbors)) {
		int tree_id = node_queue.top().tree_id;
		if (trees[tree_id].priority_search(target, node_queue, n_neighbors, heap, indexes, tau, _leaf_number)) {
			dup_count++;
		}
	}

	// Gather final results
	indices->clear();
	distances->clear();
	while (!heap.empty()) {
		indices->push_back(heap.top().index);
		distances->push_back(heap.top().dist);
		heap.pop();
	}

	// Results are in reverse order
	std::reverse(indices->begin(), indices->end());
	std::reverse(distances->begin(), distances->end());
	return double(dup_count) / double(leaf_number - _leaf_number);
}

VpTreeForest::VpTreeForest(const std::vector<DataPoint>& items, int _forest_size, int _leaf_number) {
	forest_size = _forest_size;
	leaf_number = _leaf_number;

	for (int i = 0; i < forest_size; i++) {
		VpTree tree = VpTree(items);
		trees.push_back(tree);
	}
}

VpTreeForest::~VpTreeForest() {}

double VpTreeForest::priority_search(const DataPoint& target, int n_neighbors, std::vector<int>* indices, std::vector<double>* distances) {
	// Use a priority queue to store intermediate results on
	std::priority_queue<PriorityQueueItem> heap;
	std::set<int> indexes;

	// Use a priority queue to store node to be search
	std::priority_queue<PriorityQueueItem> node_queue;
	for (int i = 0; i < forest_size; i++) {
		node_queue.push(PriorityQueueItem(0, -DBL_MAX, false, i));
	}

	// Variable that tracks the distance to the farthest point in our results
	double tau = DBL_MAX;
	//target, node_queue, n_neighbors, heap, indexes, tau, _leaf_number
	// Perform the search
	int _leaf_number = leaf_number;
	int dup_count = 0;
	while (!node_queue.empty() && (_leaf_number > 0 || int(heap.size()) < n_neighbors)) {
		int tree_id = node_queue.top().tree_id;
		if (trees[tree_id].priority_search(node_queue, target, n_neighbors, heap, indexes, tau, _leaf_number)) {
			dup_count++;
		}
	}
	// Gather final results
	indices->clear();
	distances->clear();
	while (!heap.empty()) {
		indices->push_back(heap.top().index);
		distances->push_back(heap.top().dist);
		heap.pop();
	}

	// Results are in reverse order
	std::reverse(indices->begin(), indices->end());
	std::reverse(distances->begin(), distances->end());
	return double(dup_count) / double(leaf_number - _leaf_number);
}

int VpTreeForest::least_leaf_number_search(const DataPoint& target, int n_neighbors, double max_distance, double accuracy) {
	// Use a priority queue to store intermediate results on
	std::priority_queue<PriorityQueueItem> heap;
	std::set<int> indexes;

	// Use a priority queue to store node to be search
	std::priority_queue<PriorityQueueItem> node_queue;
	for (int i = 0; i < forest_size; i++) {
		node_queue.push(PriorityQueueItem(0, -DBL_MAX, false, i));
	}

	// Variable that tracks the distance to the farthest point in our results
	double tau = DBL_MAX;
	//target, node_queue, n_neighbors, heap, indexes, tau, _leaf_number
	// Perform the search
	int _leaf_number = 0;
	int least_count = int(accuracy * n_neighbors);
	while (!node_queue.empty() && (least_count > 0 || int(heap.size()) < n_neighbors)) {
		int tree_id = node_queue.top().tree_id;
		trees[tree_id].forest_least_leaf_number_search_helper(node_queue, target, n_neighbors, heap, indexes, tau, max_distance, _leaf_number, least_count);
	}
	return _leaf_number;
}


int KdTreeForest::least_leaf_number_search(const Node& target, int n_neighbors, double max_distance, double accuracy) {
	// Use a priority queue to store intermediate results on
	std::priority_queue<PriorityQueueItem> heap;
	std::set<int> indexes;

	// Use a priority queue to store node to be search
	std::priority_queue<PriorityQueueItem> node_queue;
	for (int i = 0; i < forest_size; i++) {
		node_queue.push(PriorityQueueItem(0, -DBL_MAX, false, i));
	}

	// Variable that tracks the distance to the farthest point in our results
	double tau = DBL_MAX;

	// Perform the search
	int _leaf_number = 0;
	int least_count = int(accuracy * n_neighbors);
	while (!node_queue.empty() && (least_count > 0 || int(heap.size()) < n_neighbors)) {
		int tree_id = node_queue.top().tree_id;
		trees[tree_id].forest_least_leaf_number_search_helper(node_queue, target, n_neighbors, heap, indexes, tau, max_distance, _leaf_number, least_count);
	}
	return _leaf_number;
}