/*
 *  kd_tree.h
 *  Header file for kd_tree.
 *
 *  Created by Shouxing Xiang.
 */

#pragma once
#include <vector>
#include <queue>
#include <set>
#include "tool.h"

class Node {
public:
    int index;
    int split;
    double* data;
    int dim;
    int left;
    int right;
	Node();
	Node(double* _data, int _dim);
	Node(int _index, int _split, double* _data, int _dim, int _left, int _right);
    Node(const Node& other);
	Node& operator= (const Node& other);
	~Node();
};

class KdTree {
public:
    std::vector<Node> nodes;
    int size;
    int dim;
	int subdivide_variance_size;

    // Default constructor
	KdTree(const std::vector<Node>& items, int N, int D, int _subdivide_variance_size);

    // Destructor
	~KdTree();

	void build(int left, int right);

    // Function that uses the tree to find the k nearest neighbors of target
	void search(const Node& target, int n_neighbors, std::vector<int>* indices, std::vector<double>* distances, int leaf_number);

	//Function that uses the tree and priority queue to find the k nearest neighbors of target
	void priority_search(const Node& target, int n_neighbors, std::vector<int>* indices, std::vector<double>* distances, int leaf_number);

private:
	// Distance comparator for use in nth_element
	struct DistanceComparator
	{
		const int split;
		explicit DistanceComparator(const int split) : split(split) {}
		bool operator()(const Node& a, const Node& b) {
			return a.data[split] < b.data[split];
		}
	};

    // Helper function that searches the tree, just in the single tree
	void search(const Node& target, int cur_root, int n_neighbors, std::priority_queue<PriorityQueueItem>& heap, double& tau, int& leaf_number);

public:
	// Helper function that searches the tree, just in the single tree
	bool priority_search(const Node& target, std::priority_queue<PriorityQueueItem>& node_queue, int n_neighbors, std::priority_queue<PriorityQueueItem>& heap, 
							std::set<int>& indexes, double& tau, int& leaf_number);
	int least_leaf_number_search(const Node& target, int n_neighbors, double max_distance, double accuracy);
	void forest_least_leaf_number_search_helper(std::priority_queue<PriorityQueueItem>& node_queue, const Node& target, unsigned int k, std::priority_queue<PriorityQueueItem>& heap,
		std::set<int>& indexes, double& tau, double max_distance, int& leaf_number, int& least_count);
};

double euclidean_distance_squared(const Node& t1, const Node& t2);