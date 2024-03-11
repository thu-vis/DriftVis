/*
*  forest.h
*  Header file for forest.
*
*  Created by Shouxing Xiang.
*/

#pragma once
#include <vector>
#include <queue>
#include "kd_tree.h"
#include "vp_tree.h"
#include "tool.h"

class KdTreeForest {
public:
	KdTreeForest(const std::vector<Node>& items, int _tree_size, int _data_dim, int _forest_size, int _subdivide_variance_size, int _leaf_number);
	~KdTreeForest();
	//Function that uses the tree and priority queue to find the k nearest neighbors of target
	double priority_search(const Node& target, int n_neighbors, std::vector<int>* indices, std::vector<double>* distances);
	int least_leaf_number_search(const Node& target, int n_neighbors, double max_distance, double accuracy);
	void accuracy_trend_test(const std::vector<Node>& targets, int n_neighbors, double *max_distance);
	
	
//private:
	int tree_size;
	int data_dim;
	int forest_size;
	int subdivide_variance_size; //Choose randomly from k dimension whose variance is larger to subdivide the nodes: subdivide_variance_size
	int leaf_number;
	std::vector<KdTree> trees;
};

class VpTreeForest {
public:
	VpTreeForest(const std::vector<DataPoint>& items, int _forest_size, int _leaf_number);
	~VpTreeForest();
	//Function that uses the tree and priority queue to find the k nearest neighbors of target
	double priority_search(const DataPoint& target, int n_neighbors, std::vector<int>* indices, std::vector<double>* distances);
	int least_leaf_number_search(const DataPoint& target, int n_neighbors, double max_distance, double accuracy);


//private:
	int forest_size; 
	int leaf_number;
	std::vector<VpTree> trees;

};