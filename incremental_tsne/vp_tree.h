/*
 *  vptree.h
 *  Implementation of a vantage-point tree.
 *
 *  Created by Laurens van der Maaten.
 *  Copyright 2012, Delft University of Technology. All rights reserved.
 *
 *  Multicore version by Dmitry Ulyanov, 2016. dmitry.ulyanov.msu@gmail.com
 */


#include <cstdlib>
#include <algorithm>
#include <vector>
#include <cstdio>
#include <queue>
#include <set>
#include <limits>
#include "tool.h"


#ifndef VPTREE_H
#define VPTREE_H


class DataPoint
{
    int _D;
    int _ind;

public:
    double* _x;
	DataPoint();
	DataPoint(int D, int ind, double* x);
	DataPoint(const DataPoint& other);
	~DataPoint();

	DataPoint& operator= (const DataPoint& other);
	int index() const;
	int dimensionality() const;
	double x(int d) const;
};

double euclidean_distance_squared(const DataPoint &t1, const DataPoint &t2);


// Single node of a VP tree (has a point and radius; left children are closer to point than the radius)
class VptreeNode {
public:
	int index;              // index of point in node
	double threshold;       // radius(?)
	int left;             // points closer by than threshold
	int right;            // points farther away than threshold

	VptreeNode();

	~VptreeNode();
};


class VpTree
{
public:
    // Default constructor
	VpTree(const std::vector<DataPoint>& items);

    // Destructor
	~VpTree();

    // Function that uses the tree to find the k nearest neighbors of target
	void search(const DataPoint& target, int k, std::vector<int>* results, std::vector<double>* distances, int leaf_number);

	// Function that uses the tree and priority queue to find the k nearest neighbors of target
	void priority_search(const DataPoint& target, int k, std::vector<int>* results, std::vector<double>* distances, int leaf_number);


	std::vector<DataPoint> _items;
	int _size;
	std::vector<VptreeNode> _nodes;

private:

    // Distance comparator for use in std::nth_element
    struct DistanceComparator
    {
        const DataPoint& item;
        explicit DistanceComparator(const DataPoint& item) : item(item) {}
        bool operator()(const DataPoint& a, const DataPoint& b) {
            return euclidean_distance_squared(item, a) < euclidean_distance_squared(item, b);
        }
    };

    // Function that (recursively) fills the tree
	void buildFromPoints(int lower, int upper);

    // Helper function that searches the tree
	void search(int cur_root, const DataPoint& target, unsigned int k, std::priority_queue<PriorityQueueItem>& heap, double& tau, int& leaf_number);

public:
	// Helper function that searches the tree
	bool priority_search(std::priority_queue<PriorityQueueItem>& node_queue, const DataPoint& target, unsigned int k, std::priority_queue<PriorityQueueItem>& heap, std::set<int>& indexes, double& tau, int& leaf_number);
	int least_leaf_number_search(const DataPoint& target, int n_neighbors, double max_distance, double accuracy);
	void forest_least_leaf_number_search_helper(std::priority_queue<PriorityQueueItem>& node_queue, const DataPoint& target, unsigned int k, std::priority_queue<PriorityQueueItem>& heap, 
		std::set<int>& indexes, double& tau, double max_distance, int& leaf_number, int& least_count);
};

#endif
