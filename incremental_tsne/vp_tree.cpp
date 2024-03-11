/*
*  vp_tree.cpp
*  Implementation of vp_tree.
*
*  Created by Shouxing Xiang.
*/

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <float.h>
#include "vp_tree.h"

DataPoint::DataPoint() {
	_D = 1;
	_ind = -1;
	_x = NULL;
}

DataPoint::DataPoint(int D, int ind, double* x) {
	_D = D;
	_ind = ind;
	_x = x;
}

// this makes a deep copy -- should not free anything
DataPoint::DataPoint(const DataPoint& other) {                     
	if (this != &other) {
		_D = other.dimensionality();
		_ind = other.index();
		_x = other._x;
	}
}

// asignment should free old object
DataPoint& DataPoint::operator= (const DataPoint& other) {         
	if (this != &other) {
		_D = other.dimensionality();
		_ind = other.index();
		_x = other._x;
	}
	return *this;
}

int DataPoint::index() const { 
	return _ind; 
}

int DataPoint::dimensionality() const { 
	return _D; 
}

double DataPoint::x(int d) const { 
	return _x[d]; 
}

DataPoint::~DataPoint() {}

VptreeNode::VptreeNode() {
	index = 0;
	threshold = 0.0;
	left = -1;
	right = -1;
}

VptreeNode::~VptreeNode() {}

double euclidean_distance_squared(const DataPoint &t1, const DataPoint &t2) {
	double dd = .0;
	for (int d = 0; d < t1.dimensionality(); d++) {
		double t = (t1.x(d) - t2.x(d));
		dd += t * t;
	}
	return dd;
}

VpTree::VpTree(const std::vector<DataPoint>& items) {
	_items = items;
	_size = items.size();
	buildFromPoints(0, items.size());
}

VpTree::~VpTree() {}

// Function that uses the tree to find the k nearest neighbors of target
void VpTree::search(const DataPoint& target, int k, std::vector<int>* results, std::vector<double>* distances, int leaf_number)
{

	// Use a priority queue to store intermediate results on
	std::priority_queue<PriorityQueueItem> heap;

	// Variable that tracks the distance to the farthest point in our results
	double tau = DBL_MAX;

	// Perform the search
	search(0, target, k, heap, tau, leaf_number);

	// Gather final results
	results->clear(); distances->clear();
	while (!heap.empty()) {
		results->push_back(_items[heap.top().index].index());
		distances->push_back(heap.top().dist);
		heap.pop();
	}

	// Results are in reverse order
	std::reverse(results->begin(), results->end());
	std::reverse(distances->begin(), distances->end());
}

// Function that uses the tree and priority queue to find the k nearest neighbors of target
void VpTree::priority_search(const DataPoint& target, int k, std::vector<int>* results, std::vector<double>* distances, int leaf_number) {
	// Use a priority queue to store intermediate results on
	std::priority_queue<PriorityQueueItem> heap;

	// Use a priority queue to store node to be search
	std::priority_queue<PriorityQueueItem> node_queue;
	node_queue.push(PriorityQueueItem(0, 0, false));

	// Variable that tracks the distance to the farthest point in our results
	double tau = DBL_MAX;

	// Perform the search
	while (!node_queue.empty() && (leaf_number > 0 || int(heap.size()) < k)) {
		int cur_root = node_queue.top().index;
		node_queue.pop();

		// indicates that we're done here
		if (cur_root < 0 || cur_root >= int(_nodes.size())) {
			continue;
		}
		leaf_number--;

		// Compute distance between target and current node
		double dist = euclidean_distance_squared(_items[_nodes[cur_root].index], target);

		// If current node within radius tau
		if (dist < tau) {
			if (heap.size() == k) heap.pop();                // remove furthest node from result list (if we already have k results)
			heap.push(PriorityQueueItem(_nodes[cur_root].index, dist));           // add current node to result list
			if (heap.size() == k) tau = heap.top().dist;    // update value of tau (farthest point in result list)
		}

		// Return if we arrived at a leaf
		if (_nodes[cur_root].left == -1 && _nodes[cur_root].right == -1) {
			continue;
		}

		// If the target lies within the radius of ball
		double gap = dist - _nodes[cur_root].threshold;
		if (gap < 0) {
			node_queue.push(PriorityQueueItem(_nodes[cur_root].left, gap, false));
			if (dist + tau >= _nodes[cur_root].threshold) {        // if there can still be neighbors outside the ball, recursively search right child
				node_queue.push(PriorityQueueItem(_nodes[cur_root].right, -gap, false));
			}

			// If the target lies outsize the radius of the ball
		}
		else {
			node_queue.push(PriorityQueueItem(_nodes[cur_root].right, -gap, false));

			if (dist - tau <= _nodes[cur_root].threshold) {         // if there can still be neighbors inside the ball, recursively search left child
				node_queue.push(PriorityQueueItem(_nodes[cur_root].left, gap, false));
			}
		}
	}

	// Gather final results
	results->clear(); distances->clear();
	while (!heap.empty()) {
		results->push_back(_items[heap.top().index].index());
		distances->push_back(heap.top().dist);
		heap.pop();
	}

	// Results are in reverse order
	std::reverse(results->begin(), results->end());
	std::reverse(distances->begin(), distances->end());
}

// Function that (recursively) fills the tree
void VpTree::buildFromPoints(int lower, int upper) {
	if (upper <= lower) {     // indicates that we're done here!
		return;
	}

	// Lower index is center of current node
	VptreeNode node;
	node.index = lower;

	if (upper - lower > 1) {      // if we did not arrive at leaf yet

								  // Choose an arbitrary point and move it to the start
		int i = (int)((double)rand() / RAND_MAX * (upper - lower - 1)) + lower;
		std::swap(_items[lower], _items[i]);

		// Partition around the median distance
		int median = (upper + lower) / 2;
		std::nth_element(_items.begin() + lower + 1,
			_items.begin() + median,
			_items.begin() + upper,
			DistanceComparator(_items[lower]));

		// Threshold of the new node will be the distance to the median
		node.threshold = euclidean_distance_squared(_items[lower], _items[median]);

		// Recursively build tree
		node.index = lower;
		if (lower + 1 < median) {
			node.left = lower + 1;
		}
		if (median < upper) {
			node.right = median;
		}

		_nodes.push_back(node);
		buildFromPoints(lower + 1, median);
		buildFromPoints(median, upper);
		return;
	}

	// Return result
	_nodes.push_back(node);
	return;
}

// Helper function that searches the tree
void VpTree::search(int cur_root, const DataPoint& target, unsigned int k, std::priority_queue<PriorityQueueItem>& heap, double& tau, int& leaf_number) {
	// indicates that we're done here
	if (cur_root < 0 || cur_root >= int(_nodes.size())) {
		return;
	}

	if (leaf_number <= 0 && heap.size() == k) {
		return;
	}
    leaf_number--;

	// Compute distance between target and current node
	double dist = euclidean_distance_squared(_items[_nodes[cur_root].index], target);

	// If current node within radius tau
	if (dist < tau) {
		if (heap.size() == k) heap.pop();                // remove furthest node from result list (if we already have k results)
		heap.push(PriorityQueueItem(_nodes[cur_root].index, dist));           // add current node to result list
		if (heap.size() == k) tau = heap.top().dist;    // update value of tau (farthest point in result list)
	}

	// Return if we arrived at a leaf
	if (_nodes[cur_root].left == -1 && _nodes[cur_root].right == -1) {
		return;
	}

	// If the target lies within the radius of ball
	if (dist < _nodes[cur_root].threshold) {
		search(_nodes[cur_root].left, target, k, heap, tau, leaf_number);
		if (dist + tau >= _nodes[cur_root].threshold) {        // if there can still be neighbors outside the ball, recursively search right child
			search(_nodes[cur_root].right, target, k, heap, tau, leaf_number);
		}

		// If the target lies outsize the radius of the ball
	}
	else {
		search(_nodes[cur_root].right, target, k, heap, tau, leaf_number);
		if (dist - tau <= _nodes[cur_root].threshold) {         // if there can still be neighbors inside the ball, recursively search left child
			search(_nodes[cur_root].left, target, k, heap, tau, leaf_number);
		}
	}
}

// Helper function that searches the tree
bool VpTree::priority_search(std::priority_queue<PriorityQueueItem>& node_queue, const DataPoint& target, unsigned int k, 
	std::priority_queue<PriorityQueueItem>& heap, std::set<int>& indexes, double& tau, int& leaf_number) {
	bool flag = false;
	int cur_root = node_queue.top().index;
	int tree_id = node_queue.top().tree_id;
	node_queue.pop();

	// indicates that we're done here
	if (cur_root < 0 || cur_root >= int(_nodes.size())) {
		return flag;
	}
    leaf_number--;

	// Compute distance between target and current node
	double dist = euclidean_distance_squared(_items[_nodes[cur_root].index], target);
	flag = indexes.count(_items[_nodes[cur_root].index].index()) != 0;
	// If current node within radius tau
	if (dist < tau && !flag) {
		if (heap.size() == k) {
			indexes.erase(_items[heap.top().index].index());
			heap.pop();                // remove furthest node from result list (if we already have k results)
		}
		indexes.insert(_items[_nodes[cur_root].index].index());
		heap.push(PriorityQueueItem(_items[_nodes[cur_root].index].index(), dist));           // add current node to result list
		if (heap.size() == k) {
			tau = heap.top().dist;    // update value of tau (farthest point in result list)
		}
	}

	// Return if we arrived at a leaf
	if (_nodes[cur_root].left == -1 && _nodes[cur_root].right == -1) {
		return flag;
	}

	// If the target lies within the radius of ball
	double gap = dist - _nodes[cur_root].threshold;
	if (gap < 0) {
		node_queue.push(PriorityQueueItem(_nodes[cur_root].left, gap, false, tree_id));
		if (dist + tau >= _nodes[cur_root].threshold) {        // if there can still be neighbors outside the ball, recursively search right child
			node_queue.push(PriorityQueueItem(_nodes[cur_root].right, -gap, false, tree_id));
		}

		// If the target lies outsize the radius of the ball
	}
	else {
		node_queue.push(PriorityQueueItem(_nodes[cur_root].right, -gap, false, tree_id));

		if (dist - tau <= _nodes[cur_root].threshold) {         // if there can still be neighbors inside the ball, recursively search left child
			node_queue.push(PriorityQueueItem(_nodes[cur_root].left, gap, false, tree_id));
		}
	}
	return flag;
}

int VpTree::least_leaf_number_search(const DataPoint& target, int n_neighbors, double max_distance, double accuracy) {
	// Use a priority queue to store intermediate results on
	std::priority_queue<PriorityQueueItem> heap;

	// Use a priority queue to store node to be search
	std::priority_queue<PriorityQueueItem> node_queue;
	node_queue.push(PriorityQueueItem(0, 0, false));

	// Variable that tracks the distance to the farthest point in our results
	double tau = DBL_MAX;
	int leaf_number = 0;
	int least_count = int(n_neighbors * accuracy);
	// Perform the search
	while (!node_queue.empty() && (least_count > 0 || int(heap.size()) < n_neighbors)) {
		int cur_root = node_queue.top().index;
		node_queue.pop();

		// indicates that we're done here
		if (cur_root < 0 || cur_root >= int(_nodes.size())) {
			continue;
		}
		leaf_number++;

		// Compute distance between target and current node
		double dist = euclidean_distance_squared(_items[_nodes[cur_root].index], target);

		// If current node within radius tau
		if (dist < tau) {
			if (heap.size() == n_neighbors) heap.pop();                // remove furthest node from result list (if we already have k results)
			heap.push(PriorityQueueItem(_nodes[cur_root].index, dist));           // add current node to result list
			if (heap.size() == n_neighbors) tau = heap.top().dist;    // update value of tau (farthest point in result list)
			if (dist <= max_distance) {
				least_count--;
			}
		}

		// Return if we arrived at a leaf
		if (_nodes[cur_root].left == -1 && _nodes[cur_root].right == -1) {
			continue;
		}

		// If the target lies within the radius of ball
		double gap = dist - _nodes[cur_root].threshold;
		if (gap < 0) {
			node_queue.push(PriorityQueueItem(_nodes[cur_root].left, gap, false));
			if (dist + tau >= _nodes[cur_root].threshold) {        // if there can still be neighbors outside the ball, recursively search right child
				node_queue.push(PriorityQueueItem(_nodes[cur_root].right, -gap, false));
			}

			// If the target lies outsize the radius of the ball
		}
		else {
			node_queue.push(PriorityQueueItem(_nodes[cur_root].right, -gap, false));

			if (dist - tau <= _nodes[cur_root].threshold) {         // if there can still be neighbors inside the ball, recursively search left child
				node_queue.push(PriorityQueueItem(_nodes[cur_root].left, gap, false));
			}
		}
	}

	return leaf_number;
}

void VpTree::forest_least_leaf_number_search_helper(std::priority_queue<PriorityQueueItem>& node_queue, const DataPoint& target, unsigned int k, std::priority_queue<PriorityQueueItem>& heap, std::set<int>& indexes, double& tau, double max_distance, int& leaf_number, int& least_count) {
	int cur_root = node_queue.top().index;
	int tree_id = node_queue.top().tree_id;
	node_queue.pop();

	// indicates that we're done here
	if (cur_root < 0 || cur_root >= int(_nodes.size())) {
		return;
	}
	leaf_number++;

	// Compute distance between target and current node
	double dist = euclidean_distance_squared(_items[_nodes[cur_root].index], target);
	bool flag = indexes.count(_items[_nodes[cur_root].index].index()) != 0;
	// If current node within radius tau
	if (dist < tau && !flag) {
		if (heap.size() == k) {
			indexes.erase(_items[heap.top().index].index());
			heap.pop();                // remove furthest node from result list (if we already have k results)
		}
		indexes.insert(_items[_nodes[cur_root].index].index());
		heap.push(PriorityQueueItem(_items[_nodes[cur_root].index].index(), dist));           // add current node to result list
		if (heap.size() == k) {
			tau = heap.top().dist;    // update value of tau (farthest point in result list)
		}
		if (dist <= max_distance) {
			least_count--;
		}
	}

	// Return if we arrived at a leaf
	if (_nodes[cur_root].left == -1 && _nodes[cur_root].right == -1) {
		return;
	}

	// If the target lies within the radius of ball
	double gap = dist - _nodes[cur_root].threshold;
	if (gap < 0) {
		node_queue.push(PriorityQueueItem(_nodes[cur_root].left, gap, false, tree_id));
		if (dist + tau >= _nodes[cur_root].threshold) {        // if there can still be neighbors outside the ball, recursively search right child
			node_queue.push(PriorityQueueItem(_nodes[cur_root].right, -gap, false, tree_id));
		}

		// If the target lies outsize the radius of the ball
	}
	else {
		node_queue.push(PriorityQueueItem(_nodes[cur_root].right, -gap, false, tree_id));

		if (dist - tau <= _nodes[cur_root].threshold) {         // if there can still be neighbors inside the ball, recursively search left child
			node_queue.push(PriorityQueueItem(_nodes[cur_root].left, gap, false, tree_id));
		}
	}
	return;
}