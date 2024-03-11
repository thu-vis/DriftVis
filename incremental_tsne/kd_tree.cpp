/*
 *  kd_tree.cpp
 *  Implementation of kd_tree.
 *
 *  Created by Shouxing Xiang.
 */

#include <float.h>
#include <algorithm>
#include "kd_tree.h"

Node::Node() {
	index = -1;
	split = -1;
	data = NULL;
	dim = 0;
	left = -1;
	right = -1;
}

Node::Node(double* _data, int _dim) {
	index = -1;
	split = -1;
	data = _data;
	dim = _dim;
	left = -1;
	right = -1;
}

Node::Node(int _index, int _split, double* _data, int _dim, int _left, int _right) {
	index = _index;
	split = _split;
	data = _data;
	dim = _dim;
	left = _left;
	right = _right;
}

Node::Node(const Node& other) {
	if (this != &other) {
		index = other.index;
		split = other.split;
		data = other.data;
		dim = other.dim;
		left = other.left;
		right = other.right;
	}
}

Node& Node::operator=(const Node& other) {
	if (this != &other) {
		index = other.index;
		split = other.split;
		data = other.data;
		dim = other.dim;
		left = other.left;
		right = other.right;
	}
	return *this;
}

Node::~Node() {}

KdTree::KdTree(const std::vector<Node>& items, int N, int D, int _subdivide_variance_size) {
	nodes = items;
	size = N;
	dim = D;
	subdivide_variance_size = _subdivide_variance_size;
	build(0, N);
}

KdTree::~KdTree() {}

void KdTree::build(int left, int right) {
	if (left >= right) {
		return;
	}
	int split = 0;
	std::priority_queue<PriorityQueueItem> splits;
	for (int i = 0; i < dim; i++) {
		double delta = 0.0, mean = 0.0;
		for (int j = left; j < right; j++) {
			mean += nodes[j].data[i];
		}
		mean /= right - left;
		for (int j = left; j < right; j++) {
			delta += (nodes[j].data[i] - mean) * (nodes[j].data[i] - mean);
		}
		if (int(splits.size()) < subdivide_variance_size || delta > splits.top().dist) {
			if (int(splits.size()) >= subdivide_variance_size) {
				splits.pop();
			}
			splits.push(PriorityQueueItem(i, delta, false));
		}
	}

	int choice = int((double(rand() - 1) / RAND_MAX) * int(splits.size()));
	for (int i = 0; i < choice - 1; i++) {
		splits.pop();
	}
	split = splits.top().index;

	int median = (left + right) / 2;
	std::nth_element(nodes.begin() + left, nodes.begin() + median,
		nodes.begin() + right, DistanceComparator(split));
	std::swap(nodes[left], nodes[median]);
	nodes[left].split = split;
	if (left < median) {
		nodes[left].left = left + 1;
		build(left + 1, median + 1);
	}
	if (median + 1 < right) {
		nodes[left].right = median + 1;
		build(median + 1, right);
	}
}

double euclidean_distance_squared(const Node& t1, const Node& t2) {
	double dd = .0;
	for (int d = 0; d < t1.dim; d++) {
		double t = (t1.data[d] - t2.data[d]);
		dd += t * t;
	}
	return dd;
}

void KdTree::search(const Node& target, int n_neighbors, std::vector<int>* indices, std::vector<double>* distances, int leaf_number) {
	// Use a priority queue to store intermediate results on
	std::priority_queue<PriorityQueueItem> heap;

	// Variable that tracks the distance to the farthest point in our results
	double tau = DBL_MAX;

	// Perform the search
	search(target, 0, n_neighbors, heap, tau, leaf_number);

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
}

void KdTree::priority_search(const Node& target, int n_neighbors, std::vector<int>* indices, std::vector<double>* distances, int leaf_number) {
	// Use a priority queue to store intermediate results on
	std::priority_queue<PriorityQueueItem> heap;

	// Use a priority queue to store node to be search
	std::priority_queue<PriorityQueueItem> node_queue;
	node_queue.push(PriorityQueueItem(0, 0, false));

	// Variable that tracks the distance to the farthest point in our results
	double tau = DBL_MAX;


	// Perform the search
	while (!node_queue.empty() && (leaf_number > 0 || int(heap.size()) < n_neighbors)) {
		int cur_root = node_queue.top().index;
		node_queue.pop();

		if (cur_root >= size || cur_root < 0) {
			continue;
		}
        leaf_number--;
		double dist = euclidean_distance_squared(target, nodes[cur_root]);
		// If we arrived at a leaf
		if (nodes[cur_root].left == -1 && nodes[cur_root].right == -1) {
			if (dist < tau) {
				if (heap.size() == n_neighbors) {
					heap.pop();
				}
				heap.push(PriorityQueueItem(nodes[cur_root].index, dist));
				if (heap.size() == n_neighbors) {
					tau = heap.top().dist;
				}
			}
			continue;
		}

		int split = nodes[cur_root].split;
		double split_value = nodes[cur_root].data[split];
		double split_gap = target.data[split] - split_value;
		if (target.data[split] <= split_value) {
			node_queue.push(PriorityQueueItem(nodes[cur_root].left, -dist, false));
			if (tau > split_gap * split_gap) {
				if (dist < tau) {
					if (heap.size() == n_neighbors) {
						heap.pop();
					}
					heap.push(PriorityQueueItem(nodes[cur_root].index, dist));
					if (heap.size() == n_neighbors) {
						tau = heap.top().dist;
					}
				}
				node_queue.push(PriorityQueueItem(nodes[cur_root].right, dist, false));
			}
		}
		else {
			node_queue.push(PriorityQueueItem(nodes[cur_root].right, -dist, false));
			if (tau > split_gap * split_gap) {
				if (dist < tau) {
					if (heap.size() == n_neighbors) {
						heap.pop();
					}
					heap.push(PriorityQueueItem(nodes[cur_root].index, dist));
					if (heap.size() == n_neighbors) {
						tau = heap.top().dist;
					}
				}
				node_queue.push(PriorityQueueItem(nodes[cur_root].left, dist, false));
			}
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
}

void KdTree::search(const Node& target, int cur_root, int n_neighbors, std::priority_queue<PriorityQueueItem>& heap, double& tau, int& leaf_number) {
	if (cur_root >= size || cur_root < 0) {
		return;
	}
	if (heap.size() == n_neighbors && leaf_number <= 0) {
		return;
	}
    leaf_number--;

	// Return if we arrived at a leaf
	if (nodes[cur_root].left == -1 && nodes[cur_root].right == -1) {
		double dist = euclidean_distance_squared(target, nodes[cur_root]);
		if (dist < tau) {
			if (heap.size() == n_neighbors) {
				heap.pop();
			}
			heap.push(PriorityQueueItem(nodes[cur_root].index, dist));
			if (heap.size() == n_neighbors) {
				tau = heap.top().dist;
			}
		}
		return;
	}



	int split = nodes[cur_root].split;
	double split_value = nodes[cur_root].data[split];
	double split_gap = target.data[split] - split_value;
	if (target.data[split] <= split_value) {
		search(target, nodes[cur_root].left, n_neighbors, heap, tau, leaf_number);
		if (tau > split_gap * split_gap) {
			double dist = euclidean_distance_squared(target, nodes[cur_root]);
			if (dist < tau) {
				if (heap.size() == n_neighbors) {
					heap.pop();
				}
				heap.push(PriorityQueueItem(nodes[cur_root].index, dist));
				if (heap.size() == n_neighbors) {
					tau = heap.top().dist;
				}
			}
			search(target, nodes[cur_root].right, n_neighbors, heap, tau, leaf_number);
		}
	}
	else {
		search(target, nodes[cur_root].right, n_neighbors, heap, tau, leaf_number);
		if (tau > split_gap * split_gap) {
			double dist = euclidean_distance_squared(target, nodes[cur_root]);
			if (dist < tau) {
				if (heap.size() == n_neighbors) {
					heap.pop();
				}
				heap.push(PriorityQueueItem(nodes[cur_root].index, dist));
				if (heap.size() == n_neighbors) {
					tau = heap.top().dist;
				}
			}
			search(target, nodes[cur_root].left, n_neighbors, heap, tau, leaf_number);
		}
	}
}

bool KdTree::priority_search(const Node& target, std::priority_queue<PriorityQueueItem>& node_queue, int n_neighbors, std::priority_queue<PriorityQueueItem>& heap, 
								std::set<int>& indexes, double& tau, int& leaf_number) {
		
	bool flag = false;
	int cur_root = node_queue.top().index;
	int tree_id = node_queue.top().tree_id;
	node_queue.pop();

	if (cur_root >= size || cur_root < 0) {
		return flag;
	}
    leaf_number--;
	double dist = euclidean_distance_squared(target, nodes[cur_root]);
	flag = indexes.count(nodes[cur_root].index) != 0;
	// If we arrived at a leaf
	if (nodes[cur_root].left == -1 && nodes[cur_root].right == -1) {
		if (dist < tau && !flag) {
			if (heap.size() == n_neighbors) {
				indexes.erase(heap.top().index);
				heap.pop();
			}
			indexes.insert(nodes[cur_root].index);
			heap.push(PriorityQueueItem(nodes[cur_root].index, dist));
			if (heap.size() == n_neighbors) {
				tau = heap.top().dist;
			}
		}
		return flag;
	}

	int split = nodes[cur_root].split;
	double split_value = nodes[cur_root].data[split];
	double split_gap = target.data[split] - split_value;
	if (target.data[split] <= split_value) {
		node_queue.push(PriorityQueueItem(nodes[cur_root].left, -dist, false, tree_id));
		if (tau > split_gap * split_gap) {
			if (dist < tau && !flag) {
				if (heap.size() == n_neighbors) {
					indexes.erase(heap.top().index);
					heap.pop();
				}
				indexes.insert(nodes[cur_root].index);
				heap.push(PriorityQueueItem(nodes[cur_root].index, dist));
				if (heap.size() == n_neighbors) {
					tau = heap.top().dist;
				}
			}
			node_queue.push(PriorityQueueItem(nodes[cur_root].right, dist, false, tree_id));
		}
	}
	else {
		node_queue.push(PriorityQueueItem(nodes[cur_root].right, -dist, false, tree_id));
		if (tau > split_gap * split_gap) {
			if (dist < tau && !flag) {
				if (heap.size() == n_neighbors) {
					indexes.erase(heap.top().index);
					heap.pop();
				}
				indexes.insert(nodes[cur_root].index);
				heap.push(PriorityQueueItem(nodes[cur_root].index, dist));
				if (heap.size() == n_neighbors) {
					tau = heap.top().dist;
				}
			}
			node_queue.push(PriorityQueueItem(nodes[cur_root].left, dist, false, tree_id));
		}
	}
	return flag;
}

int KdTree::least_leaf_number_search(const Node& target, int n_neighbors, double max_distance, double accuracy) {
	// Use a priority queue to store intermediate results on
	std::priority_queue<PriorityQueueItem> heap;

	// Use a priority queue to store node to be search
	std::priority_queue<PriorityQueueItem> node_queue;
	node_queue.push(PriorityQueueItem(0, 0, false));
	int leaf_number = 0;
	int least_count = int(n_neighbors * accuracy);

	// Variable that tracks the distance to the farthest point in our results
	double tau = DBL_MAX;


	// Perform the search
	while (!node_queue.empty() && (least_count > 0 || int(heap.size()) < n_neighbors)) {
		int cur_root = node_queue.top().index;
		node_queue.pop();

		if (cur_root >= size || cur_root < 0) {
			continue;
		}
		leaf_number++;
		double dist = euclidean_distance_squared(target, nodes[cur_root]);
		// If we arrived at a leaf
		if (nodes[cur_root].left == -1 && nodes[cur_root].right == -1) {
			if (dist < tau) {
				if (heap.size() == n_neighbors) {
					heap.pop();
				}
				heap.push(PriorityQueueItem(nodes[cur_root].index, dist));
				if (heap.size() == n_neighbors) {
					tau = heap.top().dist;
				}
				if (dist < max_distance) {
					least_count--;
				}
			}
			continue;
		}

		int split = nodes[cur_root].split;
		double split_value = nodes[cur_root].data[split];
		double split_gap = target.data[split] - split_value;
		if (target.data[split] <= split_value) {
			node_queue.push(PriorityQueueItem(nodes[cur_root].left, -dist, false));
			if (tau > split_gap * split_gap) {
				if (dist < tau) {
					if (heap.size() == n_neighbors) {
						heap.pop();
					}
					heap.push(PriorityQueueItem(nodes[cur_root].index, dist));
					if (heap.size() == n_neighbors) {
						tau = heap.top().dist;
					}
					if (dist < max_distance) {
						least_count--;
					}
				}
				node_queue.push(PriorityQueueItem(nodes[cur_root].right, dist, false));
			}
		}
		else {
			node_queue.push(PriorityQueueItem(nodes[cur_root].right, -dist, false));
			if (tau > split_gap * split_gap) {
				if (dist < tau) {
					if (heap.size() == n_neighbors) {
						heap.pop();
					}
					heap.push(PriorityQueueItem(nodes[cur_root].index, dist));
					if (heap.size() == n_neighbors) {
						tau = heap.top().dist;
					}
					if (dist < max_distance) {
						least_count--;
					}
				}
				node_queue.push(PriorityQueueItem(nodes[cur_root].left, dist, false));
			}
		}
	}

	return leaf_number;
}

void KdTree::forest_least_leaf_number_search_helper(std::priority_queue<PriorityQueueItem>& node_queue, const Node& target, unsigned int n_neighbors, std::priority_queue<PriorityQueueItem>& heap,
	std::set<int>& indexes, double& tau, double max_distance, int& leaf_number, int& least_count) {

	int cur_root = node_queue.top().index;
	int tree_id = node_queue.top().tree_id;
	node_queue.pop();

	if (cur_root >= size || cur_root < 0) {
		return;
	}
	leaf_number++;
	double dist = euclidean_distance_squared(target, nodes[cur_root]);
	bool flag = indexes.count(nodes[cur_root].index) != 0;
	// If we arrived at a leaf
	if (nodes[cur_root].left == -1 && nodes[cur_root].right == -1) {
		if (dist < tau && !flag) {
			if (heap.size() == n_neighbors) {
				indexes.erase(heap.top().index);
				heap.pop();
			}
			indexes.insert(nodes[cur_root].index);
			heap.push(PriorityQueueItem(nodes[cur_root].index, dist));
			if (heap.size() == n_neighbors) {
				tau = heap.top().dist;
			}
			if (dist <= max_distance) {
				least_count--;
			}
		}
		return;
	}

	int split = nodes[cur_root].split;
	double split_value = nodes[cur_root].data[split];
	double split_gap = target.data[split] - split_value;
	if (target.data[split] <= split_value) {
		node_queue.push(PriorityQueueItem(nodes[cur_root].left, -dist, false, tree_id));
		if (tau > split_gap * split_gap) {
			if (dist < tau && !flag) {
				if (heap.size() == n_neighbors) {
					indexes.erase(heap.top().index);
					heap.pop();
				}
				indexes.insert(nodes[cur_root].index);
				heap.push(PriorityQueueItem(nodes[cur_root].index, dist));
				if (heap.size() == n_neighbors) {
					tau = heap.top().dist;
				}
				if (dist <= max_distance) {
					least_count--;
				}
			}
			node_queue.push(PriorityQueueItem(nodes[cur_root].right, dist, false, tree_id));
		}
	}
	else {
		node_queue.push(PriorityQueueItem(nodes[cur_root].right, -dist, false, tree_id));
		if (tau > split_gap * split_gap) {
			if (dist < tau && !flag) {
				if (heap.size() == n_neighbors) {
					indexes.erase(heap.top().index);
					heap.pop();
				}
				indexes.insert(nodes[cur_root].index);
				heap.push(PriorityQueueItem(nodes[cur_root].index, dist));
				if (heap.size() == n_neighbors) {
					tau = heap.top().dist;
				}
				if (dist <= max_distance) {
					least_count--;
				}
			}
			node_queue.push(PriorityQueueItem(nodes[cur_root].left, dist, false, tree_id));
		}
	}
	return;
}