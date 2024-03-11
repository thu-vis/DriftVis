/*
 *  kd_tree_forest.h
 *  Header file for kd_tree forest.
 *
 *  Created by Shouxing Xiang.
 */

#pragma once
#include <vector>
#include <queue>

class Node {
public:
    int index;
    int split;
    double* data;
    int dim;
    int left;
    int right;
    Node() {
        index = -1;
        split = -1;
        data = NULL;
        dim = 0;
        left = -1;
        right = -1;
    }
    Node(double* _data, int _dim) {
        index = -1;
        split = -1;
        data = _data;
        dim = _dim;
        left = -1;
        right = -1;
    }
    Node(int _index, int _split, double* _data, int _dim, int _left, int _right) {
        fprintf(stderr, "Node init.\n");
        index = _index;
        split = _split;
		data = new double[_dim];
		for (int i = 0; i < _dim; i++) {
			data[i] = _data[i];
		}
        dim = _dim;
        left = _left;
        right = _right;
    }
    Node(const Node& other) {
        if (this != &other) {
            index = other.index;
            split = other.split;
            data = other.data;
            dim = other.dim;
            left = other.left;
            right = other.right;
        }
    }
    Node& operator= (const Node& other) {
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
    ~Node() {}
};

class KdTree {
public:
    std::vector<Node> tree;
    int size;
    int dim;

    // Default constructor
    KdTree(const std::vector<Node>& items, int N, int D) {
        fprintf(stderr, "KdTree init.\n");
        tree = items;
        size = N;
        dim = D;
        build(0, N);
    }

    // Destructor
    ~KdTree() {}

    void build(int left, int right) {
        //fprintf(stderr, "KdTree build.\n");
        if (left >= right) {
            return;
        }
        int split = (int) ((double)rand() / RAND_MAX * dim);
        int median = (left + right) / 2;
        std::nth_element(tree.begin() + left, tree.begin() + median,
                    tree.begin() + right, DistanceComparator(split));
        std::swap(tree[left], tree[median]);
        tree[left].split = split;
        if (left < median) {
            tree[left].left = left + 1;
            build(left + 1, median + 1);
        }
        if (median + 1 < right) {
            tree[left].right = median + 1;
            build(median + 1, right);
        }
    }

    // Distance comparator for use in nth_element
    struct DistanceComparator
    {
        const int split;
        explicit DistanceComparator(const int split) : split(split) {}
        bool operator()(const Node& a, const Node& b) {
            return a.data[split] < b.data[split];
        }
    };

    // An item on the intermediate result queue
    struct HeapItem {
        HeapItem( int index, double dist) :
            index(index), dist(dist) {}
        int index;
        double dist;
        bool operator<(const HeapItem& o) const {
            return dist < o.dist;
        }
    };

    double euclidean_distance_squared(const Node& t1, const Node& t2) {
        double dd = .0;
        for (int d = 0; d < t1.dim; d++) {
            double t = (t1.data[d] - t2.data[d]);
            dd += t * t;
        }
        return dd;
    }

    // Function that uses the tree to find the k nearest neighbors of target
    void search(const Node& target, int n_neighbors, std::vector<int>* indices, std::vector<double>* distances) {
        // Use a priority queue to store intermediate results on
        std::priority_queue<HeapItem> heap;

        // Variable that tracks the distance to the farthest point in our results
        double tau = DBL_MAX;

        int leaf_number = 10000000;

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

private:
    // Helper function that searches the tree
    void search(const Node& target, int cur_root, int n_neighbors, std::priority_queue<HeapItem>& heap, double& tau, int& leaf_number) {
        if (cur_root >= size || cur_root < 0) {
            return;
        }
        if (leaf_number <= 0) {
            return;
        }
        double dist = euclidean_distance_squared(target, tree[cur_root]);
        if (dist < tau) {
            if (heap.size() == n_neighbors) {
                heap.pop();
            }
            heap.push(HeapItem(tree[cur_root].index, dist));
            if (heap.size() == n_neighbors) {
                tau = heap.top().dist;
            }
        }

        // Return if we arrived at a leaf
        if (tree[cur_root].left == -1 && tree[cur_root].right == -1) {
            leaf_number--;
            return;
        }
        int split = tree[cur_root].split;
        double split_value = tree[cur_root].data[split];
        double split_gap = target.data[split] - split_value;
        if (target.data[split] <= split_value) {
            search(target, tree[cur_root].left, n_neighbors, heap, tau, leaf_number);
            if (tau > split_gap * split_gap) {
                search(target, tree[cur_root].right, n_neighbors, heap, tau, leaf_number);
            }
        }
        else {
            search(target, tree[cur_root].right, n_neighbors, heap, tau, leaf_number);
            if (tau > split_gap * split_gap) {
                search(target, tree[cur_root].left, n_neighbors, heap, tau, leaf_number);
            }
        }
    }
};