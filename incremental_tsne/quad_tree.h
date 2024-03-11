/*
 *  quad_tree.h
 *  Header file for quad tree.
 *
 *  Created by Shouxing Xiang.
 *  Code is from Dmitry Ulyanov's Multicore-TSNE
 */

#pragma once
#include <cstdlib>
#include <vector>

static inline double min(double x, double y) { return (x <= y ? x : y); }
static inline double max(double x, double y) { return (x <= y ? y : x); }
static inline double abs_d(double x) { return (x <= 0 ? -x : x); }
static inline double sign(double x) {
    return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0));
}

class Cell {

public:
	double* mins;
	double* maxs;
	int n_dims;
	bool   containsPoint(double point[]);
	~Cell() {
		delete[] mins;
		delete[] maxs;
	}
};

class QueryParameter {
public:
	short *labels;
	short label;
	double *label_alpha;
	QueryParameter(short label, short *labels, double* label_alpha):
		label(label), labels(labels), label_alpha(label_alpha) {}
	~QueryParameter(){}
};

class QuadTree
{

	// Fixed constants
	static const int QT_NODE_CAPACITY = 1;
	

	// Properties of this node in the tree
	int QT_NO_DIMS;
	bool is_leaf;
	int size;
	double cum_size;
	double *weight;

	// Axis-aligned bounding box stored as a center with half-dimensions to represent the boundaries of this quad tree
	Cell boundary;

	// Indices in this quad tree node, corresponding center-of-mass, and list of all children
	double* data;
	double* center_of_mass;
	int index[QT_NODE_CAPACITY];

	int num_children;
	std::vector<QuadTree*> children;
public:


	QuadTree(double* inp_data, int N, int no_dims, double* _weight = NULL);
	QuadTree(QuadTree* inp_parent, double* inp_data, double* mean_Y, double* width_Y);
	~QuadTree();
	bool insert(int new_index);
	void subdivide();
	void set_weight(double* _weight);
	void compute_non_edge_forces(int point_index, double theta, double* neg_f, double* sum_Q);
	void constraint_compute_non_edge_forces(double* target, double theta, double* neg_f, double* sum_Q);
	void modified_compute_non_edge_forces(double* target, QueryParameter* para, double theta, double* neg_f, double* sum_Q);

	int summarize(int point_index, double theta, double* summary, int idx);
	int constraint_summarize(double* target, double theta, double* summary, int idx);

private:

	void init(QuadTree* inp_parent, double* inp_data, double* mean_Y, double* width_Y);
	void fill(int N);
};