/*
 *  quad_tree.cpp
 *  Implementation of quad_tree.
 *
 *  Created by Shouxing Xiang.
 *  Code is from Dmitry Ulyanov's Multicore-TSNE
 */
 
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <iostream>
#include <float.h>
#include "quad_tree.h"


// Checks whether a point lies in a cell
bool Cell::containsPoint(double point[])
{   
    for (int i = 0; i< n_dims; ++i) {
        if (point[i] < mins[i] || point[i] > maxs[i]) {
            return false;
        }
    }
    return true;
}


// Default constructor for quadtree -- build tree, too!
QuadTree::QuadTree(double* inp_data, int N, int no_dims, double* _weight)
{   
	weight = NULL;
    QT_NO_DIMS = no_dims;
	num_children = 1;
	for (int i = 0; i < QT_NO_DIMS; i++) {
		num_children *= 2;
	}
    

    // Compute mean, width, and height of current map (boundaries of QuadTree)

    double*  min_Y = new double[QT_NO_DIMS]; 
    for (int d = 0; d < QT_NO_DIMS; d++) {
        min_Y[d] =  DBL_MAX;  
    } 
    double*  max_Y = new double[QT_NO_DIMS]; 
    for (int d = 0; d < QT_NO_DIMS; d++) {
        max_Y[d] = -DBL_MAX;
    }

    for (int n = 0; n < N; n++) {
        for (int d = 0; d < QT_NO_DIMS; d++) {
            min_Y[d] = min(min_Y[d], inp_data[n * QT_NO_DIMS + d]);
            max_Y[d] = max(max_Y[d], inp_data[n * QT_NO_DIMS + d]);
        }
    }
    for (int d = 0; d < QT_NO_DIMS; d++) {
		max_Y[d] = max(max_Y[d] * (1. + 1e-3 * sign(max_Y[d])), max_Y[d] + 1e-3);
		min_Y[d] = min_Y[d] - abs_d(min_Y[d]) * 1e-3;
        //printf("dim: %d max: %f min: %f\n", d, max_Y[d], min_Y[d]);
    }

	weight = _weight;
	
    // Construct QuadTree
    init(NULL, inp_data, min_Y, max_Y);
    fill(N);
}

// Constructor for QuadTree with particular size and parent (do not fill the tree)
QuadTree::QuadTree(QuadTree* inp_parent, double* inp_data, double* min_Y, double* max_Y)
{   
	weight = NULL;
    QT_NO_DIMS = inp_parent->QT_NO_DIMS;
	num_children = 1;
	for (int i = 0; i < QT_NO_DIMS; i++) {
		num_children *= 2;
	}

    init(inp_parent, inp_data, min_Y, max_Y);
}


// Main initialization function
void QuadTree::init(QuadTree* inp_parent, double* inp_data, double* min_Y, double* max_Y)
{   
    // parent = inp_parent;
    data = inp_data;
    is_leaf = true;
    size = 0;
    cum_size = 0;
    
    boundary.mins = min_Y;
    boundary.maxs  = max_Y;
    boundary.n_dims = QT_NO_DIMS;

    index[0] = 0;

    center_of_mass = new double[QT_NO_DIMS];
    for (int i = 0; i < QT_NO_DIMS; i++) {
        center_of_mass[i] = .0;
    }
}


// Destructor for QuadTree
QuadTree::~QuadTree()
{   
    for(unsigned int i = 0; i != children.size(); i++) {
        delete children[i];
    }
    delete[] center_of_mass;
}

void QuadTree::set_weight(double* _weight) {
	weight = _weight;
}
// Insert a point into the QuadTree
bool QuadTree::insert(int new_index)
{
    // Ignore objects which do not belong in this quad tree
    double* point = data + new_index * QT_NO_DIMS;
    if (!boundary.containsPoint(point)) {
        return false;
    }

    // Online update of cumulative size and center-of-mass
    cum_size += weight ? weight[new_index] : 1;
    double mult1 = (double) (cum_size - 1) / (double) cum_size;
    double mult2 = 1.0 / (double) cum_size;
    for (int d = 0; d < QT_NO_DIMS; d++) {
        center_of_mass[d] = center_of_mass[d] * mult1 + mult2 * point[d];
    }

    // If there is space in this quad tree and it is a leaf, add the object here
    if (is_leaf && size < QT_NODE_CAPACITY) {
        index[size] = new_index;
        size++;
        return true;
    }

    // Don't add duplicates for now (this is not very nice)
    bool any_duplicate = false;
    for (int n = 0; n < size; n++) {
        bool duplicate = true;
        for (int d = 0; d < QT_NO_DIMS; d++) {
            if (abs_d(point[d] - data[index[n] * QT_NO_DIMS + d]) > DBL_EPSILON) { duplicate = false; break; }
        }
        any_duplicate = any_duplicate | duplicate;
    }
    if (any_duplicate) {
        return true;
    }
    // Otherwise, we need to subdivide the current cell
    if (is_leaf) {
        subdivide();
    }

    // Find out where the point can be inserted
    for (int i = 0; i < num_children; ++i) {
        if (children[i]->insert(new_index)) {
            return true;
        }
    }
    
    // Otherwise, the point cannot be inserted (this should never happen)
    printf("%s\n", "[t-SNE] No no, this should not happen");
	exit(1);
    return false;
}

int *get_bits(int n, int bitswanted){
  int *bits = new int[bitswanted];

  int k;
  for(k=0; k<bitswanted; k++) {
    int mask =  1 << k;
    int masked_n = n & mask;
    int thebit = masked_n >> k;
    bits[k] = thebit;
  }

  return bits;
}

// Create four children which fully divide this cell into four quads of equal area
void QuadTree::subdivide() {
    // Create children
    double *new_mins = new double[2 * QT_NO_DIMS],
		*new_maxs = new double[2 * QT_NO_DIMS];
    for(int i = 0; i < QT_NO_DIMS; ++i) {
		new_mins[i * 2] = boundary.mins[i];
		new_mins[i * 2 + 1] = (boundary.mins[i] + boundary.maxs[i]) / 2;
		new_maxs[i * 2] = (boundary.mins[i] + boundary.maxs[i]) / 2;
		new_maxs[i * 2 + 1] = boundary.maxs[i];
    }

    for (int i = 0; i < num_children; ++i) {
        int *bits = get_bits(i, QT_NO_DIMS);    

        double* min_Y = new double[QT_NO_DIMS]; 
        double* max_Y = new double[QT_NO_DIMS]; 

        // fill the means and width
        for (int d = 0; d < QT_NO_DIMS; d++) {
            min_Y[d] = new_mins[d * 2 + bits[d]];
            max_Y[d] = new_maxs[d * 2 + bits[d]];
        }
        
        QuadTree* qt = new QuadTree(this, data, min_Y, max_Y);     
		qt->set_weight(weight);   
        children.push_back(qt);
        delete[] bits; 
    }
	delete[] new_mins;
	delete[] new_maxs;

    // Move existing points to correct children
    for (int i = 0; i < size; i++) {
        // bool flag = false;
        for (int j = 0; j < num_children; j++) {
            if (children[j]->insert(index[i])) {
                // flag = true;
                break;
            }
        }
        // if (flag == false) {
        index[i] = -1;
        // }
    }
    
    // This node is not leaf now
    // Empty it
    size = 0;
    is_leaf = false;
}


// Build QuadTree on dataset
void QuadTree::fill(int N)
{
    for (int i = 0; i < N; i++) {
        insert(i);
		//printf("%d\n", i);
    }
}


// Compute summary
int QuadTree::constraint_summarize(double* target, double theta, double* summary, int idx)
{
	// Make sure that we spend no time on empty nodes or self-interactions
	if (cum_size == 0) {
		//printf("size=0 return idx=%d\n", idx);
		return idx;
	}
	// Compute distance between point and center-of-mass
	int idx_d = idx + QT_NO_DIMS;
	summary[idx_d] = .0;
	bool dup = true;

	for (int d = 0; d < QT_NO_DIMS; d++) {
		summary[idx + d] = target[d] - center_of_mass[d];
		summary[idx_d] += summary[idx + d] * summary[idx + d];
		dup = dup && (abs_d(summary[idx + d]) <= DBL_EPSILON);
	}

	if (dup && is_leaf) {
		//printf("dup and leaf return idx=%d\n", idx);
		return idx;
	}
	// Check whether we can use this node as a "summary"
	double m = -1;
	for (int i = 0; i < QT_NO_DIMS; ++i) {
		m = max(m, (boundary.maxs[i] - boundary.mins[i]) * (boundary.maxs[i] - boundary.mins[i]));
	}
	//printf("m=%f, summary[idx_d]=%f, theta*theta=%f\n", m, summary[idx_d], theta*theta);
	if (is_leaf || m / summary[idx_d] < theta * theta) {
		summary[idx_d + 1] = cum_size;
		idx += QT_NO_DIMS + 2;
		//printf("leaf or range return idx=%d\n", idx);
	}
	else {
		// Recursively apply summarize to children
		for (int i = 0; i < num_children; ++i) {
			//printf("go to child %d\n", i);
			idx = children[i]->constraint_summarize(target, theta, summary, idx);
		}
		//printf("finish summarize all children, return idx=%d\n", idx);
	}
	return idx;
}

void QuadTree::compute_non_edge_forces(int point_index, double theta, double* neg_f, double* sum_Q) {

	// Make sure that we spend no time on empty nodes or self-interactions
	if (cum_size == 0 || (is_leaf && size == 1 && index[0] == point_index)) {
		return;
	}
	// Compute distance between point and center-of-mass
	double D = .0;
	int ind = point_index * QT_NO_DIMS;

	for (int d = 0; d < QT_NO_DIMS; d++) {
		double t = data[ind + d] - center_of_mass[d];
		D += t * t;
	}

	// Check whether we can use this node as a "summary"
	double m = -1;
	for (int i = 0; i < QT_NO_DIMS; ++i) {
		m = max(m, (boundary.maxs[i] - boundary.mins[i]) / 2.0);
	}
	if (is_leaf || m / sqrt(D) < theta) {

		// Compute and add t-SNE force between point and current node
		double Q = 1.0 / (1.0 + D);
		*sum_Q += cum_size * Q;
		double mult = cum_size * Q * Q;
		for (int d = 0; d < QT_NO_DIMS; d++) {
			neg_f[d] += mult * (data[ind + d] - center_of_mass[d]);
		}
	}
	else {
		// Recursively apply Barnes-Hut to children
		for (int i = 0; i < num_children; ++i) {
			children[i]->compute_non_edge_forces(point_index, theta, neg_f, sum_Q);
		}
	}
}
void QuadTree::modified_compute_non_edge_forces(double* target, QueryParameter* para, double theta, double* neg_f, double* sum_Q) {

	// Make sure that we spend no time on empty nodes or self-interactions
	if (cum_size == 0) {
		return;
	}
	// Compute distance between point and center-of-mass
	double D = .0;

	for (int d = 0; d < QT_NO_DIMS; d++) {
		double t = target[d] - center_of_mass[d];
		D += t * t;
	}
	if (is_leaf && size == 1 && para->labels[index[0]] == para->label) {
		D = D * para->label_alpha[para->label];
	}

	// Check whether we can use this node as a "summary"
	double m = -1;
	for (int i = 0; i < QT_NO_DIMS; ++i) {
		m = max(m, (boundary.maxs[i] - boundary.mins[i]) / 2.0);
	}
	if (is_leaf || m / sqrt(D) < theta) {

		// Compute and add t-SNE force between point and current node
		double Q = 1.0 / (1.0 + D);
		*sum_Q += cum_size * Q;
		double mult = cum_size * Q * Q;
		for (int d = 0; d < QT_NO_DIMS; d++) {
			neg_f[d] += mult * (target[d] - center_of_mass[d]);
		}
	}
	else {
		// Recursively apply Barnes-Hut to children
		for (int i = 0; i < num_children; ++i) {
			children[i]->modified_compute_non_edge_forces(target, para, theta, neg_f, sum_Q);
		}
	}
}


void QuadTree::constraint_compute_non_edge_forces(double* target, double theta, double* neg_f, double* sum_Q) {

	// Make sure that we spend no time on empty nodes or self-interactions
	if (cum_size == 0) {
		return;
	}
	// Compute distance between point and center-of-mass
	double D = .0;

	for (int d = 0; d < QT_NO_DIMS; d++) {
		double t = target[d] - center_of_mass[d];
		D += t * t;
	}

	// Check whether we can use this node as a "summary"
	double m = -1;
	for (int i = 0; i < QT_NO_DIMS; ++i) {
		m = max(m, (boundary.maxs[i] - boundary.mins[i]) / 2.0);
	}
	if (is_leaf || m / sqrt(D) < theta) {

		// Compute and add t-SNE force between point and current node
		double Q = 1.0 / (1.0 + D);
		*sum_Q += cum_size * Q;
		double mult = cum_size * Q * Q;
		for (int d = 0; d < QT_NO_DIMS; d++) {
			neg_f[d] += mult * (target[d] - center_of_mass[d]);
		}
	}
	else {
		// Recursively apply Barnes-Hut to children
		for (int i = 0; i < num_children; ++i) {
			children[i]->constraint_compute_non_edge_forces(target, theta, neg_f, sum_Q);
		}
	}
}

int QuadTree::summarize(int point_index, double theta, double* summary, int idx)
{
	// Make sure that we spend no time on empty nodes or self-interactions
	if (cum_size == 0) {
		//printf("size=0 return idx=%d\n", idx);
		return idx;
	}
	// Compute distance between point and center-of-mass
	int idx_d = idx + QT_NO_DIMS;
	summary[idx_d] = .0;
	int ind = point_index * QT_NO_DIMS;
	bool dup = true;

	for (int d = 0; d < QT_NO_DIMS; d++) {
		summary[idx + d] = data[ind + d] - center_of_mass[d];
		summary[idx_d] += summary[idx + d] * summary[idx + d];
		dup = dup && (abs_d(summary[idx + d]) <= DBL_EPSILON);
	}

	if (dup && is_leaf) {
		//printf("dup and leaf return idx=%d\n", idx);
		return idx;
	}
	// Check whether we can use this node as a "summary"
	double m = -1;
	for (int i = 0; i < QT_NO_DIMS; ++i) {
		m = max(m, (boundary.maxs[i] - boundary.mins[i]) * (boundary.maxs[i] - boundary.mins[i]));
	}
	//printf("m=%f, summary[idx_d]=%f, theta*theta=%f\n", m, summary[idx_d], theta*theta);
	if (is_leaf || m / summary[idx_d] < theta * theta) {
		summary[idx_d + 1] = cum_size;
		idx += QT_NO_DIMS + 2;
		//printf("leaf or range return idx=%d\n", idx);
	}
	else {
		// Recursively apply summarize to children
		for (int i = 0; i < num_children; ++i) {
			//printf("go to child %d\n", i);
			idx = children[i]->summarize(point_index, theta, summary, idx);
		}
		//printf("finish summarize all children, return idx=%d\n", idx);
	}
	return idx;
}