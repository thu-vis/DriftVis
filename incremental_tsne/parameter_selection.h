/*
 *  parameter_selection.h
 *  Header file for parameter selection function.
 *
 *  Created by Shouxing Xiang.
 */

#pragma once

class ParameterSelection
{
public:
	ParameterSelection(double* _data, int N, int D, int n_neighbor, double build_time_weight, double memory_weight, bool if_sample);
	ParameterSelection(double* _data, int N, int D, int n_neighbor, bool if_sample);
	~ParameterSelection();
	void set_weight(double build_time_weight, double memory_weight);
	void parameter_selection(double accuracy, int& knn_tree, int& forest_size, int& leaf_number);


private:
	double build_time_weight;
	double memory_weight;
	double accuracy;
	double *data;
	int* sample_indexes;
	int N, D, n_neighbor, sample_N, sample_n_neighbor;
	double *true_knn_distance;
	double *true_distances;
	int* true_neighbors;
	double vp_tree_selection(int& leaf_number, double _memory_weight=0.0, double _opt_cost=1.0);
	double kd_tree_selection(int& leaf_number, double _memory_weight=0.0, double _opt_cost=1.0);
	double vp_forest_selection(int& forest_size, int& leaf_number, double _memory_weight=0.0, double _opt_cost=1.0);
	double kd_forest_selection(int& forest_size, int& leaf_number, double _memory_weight=0.0, double _opt_cost=1.0);
	void init_sampled_data();
	void compute_true_knn_distance();
	double compute_neighbor_accuracy(int* neighbors1, double* distances1, int* neighbors2, double* distances2, int N, int D);
};
