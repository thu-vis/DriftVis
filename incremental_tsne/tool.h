/*
*  tool.h
*  Header file for tool.
*  Contain some structures, functions and variables
*
*  Created by Shouxing Xiang.
*/

// An item on the intermediate result queue
#pragma once

class PriorityQueueItem {
public:
	PriorityQueueItem(int index, double dist, bool dir=true, int tree_id=0) :
		index(index), dist(dist), dir(dir), tree_id(tree_id) {}
	int index, tree_id;
	double dist;
	bool dir;
	bool operator<(const PriorityQueueItem& o) const {
		if (dir) {
			return dist < o.dist;
		}
		else {
			return dist > o.dist;
		}
	}
};
