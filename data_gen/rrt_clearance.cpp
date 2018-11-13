#include <math.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <ctime>
#include <iostream>
#include <algorithm>    // std::min_element, std::max_element


//opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

//kd-tree nanoflann
#include "nanoflann.hpp"
#include "utils.h"

using namespace cv;
using namespace std;
using namespace nanoflann;



struct approx_circel { 
    double x;
    double y;
    double r;
};




template <typename num_t>
class RRT_custem 
{
	public:
		
		struct Point
		{
			double  x,y,t;
			int idx;
			int parent_idx;
			std::vector<int> children_idx;
		};
		

		PointCloud_se2<num_t> cloud;
		typedef KDTreeSingleIndexDynamicAdaptor<
		SE2_Adaptor<double, PointCloud_se2<double> > ,
		PointCloud_se2<double>,
		3 /* dim */
		> my_kd_tree_t;

		my_kd_tree_t* kd_tree;//(3 /*dim*/, cloud, KDTreeSingleIndexAdaptorParams(10 /* max leaf */) );;

		int inter_steps=6;
		int  MAX_DIS; //max distance away from sampling
		int MAX_POINTS=3000000;
		int MAX_ITER=1000000;
		int totpoints=0;
		std::vector<std::vector<approx_circel>> obstacles;
		std::vector<approx_circel> object;
		int solution=0;
		double EPSILON;
		int iter=0;
		int IMAGESIZE;
		double MIN_DIS=0.005;

		double mean_x_object;
		double mean_y_object;

		Mat obstical_map;

		

		RRT_custem()
		{
			
			// construct a kd-tree index:
			kd_tree = new my_kd_tree_t(3 /*dim*/, cloud, KDTreeSingleIndexAdaptorParams(10 /* max leaf */) );
			//root
			Point root;
			root.x=0.0;
			root.y=0.0;
			root.t=0.0;
			root.idx=totpoints;
			root.parent_idx=-1;
			generateRandomPointCloud_se2(cloud, MAX_POINTS);			
			
			add_point_to_tree(root);
			srand (time(NULL));

		}



		~RRT_custem(){}

		void add_point_to_tree(Point apoint)
		{
			cloud.pts[totpoints].x=apoint.x;
			cloud.pts[totpoints].y=apoint.y;
			cloud.pts[totpoints].t=apoint.t;
			cloud.pts[totpoints].idx=apoint.idx;
			cloud.pts[totpoints].parent_idx=apoint.parent_idx;
			kd_tree->addPoints(totpoints, totpoints+1);
			totpoints++;
		}

		Point get_point_from_tree(int idx)
		{
			Point tree_point;
			tree_point.x=kd_tree->kdtree_get_pt(idx,0);
			tree_point.y=kd_tree->kdtree_get_pt(idx,1);
			tree_point.t=kd_tree->kdtree_get_pt(idx,2);
			tree_point.idx=kd_tree->kdtree_get_pt(idx,3);
			tree_point.parent_idx=kd_tree->kdtree_get_pt(idx,4);

			return tree_point;
		}

		std::vector<int> go_deeper(std::vector<int> badnodes,Point dnode)
		{
			for(int i=0;i<cloud.pts[dnode.idx].children_idx.size();i++)
				{
					//add children to the badnodes
					Point temp=get_point_from_tree(cloud.pts[dnode.idx].children_idx[i]);
					badnodes = go_deeper(badnodes,temp);
				}
				badnodes.push_back(dnode.idx);

				cloud.pts[dnode.idx].x=-7777;
				cloud.pts[dnode.idx].y=-7777;
				cloud.pts[dnode.idx].t=-7777;

				return badnodes;
		}



		void check_all_the_states()
		{
			std::vector<int> badnodes;
			//go through the structure and follow the children
			
			for(int i=0;i<totpoints;i++)
			{
				Point temp=get_point_from_tree(i);
				if(!valid_node(temp))
				{
					//go kill all the children
					badnodes = go_deeper(badnodes,temp);
				}
					
			}
			//now delete all the nodes start with highes idx
			std::cout << "found: " << badnodes.size() << " nodes to prune" << std::endl;
			
			for (int i=0;i< badnodes.size();i++)
			{
				//earse the children idx in the parent node
				int parentnodeidx=cloud.pts[badnodes[i]].parent_idx;
				for (int j=0;j<cloud.pts[parentnodeidx].children_idx.size();j++)
				{
					if(cloud.pts[parentnodeidx].children_idx[j]==badnodes[i])
					{
						// erase the cild idx element
						cloud.pts[parentnodeidx].children_idx.erase (cloud.pts[parentnodeidx].children_idx.begin()+j);
  						break;
					}
				}

					
				cloud.pts[badnodes[i]].x=-7777;
				cloud.pts[badnodes[i]].y=-7777;
				cloud.pts[badnodes[i]].t=-7777;
				cloud.pts[badnodes[i]].idx=-7777;
				cloud.pts[badnodes[i]].parent_idx=-7777;
				//cloud.pts[badnodes[i]].children_idx.clear();
				
			}
			//adjust totnods count
			std::cout << "Prunded: " << badnodes.size() << " nodes" << std::endl;
			//totpoints=totpoints;//-badnodes.size()-1;
		}

		void calc_means()
		{
			double sum_x=0;
			double sum_y=0;
			for(int i=0;i< object.size();i++)
			{
				sum_x+=object[i].x;
				sum_y+=object[i].y;
			}
			mean_x_object=sum_x/object.size();
			mean_y_object=sum_y/object.size();
		}

		double fRand(double fMin, double fMax)
		{
		    double f = (double)rand() / RAND_MAX;
		    return fMin + f * (fMax - fMin);
		}

		double angle_difference( double angle1, double angle2 )
		{
			double diff;
		    if (angle2 > angle1 && angle2 - angle1 <= M_PI )
				diff=angle2 - angle1;
			else if (angle2 > angle1 && angle2 - angle1 > M_PI )
			  	diff=abs(angle1-angle2);
			else if (angle1 > angle2 && angle1 - angle2 <= M_PI )
			    diff=angle1 - angle2;
			else if (angle1 > angle2 && angle1 - angle2 > M_PI )
				diff=abs(angle2-angle1);
		}

		

		//sample a randem config
		Point sample_rnode(Point base)
		{
			//std::cout << "sample random x  " << fRand(-MAX_DIS/2 ,MAX_DIS/2 ) << std::endl;
			Point rnode;
			rnode.x= base.x + fRand(-(MAX_DIS)/2 ,(MAX_DIS)/2 );
			rnode.y= base.y +fRand(-(MAX_DIS)/2 ,(MAX_DIS)/2 );
			rnode.t= fRand(-M_PI,M_PI);
			return rnode;
		}

	
		std::vector<approx_circel> augment_object(Point anode)
		{
			//augment the object given the coordi
			std::vector<approx_circel> a_object;
			for(int i=0;i<object.size();i++)
			{
				approx_circel a_c_temp=object[i];
				//subtract the mean then rotate it!
				double xo=a_c_temp.x-mean_x_object;
				double yo=a_c_temp.y-mean_y_object;
				a_c_temp.x=cos(anode.t)*xo-sin(anode.t)*yo;
				a_c_temp.y=sin(anode.t)*xo+cos(anode.t)*yo;
				//add mean again and the new offset
				a_c_temp.x+=mean_x_object+anode.x;
				a_c_temp.y+= mean_y_object+anode.y;
				a_object.push_back(a_c_temp);
			}
			return a_object;
		}

		//check if node is valid
		bool valid_node(Point vnode)
		{
			std::vector<approx_circel> a_object=augment_object(vnode);
			for(int i=0;i<a_object.size();i++)
			{
				approx_circel object_c=a_object[i];
				//std::cout << "validX? " << object_c.x << std::endl;
				for(int j=0;j<obstacles.size();j++)
				{
					std::vector<approx_circel> obstacle_temp=obstacles[j];
					for(int k=0;k<obstacle_temp.size();k++)
					{
						approx_circel obstacle_c=obstacle_temp[k];
						obstacle_c.r+=EPSILON;
						if(circle_intersection(object_c,obstacle_c))
						{							
							return false;
						}	
					}
				}
			}
			return true;
		}

	

		bool circle_intersection(approx_circel c1, approx_circel c2)
		{
			//check if inside
			double d= (pow(c1.x-c2.x,2)+pow(c1.y-c2.y,2));
			if(pow(c1.r,2)>d+pow(c2.r,2))
				return true;
			if(pow(c2.r,2)>d+pow(c1.r,2))
				return true;
			//check if c1 and c2 intersect
			if(pow(c1.r-c2.r,2)<=d && d<= pow(c1.r+c2.r,2))
				return true;

			return false;
		}

		bool valide_edge(Point snode, Point nnode)
		{
			//performe inter_steps interpolations
			//calc deltas
			double dx=(nnode.x-snode.x)/inter_steps;			
			double dy=(nnode.y-snode.y)/inter_steps;
			double dt=(nnode.t-snode.t)/inter_steps;
			
			for(int k=0;k<inter_steps;k++)
			{
				Point tnode;
				tnode.x=snode.x+dx*k;
				tnode.y=snode.y+dy*k;
				tnode.t=snode.t+dt*k;

				std::vector<approx_circel> a_object=augment_object(tnode);
		
				for(int i=0;i<a_object.size();i++)
				{
					approx_circel object_c=a_object[i];
					
					for(int j=0;j<obstacles.size();j++)
					{
						std::vector<approx_circel> obstacle_temp=obstacles[j];
						for(int k=0;k<obstacle_temp.size();k++)
						{
							approx_circel obstacle_c=obstacle_temp[k];
							obstacle_c.r+=EPSILON;
							if(circle_intersection(object_c,obstacle_c))
							{
								return false;
							}								
						}
					}
				}
			}			
			return true;
		}

		//check if it is a goal state
		bool is_goal(Point gnode, double escape_r)
		{

			if(pow(escape_r,2)<(pow(gnode.x,2)+pow(gnode.y,2)))
				return true;
		}

		//performe the RRT
		void run()
		{
			std::cout << " starting tree building " << totpoints <<  std::endl;
			//sample a  valid random node
			bool fin=false;
			int idxnnode;
			while(!fin)
			{
				bool fvalid_edge=false;
				Point rnode;
				Point nnode;			
				while(!fvalid_edge)
				{
					
					//get a base node
					bool fvalid_node=false;
					while(!fvalid_node)
					{
						if(iter>MAX_ITER)
							break;
						int u=1;
						bool move_on=false;
						while(!move_on)
						{
							int idxbasenode=totpoints-u;// rand()%totnodes;
							rnode = sample_rnode(get_point_from_tree(idxbasenode)); //TODO: offset calc
							if(get_point_from_tree(idxbasenode).x>-1000)
								move_on=true;
							u++;
						}					
						
						//check if random node is good
						fvalid_node=valid_node(rnode);
						iter++;

					}
					
					if(iter>MAX_ITER)
						break;
					
					// //find nearest node
					// do a knn search
					const size_t num_results = 1;
					size_t ret_index;
					num_t out_dist_sqr;
					nanoflann::KNNResultSet<num_t> resultSet(num_results);
					resultSet.init(&ret_index, &out_dist_sqr );
					double query_pt[3] = { rnode.x, rnode.y, rnode.t };
					kd_tree->findNeighbors(resultSet, query_pt, nanoflann::SearchParams(10));
					idxnnode=ret_index;
					if(idxnnode>=0)
						fvalid_edge=valide_edge(get_point_from_tree(idxnnode),rnode);
				}
				
				if(iter>MAX_ITER)
				{
					fin =true;
					totpoints++;
					break;
				}
				// //add node to tree
				//Relations *trnode= new Relations;
				rnode.idx=totpoints;
				rnode.parent_idx=idxnnode;
				cloud.pts[idxnnode].children_idx.push_back(totpoints);
				
				add_point_to_tree(rnode);
				if(is_goal(rnode,IMAGESIZE))
				{
					fin=true;
					solution=1;
				}				
			}
		}
};
