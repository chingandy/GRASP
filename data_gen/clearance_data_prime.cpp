#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <math.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <ctime>
#include <algorithm>    // std::min_element, std::max_element
#include "boost/random.hpp" 

//opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

//helpers
#include "rrt_clearance.cpp"
#include "shapes.cpp"

using namespace cv;



//check if circle intersect
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
}

//check if collision free
bool collision_free(std::vector<approx_circel> objects, std::vector<std::vector<approx_circel>> obstacles,double EPSILON)
{
	for(int i=0;i<objects.size();i++)
	{
		approx_circel object_c=objects[i];
		for(int j=0;j<obstacles.size();j++)
		{
			std::vector<approx_circel> obstacle_temp=obstacles[j];
			for(int k=0;k<obstacle_temp.size();k++)
			{
				approx_circel obstacle_c=obstacle_temp[k];
				obstacle_c.r+=EPSILON;
				if(circle_intersection(object_c,obstacle_c))
					return false;
			}
		}
	}
	return true;
}




int main(int argc, char* argv[]) {

	// Check the number of parameters
    if (argc < 4) {
        // Tell the user how to run the program
        std::cerr << "Usage: " << argv[0] << " StartIDX" << " EndIDX" << " NumConfigs" << std::endl;
        /* "Usage messages" are a conventional way of telling the user
         * how to run a program if they enter the command incorrectly.
         */
        return 1;
    }
	//HYPERPARAMS
	int STARTIDX=atoi(argv[1]);
	int ENDIDX=atoi(argv[2]);
	int NUMCONFIGS=atoi(argv[3]);
	std::string DATASET="./dataset_" + std::to_string(STARTIDX) + "_" + std::to_string(ENDIDX) + ".txt";	

	int SCALEFACTOR=1;
	int IMGSIZE=64*SCALEFACTOR;
	int GRIDSTEP=1*SCALEFACTOR;	
	int D_OBJ=IMGSIZE/4;
	int NUM_GRIPPERS=4;
	int GRIPPER_D=6;
	int MAX_NUMER_OF_ATTAMPS=5000;
	int EPSILON_MAX=IMGSIZE/2; 
	//double COVERFACTOR=0.9;
	//double MINCIRCLE=2*SCALEFACTOR;
	int MAX_B_SEARCH=6;
	int RRT_MAX_SAMPLES=1000000;

	typedef boost::uniform_int<> NumberDistribution; 
	typedef boost::mt19937 RandomNumberGenerator; 
	typedef boost::variate_generator<RandomNumberGenerator&, 
                                   NumberDistribution> Generator; 
 
	NumberDistribution distribution(-IMGSIZE/2+GRIPPER_D/2,IMGSIZE/2-GRIPPER_D/2); 
	NumberDistribution distribution2(2,10.99);
	NumberDistribution distribution_x_y(8,IMGSIZE-8);
	NumberDistribution distribution_t(0,2*M_PI);
	NumberDistribution distribution_r(2,10);
	NumberDistribution distribution_objectparts(1,4);  
	RandomNumberGenerator generator; 
	Generator numberGenerator(generator, distribution); 
	Generator numberGenerator2(generator, distribution2); 
	Generator numberGenerator_x_y(generator, distribution_x_y); 
	Generator numberGenerator_t(generator, distribution_t); 
	Generator numberGenerator_r(generator, distribution_r); 
	Generator numberGenerator_objectparts(generator, distribution_objectparts); 
	generator.seed(std::time(0)*ENDIDX); // seed with the current time 

	srand (time(NULL)*ENDIDX);	

	for(int s=STARTIDX;s< ENDIDX;s++)
	{
		// build the magicalk super object ...
		//how many 
		std::vector<approx_circel> object_c_vec_simple;

		int num_obj_parts=rint(numberGenerator_objectparts());
		for(int i=0;i< num_obj_parts;i++)
		{
			int xstart;
			int ystart;
			//select random point form vec as startpoint
			bool inside_ws=false;
			while(!inside_ws){
				if(object_c_vec_simple.size()==0)
				{
					xstart=32;//rint(numberGenerator_x_y());
					ystart=32;//rint(numberGenerator_x_y());
				}
				else
				{
					int s_id=rand() % object_c_vec_simple.size();
					xstart=object_c_vec_simple[s_id].x;
					ystart=object_c_vec_simple[s_id].y;
					//std::cout << "xs: " << xstart << " ys: " << ystart << std::endl;
				}
				//check if it is valid point				
				if(xstart<IMGSIZE-8 && xstart>8 && ystart<IMGSIZE-8 && ystart>8)
				{
					inside_ws=true;
					std::cout << "xs: " << xstart << " ys: " << ystart << std::endl;
					//int a=1/0;
				}					
			}			

			std::vector<approx_circel> object_c_vec_temp;
			int num_id=rint(numberGenerator_objectparts());
			switch(num_id) {
	      		case 1 :
	      			// build circle
	      			object_c_vec_temp=c_disk(xstart,ystart,rint(numberGenerator_r()));
			        break;
			    case 2 :
	      			// build circle
	      			object_c_vec_temp=c_line(xstart,ystart,numberGenerator_t());
			        break;
			    case 3 :
	      			// build circle
			    	object_c_vec_temp=c_sqr(xstart,ystart,numberGenerator_t());
	      			
			        break;
			    case 4 :
	      			// build circle
			    	object_c_vec_temp=c_triangle(xstart,ystart,numberGenerator_t());	      			
			        break;
			    case 5 :
	      			// build circle
	      			object_c_vec_temp=c_l_shape(xstart,ystart,numberGenerator_t());
			        break;
			    case 6 :
	      			// build circle
	      			object_c_vec_temp=c_c_shape(xstart,ystart,numberGenerator_t());
			        break;
			    case 7 :
	      			// build circle
	      			object_c_vec_temp=c_hc_shape(xstart,ystart,numberGenerator_t());
			        break;
			    case 8 :
	      			// build circle
	      			object_c_vec_temp=c_u_shape(xstart,ystart,numberGenerator_t());
			        break;
			      
		      default :
		         cout << "Invalid ID" << endl;
	   		}

	   		for(int j=0;j<object_c_vec_temp.size();j++)
	   		{
	   			object_c_vec_simple.push_back(object_c_vec_temp[j]);
	   		}
		}
		
		std::cout << "Object with  " << num_obj_parts << " primals, num circles:" << object_c_vec_simple.size() << std::endl;
		
		//debug
		// for(int h=0;h<object_c_vec_simple.size();h++)
		// {
		// 	std::cout << h << " : " <<  object_c_vec_simple[h].x << " , " <<  object_c_vec_simple[h].y << " , " <<  object_c_vec_simple[h].r << std::endl;
		// }
	
		
		int good_config_counter=0;
		int attamps_count=0;
		int collison_detected=-1;
		while(good_config_counter<NUMCONFIGS && attamps_count< MAX_NUMER_OF_ATTAMPS)
		{
			attamps_count++;

			int num_grippers_want=NUM_GRIPPERS;
			int num_grippers_got=0;
			int exit_count=0;
			std::vector<std::vector<approx_circel>> grip_vec_vec;
			std::cout << "Constructing " << num_grippers_want << " configs" << std::endl;
			bool found_config=false;
			std::vector<std::vector<approx_circel>> gripper_c_vec_final;
			

			//build grippers 
			while(num_grippers_got<num_grippers_want && exit_count< MAX_NUMER_OF_ATTAMPS)
			{
				try
				{	
					//make gripper 
					int dx = numberGenerator();
					int dy = numberGenerator();
					approx_circel gripper_disk;
					gripper_disk.x=dx+IMGSIZE/2;
					gripper_disk.y=dy+IMGSIZE/2;
					gripper_disk.r=GRIPPER_D/2;					
					std::vector<approx_circel> gripper_vec_temp;
					gripper_vec_temp.push_back(gripper_disk);
					std::vector<std::vector<approx_circel>> grip_vec_vec_temp;
					grip_vec_vec_temp.push_back(gripper_vec_temp);
					if(collision_free(object_c_vec_simple,grip_vec_vec_temp,1))
					{
						//check against oher grippers
						if(grip_vec_vec.size()==0)
						{													
							grip_vec_vec.push_back(gripper_vec_temp);
			        		num_grippers_got++;	
						}
						else
						{
							bool selfintersection_grippers=false;
							for(int g=0;g< grip_vec_vec.size();g++)
							{
								std::vector<approx_circel> gripper_vec_temp2=grip_vec_vec[g];
								std::vector<std::vector<approx_circel>> grip_vec_vec_temp2;
								grip_vec_vec_temp2.push_back(gripper_vec_temp2);
								if(!collision_free(gripper_vec_temp,grip_vec_vec_temp2,1))
								{
									selfintersection_grippers=true;
									break;
								}
							}
							if(!selfintersection_grippers)
							{
								grip_vec_vec.push_back(gripper_vec_temp);
				        		num_grippers_got++;
							}
						}
					}			
				}
				catch (const std::exception& e) {
		                std::cout << "CGAL exeption: " << e.what() << std::endl; 
		      	}
				exit_count++;
				//std::cout << num_grippers_got << "/" << num_grippers_want << " exitc: " << exit_count << std::endl;
			}

			//we got it!
			std::cout << " NICE: " << grip_vec_vec.size() << std::endl;
				
			if(grip_vec_vec.size()>0)
			{
				RRT_custem<double> rrt_search;
				int superitercounter=0;
				//set object and obstical
				rrt_search.object=object_c_vec_simple;
				rrt_search.obstacles=grip_vec_vec;
				//set EPSILON
				double next_epsilon=0;
				rrt_search.EPSILON=next_epsilon;
				rrt_search.IMAGESIZE=IMGSIZE;
				rrt_search.MAX_DIS=IMGSIZE/2;
				rrt_search.MAX_ITER=RRT_MAX_SAMPLES;
				//rrt_search.generate_obstical_map();
				rrt_search.calc_means();			

				//start with presearch
				std::cout << "starting rrt" << std::endl;
				rrt_search.solution=0;
				rrt_search.run();
				std::cout << "build a tree with " << rrt_search.totpoints  << " using sampling: " << rrt_search.iter <<  std::endl;
			

//debug visulize
// //*******************************************************************************


// 			Mat image_ultra( IMGSIZE*3, IMGSIZE*3, CV_8UC4,Scalar(255, 255, 255, 0 ) ); 
// 	std::cout << "DRAW IT!!!" << std::endl;
				
// 	//draw obstacels
// 	//augment obsticals and object so thy are in the middel again
// 	std::vector<std::vector<approx_circel>> caging_tools=rrt_search.obstacles;
// 	for(int i=0;i<caging_tools.size();i++)
// 	{
// 		std::vector<approx_circel> caging_tools_drawtemp=caging_tools[i];
// 		std::vector<approx_circel> caging_tools_drawtemp_augmented;
// 		for(int j=0;j<caging_tools_drawtemp.size();j++)
// 		{
// 			approx_circel grip_temp=caging_tools_drawtemp[j];
// 			grip_temp.x=grip_temp.x-IMGSIZE/2+IMGSIZE*3/2;
// 			grip_temp.y=grip_temp.y-IMGSIZE/2+IMGSIZE*3/2;
// 			grip_temp.r+=rrt_search.EPSILON;
// 			approx_circel circle_draw;
			
// 			circle( image_ultra, Point( grip_temp.x, grip_temp.y ), grip_temp.r, Scalar(0, 0, 255, 0 ), CV_FILLED, 8 );//CV_FILLED
			
		
// 		}				
// 	}
// 	for(int i=0;i<caging_tools.size();i++)
// 	{
// 		std::vector<approx_circel> caging_tools_drawtemp=caging_tools[i];
// 		std::vector<approx_circel> caging_tools_drawtemp_augmented;
// 		for(int j=0;j<caging_tools_drawtemp.size();j++)
// 		{
// 			approx_circel grip_temp=caging_tools_drawtemp[j];
// 			grip_temp.x=grip_temp.x-IMGSIZE/2+IMGSIZE*3/2;
// 			grip_temp.y=grip_temp.y-IMGSIZE/2+IMGSIZE*3/2;
// 			approx_circel circle_draw;
			
// 			circle( image_ultra, Point( grip_temp.x, grip_temp.y ), grip_temp.r, Scalar(0, 0, 0, 0 ), CV_FILLED, 8 );//CV_FILLED
			
		
// 		}				
// 	}

// 	// //draw object (circles)
// 	approx_circel circle_draw;
// 	std::vector<approx_circel> aug_obj_drawing;
// 	std::vector<approx_circel> object_c_vec_simple=rrt_search.object;
// 	for(int d=0;d<object_c_vec_simple.size();d++)
// 	{
// 		circle_draw=object_c_vec_simple[d]; 
// 		circle_draw.x+=-IMGSIZE/2+IMGSIZE*3/2;
// 		circle_draw.y+=-IMGSIZE/2+IMGSIZE*3/2;
// 		aug_obj_drawing.push_back(circle_draw);
		
// 		circle( image_ultra, Point( circle_draw.x, circle_draw.y ), circle_draw.r, Scalar(0, 255, 0, 0 ), CV_FILLED, 8 );//CV_FILLED
// 	}
// 	//draw the goal circle
// 	circle( image_ultra, Point( IMGSIZE*3/2, IMGSIZE*3/2), IMGSIZE, Scalar( 0, 0, 0 ), 3, 8 );//CV_FILLED
// 	//image_rrt=draw_circles(image_rrt,object_c_vec_simple);

// 	//draw the rrt nodes
// 	double sum_x=0;
// 	double sum_y=0;
// 	for(int i=0;i< object_c_vec_simple.size();i++)
// 	{
// 		sum_x+=object_c_vec_simple[i].x;
// 		sum_y+=object_c_vec_simple[i].y;
// 	}
// 	double mean_x_object=sum_x/object_c_vec_simple.size();
// 	double mean_y_object=sum_y/object_c_vec_simple.size();
	
// 	for(int o=0;o< rrt_search.totpoints;o++)
// 	{
// 		//show the image
// 		//imshow("Imagerrt",image_rrt);				
// 		//waitKey( 0 );
// 		//draw a center point
// 		int x_temp=rrt_search.get_point_from_tree(o).x;
// 		int y_temp=rrt_search.get_point_from_tree(o).y;
// 		circle( image_ultra, Point( x_temp-IMGSIZE/2 +IMGSIZE*3/2+mean_x_object, mean_y_object + y_temp-IMGSIZE/2 +IMGSIZE*3/2), 1, Scalar( 100, 100, 100,0 ), CV_FILLED, 8 );//CV_FILLED
			
// 		//draw the branches
// 		//check that there is a Parent node
// 		if (rrt_search.cloud.pts[o].parent_idx >-1)
// 		{
// 			double tp1x=rrt_search.get_point_from_tree(rrt_search.cloud.pts[o].idx).x-IMGSIZE/2 +IMGSIZE*3/2+mean_x_object;
// 			double tp1y=mean_y_object + rrt_search.get_point_from_tree(rrt_search.cloud.pts[o].idx).y-IMGSIZE/2 +IMGSIZE*3/2;
// 			double tp2x=rrt_search.get_point_from_tree(rrt_search.cloud.pts[o].parent_idx).x-IMGSIZE/2 +IMGSIZE*3/2+mean_x_object;
// 			double tp2y=mean_y_object + rrt_search.get_point_from_tree(rrt_search.cloud.pts[o].parent_idx).y-IMGSIZE/2 +IMGSIZE*3/2;
// 			line(image_ultra,Point(tp1x,tp1y), Point(tp2x,tp2y),Scalar( 255, 0, 0,0 ), 1, 8);
// 		}

// 	}
// imwrite("./"+ std::to_string(s) + ".jpg",image_ultra);
// imshow("showit",image_ultra);
// waitKey(0);


// //*************************************************************


			//check solution
			if (rrt_search.solution==0)
			{
				//done
				// std::cout << "cagescore= " << (1-(next_epsilon/EPSILON_MAX))<< std::endl;
				// //next_epsilon=EPSILON_MAX;
				// std::cout << "cagescore= " <<  std::endl;
			}
			else
			{
				//we want to keep the tree we grew so far and only change epsilon
				
					//run binary search	
					
					double eps_right=EPSILON_MAX;
					double eps_left=0.0;
					next_epsilon=(eps_right+eps_left)/2;
					bool foundpath=false;
					bool closedpath=false;
					for(int e=0;e<MAX_B_SEARCH;e++)
					{

						//collsision check
						if(collision_free(object_c_vec_simple,grip_vec_vec,next_epsilon))
						{
							//set it up!
							std::cout << "starting rrt with eps: " << next_epsilon << std::endl;
							
							//set EPSILON
							//if the epsilon gets bigger we need to eliminate invalid states
							if(next_epsilon>rrt_search.EPSILON)
							{
								rrt_search.EPSILON=next_epsilon;
								rrt_search.check_all_the_states();
							}


							rrt_search.EPSILON=next_epsilon;
							superitercounter+=rrt_search.iter;
							rrt_search.iter=0; //reset the iter
							rrt_search.MAX_ITER=RRT_MAX_SAMPLES; // with the 4mm we have another mil should be enough
							//rrt_search.totpoints--;
							rrt_search.solution=0;
							std::cout << "starting rrt with "<< rrt_search.totpoints << " already in tree" << std::endl;
					
							rrt_search.run();
							std::cout << "build a tree with " << rrt_search.totpoints << " sampling: " << rrt_search.iter <<  std::endl;
						

							if(rrt_search.solution==1)
							{
								//found a waz out -> make epsilon bigger
								eps_left=next_epsilon;
								next_epsilon=(eps_right+eps_left)/2;
								foundpath=true;
							}
							else
							{
								// no waz out make epsilon smaller
								eps_right=next_epsilon;
								next_epsilon=(eps_right+eps_left)/2;
								closedpath=true;								
							}
						}
						else
						{
							//if collison make epsi smaller
							eps_right=next_epsilon;
							next_epsilon=(eps_right+eps_left)/2;
						}					

					}
					if(foundpath && closedpath)
					{
						collison_detected=0;
					}
					else
					{
						collison_detected=1;
					}
					if(collison_detected==0)
						good_config_counter++;

					std::cout << "Done with Epsilons: " << next_epsilon << " cagescore: " << (1-(next_epsilon/EPSILON_MAX)) <<  std::endl;
				
				}
			
			//std::cout << "basi: " << std::endl;
			//save results and cagescore
			std::stringstream save_object_name;          
            save_object_name  << "O_" << num_obj_parts << "_" << std::setfill('0') << std::setw(5) << s << ".jpg";
            
			std::stringstream save_gripper_name;          
            save_gripper_name  <<  "O_" << num_obj_parts<< "_" << std::setfill('0') << std::setw(5) << s << "_C_" 
            << std::setfill('0') << std::setw(5) << good_config_counter << "_" << attamps_count << ".jpg";
  
			//write to statistic file
			std::ofstream outfile;
			outfile.open(DATASET, std::ios_base::app);
			//Write gripper configs
			outfile << save_object_name.str() << "," << object_c_vec_simple.size() << ",";
            for (int x=0;x< object_c_vec_simple.size();x++)
            {
            	approx_circel obj_draw=object_c_vec_simple[x];
			    outfile << obj_draw.x << ","<< obj_draw.y << "," << obj_draw.r << ",";		        
			    
            }            
		    //gripper
		    outfile << save_gripper_name.str() << "," << grip_vec_vec.size() << ",";
		    //loop over all grippers
		    for(int x=0;x<grip_vec_vec.size();x++)
		    {
		    	std::vector<approx_circel> gripper_vec_temp=grip_vec_vec[x];
		    	for (int k=0;k<gripper_vec_temp.size();k++)
			    {        
			      approx_circel gripper_draw=gripper_vec_temp[k];
			      outfile << gripper_draw.x << ","<< gripper_draw.y << "," << gripper_draw.r << ",";			        
			    }
		    }
		    outfile << ",-," ;               
            
            outfile << save_object_name.str() << "," << save_gripper_name.str() << "," << next_epsilon << "," << (1-(next_epsilon/EPSILON_MAX)) << "," 
            << collison_detected << "," << rrt_search.totpoints << "," << superitercounter << "\n"; 
             
		}

		}

	}

  return 0;
}
