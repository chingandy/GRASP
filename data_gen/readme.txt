Dependencies:
opencv 2.4.11 (https://gist.github.com/dynamicguy/3d1fce8dae65e765f7c4)

compile example:

g++ clearance_data_prime.cpp -L//home/mcw/src/opencv-2.4.11/build/lib -lopencv_core -lopencv_highgui  -lz -lpng -std=c++11


g++ clearance_data_prime.cpp -L//usr/local/Cellar/opencv/3.4.3/lib/ -lopencv_core -lopencv_highgui  -lz -lpng -std=c++11



When its build you can "chmod +x run_8.sh" and then execute it to start 8 parallel instances of the data generation. For fewer or more just remove or add some in the "8.sh" file. 
It uses "parallel" (https://www.gnu.org/software/parallel/) to start the different instances.

After starting that a bunch of text files should appear, (that's the data).