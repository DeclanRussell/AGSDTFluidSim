# AGSDTFluidSim
A fluid simulation for my advanced graphics software development techniques unit. 
This is an SPH Fluid simulation that uses dynamic parallelism and fancy shading techniques.

Instalation and running:

For this simulation to run your GPU will need the to have compatibility with cuda and dynamic
Parallelism. This will require an NVidia chip of compute caperbility 3.5 or higher!
It should work on both mac and linux but I have not tested for mac so don't hold me to this!

Step 1: You will need to tweak your gencode caperbility in the .pro file to your
	hardware configuration.
Step 2: (optional) Depending on what version on cuda you are running you may have to change
	the path to your cuda library in the .pro aswell. Im running 6.5 so if you are the
	same you should be fine.
Step 3: In the AGSDTFluidSim directory run,
	
	qmake
	make clean
	make

step 4: You should now have a working executable. Please contact me if you cant get it
	working. 


Documentation and Report:

The documation and report of this project are in the form of a doxygen file. 
To access this you can either run the program and open it with the "Open Documenation"
button or open index.html which you will find in doc/html/ 
 

Enjoy! 

Dec
