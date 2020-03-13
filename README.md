# MachineLearning-IsingModel2D
MachineLearning to 2D Ising model

### General purpose of the code
Current version commited generates Ising configurations in a square lattice of size to be chosen. The generated files are written into sub-directories named accordingly. The functions for reading the data is also written and the way to use the function is demonstrated.
#### Floating Point Issue
In the following link, we can see the explanation about the problem. This issue is encountered at T~1 region, and we end up with the mean magnetization <m>~1.002 which is an impossible value.
https://docs.python.org/2/tutorial/floatingpoint.html#floating-point-arithmetic-issues-and-limitations
##### Note about the data stored
Currently, it only stores means of energy and magnetization; and the lattice data. 
##### Goals for the next commit
The next commit will include converting the data into images, saving the images and reading the image files.
##### Note about huge-amount of data generation
In the collect_monte_carlo_data(...) function, there is a part that has been commented-out. 
This part is written in case we generate great amounts of data, and therefore in order to make sure that even if something goes wrong (somebody plugging the computer off or etc.), each data is also saved seperately. For the relatively small amount of generating data, this part can stay commented-out. 
