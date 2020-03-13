# MachineLearning-IsingModel2D
MachineLearning to 2D Ising model

### General purpose of the code
Current version commited generates Ising configurations in a square lattice of size to be chosen. The generated files are written into sub-directories named accordingly. The functions for reading the data is also written and the way to use the function is demonstrated.
#### Floating Point Issue
In the following link, we can see the explanation about the problem. This issue is encountered at T~1 region, and we end up with the < m >~1.002 which is an impossible value.
https://docs.python.org/2/tutorial/floatingpoint.html#floating-point-arithmetic-issues-and-limitations
##### Note about the data stored
Currently, it stores the configuration as a .png file and a .pkl file in which there is a dictionary. The dictionary has the key-value pairs of configuration, mean magnetization and mean energy. 
##### Goals for the next commit
The next commit will be one-line reading the data function.

