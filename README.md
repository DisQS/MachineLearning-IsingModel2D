# MachineLearning-IsingModel2D
MachineLearning to 2D Ising model

### General purpose of the code
Current version commited generates Ising configurations in a square lattice of size to be chosen. The generated files are written into sub-directories named accordingly. The functions for reading the data is also written and the way to use the function is demonstrated.
##### Note about the data stored
Currently, it stores the configuration as a .png file and a .pkl file in which there is a dictionary. The dictionary has the key-value pairs of configuration, mean magnetization and mean energy. 
##### Goals for the next commit
Lose the directory for Ising Data, write and read directly in the current directory. Read data to return two numpy arrays: configurations and labels; so that we are ready to start with machine learning.
