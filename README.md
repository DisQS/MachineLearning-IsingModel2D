# MachineLearning-IsingModel2D
MachineLearning to 2D Ising model

### General purpose of the code
Current version commited generates Ising configurations in a square lattice of size to be chosen. The generated files are written into sub-directories named accordingly. The functions for reading the data is also written and the way to use the function is demonstrated.
##### Note about the data stored
Currently, it stores the configuration as a .png file and a .pkl file in which there is a dictionary. The dictionary has the key-value pairs of configuration, mean magnetization and mean energy. 
##### Goals for the next commit
Thermalization will be distinguished and will be done once. For one T, we will get more data in one run by taking data after each autocorrelation step. (Around double the autocorrelation time). The naming files will be changed accordingly. Lose the directory for Ising Data, write and read directly in the current directory. Make sure T_init = T_final works for the current data collection functions. 
