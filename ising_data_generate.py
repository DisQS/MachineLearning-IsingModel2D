import numpy as np
import matplotlib.pyplot as plt
import random
import pickle as pkl
import imageio
import time
import os

# ----THAT SHOULD NOT BE CHANGED----

MAIN_DIRECTORY = "ising_data"


# This is the scale we are using in order to prevent floating 
# point errors in temperature values. Below is a default value

TEMPERATURE_SCALE = 1000


class IsingLattice:
    
    # Initializer. Parameter n corresponds to the lattice size. 
    
    def __init__(self,lattice_size,J,h):
        
        # In order to easily access the parameters: 
        self.lattice_size = lattice_size
        self.num_sites = lattice_size*lattice_size
        self.J = J
        self.h = h
        
        # We randomly initialize the lattice with 0's and 1's
        lattice_state = np.random.choice([1,-1],size=(self.lattice_size,self.lattice_size))
        
        # We store the configuration 
        self.lattice_state = lattice_state
    
    # The Methods 
    # Plot function. This will help us easily see the lattice configuration.
    
    def plot_lattice(self, print_info=False): # print_info is Boolean. If it is true then we print info.
        
        plt.figure()
        plt.imshow(self.lattice_state)
        plt.show()
        if print_info:
            self.print_info()
    
    # Now we define print_info() method. It will print all the information about the lattice.
    
    def print_info(self):
        
        print("Lattice size: ", self.lattice_size , "x", self.lattice_size, ". J: ", self.J, " h: ", self.h )
    
    # A spin flipper at site (i,j) method
    
    def flip_spin(self,i,j):
        self.lattice_state[i,j] *= -1
        
    # Calculating energy of one spin at site (i,j)
        
    def spin_energy(self,i,j):
        
        # Spin at (i,j)
        spin_ij = self.lattice_state[i,j]
        
        # Now we need to deal with the boundary spins. 
        # We apply periodic boundary conditions.  
        sum_neighbouring_spins = self.lattice_state[(i+1)%self.lattice_size, j] + \
                                 self.lattice_state[i, (j+1)%self.lattice_size] + \
                                 self.lattice_state[(i-1)%self.lattice_size, j] + \
                                 self.lattice_state[i, (j-1)%self.lattice_size]
        
        # We calculate the energy terms for site 
        interaction_term = (- self.J * spin_ij * sum_neighbouring_spins)
        
        # This part is added so that in case 
        # there is no external magnetic field, i.e. h = 0
        # then we do not need the computer to do the computation
        # for the magnetic term. 
        if self.h == 0:
            return interaction_term
        else:
            magnetic_field_term = - (self.h * spin_ij)
            return magnetic_field_term + interaction_term
    
    # Calculating Total Lattice Energy
    
    def energy(self):
        
        # Initialize energy as 0.
        E = 0.0
        
        # We iterate through the lattice
        for i in np.arange(self.lattice_size):
            for j in np.arange(self.lattice_size):
                E = E + self.spin_energy(i,j)
                
        # But we counted neighbours twice here. So we need to correctly return. 
        # We divide by two 
        E = E / (2.0) / self.num_sites
        if self.h==0:
            return E
        else: 
            # We add the magnetic field term |IS THERE A 1/2 FACTOR HERE?|
            E = (E - self.h * np.sum(self.lattice_state)) / self.num_sites
            return E
    
    # Net magnetization
    
    def magnetization(self):
        return  np.abs(np.sum(self.lattice_state))/ (self.num_sites)
        

# Boltzmann constant is fixed to 1.
def scan_lattice(ising_lattice, T):
    
    for k in np.arange(ising_lattice.num_sites):
        # We choose a random site
        lattice_size_power = int(np.log2(ising_lattice.lattice_size))
        i = random.getrandbits(lattice_size_power)
        j = random.getrandbits(lattice_size_power)

        
        # We calculate the energy difference if we flip
        E_initial = ising_lattice.spin_energy(i,j)
        ising_lattice.flip_spin(i,j)
        E_final = ising_lattice.spin_energy(i,j)
        delta_E = E_final - E_initial
        
        # For convenience we flip it back to the original
        ising_lattice.flip_spin(i,j)
        
        # Then we should flip the spin
        
        if delta_E<0 or np.random.rand()<np.exp(-delta_E/T):
            # If the Metropolis Criteria holds, swap. 
            ising_lattice.flip_spin(i,j)
    
def monte_carlo_simulation(ising_lattice, T, num_scans, num_scans_4_equilibrium, frequency_sweeps_to_collect_magnetization, plot_result = False,print_info=False):

    start_time = time.time()
    
    # The first three arguments are self-explanatory. 
    # The last one is the number of scans we need to do
    # Before we reach equilibrium. Therefore we do not
    # need to collect data at these steps. 
    if print_info:
        ising_lattice.print_info()
    
    # We start by collecting <E> and <m> data. In order to 
    # calculate these, we record energy and magnetization 
    # after we reach equilibrium.
    
    # The total number of records, both first and last point included
    TOTAL_NUM_RECORDS = int(num_scans/frequency_sweeps_to_collect_magnetization)+1
    energy_records = np.zeros(TOTAL_NUM_RECORDS)
    magnetization_records = np.zeros(TOTAL_NUM_RECORDS)
    increment_records = 0
    
    # We will return this n-dimensional 
    lattice_configs = np.zeros((TOTAL_NUM_RECORDS,\
                               ising_lattice.lattice_size,\
                               ising_lattice.lattice_size))
    
    for equ in np.arange(num_scans_4_equilibrium):
        scan_lattice(ising_lattice,T)

    for k in np.arange(num_scans):
        scan_lattice(ising_lattice, T)
        if k%frequency_sweeps_to_collect_magnetization==0:
            energy_records[increment_records] = ising_lattice.energy()
            magnetization_records[increment_records] = ising_lattice.magnetization()
            lattice_configs[increment_records] = ising_lattice.lattice_state
            increment_records += 1
    
    # Now we can get the <E> and <m>
    
  
    
    
    print("For T = ", T, "Simulation is executed in: ", " %s seconds " % round(time.time() - start_time,2))
    
    if plot_result:
        ising_lattice.plot_lattice()
    
    
    return lattice_configs, energy_records, magnetization_records


def file_name(lattice_size,J,h,T,seed):
    return f'SQ_L_{lattice_size}_J_{J:.2f}_h_{h:.2f}_T_{T}_s_{seed}'


def write_to_sub_directory(quantity, file_name):
    
    # Let us check if the path to data exists
    if not(os.path.exists(MAIN_DIRECTORY)):
        os.mkdir(MAIN_DIRECTORY)        
    
    # We change directories accordingly
    os.chdir(MAIN_DIRECTORY)
    if not(os.path.exists(file_name)):
        os.mkdir(file_name)
    os.chdir(file_name)
    
    # Now save with pickle
    file_name_pkl = file_name + ".pkl"
    open_file = open(file_name_pkl,"wb")
    pkl.dump(quantity, open_file)
    open_file.close()
    
    # We go up into the original directory
    os.chdir('..')
    os.chdir('..')

def save_image_to_sub_directory(data, directory_name, file_name):
    
    # Let us check if the path to data exists
    if not(os.path.exists(MAIN_DIRECTORY)):
        os.mkdir(MAIN_DIRECTORY)
    
    # We change directories accordingly
    os.chdir(MAIN_DIRECTORY)
    
    # We check if it exists, if not we make directory
    if not(os.path.exists(directory_name)):
        os.mkdir(directory_name)
    os.chdir(directory_name)
    
    # Now save image
    file_name_img = file_name + ".png"
    imageio.imwrite(file_name_img, data)
    
    # We go up into the original directory
    os.chdir('..')
    os.chdir('..')

def collect_monte_carlo_data(lattice_size,J,h, T_init, T_final, T_increment, num_scans, num_scans_4_equilibrium, frequency_sweeps_to_collect_magnetization):

    print("Lattice size: ", lattice_size , "x", lattice_size, ". J: ", J, " h: ", h , "\n")
    
    # Let's scale it up
    # T array is going to be then
    T = np.arange(T_init*TEMPERATURE_SCALE, (T_final+T_increment)*TEMPERATURE_SCALE, T_increment*TEMPERATURE_SCALE).astype(int)
    
    # Number of samples are calculated
    # since we take one sample for each T
    NUM_SAMPLES = T.size
    
    # We run through T's 
    for i in np.arange(NUM_SAMPLES):
        file_name_lattice = file_name(lattice_size,J,h,T[i],SEED)
        path = MAIN_DIRECTORY + "/" +file_name_lattice
        scale_down_T = T[i]/TEMPERATURE_SCALE
        if os.path.exists(path):
            print("Simulation ", i+1, "/", NUM_SAMPLES, ": ", "T = ", scale_down_T, " Already exists!")
            continue
        print("Simulation ", i+1, "/", NUM_SAMPLES, ": ")
        
        # Each time generate a new random initial lattice configuration
        ising_lattice = IsingLattice(lattice_size, J,h)
        
        # Now we go through with the Monte-Carlo Simulation
        lattice_configs, energy_records, magnetization_records = monte_carlo_simulation(ising_lattice,scale_down_T,num_scans,num_scans_4_equilibrium,frequency_sweeps_to_collect_magnetization)
                
        # We write these down to a file
        # We create a dictionary with the following key-value pairs
        data_sample = {'lattice_configuration' : lattice_configs,
                       'energy' : energy_records,
                       'magnetization' : magnetization_records,
        } 
        write_to_sub_directory(data_sample,file_name_lattice)
        for img in np.arange(lattice_configs.shape[0]):
            save_image_to_sub_directory(lattice_configs[img].astype(np.uint8), file_name_lattice, file_name_lattice+"_n_"+f"%d"% (img*frequency_sweeps_to_collect_magnetization))
        




















##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
################## BELOW THIS PART IS TO BE CHANGED ACCORDING TO THE DATA WE NEED TO GENERATE ################

# HERE IS AN EXAMPLE ON HOW TO USE THE FUNCTION
# IT GENERATES FOR ONE SEED

# Lattice size, J, h are physical parameters. T_init=T_final is allowed and gets only one temperature data.
# T_increment CANNOT BE ZERO
# num_scans is the number of sweeps we do AFTER thermalization
# num_scans_4_equilibrium is the number of sweeps TO thermalize the system
# frequency_sweeps_to_collect_magnetization is the frequency of saving the configurations.
# e.g. save each 50th configuration after the thermalization
# SEED is a global variable for convenience and better control
# For one (seed,temperature) tuple with the number of sweeps and frequency left unchanged
# We end up with 21 different configurations saved as .pkl files. 

SEED = 2
collect_monte_carlo_data(lattice_size = 16 ,
                         J = 1.0 , 
                         h = 0.0 ,
                         T_init = 2.26 ,
                         T_final = 2.35 ,
                         T_increment = 0.01 ,
                         num_scans = 1000 ,
                         num_scans_4_equilibrium = 1000 ,
                         frequency_sweeps_to_collect_magnetization = 50)

