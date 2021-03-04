
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle as pkl
import imageio
import time
import os
import sys

###############################################################################
# For size 10, 100 CORR_SIZE = 20, 1031
CORR_SIZE = 1031
INF_TEMP = 100

###############################################################################
###############################################################################
# CLASS DEFINITION
class IsingLattice:
    
###############################################################################
    # Initializer. Parameter n corresponds to the lattice size. 
    def __init__(self,lattice_size,J,h):
        
        # In order to easily access the parameters: 
        self.lattice_size = lattice_size
        self.num_sites = lattice_size*lattice_size
        self.J = J
        self.h = h
        
        # We randomly initialize the lattice with 0's and 1's
        lattice_state = np.zeros((self.lattice_size,self.lattice_size))
        for i in np.arange(self.lattice_size):
            for j in np.arange(self.lattice_size):
                lattice_state[i][j] = random.getrandbits(1)

        lattice_state = np.where(lattice_state==0, -1, lattice_state)
        
        # We store the configuration 
        self.lattice_state = lattice_state
    
    # THE METHODS 

###############################################################################
    # Plot function. This will help us easily see the lattice configuration.
    def plot_lattice(self, print_info=False): 
        # print_info is Boolean. If it is true then we print info.
        
        plt.figure()
        plt.imshow(self.lattice_state)
        plt.show()
        if print_info:
            self.print_info()

###############################################################################
    # Make the initialized configuration a ferromagnet
    #def make_ferromagnetic(self):
    #    spin_direction = random.getrandbits(1)
    #    if spin_direction == 0:
    #        spin_direction = -1
    #    self.lattice_state = np.full((self.lattice_size, self.lattice_size),spin_direction)

###############################################################################
    # Now we define print_info() method. 
    # It will print all the information about the lattice.
    def print_info(self):
        
        print("Lattice size: ", self.lattice_size , \
            "x", self.lattice_size, ". J: ", self.J, " h: ", self.h )
    
###############################################################################
    # A spin flipper at site (i,j) method
    def flip_spin(self,i,j):
        self.lattice_state[i,j] *= -1
        
###############################################################################
    # Calculating energy of one spin at site (i,j)
    def spin_energy(self,i,j):
        
        # Spin at (i,j)
        spin_ij = self.lattice_state[i,j]
        
        # Now we need to deal with the boundary spins. 
        # We apply periodic boundary conditions.  
        sum_neighbouring_spins = \
            self.lattice_state[(i+1)%self.lattice_size, j] + \
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
    
###############################################################################
    # Calculating Total Lattice Energy
    def energy(self):
        
        # Initialize energy as 0.
        E = 0.0
        
        # We iterate through the lattice
        for i in np.arange(self.lattice_size):
            for j in np.arange(self.lattice_size):
                E = E + self.spin_energy(i,j)
                
        # But we counted neighbours twice here.
        #  So we need to correctly return. 
        # We divide by two 
        E = E / (2.0) / self.num_sites
        if self.h==0:
            return E
        else: 
            # We add the magnetic field term |IS THERE A 1/2 FACTOR HERE?
            E = (E - self.h * np.sum(self.lattice_state)) / self.num_sites
            return E
    
###############################################################################
    # Net magnetization
    def magnetization(self):
        return  np.sum(self.lattice_state)/ (self.num_sites)
        
###############################################################################
    # Correlation Function 
    def correlation_function(self, plot=False):
        
        counter = 0
        
        correlation_function = np.zeros(self.num_sites**2)
        r_sq = np.zeros(self.num_sites**2).astype(int)
        
        for i in np.arange(self.lattice_size):
            for j in np.arange(self.lattice_size):
                for k in np.arange(i,self.lattice_size):
                    check_var = 0
                    if i==k:
                        check_var = j
                    for l in np.arange(check_var,self.lattice_size):
                        x_distance = abs(j-l)
                        y_distance = abs(i-k)
                        
                        
                        if x_distance > self.lattice_size/2:
                            x_distance = abs(self.lattice_size - x_distance)
                        if y_distance > self.lattice_size/2:
                            y_distance = abs(self.lattice_size - y_distance) 
                            
                        distance = x_distance**2 + y_distance**2
                        
                        r_sq[counter] = distance
                        correlation_function[counter] = self.lattice_state[i,j] * self.lattice_state[k,l]
                        
                        counter += 1
        
        corr = correlation_function[0:counter]
        dist = r_sq[0:counter]

        sort_ind = np.argsort(dist)
        sorted_d = np.sort(dist)
        sorted_c = corr[sort_ind]
        
        unique_d, unique_indices_d = np.unique(sorted_d, return_index=True)
        averaged_c = np.zeros(unique_d.size)


        for i in np.arange(averaged_c.size-1):
            denom = unique_indices_d[i+1]-unique_indices_d[i]
            averaged_c[i] = np.sum(sorted_c[unique_indices_d[i]:unique_indices_d[i+1]])/denom

        denom = unique_indices_d[unique_indices_d.size-1]-unique_indices_d[unique_indices_d.size-2]
        averaged_c[averaged_c.size-1] = np.sum(sorted_c[unique_indices_d[unique_indices_d.size-2]:unique_indices_d[unique_indices_d.size-1]]) / denom
        
        if plot:
            plt.plot(np.sqrt(unique_d),averaged_c)

        return averaged_c - self.magnetization()**2, np.sqrt(unique_d)
      
###############################################################################
###############################################################################
# END OF CLASS

###############################################################################

# DATA CLEANING

def get_index(data):
    for i in np.arange(data.size):
        if data[i] < 10**(-8):
            index = i
            break
    return index

def data2cut(data,index):
    return data[0:index]

def distance2cut(distance,index):
    return distance[0:index]

###############################################################################
# CORRELATION LENGTH CALCULATION
def get_cor_len(cor_func, dist, index):
    cor_func_cut = data2cut(cor_func,index)
    distance_cut = distance2cut(dist,index)

    correlation_negative_check = np.sum(cor_func_cut* \
                (distance_cut**2))
    correlation_zero_check = np.sum(6*cor_func_cut)
    if correlation_zero_check == 0:
        correlation_length = 0
    elif correlation_negative_check/correlation_zero_check<0:
        correlation_length = 0
    else:
        correlation_length = np.sqrt(correlation_negative_check/correlation_zero_check)
    return correlation_length

###############################################################################

# Boltzmann constant is fixed to 1.
def scan_lattice(ising_lattice, temperature):

    for k in np.arange(ising_lattice.num_sites):
        
        # POWER OF 2 CASE
        # We choose a random site
        #lattice_size_power = int(np.log2(ising_lattice.lattice_size))
        #i = random.getrandbits(lattice_size_power)
        #j = random.getrandbits(lattice_size_power)

        # RANDOM INT CASE        
        # We choose a random site  
        i = int(ising_lattice.lattice_size * random.random())      
        j = int(ising_lattice.lattice_size * random.random())  

        # We calculate the energy difference if we flip
        energy_initial = ising_lattice.spin_energy(i,j)
        ising_lattice.flip_spin(i,j)
        energy_final = ising_lattice.spin_energy(i,j)
        energy_change = energy_final - energy_initial
        # For convenience we flip it back to the original
        ising_lattice.flip_spin(i,j)
        
        # Then we should flip the spin
        if temperature != 0:
            if energy_change<=0 or \
                random.uniform(0, 1) <= np.exp(-energy_change/temperature):
                # If the Metropolis Criteria holds, swap. 
                ising_lattice.flip_spin(i,j)

###############################################################################

###############################################################################
def monte_carlo_simulation(ising_lattice,\
                           temperature, num_scans,\
                           frequency_sweeps_to_collect_magnetization,\
                           plot_result = False,print_info=False):

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
    TOTAL_NUM_RECORDS = \
        int(num_scans/frequency_sweeps_to_collect_magnetization)+1
    energy_records = np.zeros(TOTAL_NUM_RECORDS)
    magnetization_records = np.zeros(TOTAL_NUM_RECORDS)
    correlation_function_records = np.zeros([TOTAL_NUM_RECORDS,CORR_SIZE])
    correlation_length_records = np.zeros(TOTAL_NUM_RECORDS)
    increment_records = 0
    
    # We will return this n-dimensional 
    lattice_configs = np.zeros((TOTAL_NUM_RECORDS,\
                               ising_lattice.lattice_size,\
                               ising_lattice.lattice_size))

    for k in np.arange(num_scans+frequency_sweeps_to_collect_magnetization):
        scan_lattice(ising_lattice, temperature)
        if k%frequency_sweeps_to_collect_magnetization==0:
            energy_records[increment_records] = ising_lattice.energy()
            magnetization_records[increment_records] = \
                ising_lattice.magnetization()
            lattice_configs[increment_records] = ising_lattice.lattice_state

            correlations , distances = ising_lattice.correlation_function(True)
            correlation_function_records[increment_records] = \
                correlations
            
            # Correlation Length
            index = get_index(correlations)
            correlation_length_records[increment_records] = \
                get_cor_len(correlations,distances,index)
            
            increment_records += 1
            # LOG feature
            print(increment_records ," / ", TOTAL_NUM_RECORDS, " samples saved.")
    

    # Now we can get the <E> and <m>
    print("For temperature= ", temperature, "MC simulation is executed in: ", \
        " %s seconds " % round(time.time() - start_time,2))
    
    if plot_result:
        ising_lattice.plot_lattice()
    
    return lattice_configs, energy_records, magnetization_records, \
        correlation_function_records, correlation_length_records, distances



###############################################################################
def dir_name(lattice_size,J,h,temperature):
    return f'SQ_L_{lattice_size}_J_{J:.2f}_h_{h:.2f}_T_{temperature:.2f}'

###############################################################################
def file_name(lattice_size,J,h,temperature,seed):
    return f'SQ_L_{lattice_size}_J_{J:.2f}_h_{h:.2f}_T_{temperature:.2f}_s_{seed}'

###############################################################################
def write_to_sub_directory(quantity, dir_name,file_name):
    
    if not(os.path.exists(dir_name)):
        os.mkdir(dir_name)
    os.chdir(dir_name)
    
    # Now save with pickle
    open_file = open(file_name,"wb")
    pkl.dump(quantity, open_file)
    open_file.close()
    
    # We go up into the original directory
    os.chdir('..')

###############################################################################
def write_txt_files(quantity, dir_name, file_name):
    # FIRST ENERGY THEN MAGNETIZATION
    if not(os.path.exists(dir_name)):
        os.mkdir(dir_name)
    os.chdir(dir_name)

    np.savetxt(file_name, quantity, fmt='%1.3f')
    # We go up into the original directory
    os.chdir('..')

###############################################################################
def save_image_to_sub_directory(data, directory_name, file_name):
    
    # We check if it exists, if not we make directory
    if not(os.path.exists(directory_name)):
        os.mkdir(directory_name)
    os.chdir(directory_name)
    
    # Now save image
    file_name_img = file_name + ".png"
    imageio.imwrite(file_name_img, data)
    
    # We go up into the original directory
    os.chdir('..')
###############################################################################
def thermalize(ising_lattice, num_scans,from_T, to_T):
     # We thermalize for the  T value at hand
    print("Equilibrating to T = ", "%.2f" % round(to_T, 2), "starting from T = ", "%.2f" % round(from_T, 2))
    for k in np.linspace(from_T,to_T,num=num_scans):
        scan_lattice(ising_lattice,k)
    # LOG feature
    print("Reached T=", "%.2f" % round(to_T, 2), "Beginning to collect data.")

###############################################################################
def collect_monte_carlo_data(lattice_size,J,h, \
                             num_scans, temperature, thermalization_scans,\
                             frequency_sweeps_to_collect_magnetization):
    random.seed(SEED)
    print("Lattice size: ", lattice_size ,\
          "x", lattice_size, ", J= ", J, ", h= ", h, ", SEED=", SEED, "\n")
    temperature = np.append(INF_TEMP, temperature)
    TEMPERATURE_SCALE = 1000
    # Let's scale it up
    # T array is going to be then
    temperature_scaled = temperature*TEMPERATURE_SCALE
    
    if np.where(temperature==0)[0].size != 0:
        raise ValueError("ValueError exception thrown. Monte-Carlo does not \
            work properly at T=0.")
    elif np.where(temperature<0)[0].size != 0:
        raise ValueError("ValueError exception thrown. T cannot be a \
            negative value.")
    ###############################################################################
    # One time generate a new random initial lattice configuration
    ising_lattice = IsingLattice(lattice_size, J,h)

    ###############################################################################
    # Number of temperatures to be calculated
    # since we take one sample for each T
    NUM_TEMPS = temperature.size
    
    # We run through T's 
    for i in np.arange(NUM_TEMPS-1):
        file_name_lattice = file_name(lattice_size,J,h,temperature[i+1],SEED)
        dir_name_data = dir_name(lattice_size,J,h,temperature[i+1])
        
        TOTAL_NUM_CONFIGURATIONS = \
        int(num_scans/frequency_sweeps_to_collect_magnetization)+1
        file_exists = np.zeros(TOTAL_NUM_CONFIGURATIONS, dtype=bool)

        for configs in np.arange(TOTAL_NUM_CONFIGURATIONS):
            file_name_existence = file_name_lattice + "_n_"+f"%d"% \
                            (configs*frequency_sweeps_to_collect_magnetization)
            if os.path.isfile(dir_name_data + "/" +file_name_existence + ".pkl"):
                file_exists[configs] = 1  
        if os.path.exists(dir_name_data) and not(np.all(file_exists)):
            print((np.argwhere(file_exists==False)[0][0]),\
                "Previous configurations for SEED = ", SEED, " with L = ",lattice_size," T = " \
                ,"%.2f" % round(temperature[i+1], 2)," J = ",J," h = ",h, "\n")

        if np.all(file_exists):
            print("ALL requested configurations for SEED = ", SEED, " with L = ",lattice_size," T = " \
                ,"%.2f" % round(temperature[i+1], 2)," J = ",J," h = ",h, " already exist! \n")
            continue
        
        # We thermalize for the  T value at hand
        thermalize(ising_lattice,thermalization_scans, temperature[i], temperature[i+1])

        ###############################################################################
        print("START - MC simulation ", i+1, "/", NUM_TEMPS-1, ", T = ", "%.2f" % round(temperature[i+1], 2))

        ###############################################################################
        # Now we go through with the Monte-Carlo Simulation
        lattice_configs, energy_records, magnetization_records, correlation_function_records, correlation_length_records, distances = \
            monte_carlo_simulation(ising_lattice,\
                                   temperature[i+1],\
                                   num_scans,\
                                   frequency_sweeps_to_collect_magnetization)

        ###############################################################################
        # We write these down to a file
        # We create a dictionary with the following key-value pairs
        print("START - MC simulation ", i+1, "/", NUM_TEMPS-1, ", T = ", "%.2f" % round(temperature[i+1], 2))

        for img in np.arange(TOTAL_NUM_CONFIGURATIONS):
            file_name_img = file_name_lattice+"_n_"+f"%d"% \
                            (img*frequency_sweeps_to_collect_magnetization)
            file_name_txt = file_name_img + ".txt"
            
            data_sample = {'configuration' : lattice_configs[img],
                       'energy' : energy_records[img],
                       'magnetization' : magnetization_records[img],
                       'correlation_length' : correlation_length_records[img],
                       'correlation_function' : correlation_function_records[img],
                       'distances' : distances}
            
            txt_data = np.array([data_sample['energy'],\
                 data_sample['magnetization'], data_sample['correlation_length']])
            correlation_function_txt_data = np.array(data_sample['correlation_function'])
            file_name_pkl = file_name_img + ".pkl"
            file_name_correlation_function = file_name_img + "_correlation_function.txt"
            file_name_distances = file_name_img + "_distances.txt"
            
            write_to_sub_directory(data_sample,dir_name_data,file_name_pkl)
            save_image_to_sub_directory(lattice_configs[img].astype(np.uint8),\
                                        dir_name_data, file_name_img)
            write_txt_files(txt_data, dir_name_data,\
                file_name_txt)
            write_txt_files(correlation_function_txt_data, dir_name_data, file_name_correlation_function)
            write_txt_files(distances, dir_name_data, file_name_distances)

        print("END --- MC simulation ", i+1, "/", NUM_TEMPS-1, ", T = ", "%.2f" % round(temperature[i+1], 2), "\n")


###############################################################################
##BELOW THIS PART IS TO BE CHANGED ACCORDING TO THE DATA WE NEED TO GENERATE ##

# HERE IS AN EXAMPLE ON HOW TO USE THE FUNCTION
# IT GENERATES FOR ONE SEED

# Lattice size, J, h are physical parameters. 
# T_init=T_final is allowed and gets only one temperature data.
# T_increment CANNOT BE ZERO
# num_scans is the number of sweeps we do AFTER thermalization
# num_scans_4_equilibrium is the number of sweeps TO thermalize the system
# frequency_sweeps_to_collect_magnetization
#  is the frequency of saving the configurations.
# e.g. save each 50th configuration after the thermalization
# SEED is a global variable for convenience and better control
# For one (seed,temperature) tuple
# - with the number of sweeps and frequency left unchanged -
# We end up with 21 different configurations saved as .pkl files. 

if ( len(sys.argv) == 4 ):
    SEED = int(sys.argv[1])
    lattice_size = int(sys.argv[2])
    number_configs = int(sys.argv[3])
    temperature = np.array([10,7,4,3.9,3.8,3.7,3.6,3.5,3.4,3.3,3.2,3.1,3.0,2.9,2.8,2.7,2.6,2.5,2.4,2.3,2.2,2.1,2.0,1.9,1.8,1.7,1.6,1.5,1.0,0.5])
    sweep_steps = 1000

    collect_monte_carlo_data(lattice_size = lattice_size ,
                             J = 1.0 , 
                             h = 0.0 ,
                             num_scans = sweep_steps * (number_configs-1),
                             temperature = temperature ,
                             thermalization_scans = 1000 ,
                             frequency_sweeps_to_collect_magnetization = sweep_steps)

else:
    print ('Number of arguments:', len(sys.argv), 'arguments is less than expected (3) --- ABORTING!')
    print ('Usage: python '+sys.argv[0],' seed lattice_size number_of_configurations')
    #print ('Argument List:', str(sys.argv))


