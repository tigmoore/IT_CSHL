###############################
'''
This script converts a .mat file to a dictionary that is a .pkl file (that is much smaller and only contains 
the necessary information to analyze spikes)
The dictionary can then be loaded into other scripts for analysis
'''
###############################

import mat73
import numpy as np
import sys
import pickle

# Load the .mat file
file_name = str(sys.argv[1])
file_to_name = sys.argv[1].split('_')[1] + '_' + sys.argv[1].split('_')[2] + '_' + sys.argv[1].split('_')[3].split('.')[0]
mat_contents = mat73.loadmat(file_name)


def explore_dict(d, depth=1):
    if not isinstance(d, dict) or depth == 0:
        return
    for key, value in d.items():
        print(f'{"  " * (3 - depth)}Key: {key}, Type of value: {type(value)}')
        if isinstance(value, dict):
            explore_dict(value, depth - 1)



keys_to_extract = ['itemnames', 'itempos', 'items', 'spikes', 'trialnum', 'neuron_id', 'fields']

# New dictionary with selected key-value pairs (only necessary information to analyze spikes)
spike_dict = {key: mat_contents['L2_str'][key] for key in keys_to_extract}


with open(f'spike_dict_{file_to_name}.pkl', 'wb') as file: 
    pickle.dump(spike_dict, file)

