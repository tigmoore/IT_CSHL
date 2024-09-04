# IT_CSHL
This repo analyizes spike trains recorded from macaque IT cortex as well as videoMAE responses to the same video clips

Steps for preprocessing IT data:
### 1. Convert .mat to python compatible structures for data files using mat_2_py.py 
Dependencies: 
```pip install mat73, numpy, sys, pickle, scikit-learn``` 

The original IT data are in .mat files, in order to convert these to python files and extract just the information we need, use mat_2_py. 
This script can be run from command line with the name of the .mat file as the first and only argument. Run for each recording session .mat file.
i.e. from command line: ```$ mat_2_py.py path/to/monkey_data.mat```

This will create a pickle file for example named 'spike_dict_fixshortvid1_coco.pkl'

### 2.  Filter out responsive neurons with real_stats.ipynb
A pickle file is created from mat_2_py.py. The pickle file is a dictionary with ALL neuron (or channel) names, spike timings, video clip names, etc. 
To extract only responsive neurons through a signal correlation analysis: run notebook real_stats.ipynb 

The structure of the dicts of responsive neurons are nested as follows:
a key-value pair would be: 'neuron_id': nested_spike_timings.

The nested spike timings will have shape = (num_responsive_neurons, num_video_clips, num_repeats_for_given_clip, num_spikes_recorded)

### 3. Heatmaps and analysis 
Run the notebook heatmaps.ipynb
This bins the spike timings, (bin sizes can be adjusted), takes the average response over repeats, normalizes response to clips per neuron.

Visualize avg firing rates for all responsive neurons and firing rates for **individual** responsive neurons

### 4. Steps for ANN activation extraction
Extract activations from VideoMAE transformer model for video data 
create a venv 
pip install requirements.txt in virtual environment
ensure paths to movie frame data and grey_image.png are correct,
MOVIE_DATA is a directory of all pngs for the video clips 
run ```videomae_grey_screen_extract.py```
This script treats the transformer as a sliding window over the video clip padded by 16 frames of grey image on either end, and extracts the final temporal token averaged response as a numpy variable 'hidden_states.npy'. 
![model_diagram](https://github.com/user-attachments/assets/99aacc72-afee-4852-bc8e-30ac8123baf3)

The shape of 'hidden_states.npy'  = (total_num_frames, num_hidden_units)

Finally, visualize the ANN activations with notebook ```grey_visualization.ipynb``` by loading in 'hidden_states.npy' 

email me with any questions :)
