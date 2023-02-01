import random
import tarfile

import numpy as np
import pyvista as pv
import pynoddy.output
import pynoddy.history
import os
import gzip
import time
from tqdm import tqdm
import pickle
import copy

path = r'E:\Code\duanjc\PyCode\GeoScience\geomodel_workshop\test'
model_name = '20-09-04-16-00-26-664297926'

output_dir = path
his_file = os.path.join(path, model_name) + '.his'
output_path = os.path.join(output_dir, model_name)
pynoddy.compute_model(his_file, output_path)
pynoddy.compute_model(his_file, output_path, sim_type='GEOPHYSICS')
nout = pynoddy.output.NoddyOutput(output_path)
nout_geophysics = pynoddy.output.NoddyGeophysics(output_path)
# nout.export_to_vtk()
nout_geophysics.grv_data
vtr_path = os.path.join(path, model_name) + '.vtr'
vtr_model = pv.read(vtr_path)
vtr_model.plot()