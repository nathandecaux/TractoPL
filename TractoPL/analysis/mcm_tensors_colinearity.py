from TractoPL.data.loader import Dataset, BIDSFile
import pickle
import os
from TractoPL.data.vtk_loader import load_vtk_streamlines
import numpy as np

mcm_pipeline='mcm_tensors_staniz'

if os.path.exists('/tmp/actidep_ds.pkl'):
    with open('/tmp/actidep_ds.pkl','rb') as f:
        ds = pickle.load(f)
else: 
    ds = Dataset()
    with open('/tmp/actidep_ds.pkl','wb') as f:
        pickle.dump(ds,f)


mat_files = ds.get_global(desc='tensors',datatype='mat',extension='.vtk')


bundle_list = list(set([f.bundle for f in mat_files]))

print(bundle_list)

df=[]
for bundle in bundle_list:
    print(f'Processing bundle: {bundle}')
    files = [f for f in mat_files if f.bundle == bundle]
    for f in files:
        print(f'  Loading file: {f.path}')
        tensors,array_dict = load_vtk_streamlines(f.path,reshape_scalars=True)
        print(f'    Loaded {len(tensors)} streamlines with tensors shape: {[len(t) for t in tensors]}')
        modified_voxel=array_dict['ModifiedMostColinear']
        angle_with_sl = array_dict['Tensor_Angle_MostColinear']
        fa_vox= array_dict['Tensor_FA_MostColinear']

        #add in the data frame
        for sl_idx, ( mod_vox, angle_vox, fa_vox) in enumerate(zip( modified_voxel, angle_with_sl, fa_vox)):
            for point_idx, ( mv, av, fv) in enumerate(zip( mod_vox, angle_vox, fa_vox)):
                df.append({
                    'subject': f.subject,
                    'session': f.session,
                    'bundle': f.bundle,
                    'streamline_index': sl_idx,
                    'point_index': point_idx,
                    'modified_most_colinear': mv,
                    'angle_with_streamline': av,
                    'fa_most_colinear': fv
                })

import pandas as pd
df = pd.DataFrame(df)
output_csv_path = '/home/ndecaux/Code/actiDep/analysis/mcm_tensors_colinearity_results.csv'
df.to_csv(output_csv_path, index=False)
print(f'Results saved to {output_csv_path}')
