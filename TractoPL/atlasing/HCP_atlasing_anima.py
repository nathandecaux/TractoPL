import os 
from pprint import pprint
from os.path import join as opj
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import xml.etree.ElementTree as ET
import multiprocessing
from functools import partial
import traceback
import tempfile
import dipy


base_dir = "/local/tdurante/atlas/groupe1/"
sub_list = "/local/tdurante/atlas/groupe1/sujets.txt"
bundle_list = "/local/tdurante/atlas/groupe1/bundles.txt"
# bundle_src="/data/HCP_Data/HCP105_Zenodo/"
bundle_src="/data/HCP_Data/Diffusion_Data_Preprocessed/"
# bundle_src='/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/'
anat_src="/data/HCP_Data/Structural_Data_Preprocessed/"
output_dir="/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/inGroupe1Space/"
fa_src="/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/"
#Bundle path example /data/HCP_Data/HCP105_Zenodo/599469/tracts/AF_left.vtp


with open(sub_list, 'r') as f:
    subjects = f.read().splitlines()

print(len(subjects))

with open(bundle_list, 'r') as f:
    bundles = f.read().splitlines()

subjects = {k+1:v for k,v in enumerate(subjects)}

def create_fake_rigid_transform():
    """
    Create a fake rigid transformation matrix for testing purposes.
    """
    transform_matrix = np.eye(4)
    transform_matrix[0, 3] = 10  # Translate x by 10
    transform_matrix[1, 3] = -11  # Translate y by -5
    transform_matrix[2, 3] = 12  # Translate z by 2
    return transform_matrix

def read_transformation_xml(xml_file):
    """Read and parse an XML transformation file."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    transformations = []
    for transform in root.findall('Transformation'):
        transform_type = transform.find('Type').text
        inversion = bool(int(transform.find('Inversion').text))
        path = transform.find('Path').text
        
        transformations.append({
            'type': transform_type,
            'inversion': inversion,
            'path': path
        })
    
    return transformations

def convert_transform_matrix_to_nibabel(transform_matrix):
    """
    Convert a transformation matrix from SimpleITK to nibabel format.
    SimpleITK uses LPS (Left-Posterior-Superior) coordinate system,
    while nibabel uses RAS (Right-Anterior-Superior).
    This function handles the conversion between these coordinate systems.
    """
    # Create the coordinate system conversion matrix (LPS to RAS)
    lps_to_ras = np.array([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    # Convert the matrix by applying the coordinate system transformation
    nib_matrix = lps_to_ras @ transform_matrix @ np.linalg.inv(lps_to_ras)
    
    return nib_matrix

def apply_transform_to_tract(tract_path, ref_path, transform_xml_path, output_dir=None, transform_ref=False):
    """
    Apply transformations defined in XML file to a tractography file and optionally to the reference image.
    
    Parameters:
    -----------
    tract_path : str
        Path to the tract file (.vtp)
    ref_path : str
        Path to the reference image (.nii.gz)
    transform_xml_path : str
        Path to the transformation XML file
    output_dir : str, optional
        Directory to save output files. If None, uses the same directory as input files.
    transform_ref : bool, optional
        Whether to transform the reference image as well.
        
    Returns:
    --------
    dict
        Dictionary with paths to transformed files
    """
    base_dir = os.path.dirname(os.path.dirname(transform_xml_path))
    os.chdir(base_dir)
    tract_name = os.path.basename(tract_path).replace('.vtp', '')
    
    # Read the transformation XML file
    # transformations = read_transformation_xml(transform_xml_path)
    
    # Prepare output directory
    if output_dir is None:
        output_dir = os.path.dirname(tract_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the reference image if needed
    # moved_ref_path=f"{output_dir}/transformed_ref.nii.gz" if transform_ref else None
    
    if transform_ref:
        # Use animaApplyTransformSerie
        print(f"Transforming reference image: {ref_path}")
        trans_cmd = f"animaApplyTransformSerie -i {ref_path} -g {ref_path} -o {output_dir}/transformed_anat.nii.gz -t {transform_xml_path}"
        os.system(trans_cmd)

    return True

def apply_transform_to_FA(fa_path, transform_xml_path, output_dir=None):
    """
    Apply transformations defined in XML file to a DTI image.
    Parameters:
    -----------
    fa_path : str
        Path to the FA image (.nii.gz)
    transform_xml_path : str
        Path to the transformation XML file
    output_dir : str, optional
        Directory to save output files. If None, uses the same directory as input files.
    Returns:
    --------
    dict
        Dictionary with paths to transformed files
    """
    base_dir = os.path.dirname(os.path.dirname(transform_xml_path))
    os.chdir(base_dir)
    fa_name = os.path.basename(fa_path).replace('.nii.gz', '')

    if output_dir is None:
        output_dir = os.path.dirname(fa_path)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Transforming FA image: {fa_path}")
    trans_cmd = f"animaApplyTransformSerie -i {fa_path} -g {output_dir}/transformed_anat.nii.gz -o {output_dir}/transformed_fa.nii.gz -t {transform_xml_path}"
    os.system(trans_cmd)

    # #Use animaFibersApplyTransformSerie
    # trans_cmd = f"animaFibersApplyTransformSerie -i {tract_path} -o {output_dir}/transformed_{tract_name}.vtk -t {transform_xml_path}"

    # print(trans_cmd)

    # os.system(trans_cmd)

    # print(f"Transformation completed, output saved to {output_dir}/transformed_{tract_name}.vtk")

    # print(f"Converting transformed VTP to TRK")

    # convert_cmd = f"dipy_convert_tractogram {output_dir}/transformed_{tract_name}.vtk --out_tractogram {output_dir}/transformed_{tract_name}.trk --reference {ref_path} --force"

    # os.system(convert_cmd)

    # return {
    #     'tract': f"{output_dir}/transformed_{tract_name}",
    # }

# Fonction auxiliaire pour traiter un faisceau (à ajouter avant la boucle principale)
def process_bundle(bundle, bundle_src, sub_name, ref_path, transfo, sub_out_dir):
    bundle_path = opj(bundle_src, sub_name, "Tracks", f"{bundle}.vtp")
    excepted_out_bundle = opj(sub_out_dir, f"transformed_{bundle}.vtp")
    
    if os.path.exists(excepted_out_bundle):
        print(f"Bundle {bundle} already processed, skipping")
        return
    
    try:
        print(f"Processing bundle: {bundle}")
        res = apply_transform_to_tract(bundle_path, ref_path, transfo, output_dir=sub_out_dir, transform_ref=False)
        print(f"Bundle {bundle} processed successfully")
    except Exception as e:
        error_file = opj(sub_out_dir, f"error_{bundle}.txt")
        with open(error_file, 'w') as f:
            f.write(str(e))
        print(f"Error processing bundle {bundle}, check {error_file}")
        # Print the full error message to console
        print(f"Detailed error: {str(e)}")
        print(traceback.format_exc())


# Only process the subject 599671
#Get hostname
hostname = os.uname()[1]
print(f"Hostname: {hostname}")
# if hostname == "calcarine":
#     #Get the first half of the subjects
#     subjects = list(subjects.items())[:len(subjects)//2]
# else:
#     #Get the second half of the subjects
#     subjects = list(subjects.items())[len(subjects)//2:]

# # Convert to dictionary
# subjects = {k:v for k,v in subjects}
# subjects = {k:v for k,v in subjects if v == '871964'}

list_transformed_fa = []

for sub_id, sub_name in subjects.items():
    transfo = opj(base_dir, f'Transfos/DTI_{sub_name}.xml')
    ref_path = opj(anat_src, sub_name, "Images", "T1w_acpc_dc_restore_brain.nii.gz")
    ref_transformed_path = opj(output_dir, sub_name, "transformed_anat.nii.gz")
    fa_path= opj(fa_src, sub_name, "dti", f"FA_{sub_name}.nii.gz")
    sub_out_dir = opj(output_dir, sub_name)
    os.makedirs(sub_out_dir, exist_ok=True)
    print(f"Processing subject {sub_id}: {sub_name}")
    
    # # Traiter le premier faisceau séparément pour transformer la référence
    # first_bundle = bundles[0]
    # # first_bundle= 'CC'
    # first_bundle_path = opj(bundle_src, sub_name, "Tracks", f"{first_bundle}.vtp")
    # excepted_out_bundle = opj(sub_out_dir, f"transformed_{first_bundle}.vtp")
    
    # if not os.path.exists(excepted_out_bundle):
    #     try:
    #         print(f"Processing first bundle with reference transformation: {first_bundle}")
    #         res = apply_transform_to_tract(first_bundle_path, ref_path, transfo, output_dir=sub_out_dir, transform_ref=True)
    #         print("Results:")
    #         pprint(res)
    #     except Exception as e:
    #         error_file = opj(sub_out_dir, f"error_{first_bundle}.txt")
    #         with open(error_file, 'w') as f:
    #             f.write(str(e))
    #         print(f"Error processing bundle {first_bundle}, check {error_file}")
    #         print(f"Detailed error: {str(e)}")
    
    # # Traiter les autres faisceaux en parallèle sans transformer la référence
    # remaining_bundles = bundles[1:] if bundles else []
    # if remaining_bundles:
    #     # Créer une fonction partielle avec les arguments constants
    #     process_bundle_partial = partial(
    #         process_bundle, 
    #         bundle_src=bundle_src,
    #         sub_name=sub_name,
    #         ref_path=ref_path,
    #         transfo=transfo,
    #         sub_out_dir=sub_out_dir
    #     )
        
    #     # Utiliser un pool de processus pour traiter les faisceaux en parallèle
    #     with multiprocessing.Pool(processes=min(multiprocessing.cpu_count(), len(remaining_bundles))) as pool:
    #         pool.map(process_bundle_partial, remaining_bundles)
    
    # print("\n\n")

    #Apply transform to FA
    transformed_fa_path = opj(sub_out_dir, "transformed_fa.nii.gz")
    list_transformed_fa.append(transformed_fa_path)

    if os.path.exists(transformed_fa_path):
        print(f"FA already transformed for subject {sub_name}, skipping")
        continue
    try:
        print(f"Transforming FA for subject {sub_name}")
        res = apply_transform_to_FA(fa_path, transfo, output_dir=sub_out_dir)
        print("Results:")
        pprint(res)
    except Exception as e:
        error_file = opj(sub_out_dir, f"error_fa.txt")
        with open(error_file, 'w') as f:
            f.write(str(e))
        print(f"Error transforming FA for subject {sub_name}, check {error_file}")
        print(f"Detailed error: {str(e)}")


print("All transformed FA paths:")
pprint(list_transformed_fa)

# cmd = f'animaAverageImages -i {" ".join(list_transformed_fa)} -o {opj(output_dir,'Atlas', "average_fa.nii.gz")}'
#Use fsl to compute average

out_average_fa=opj(output_dir,"Atlas", "average_fa.nii.gz")

cmd = f'fslmaths {" -add ".join(list_transformed_fa)} -div {len(list_transformed_fa)} {out_average_fa}'
if not os.path.exists(out_average_fa):
    print("Computing average FA")
    print(cmd)
    os.system(cmd)
else:
    print("Average FA already computed, skipping")


out_fiber_count=opj(output_dir,"Atlas", 'fiber_count')
os.makedirs(out_fiber_count, exist_ok=True)

out_bundle_mask_dir=opj(output_dir,"Atlas", 'FA')
os.makedirs(out_bundle_mask_dir, exist_ok=True)

for bundle in bundles:
    out_bundle = opj(output_dir, "Atlas",'vtk', f"summed_{bundle}.vtk")
    if os.path.exists(out_bundle):
        print(f"Getting bundle {bundle} mask")
        out_fiber_count_bundle=opj(out_fiber_count, f"fiber_count_{bundle}.nii.gz")
        cmd=f'animaFibersCounter -i {out_bundle} -g {out_average_fa} -o {out_fiber_count_bundle} -P'
        if not os.path.exists(out_fiber_count_bundle):
            print(cmd)
            os.system(cmd)
        else:
            print(f"Fiber count for bundle {bundle} already computed, skipping")

        out_bundle_mask=opj(out_bundle_mask_dir, f"FA_{bundle}.nii.gz")

        if os.path.exists(out_bundle_mask):
            print(f"Bundle mask for {bundle} already computed, skipping")
            continue
        fiber_prob = nib.load(out_fiber_count_bundle)
        binary_mask = fiber_prob.get_fdata()> 0

        fa_data = nib.load(out_average_fa)
        fa_masked = fa_data.get_fdata() * binary_mask
        fa_img = nib.Nifti1Image(fa_masked, fa_data.affine, fa_data.header)
        nib.save(fa_img, out_bundle_mask)
        print(f"Bundle mask saved to {out_bundle_mask}")
        
    
