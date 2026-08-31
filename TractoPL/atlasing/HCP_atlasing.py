import os 
from pprint import pprint
from os.path import join as opj
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import xml.etree.ElementTree as ET
from dipy.io.streamline import load_trk, save_trk
from dipy.tracking.streamline import transform_streamlines
from dipy.align.imwarp import DiffeomorphicMap
from dipy.align import read_mapping
import multiprocessing
from functools import partial
import traceback

base_dir = "/local/tdurante/atlas/groupe1/"
sub_list = "/local/tdurante/atlas/groupe1/sujets.txt"
bundle_list = "/local/tdurante/atlas/groupe1/bundles.txt"
# bundle_src="/data/HCP_Data/HCP105_Zenodo/"
bundle_src='/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/'
anat_src="/data/HCP_Data/Structural_Data_Preprocessed/"
output_dir="/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/inGroupe1Space/"
#Bundle path example /data/HCP_Data/HCP105_Zenodo/599469/tracts/AF_left.trk


with open(sub_list, 'r') as f:
    subjects = f.read().splitlines()

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
    """
    # Create the coordinate system conversion matrices
    lps_to_ras = np.array([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    # Convert the matrix
    # First convert from LPS to RAS, then apply inverse for tract coordinates
    ras_matrix = lps_to_ras @ transform_matrix @ np.linalg.inv(lps_to_ras)
    
    return np.linalg.inv(ras_matrix)

def apply_transform_to_tract(tract_path, ref_path, transform_xml_path, output_dir=None, transform_ref=False):
    """
    Apply transformations defined in XML file to a tractography file and optionally to the reference image.
    
    Parameters:
    -----------
    tract_path : str
        Path to the tract file (.trk)
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
    
    # Read the transformation XML file
    transformations = read_transformation_xml(transform_xml_path)
    
    # Prepare output directory
    if output_dir is None:
        output_dir = os.path.dirname(tract_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the reference image if needed
    ref_sitk = None
    if transform_ref:
        print(f"Loading reference image: {ref_path}")
        ref_sitk = sitk.ReadImage(ref_path)
    
    # Load the tractogram
    print(f"Loading tractogram: {tract_path}")
    tractogram = load_trk(tract_path, ref_path, trk_header_check=False)
    transformed_image = ref_sitk
    
    # Apply transformations in sequence
    for transform in transformations:
        transform_path = os.path.join(base_dir, transform['path'])
        print(f"Processing transformation: {transform_path}")
        
        if transform['type'] == 'linear':
            print(f"Applying linear transformation from {transform_path}")
            
            # Parse the ITK transform file
            sitk_transform = sitk.ReadTransform(transform_path)
            transform_matrix = np.eye(4)
            
            # Get the 3x3 rotation matrix
            rotation = np.array(sitk_transform.GetParameters()[:9]).reshape(3, 3)
            # Get the translation vector
            translation = np.array(sitk_transform.GetParameters()[9:12])
            
            # Construct the 4x4 affine matrix
            transform_matrix[:3, :3] = rotation
            transform_matrix[:3, 3] = translation
            
            # Apply inversion if needed
            if transform['inversion']:
                transform_matrix = np.linalg.inv(transform_matrix)
            
            # Convert matrix to nibabel format and apply to streamlines
            nib_matrix = convert_transform_matrix_to_nibabel(transform_matrix)
            # tractogram.streamlines = transform_streamlines(tractogram.streamlines, nib_matrix)
            
            # Apply the linear transformation to the image if needed
            if transform_ref and transformed_image is not None:
                # Create and apply SimpleITK transform
                sitk_affine = sitk.AffineTransform(3)
                sitk_affine.SetMatrix(transform_matrix[:3, :3].flatten())
                sitk_affine.SetTranslation(transform_matrix[:3, 3])
                
                if transform['inversion']:
                    sitk_affine = sitk_affine.GetInverse()
                
                # Resample the image
                transformed_image = sitk.Resample(transformed_image, transformed_image, sitk_affine,
                                                 sitk.sitkLinear, 0.0, transformed_image.GetPixelID())
                
                # ref_name = os.path.basename(ref_path)
                # output_image_path = os.path.join(output_dir, f"transformed_{ref_name}")
                # sitk.WriteImage(transformed_image, output_image_path)
                # print(f"Transformed reference image saved to: {output_image_path}")
                
            
        elif transform['type'] == 'svf':
            print(f"Applying non-linear transformation from {transform_path}")
            
            # Load the displacement field
            disp_img = sitk.ReadImage(transform_path)

            #Save disp_img in output_dir as nii.gz
            disp_img_path = os.path.join(output_dir, f"disp_{os.path.basename(transform_path).replace('.nrrd', '.nii.gz')}")
            sitk.WriteImage(disp_img, disp_img_path)
            print(f"Displacement field saved to: {disp_img_path}")

            transform_path = disp_img_path

            if transform_ref and transformed_image is not None:
                disp_img = sitk.Cast(disp_img, sitk.sitkVectorFloat64)
                
                # Create a displacement field transform
                displacement_transform = sitk.DisplacementFieldTransform(disp_img)
                
                if transform['inversion']:
                    displacement_transform = displacement_transform.GetInverse()
                
                # Apply transform to image
                transformed_image = sitk.Resample(transformed_image, transformed_image, displacement_transform,
                                                sitk.sitkLinear, 0.0, transformed_image.GetPixelID())
                
                ref_name = os.path.basename(ref_path)
                output_image_path = os.path.join(output_dir, f"transformed_{ref_name}")
                sitk.WriteImage(transformed_image, output_image_path)
                print(f"Transformed reference image saved to: {output_image_path}")

            static_img=os.path.join(output_dir, f"transformed_{ref_name}")

            dipy_transform= read_mapping(transform_path, ref_path, static_img)
            coord2world = np.eye(4)
            coord2world[0, 0] = -1
            coord2world[1, 1] = -1
            coord2world[2, 2] = 1
            tractogram.streamlines = dipy_transform.transform_points_inverse(tractogram.streamlines)
            tractogram.streamlines = transform_streamlines(tractogram.streamlines, nib_matrix)
        
    
    # Save the transformed tractogram
    tract_name = os.path.basename(tract_path)
    output_tract = os.path.join(output_dir, f"transformed_{tract_name}")
    save_trk(tractogram, output_tract, bbox_valid_check=False)
    print(f"Transformed tractogram saved to: {output_tract}")
    
    # Save the transformed image if needed
    output_image_path = None
    return {
        'tract': output_tract,
        'reference': output_image_path
    }

# Fonction auxiliaire pour traiter un faisceau (à ajouter avant la boucle principale)
def process_bundle(bundle, bundle_src, sub_name, ref_path, transfo, sub_out_dir):
    bundle_path = opj(bundle_src, sub_name, "tracts", f"{bundle}.trk")
    excepted_out_bundle = opj(sub_out_dir, f"transformed_{bundle}.trk")
    
    if os.path.exists(excepted_out_bundle):
        print(f"Bundle {bundle} already processed, skipping")
        return
    
    print(f"Starting processing for bundle: {bundle}")
    print(f"- Bundle path: {bundle_path}")
    print(f"- Reference path: {ref_path}")
    print(f"- Transform path: {transfo}")
    print(f"- Output directory: {sub_out_dir}")
    
    try:
        print(f"Processing bundle: {bundle}")
        # Activer un niveau de journalisation détaillé
        res = apply_transform_to_tract(bundle_path, ref_path, transfo, output_dir=sub_out_dir, transform_ref=False)
        print(f"Bundle {bundle} processed successfully")
        return res
    except Exception as e:
        error_file = opj(sub_out_dir, f"error_{bundle}.txt")
        with open(error_file, 'w') as f:
            f.write(f"Error processing bundle {bundle}:\n")
            f.write(str(e) + "\n\n")
            f.write("Full traceback:\n")
            f.write(traceback.format_exc())
        print(f"Error processing bundle {bundle}, check {error_file}")
        # Print the full error message to console
        print(f"Detailed error: {str(e)}")
        print("Full traceback:")
        traceback.print_exc()
        # Réexcepter l'erreur pour qu'elle soit propagée ou traitée à un niveau supérieur
        raise

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
subjects = list(subjects.items())
# Convert to dictionary
# subjects = {k:v for k,v in subjects}
subjects = {k:v for k,v in subjects if v == '871964'}

for sub_id, sub_name in subjects.items():
    transfo = opj(base_dir, f'Transfos/DTI_{sub_name}.xml')
    ref_path = opj(anat_src, sub_name, "Images", "T1w_acpc_dc_restore_brain.nii.gz")
    sub_out_dir = opj(output_dir, sub_name)
    os.makedirs(sub_out_dir, exist_ok=True)
    print(f"Processing subject {sub_id}: {sub_name}")
    
    # Traiter le premier faisceau séparément pour transformer la référence
    # first_bundle = bundles[0]
    first_bundle= 'CC'
    first_bundle_path = opj(bundle_src, sub_name, "tracts", f"{first_bundle}.trk")
    excepted_out_bundle = opj(sub_out_dir, f"transformed_{first_bundle}.trk")
    
    if not os.path.exists(excepted_out_bundle):
        try:
            print(f"Processing first bundle with reference transformation: {first_bundle}")
            res = apply_transform_to_tract(first_bundle_path, ref_path, transfo, output_dir=sub_out_dir, transform_ref=True)
            print("Results:")
            pprint(res)
        except Exception as e:
            error_file = opj(sub_out_dir, f"error_{first_bundle}.txt")
            with open(error_file, 'w') as f:
                f.write(str(e))
            print(f"Error processing bundle {first_bundle}, check {error_file}")
            print(f"Detailed error: {str(e)}")
            print("Full traceback:")
            traceback.print_exc()
    
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


