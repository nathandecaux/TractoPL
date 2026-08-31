#!/usr/bin/env python
import os, sys, argparse, subprocess, shutil, json
from pathlib import Path
import numpy as np
import ants

def log(m):
    print(f"[tractseg_flirt] {m}", flush=True)


def run(cmd):
    print('Running command: ' + ' '.join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        log(f"Erreur lors de l'exécution de la commande: {' '.join(cmd)}")
        log(f"Sortie standard: {result.stdout}")
        log(f"Erreur standard: {result.stderr}")
        sys.exit(result.returncode)


def parse_args():
    p = argparse.ArgumentParser(description="FLIRT registration + TractSeg (pipeline shell)")
    p.add_argument('--dwi', required=True, help='Diffusion.nii.gz')
    p.add_argument('--bvals', required=True)
    p.add_argument('--bvecs', required=True)
    p.add_argument('--atlas', required=True, help='Template FA MNI (atlas.nii.gz)')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--brain_mask', help='Masque nodif (optionnel)')
    p.add_argument('--register_on', choices=['fa', 'b0','mask'], default='fa')
    p.add_argument('--invert_bvec_y_input', action='store_true')
    p.add_argument('--tractseg_args', default='--output_type tract_segmentation')
    p.add_argument('--skip_tractseg', action='store_true')
    p.add_argument('--save_affine_json', action='store_true')
    return p.parse_args()

# def do_inverse_bvec(bvec_tmp):
#     print(f"Inverting bvec file {bvec_tmp}")
#     bvec = np.loadtxt(bvec_tmp)
#     bvec[0, :] = -bvec[0, :]
#     np.savetxt(bvec_tmp, bvec, fmt='%.18e')
def invert_bvec_y(bvec_in, bvec_out):
    arr = np.loadtxt(bvec_in)
    # Formats possibles: 3xN ou Nx3; standard FSL = 3 lignes
    arr[0, :] = -arr[0, :]
    np.savetxt(bvec_out, arr, fmt='%.8f')
    return bvec_out


def extract_affine_matrix(mat_path):
    try:
        M = np.loadtxt(mat_path)
        return M.tolist()
    except Exception as e:
        log(f"Impossible de lire la matrice FLIRT: {e}")
        return None


def run_tractseg(dwi_path, bvals_path, bvecs_path, out_dir, brain_mask_path, extra_args):
    cmd = ['TractSeg', '-i', str(dwi_path), '-o', str(out_dir), '--raw_diffusion_input', '--bvals', str(bvals_path), '--bvecs', str(bvecs_path)] + extra_args.split()
    cmd += ['--csd_type','csd']
    log("Commande TractSeg: " + " ".join(cmd))
    run(cmd)

def apply_transform_to_bvecs(bvecs_in, mat_path, fa_path, template, bvecs_out):
    """
    Applique la composante de rotation d'une matrice affine ANTs (ITK .mat) aux bvecs,
    en utilisant ants pour charger la transformation.
    """
    # Charger la transformation avec ants
    tx = ants.read_transform(str(mat_path))
    # Extraire la matrice affine 3x3 (rotation)
    if hasattr(tx, 'parameters') and len(tx.parameters) >= 9:
        rot = np.array(tx.parameters[:9]).reshape(3, 3)
    elif hasattr(tx, 'matrix') and tx.matrix.shape == (3, 3):
        rot = tx.matrix
    else:
        raise ValueError("Impossible d'extraire la matrice de rotation de la transformation ANTs.")

    # Charger les bvecs (format FSL: 3 lignes, N colonnes)
    bvecs = np.loadtxt(bvecs_in)
    if bvecs.shape[0] != 3 and bvecs.shape[1] == 3:
        bvecs = bvecs.T  # s'assurer d'avoir 3xN

    # Appliquer la rotation à chaque vecteur
    rotated = rot @ bvecs

    # Normaliser chaque vecteur (important pour rester sur la sphère unité)
    norms = np.linalg.norm(rotated, axis=0)
    norms[norms == 0] = 1  # éviter division par zéro
    rotated /= norms

    # Sauvegarder
    np.savetxt(bvecs_out, rotated, fmt='%.8f')
    return bvecs_out


def main():
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    dwi = Path(args.dwi)
    bvals = Path(args.bvals)
    bvecs = Path(args.bvecs)
    template = Path(args.atlas)

    # Préparation noms sortie
    fa_path = out_dir / 'FA.nii.gz'
    fa_mni_path = out_dir / 'FA_MNI.nii.gz'
    mat_path = out_dir / 'FA_2_MNI.mat'
    dwi_mni_path = out_dir / 'Diffusion_MNI.nii.gz'
    bvals_mni_path = out_dir / 'Diffusion_MNI.bvals'
    bvecs_mni_path = out_dir / 'Diffusion_MNI.bvecs'

    # # Option inversion Y bvec avant pipeline
    bvecs_used = bvecs
    if args.invert_bvec_y_input:
        bvecs_inv = out_dir / 'Diffusion_invY.bvecs'
        invert_bvec_y(bvecs, bvecs_inv)
        bvecs_used = bvecs_inv
        log('Bvec Y inversé -> ' + str(bvecs_used))

    if args.register_on == 'b0':
        log('Calcul b0 pour enregistrement')
        b0_path = out_dir / 'FA.nii.gz'
        run(['fslroi', str(dwi), str(b0_path), '0', '1'])
        log(f"Enregistrement sur b0: {b0_path}")
    else:
        # 1. calc_FA
        calc_cmd = ['calc_FA', '-i', str(dwi), '-o', str(fa_path), '--bvals', str(bvals), '--bvecs', str(bvecs_used)]
        if args.brain_mask:
            calc_cmd += ['--brain_mask', str(args.brain_mask)]
        run(calc_cmd)

    
    # if args.register_on == 'mask':
    #     #Compute brain mask from FA
    #     if args.brain_mask:
    #         brain_mask_path = Path(args.brain_mask)
    #     else:
    #         brain_mask_path = out_dir / 'brain_mask.nii.gz'
    #         run(['bet', str(fa_path), str(brain_mask_path), '-m', '-f', '0.3'])

    #     #Compute brain mask from template
    #     template_mask_path = out_dir / 'template_brain_mask.nii.gz'
    #     run(['bet', str(template), str(template_mask_path), '-m', '-f', '0.3'])
    #     #Use brain masks in FLIRT
    #     flirt_fa_cmd = [
    #         'flirt', '-ref', str(template), '-in', str(fa_path),
    #         '-out', str(fa_mni_path), '-omat', str(mat_path),
    #         '-dof', '3', '-cost', 'leastsq', '-searchcost', 'leastsq',
    #         '-inweight', str(brain_mask_path), '-refweight', str(template_mask_path)
    #     ]
    #     run(flirt_fa_cmd)
    # else:
    if True:
        # 2. ANTs FA -> template
        flirt_fa_cmd = [
            'antsRegistrationSyN.sh', '-d', '3', '-f', str(template), '-m', str(fa_path),
            '-o', str(fa_mni_path.with_suffix('')),'-t','r'
        ]
        run(flirt_fa_cmd)

    # Rename os.path.join(out_dir,'FA_MNI.nii0GenericAffine.mat') to mat_path
    shutil.move(str(out_dir / 'FA_MNI.nii0GenericAffine.mat'), str(mat_path))
    
    # 3. ANTs apply transform DWI
    flirt_dwi_cmd = [
        'antsApplyTransforms', '-d', '3', '-i', str(dwi), '-r', str(template),
        '-o', str(dwi_mni_path), '-t', str(mat_path), '-u', 'float','--float', '-e',"3"
    ]
    run(flirt_dwi_cmd)

    # 4. Copier bvals
    if bvals != bvals_mni_path:
        shutil.copy(bvals, bvals_mni_path)

    # 5. rotate_bvecs sur bvec utilisé (inversé ou non)
    # rotate_cmd = [
    #     'rotate_bvecs', '-i', str(bvecs_used), '-t', str(mat_path), '-o', str(bvecs_mni_path)
    # ]
    # run(rotate_cmd)
    apply_transform_to_bvecs(bvecs_used, mat_path, fa_path, template, bvecs_mni_path)

    # # Sauvegarde matrice JSON si demandé
    # if args.save_affine_json:
    #     M = extract_affine_matrix(mat_path)
    #     if M is not None:
    #         with open(out_dir / 'affine_map.json', 'w') as f:
    #             json.dump({'flirt_affine': M}, f, indent=2)



    # # # 6. TractSeg
    if not args.skip_tractseg:
        run_tractseg(dwi_mni_path, bvals_mni_path, bvecs_mni_path, out_dir, args.brain_mask, args.tractseg_args)
    else:
        log('TractSeg ignoré (--skip_tractseg)')
    #Move bundle_segmentation if exists to tractseg_output/bundle_segmentation
    bundle_segmentation_path = out_dir / 'bundle_segmentations'
    tractseg_output_dir = out_dir / 'tractseg_output'
    if bundle_segmentation_path.exists():
        tractseg_output_dir.mkdir(exist_ok=True)
        dest_path = tractseg_output_dir / 'bundle_segmentations'
        if dest_path.exists():
            shutil.rmtree(dest_path)
        shutil.move(str(bundle_segmentation_path), str(dest_path))
        log(f"Déplacé {bundle_segmentation_path} vers {dest_path}")
    # 7. Endings segmentation
    os.chdir(out_dir)
    #flip_peaks -i my_peaks.nii.gz -o my_peaks_flip_y.nii.gz -a y
    # run(['TractSeg', '-i', 'peaks.nii.gz', '--output_type', 'tract_segmentation'])
    run(['TractSeg', '-i', 'peaks.nii.gz', '--output_type', 'endings_segmentation'])
    run(['TractSeg', '-i', 'peaks.nii.gz', '--output_type', 'TOM'])
    run(['Tracking', '-i', 'peaks.nii.gz'])

    log('Pipeline FLIRT terminé')


if __name__ == '__main__':
    main()

