# -*- coding: utf-8 -*-
"""
@author: Liu, XuYang
"""

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import align
from tqdm import tqdm
import warnings
 
warnings.filterwarnings("ignore")

def alignDcd2Ref(psf_file_path, dcd_file_path,  aligned_npz,  aligned_selection, frame_to_align=0):
    position_list = []
    psf_temp = mda.Universe(psf_file_path)
    dcd_to_align = mda.Universe(psf_file_path, dcd_file_path)
    dcd_to_align.trajectory[frame_to_align]
    B = dcd_to_align.select_atoms(aligned_selection).positions

    B_center = B.mean(0, keepdims=True)
    print('.........................aligning.........................')
    for i in tqdm(dcd_to_align.trajectory):
        psf_atom = psf_temp.load_new(i.positions)
        coor_selected = psf_atom.select_atoms(aligned_selection).positions

        # cal rotation matrix
        A = psf_atom.select_atoms(aligned_selection).positions
        A_center = A.mean(0, keepdims=True)
        R, _ = mda.analysis.align.rotation_matrix(A - A_center, B - B_center)

        # transform selected coordinates
        coor_selected_transformed = (coor_selected - A_center) @ R.T

        position_list.append(coor_selected_transformed[None, ...].copy())

    Carte_coor = np.vstack(position_list)
    np.savez(aligned_npz, position=Carte_coor)
    print('.........................aligning done.........................')
    return Carte_coor
