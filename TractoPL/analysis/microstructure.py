import os
import numpy as np
import pandas as pd
import sys
import pathlib
from subprocess import call
from TractoPL.set_config import set_config
from TractoPL.data.loader import Subject
from TractoPL.data.io import copy2nii, move2nii, copy_list, copy_from_dict
from TractoPL.utils.tools import del_key, upt_dict, create_pipeline_description, CLIArg
from TractoPL.utils.mcm import mcm_estimator, update_mcm_info_file, add_mcm_to_tracts
import tempfile
import glob
import shutil
import vtk 


class MCMVTKReader:
    """Reader for VTK files containing MultiCompartmentModel (MCM) data from anima.
    
    This reader is designed to efficiently read large VTK files by loading header and 
    metadata information first and providing methods to access specific points and 
    cells without loading the entire file into memory.
    """
    
    def __init__(self, vtk_file_path):
        """Initialize the MCM VTK reader with a file path.
        
        Args:
            vtk_file_path (str): Path to the VTK file
        """
        self.vtk_file_path = vtk_file_path
        self.reader = vtk.vtkPolyDataReader()
        self.reader.SetFileName(vtk_file_path)
        
        # Don't read all data at once to save memory
        self.reader.ReadAllScalarsOff()
        self.reader.ReadAllVectorsOff()
        self.reader.ReadAllTensorsOff()
        self.reader.ReadAllFieldsOff()
        self.reader.ReadAllNormalsOff()

        # Just read header information
        self.reader.Update()
        self.polydata = self.reader.GetOutput()
        
        # Cache metadata
        self._initialize_metadata()
        
    def _initialize_metadata(self):
        """Initialize metadata from the VTK file."""
        self.n_points = self.polydata.GetNumberOfPoints()
        self.n_cells = self.polydata.GetNumberOfCells()
        
        # Get array information
        self.point_data = self.polydata.GetPointData()
        self.n_arrays = self.point_data.GetNumberOfArrays()
        
        # Cache array names and their indices
        self.array_names = {}
        for i in range(self.n_arrays):
            array = self.point_data.GetArray(i)
            self.array_names[array.GetName()] = i
            
        # Check if this is an MCM file based on arrays
        self.is_mcm = "MostColinearIndex" in self.array_names
        
        # Identify compartment data
        self.compartment_weights = []
        self.compartment_params = []
        
        # Group arrays by type
        for name in self.array_names.keys():
            if name.endswith("Weight"):
                self.compartment_weights.append(name)
            elif "Parameter" in name:
                self.compartment_params.append(name)
    
    def get_metadata(self):
        """Get metadata about the VTK file.
        
        Returns:
            dict: Dictionary containing metadata
        """
        return {
            "n_points": self.n_points,
            "n_cells": self.n_cells,
            "n_arrays": self.n_arrays,
            "array_names": list(self.array_names.keys()),
            "is_mcm": self.is_mcm,
            "compartment_weights": self.compartment_weights,
            "compartment_params": self.compartment_params
        }
    
    def get_point(self, point_id):
        """Get the coordinates of a specific point.
        
        Args:
            point_id (int): Point ID
            
        Returns:
            tuple: (x, y, z) coordinates
        """
        if point_id >= self.n_points:
            raise ValueError(f"Point ID {point_id} out of range (max: {self.n_points-1})")
        return self.polydata.GetPoint(point_id)
    
    def get_cell(self, cell_id):
        """Get a specific cell.
        
        Args:
            cell_id (int): Cell ID
            
        Returns:
            vtkCell: Cell object
        """
        if cell_id >= self.n_cells:
            raise ValueError(f"Cell ID {cell_id} out of range (max: {self.n_cells-1})")
        return self.polydata.GetCell(cell_id)
    
    def get_cell_points(self, cell_id):
        """Get all points in a specific cell.
        
        Args:
            cell_id (int): Cell ID
            
        Returns:
            list: List of point coordinates in the cell
        """
        cell = self.get_cell(cell_id)
        points = []
        for i in range(cell.GetNumberOfPoints()):
            point_id = cell.GetPointId(i)
            points.append(self.get_point(point_id))
        return points
    
    def get_point_data(self, point_id, array_name):
        """Get data for a specific point and array.
        
        Args:
            point_id (int): Point ID
            array_name (str): Name of the array
            
        Returns:
            float or array: Data value
        """
        if point_id >= self.n_points:
            raise ValueError(f"Point ID {point_id} out of range (max: {self.n_points-1})")
        if array_name not in self.array_names:
            raise ValueError(f"Array {array_name} not found in point data")
        
        array = self.point_data.GetArray(self.array_names[array_name])
        return array.GetValue(point_id)
    
    def get_mcm_data_for_point(self, point_id):
        """Get all MCM data for a specific point.
        
        Args:
            point_id (int): Point ID
            
        Returns:
            dict: Dictionary containing MCM data
        """
        if not self.is_mcm:
            raise ValueError("This is not an MCM file")
        
        result = {
            "colinear_index": self.get_point_data(point_id, "MostColinearIndex")
        }
        
        # Add compartment weights
        for weight_name in self.compartment_weights:
            result[weight_name] = self.get_point_data(point_id, weight_name)
        
        # Add compartment parameters
        for param_name in self.compartment_params:
            result[param_name] = self.get_point_data(point_id, param_name)
        
        return result
    
    def get_streamline_data(self, cell_id, array_name=None):
        """Get data for a specific streamline (cell).
        
        Args:
            cell_id (int): Cell ID
            array_name (str, optional): Name of the array to extract. If None, returns positions only.
            
        Returns:
            dict: Dictionary with positions and optional data
        """
        cell = self.get_cell(cell_id)
        result = {"positions": []}
        
        if array_name:
            if array_name not in self.array_names:
                raise ValueError(f"Array {array_name} not found in point data")
            result["data"] = []
        
        for i in range(cell.GetNumberOfPoints()):
            point_id = cell.GetPointId(i)
            result["positions"].append(self.get_point(point_id))
            
            if array_name:
                result["data"].append(self.get_point_data(point_id, array_name))
                
        return result
    
    def get_all_array_names(self):
        """Get all array names in the file.
        
        Returns:
            list: List of array names
        """
        return list(self.array_names.keys())
    
    def extract_streamlines(self, start=0, count=None):
        """Extract a subset of streamlines.
        
        Args:
            start (int): Starting cell index
            count (int, optional): Number of cells to extract. If None, extracts all from start.
            
        Returns:
            dict: Dictionary with streamlines data
        """
        if count is None:
            count = self.n_cells - start
            
        end = min(start + count, self.n_cells)
        
        streamlines = []
        for i in range(start, end):
            cell = self.get_cell(i)
            points = []
            for j in range(cell.GetNumberOfPoints()):
                point_id = cell.GetPointId(j)
                points.append(self.get_point(point_id))
            streamlines.append(points)
            
        return streamlines


# Exemple d'utilisation de la classe MCMVTKReader
if __name__ == "__main__":
    # vtk_file = "/home/ndecaux/Data/actidep_bids/derivatives/mcm_tensors/sub-03011/tracto/sub-03011_desc-normalized_label-WM_model-MCM_tracto.vtk"

    vtk_file = "/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids/derivatives/mcm_tensors_staniz/sub-01002/tracto/sub-01002_desc-cleaned_model-MCM_tracto.vtk"    
    # Initialiser le lecteur
    reader = MCMVTKReader(vtk_file)
    
    # Afficher les métadonnées
    print("Métadonnées:")
    metadata = reader.get_metadata()
    for key, value in metadata.items():
        if isinstance(value, list) and len(value) > 10:
            print(f"{key}: {value[:5]} ... ({len(value)} éléments)")
        else:
            print(f"{key}: {value}")
    
    # Afficher le premier point
    print("\nPremier point:", reader.get_point(0))
    
    # Afficher les données de la première cellule
    print("\nPremière cellule (premiers 3 points):")
    cell_points = reader.get_cell_points(0)
    for i, point in enumerate(cell_points[:3]):
        print(f"Point {i}: {point}")
    
    if reader.is_mcm:
        # Afficher les données MCM pour le premier point
        print("\nDonnées MCM pour le premier point:")
        mcm_data = reader.get_mcm_data_for_point(0)
        print(f"Indice le plus colinéaire: {mcm_data['colinear_index']}")
        
        # Afficher quelques poids de compartiments
        for weight_name in reader.compartment_weights[:3]:
            print(f"{weight_name}: {mcm_data[weight_name]}")
        
        # Afficher quelques paramètres
        for param_name in reader.compartment_params[:3]:
            print(f"{param_name}: {mcm_data[param_name]}")
    
    # Extraire quelques streamlines
    print("\nExtraction de 2 streamlines:")
    streamlines = reader.extract_streamlines(0, 2)
    for i, streamline in enumerate(streamlines):
        print(f"Streamline {i}: {len(streamline)} points")
        print(f"  Premier point: {streamline[0]}")
        print(f"  Dernier point: {streamline[-1]}")

