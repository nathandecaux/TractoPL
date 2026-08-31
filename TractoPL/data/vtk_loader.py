import vtk
from vtk.util.numpy_support import numpy_to_vtk
import numpy as np

def load_vtk_streamlines(vtk_file_path,reshape_scalars=False):
    """
    Load streamlines and associated scalar data from a VTK polydata file.
    This function reads a VTK file containing streamline data (polylines) and extracts
    both the geometric coordinates of the streamlines and any associated scalar arrays
    stored as point data.
    Parameters
    ----------
    vtk_file_path : str
        Path to the VTK polydata file to be loaded.
    reshape_scalars : bool, optional
        If True, reshape scalar arrays to match the structure of streamlines,
        where each streamline has its corresponding scalar values grouped together.
        If False, return scalar arrays as flat numpy arrays.
        Default is False.
    Returns
    -------
    streamlines : list of numpy.ndarray
        A list where each element is a numpy array of shape (n_points, 3) representing
        the 3D coordinates of points along a streamline.
    scalar_dict : dict
        A dictionary mapping scalar array names (str) to their values.
        - If reshape_scalars is False: values are 1D numpy arrays containing all
          scalar values in point order.
        - If reshape_scalars is True: values are lists of numpy arrays, where each
          array corresponds to the scalar values for one streamline.
    """
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_file_path)
    reader.Update()
    polydata = reader.GetOutput()
    lines = polydata.GetLines()
    streamlines = []
    lines.InitTraversal()
    id_list = vtk.vtkIdList()
    
    # Track sequential point ranges for each streamline (not VTK point IDs,
    # which may be non-contiguous or 1-based and would be out of range when
    # used to index into the sequential scalar arrays).
    point_ranges = []
    seq = 0

    while lines.GetNextCell(id_list):
        pts = []
        n_pts = id_list.GetNumberOfIds()
        for j in range(n_pts):
            pid = id_list.GetId(j)
            pts.append(polydata.GetPoint(pid))
        streamlines.append(np.array(pts))
        point_ranges.append((seq, seq + n_pts))
        seq += n_pts
    
    # Load scalar arrays
    scalar_dict = {}
    point_data = polydata.GetPointData()
    for i in range(point_data.GetNumberOfArrays()):
        array = point_data.GetArray(i)
        array_name = array.GetName()
        scalar_values = [array.GetValue(j) for j in range(array.GetNumberOfTuples())]
        scalar_dict[array_name] = np.array(scalar_values)

    if reshape_scalars:
        # Reshape scalar arrays to match streamlines using sequential ranges
        reshaped_scalars = {}
        for name, values in scalar_dict.items():
            reshaped_list = []
            for start, end in point_ranges:
                reshaped_list.append(values[start:end])
            reshaped_scalars[name] = reshaped_list
        scalar_dict = reshaped_scalars
    
    return streamlines, scalar_dict

def load_vtk(vtk_file_path,reshape_scalars=False):
    return load_vtk_streamlines(vtk_file_path,reshape_scalars=reshape_scalars)
def save_vtk(streamlines, output_vtk_path,scalar_dict=None):
    """
    Save streamlines to a VTK file format with optional scalar data.

    This function converts streamlines (fiber tracts) into VTK polydata format and writes
    them to a file. Optionally, scalar values can be associated with the points of the
    streamlines.

    Parameters
    ----------
    streamlines : list of array-like
        A list of streamlines, where each streamline is an array of 3D points (x, y, z).
        Each point represents a coordinate along the streamline trajectory.
    output_vtk_path : str
        The file path where the VTK file will be saved.
    scalar_dict : dict, optional
        A dictionary mapping scalar names (str) to scalar values. The scalar values can be:
        - A numpy array of values for each point across all streamlines
        - A list of arrays that will be concatenated
        The values will be flattened and added as point data to the polydata.
        Default is None.

    Returns
    -------
    None
        The function writes the data to disk but does not return a value.
    """
    polydata = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()

    point_id = 0
    for sl in streamlines:
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(len(sl))
        for i, pt in enumerate(sl):
            points.InsertNextPoint(pt)
            line.GetPointIds().SetId(i, point_id)
            point_id += 1
        lines.InsertNextCell(line)

    polydata.SetPoints(points)
    polydata.SetLines(lines)

    if scalar_dict:
        for scalar_name, scalar_values in scalar_dict.items():
            if isinstance(scalar_values, list):
                if len(scalar_values) > 0 and isinstance(scalar_values[0], np.ndarray):
                    scalar_values = np.concatenate(scalar_values)
                else:
                    scalar_values = np.array(scalar_values)
            else:
                scalar_values = np.asarray(scalar_values)
            # Convert to float64 to ensure VTK compatibility
            scalar_values = scalar_values.astype(np.float64).flatten()
            vtk_array = numpy_to_vtk(scalar_values, deep=1)
            vtk_array.SetName(scalar_name)
            polydata.GetPointData().AddArray(vtk_array)

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_vtk_path)
    writer.SetInputData(polydata)
    writer.SetFileTypeToBinary()  # Use binary format to avoid ASCII parsing issues
    writer.Write()

if __name__ == "__main__":

    streamlines, scalars = load_vtk_streamlines('/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids/derivatives/mcm_tensors_staniz/sub-01001/tracto/sub-01001_bundle-CSTleft_desc-cleaned_model-MCM_tracto.vtk', reshape_scalars=True)

    print(f"Loaded {len(streamlines)} streamlines.")
    for scalar_name, scalar_values in scalars.items():
        print(f"Scalar array '{scalar_name}' has shape {len(scalar_values)}.")

    #Get point index of each streamlines
    point_indices = []
    for i, sl in enumerate(streamlines):
        point_indices.append(list(range(len(sl))))

    point_indices = np.concatenate(point_indices)

    save_vtk(streamlines, '/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids/derivatives/mcm_tensors_staniz/sub-01001/tracto/test_output.vtk', scalar_dict={'point_index': point_indices})