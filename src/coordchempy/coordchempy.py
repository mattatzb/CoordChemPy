# %%
from mendeleev import element
import numpy as np
import py3Dmol
import os
from typing import List, Tuple
import numba
from numba import jit

# Calculate the absolute paths to the data files
file_path_1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'tmQM_X1.xyz'))
file_path_2 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'tmQM_X2.xyz'))

def calculate_distance(coord1, coord2):
    """
    Calculate the Euclidean distance between two 3D coordinates.

    Args:
        coord1 (tuple): The coordinates of the first point.
        coord2 (tuple): The coordinates of the second point.

    Returns:
        float: The Euclidean distance between the two points.
    """
    return np.linalg.norm(np.array(coord1) - np.array(coord2))

def get_covalent_radius(atom_symbol):
    """
    Get the covalent radius of an atom in angstroms using the Mendeleev package.

    Args:
        atom_symbol (str): The symbol of the atom.

    Returns:
        float: The covalent radius of the atom in angstroms.
    """
    radius_pm = element(atom_symbol).covalent_radius
    return radius_pm / 100.0  # Convert pm to Å

def read_lines_around_keyword(keyword, filenames=None, lines_before=1):
    """
    Read lines around a keyword from specified filenames.

    Args:
        keyword (str): The keyword to search for.
        filenames (list, optional): List of filenames to search in. Defaults to ['tmQM_X1.xyz', 'tmQM_X2.xyz'].
        lines_before (int, optional): Number of lines to include before the keyword line. Defaults to 1.

    Returns:
        tuple: A tuple containing the lines around the keyword, the number of lines after the keyword, and the total charge.
    """
    if filenames is None:
        filenames = [file_path_1, file_path_2]
    data = []  # To store the lines around the keyword
    lines_after = 0  # Initialize lines_after to 0
    total_charge = None  # Initialize total_charge
    
    for filename in filenames:
        with open(filename, 'r') as file:
            previous_lines = []  # To store the lines before the keyword
            keyword_found = False
            for line in file:
                if keyword in line:
                    keyword_found = True
                    data.extend(previous_lines[-lines_before:])  # Add lines before the keyword
                    data.append(line.strip())  # Add the line containing the keyword
                    lines_after = int(previous_lines[-1].strip())
                    # Search for the charge value on the line containing the keyword
                    if ' q =' in line:
                        total_charge = line.split('q = ')[1].split()[0].strip()  # Extract the charge value
                    for _ in range(lines_after):
                        next_line = next(file, None)  # Move to the line after the keyword
                        if next_line:
                            data.append(next_line.strip())  # Add lines after the keyword
                    break  # Stop reading the file after finding the keyword
                previous_lines.append(line.strip())
        
        if keyword_found:
            break  # Stop searching in other files if keyword is found
    
    if not keyword_found:
        return "CSD code incorrect or not present in database", 0, None
    
    # Concatenate the lines into a single string
    return '\n'.join(data), lines_after, total_charge

def read_xyz(keyword):  
    """
    Read XYZ data around a keyword.

    Args:
        keyword (str): The keyword to search for.

    Returns:
        tuple: A tuple containing lists of atoms and their coordinates.
    """
    # Call read_lines_around_keyword with default filenames to get the number of atom
    xyz_data, _, _ = read_lines_around_keyword(keyword)
    
    # Split the XYZ data into lines
    xyz_data_lines = xyz_data.split('\n')

    # Parse the XYZ data starting from the line after the CSD code
    atoms = []
    coordinates = []
    for line in xyz_data_lines[2:]:  # Skip the first two lines
        parts = line.split()
        if len(parts) == 4:  # Ensure the line contains atom data
            atoms.append(parts[0])
            coordinates.append((float(parts[1]), float(parts[2]), float(parts[3])))
    
    return atoms, coordinates

def infer_bonds(atoms, coordinates, tolerance=0.4):
    """
    Infer bonds between atoms based on distances and covalent radii.

    Args:
        atoms (list): List of atom symbols.
        coordinates (list): List of atom coordinates.
        tolerance (float, optional): Tolerance factor for bond determination. Defaults to 0.4.

    Returns:
        list: List of tuples representing bonded atom indices.
    """
    bonds = []  # Initialize an empty list to store the bonds.
    
    num_atoms = len(atoms)  # Get the total number of atoms in the molecule.
    
    # Iterate over each pair of atoms to check for potential bonds.
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            atom1, atom2 = atoms[i], atoms[j]  # Get the symbols of the two atoms being considered.
            coord1, coord2 = coordinates[i], coordinates[j]  # Get the 3D coordinates of these atoms.
            
            distance = calculate_distance(coord1, coord2)  # Calculate the Euclidean distance between the two atoms.
            
            # Calculate the maximum allowable distance for a bond, which is the sum of their covalent radii plus a tolerance.
            max_distance = get_covalent_radius(atom1) + get_covalent_radius(atom2) + tolerance
            
            # If the actual distance is less than or equal to the maximum allowable distance, a bond is inferred.
            if distance <= max_distance:
                bonds.append((i, j))  # Add the indices of the bonded atoms as a tuple to the bonds list.
    
    return bonds  # Return the list of inferred bonds.

def find_central_atom(atoms):
    """
    Find the central atom in a molecule based on transition metal criteria.

    Args:
        atoms (list): List of atom symbols.

    Returns:
        tuple: A tuple containing the symbol and index of the central atom.
    """
    # Define a set of atomic numbers corresponding to transition metals.
    transition_metal_atomic_numbers = set(range(21, 31)) | set(range(39, 49)) | set(range(72, 81)) | set(range(104, 113)) | {57, 89}
    
    # Iterate over the list of atoms, with 'i' as the index and 'atom' as the atom symbol.
    for i, atom in enumerate(atoms):
        # Check if the atomic number of the current atom is in the set of transition metal atomic numbers.
        if element(atom).atomic_number in transition_metal_atomic_numbers:
            return atom, i  # Return the atom symbol and its index if it's a transition metal.
    
    # If no transition metal is found, raise a ValueError.
    raise ValueError("No central atom found in the molecule.")

def find_ligands(atoms, coordinates, tolerance=0.4):
    """
    Identify the atoms directly linked to the central atom.

    Args:
        atoms (list): List of atom symbols.
        coordinates (list): List of atom coordinates.
        tolerance (float, optional): Tolerance factor for bond determination. Defaults to 0.4.

    Returns:
        list: List of the atom symbols linked to the central one.
    """
    # Find the central atom and its index in the list of atoms.
    _, central_atom_index = find_central_atom(atoms)
    
    # Infer bonds between atoms based on distances and covalent radii.
    bonds = infer_bonds(atoms, coordinates, tolerance)
    
    # Initialize an empty list to store the symbols of the ligands.
    ligands = []
    
    # Iterate over the list of inferred bonds.
    for bond in bonds:
        # Check if the central atom index is part of the current bond.
        if central_atom_index in bond:
            # Determine the index of the ligand atom in the bond.
            ligand_index = bond[1] if bond[0] == central_atom_index else bond[0]
            # Append the symbol of the ligand atom to the ligands list.
            ligands.append(atoms[ligand_index])
    
    # Return the list of ligand atom symbols.
    return ligands

@jit(nopython=True)
def find_actual_ligand_count(atoms, bonds, central_atom_index):
    """
    Find the actual count of ligands bonded to the central atom.

    Args:
        atoms (list): List of atom symbols.
        coordinates (list): List of atom coordinates.
        bonds (list): List of bonded atom indices.
        central_atom_index (int): Index of the central atom.

    Returns:
        int: The count of unique ligands.
    """
    # Create a graph adjacency list
    graph = {i: [] for i in range(len(atoms))}
    for bond in bonds:
        atom1_index, atom2_index = bond
        graph[atom1_index].append(atom2_index)
        graph[atom2_index].append(atom1_index)

    # Identify atoms directly bonded to the central atom
    directly_bonded = [bond[1] if bond[0] == central_atom_index else bond[0] for bond in bonds if central_atom_index in bond]

    # Function to perform DFS and identify connected ligand atoms
    def dfs(node, visited):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited and neighbor != central_atom_index:
                dfs(neighbor, visited)

    # Find unique ligands using DFS
    unique_ligands = []
    visited = set()
    for atom in directly_bonded:
        if atom not in visited:
            ligand_group = set()
            dfs(atom, ligand_group)
            unique_ligands.append(ligand_group)
            visited |= ligand_group

    return len(unique_ligands)

@jit(nopython=True)
def calculate_angle(coord1, coord2, coord3):
    """
    Calculate the angle between three points in 3D space.

    Args:
        coord1 (tuple): Coordinates of the first point.
        coord2 (tuple): Coordinates of the second point.
        coord3 (tuple): Coordinates of the third point.

    Returns:
        float: The angle in degrees.
    """
    # Convert coordinates to numpy arrays
    v1 = np.array(coord1) - np.array(coord2)
    v2 = np.array(coord3) - np.array(coord2)
    
    # Calculate the dot product and magnitudes of the vectors
    dot_product = np.dot(v1, v2)
    mag_v1 = np.linalg.norm(v1)
    mag_v2 = np.linalg.norm(v2)
    
    # Calculate the cosine of the angle
    cos_theta = dot_product / (mag_v1 * mag_v2)
    
    # Calculate the angle in radians and then convert to degrees
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def determine_geometry(angles, coordination_number):
    """
    Determine the molecular geometry based on bond angles and coordination number.

    Args:
        angles (list): List of bond angles.
        coordination_number (int): The coordination number of the central atom.

    Returns:
        str: The name of the molecular geometry.
    """
    # Check if the coordination number is 2.
    if coordination_number == 2:
        # For coordination number 2, if there is exactly one angle between 160 and 180 degrees, it's linear.
        if len(angles) == 1 and 160 <= angles[0] <= 180:
            return 'linear'

    # Check if the coordination number is 3.
    if coordination_number == 3:
        # For coordination number 3, if there are three angles between 95 and 145 degrees, it's trigonal planar.
        if len(angles) == 3 and all(95 <= angle <= 145 for angle in angles):
            return 'trigonal planar'

    # Check if the coordination number is 4.
    if coordination_number == 4:
        # For coordination number 4, if there are six angles, it could be square planar or tetrahedral.
        if len(angles) == 6:
            # If two angles are between 160 and 180 degrees and the rest are between 65 and 115 degrees, it's square planar.
            if sum(160 <= angle <= 180 for angle in angles) == 2 and all(65 <= angle <= 115 for angle in angles if not (160 <= angle <= 180)):
                return 'square planar'
            # If all angles are between 85 and 135 degrees, it's tetrahedral.
            elif all(85 <= angle <= 135 for angle in angles):
                return 'tetrahedral'

    # Check if the coordination number is 5.
    if coordination_number == 5:
        # For coordination number 5, if there are ten angles, it could be trigonal bipyramidal, square pyramidal, or seesaw.
        if len(angles) == 10:
            # If one angle is between 160 and 180 degrees, three angles are between 95 and 145 degrees, and the rest are between 65 and 115 degrees, it's trigonal bipyramidal.
            if sum(160 <= angle <= 180 for angle in angles) == 1 and sum(95 <= angle <= 145 for angle in angles) == 3 and all(65 <= angle <= 115 for angle in angles if not (95 <= angle <= 145 or 160 <= angle <= 180)):
                return 'trigonal bipyramidal'
            # If any angle is between 55 and 95 degrees and the rest are between 75 and 115 degrees, it's square pyramidal.
            elif any(55 <= angle <= 95 for angle in angles) and all(75 <= angle <= 115 for angle in angles if angle > 95):
                return 'square pyramidal'
            # If there are three distinct angle values, any angle is between 65 and 95 degrees, and the maximum angle is between 105 and 135 degrees, it's seesaw.
            elif len(set([round(angle) for angle in angles])) == 3 and any(65 <= angle <= 95 for angle in angles) and 105 <= max(angles) <= 135:
                return 'seesaw'

    # Check if the coordination number is 6.
    if coordination_number == 6:
        # For coordination number 6, if there are fifteen angles, it could be octahedral.
        if len(angles) == 15:
            # If three angles are between 160 and 180 degrees and the rest are between 65 and 115 degrees, it's octahedral.
            if sum(160 <= angle <= 180 for angle in angles) == 3 and all(65 <= angle <= 115 for angle in angles if not (160 <= angle <= 180)):
                return 'octahedral'

    # If no known geometry matches, return 'unknown'.
    return 'unknown'

def calculate_angles_and_geometry(atoms, coordinates):
    """
    Calculate bond angles and determine molecular geometry.

    Args:
        atoms (list): List of atom symbols.
        coordinates (list): List of atom coordinates.

    Returns:
        tuple: A tuple containing lists of bond angles and the name of the molecular geometry.
    """
    # Find the central atom and its index in the molecule.
    _, central_atom_index = find_central_atom(atoms)

    # Infer the bonds between atoms based on their coordinates and covalent radii.
    bonds = infer_bonds(atoms, coordinates)

    # Identify the indices of the ligands directly bonded to the central atom.
    ligand_indices = [i for i, atom in enumerate(atoms) if (central_atom_index, i) in bonds or (i, central_atom_index) in bonds]

    # Initialize an empty list to store the bond angles.
    angles = []

    # Calculate bond angles between each pair of ligands.
    for i in range(len(ligand_indices)):
        for j in range(i + 1, len(ligand_indices)):
            # Calculate the bond angle between ligand i, the central atom, and ligand j.
            angle = calculate_angle(coordinates[ligand_indices[i]], coordinates[central_atom_index], coordinates[ligand_indices[j]])
            # Append the calculated angle to the angles list.
            angles.append(angle)

    # Determine the coordination number based on the number of ligands.
    coordination_number = len(ligand_indices)

    # Determine the molecular geometry based on the calculated bond angles and coordination number.
    geometry = determine_geometry(angles, coordination_number)

    # Return the list of bond angles and the name of the molecular geometry.
    return angles, geometry

def cn(keyword):
    """
    Return the coordination number

    Args:
        keyword (str): The keyword used to identify the molecule.
    
    Returns:
        int : the coordination number
    """
    atoms, coordinates = read_xyz(keyword)
    cn = len(find_ligands(atoms, coordinates))
    return cn

def charge(keyword):
    """
    Return the total charge of the coordinate compound

    Args:
        keyword (str): The keyword used to identify the molecule.
    
    Returns:
        int : the charge of the coordinate compound
    """
    _, _, total_charge = read_lines_around_keyword(keyword)
    return int(total_charge)

def visualize_label(keyword):
    """
    Visualize the molecule specified by the keyword with the label of the atoms.

    Args:
        keyword (str): The keyword used to identify the molecule.
    """
    atoms, _ = read_xyz(keyword)
    xyz_data, _, _ = read_lines_around_keyword(keyword)

    # Initialize a viewer
    viewer = py3Dmol.view()

    # Add a model from the .xyz data
    viewer.addModel(xyz_data, 'xyz')

    # Set the style
    viewer.setStyle({'stick': {}})

    # Label each atom
    for i, atom in enumerate(atoms):
        coord = coordinates[i]
        label = f"{atom}"
        viewer.addLabel(label, {'position': {'x': coord[0], 'y': coord[1], 'z': coord[2]},
                'backgroundColor': 'rgba(255, 255, 255, 0.5)',  # Light background
                'fontColor': 'black',
                'fontSize': 10,  # Smaller font size
                'padding': 0,
                'borderThickness': 0})

    # Center the molecule
    viewer.zoomTo()

    # Render the viewer
    viewer.show()

def visualize (keyword):
    """
    Visualize the molecule specified by the keyword.

    Args:
        keyword (str): The keyword used to identify the molecule.
    """
    atoms, _ = read_xyz(keyword)
    xyz_data, _, _ = read_lines_around_keyword(keyword)

    # Initialize a viewer
    viewer = py3Dmol.view()

    # Add a model from the .xyz data
    viewer.addModel(xyz_data, 'xyz')

    # Set the style
    viewer.setStyle({'stick': {}})

    # Center the molecule
    viewer.zoomTo()

    # Render the viewer
    viewer.show()

def visualize_all_data(keyword):
    """
    Visualize the molecule specified by the keyword and print its characteristics.

    Args:
        keyword (str): The keyword used to identify the molecule.
    """
    atoms, coordinates = read_xyz(keyword)
    cn = len(find_ligands(atoms, coordinates))
    bonds = infer_bonds(atoms, coordinates)
    central_atom_symbol, central_atom_index = find_central_atom(atoms)
    ligand_counts = find_actual_ligand_count(atoms, bonds, central_atom_index)
    angles, geometry = calculate_angles_and_geometry(atoms, coordinates)
    xyz_data, _, total_charge = read_lines_around_keyword(keyword)
    
    # Print the symbol of the central atom
    print(f"The symbol of the central atom is {central_atom_symbol}.")

    # Print the total charge if available
    print(f"The total charge of the complex is {total_charge}.")

    # Print the coordination number and the number of ligands
    if cn != ligand_counts:
        print(f"The coordination number of the transition metal is {cn} and the number of ligands is {ligand_counts}.")
    if cn == ligand_counts:
        print(f"The coordination number of the transition metal and the number of ligands is {cn}.")

    # Print the angles and geometry
    print(f"The angles measure {angles} and the geometry of the molecule is {geometry}.")

    # Initialize a viewer
    viewer = py3Dmol.view()

    # Add a model from the .xyz data
    viewer.addModel(xyz_data, 'xyz')

    # Set the style
    viewer.setStyle({'stick': {}})

    # Label each atom
    for i, atom in enumerate(atoms):
        coord = coordinates[i]
        label = f"{atom}"
        viewer.addLabel(label, {'position': {'x': coord[0], 'y': coord[1], 'z': coord[2]},
                'backgroundColor': 'rgba(255, 255, 255, 0.5)',  # Light background
                'fontColor': 'black',
                'fontSize': 10,  # Smaller font size
                'padding': 0,
                'borderThickness': 0})

    # Center the molecule
    viewer.zoomTo()

    # Render the viewer
    viewer.show()
