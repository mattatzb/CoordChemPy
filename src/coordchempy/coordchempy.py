# %%
from mendeleev import element
import numpy as np
import py3Dmol
import os
from typing import List, Tuple

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
    xyz_data, _, total_charge = read_lines_around_keyword(keyword)
    
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
    bonds = []
    num_atoms = len(atoms)
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            atom1, atom2 = atoms[i], atoms[j]
            coord1, coord2 = coordinates[i], coordinates[j]
            distance = calculate_distance(coord1, coord2)
            max_distance = get_covalent_radius(atom1) + get_covalent_radius(atom2) + tolerance
            if distance <= max_distance:
                bonds.append((i, j))
    return bonds

def find_central_atom(atoms):
    """
    Find the central atom in a molecule based on transition metal criteria.

    Args:
        atoms (list): List of atom symbols.

    Returns:
        tuple: A tuple containing the symbol and index of the central atom.
    """
    transition_metal_atomic_numbers = set(range(21, 31)) | set(range(39, 49)) | set(range(72, 81)) | set(range(104, 113)) | {57, 89}
    for i, atom in enumerate(atoms):
        if element(atom).atomic_number in transition_metal_atomic_numbers:
            return atom, i
    raise ValueError("No central atom found in the molecule.")

def find_ligands(atoms, coordinates, tolerance=0.4):
    """
    Identify the atoms directly linked to the central atom.

    Args:
        atoms (list): List of atom symbols.
        coordinates (list): List of atom coordinates.
        tolerance (float, optional): Tolerance factor for bond determination. Defaults to 0.4.

    Returns:
        list: List of the  atom symbol linked to the central one.
    """
    _, central_atom_index = find_central_atom(atoms)
    bonds = infer_bonds(atoms, coordinates, tolerance)
    ligands = []
    for bond in bonds:
        if central_atom_index in bond:
            ligand_index = bond[1] if bond[0] == central_atom_index else bond[0]
            ligands.append(atoms[ligand_index])
    return ligands

def find_actual_ligand_count(atoms, coordinates, bonds, central_atom_index):
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
    if coordination_number == 2:
        if len(angles) == 1 and 160 <= angles[0] <= 180:
            return 'linear'

    if coordination_number == 3:
        if len(angles) == 3 and all(95 <= angle <= 145 for angle in angles):
            return 'trigonal planar'

    if coordination_number == 4:
        if len(angles) == 6:
            if sum(160 <= angle <= 180 for angle in angles) == 2 and all(65 <= angle <= 115 for angle in angles if not (160 <= angle <= 180)):
                return 'square planar'
            elif all(85 <= angle <= 135 for angle in angles):
                return 'tetrahedral'

    if coordination_number == 5:
        if len(angles) == 10:
            if sum(160 <= angle <= 180 for angle in angles) == 1 and sum(95 <= angle <= 145 for angle in angles) == 3 and all(65 <= angle <= 115 for angle in angles if not (95 <= angle <= 145 or 160 <= angle <= 180)):
                return 'trigonal bipyramidal'
            elif any(55 <= angle <= 95 for angle in angles) and all(75 <= angle <= 115 for angle in angles if angle > 95):
                return 'square pyramidal'
            elif len(set([round(angle) for angle in angles])) == 3 and any(65 <= angle <= 95 for angle in angles) and 105 <= max(angles) <= 135:
                return 'seesaw'

    if coordination_number == 6:
        if len(angles) == 15:
            if sum(160 <= angle <= 180 for angle in angles) == 3 and all(65 <= angle <= 115 for angle in angles if not (160 <= angle <= 180)):
                return 'octahedral'

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
    _, central_atom_index = find_central_atom(atoms)
    bonds = infer_bonds(atoms, coordinates)

    ligand_indices = [i for i, atom in enumerate(atoms) if (central_atom_index, i) in bonds or (i, central_atom_index) in bonds]

    angles = []
    for i in range(len(ligand_indices)):
        for j in range(i + 1, len(ligand_indices)):
            angle = calculate_angle(coordinates[ligand_indices[i]], coordinates[central_atom_index],  coordinates[ligand_indices[j]])
            angles.append(angle)

    coordination_number = len(ligand_indices)

    geometry = determine_geometry(angles, coordination_number)
        
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
    ligand_counts = find_actual_ligand_count(atoms, coordinates, bonds, central_atom_index)
    angles, geometry = calculate_angles_and_geometry(atoms, coordinates)
    xyz_data, lines_after, total_charge = read_lines_around_keyword(keyword)
    
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
