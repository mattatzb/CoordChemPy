"""Allows to visualize a coordination compound and to give some of its characteristics. The CSD code of the wanted complex is needed and it has to be in the data base used.."""

from __future__ import annotations

__version__ = "0.0.1"

from .coordchempy import (
    calculate_distance,
    get_covalent_radius,
    read_lines_around_keyword,
    read_xyz,
    infer_bonds,
    find_central_atom,
    find_ligands,
    find_actual_ligand_count,
    calculate_angle,
    determine_geometry,
    calculate_angles_and_geometry,
    cn,
    charge,
    visualize_label,
    visualize,
    visualize_all_data
)

__all__ = [
    'calculate_distance',
    'get_covalent_radius',
    'read_lines_around_keyword',
    'read_xyz',
    'infer_bonds',
    'find_central_atom',
    'find_ligands',
    'find_actual_ligand_count',
    'calculate_angle',
    'determine_geometry',
    'calculate_angles_and_geometry',
    'cn',
    'charge',
    'visualize_label',
    'visualize',
    'visualize_all_data'
]
