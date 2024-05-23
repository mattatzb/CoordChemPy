from coordchempy import (
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

import pytest

# Data expected 
TEST_DATA = {
    'KUMBAX': {
        'expected_charge': 0,
        'expected_cn': 4,
        'expected_coordinates': [
            [9.93116755586730, 7.63893951636350, 14.06242636466981],
            [11.89023979210495, 8.26809372013789, 14.78070769423852],
            [10.69692689636514, 6.82343000073777, 12.20142095795944],
            [9.55873451491081, 5.72308113381907, 15.01791478924112],
            [9.18745152381335, 10.43497128550873, 13.64374044262446],
            [7.03097950471707, 8.23227786974568, 13.12728192739099],
            [9.25234511169237, 8.57811694985959, 16.15552678744744],
            [8.45025231359200, 8.65374882645531, 10.68451538856048],
            [8.60945712626762, 8.80522489738401, 13.57756428776594],
            [9.85087491681459, 10.76068036339107, 15.41221909381012],
            [9.83870919713564, 9.80740991922912, 16.43682184293685],
            [10.39251247437480, 10.09666432651581, 17.67839437209156],
            [10.41253609780200, 9.35948418163370, 18.46503923730392],
            [10.96117656051414, 11.33722591444766, 17.91502738832495],
            [11.39547361532039, 11.54450104302724, 18.88140230163064],
            [10.97579487791026, 12.29498861999595, 16.91968906531651],
            [11.41784729142005, 13.26297555206184, 17.09996252843956],
            [10.42691139657310, 11.99459916454058, 15.68575587411511],
            [10.44271249909679, 12.73806353568821, 14.90309322465675],
            [8.88630601479674, 7.78193179057208, 17.27677789361978],
            [9.76203721597875, 7.31082085399500, 17.73613244226454],
            [8.22654926977827, 7.00014991893794, 16.89577620018798],
            [8.35246360836880, 8.38848596460022, 18.01547796084803],
            [7.86230930553524, 11.75613207407416, 13.27664820409585],
            [7.02058338513370, 11.68026452003058, 13.95585135033263],
            [7.47980596382085, 11.64790825644465, 12.26862767364380],
            [8.27569543278478, 12.75377215030569, 13.37233517109020],
            [10.67138251410527, 10.60365401154867, 12.43434820171340],
            [11.09231062485643, 9.62113243209324, 12.20796664297703],
            [11.46443114936509, 11.21349773044500, 12.85897002696793],
            [10.36976537800108, 11.06103867120100, 11.49691128419148],
            [5.85036373919955, 8.46269004470539, 14.61967029583179],
            [5.72591679833245, 9.51030157446582, 14.87358907868362],
            [6.24602722812008, 7.95505241646530, 15.49541133981302],
            [4.86895324727909, 8.04917781394758, 14.41433969765355],
            [7.16822998136751, 6.37097809722052, 12.70704306543974],
            [8.04798094304137, 6.17610398631367, 12.09436096713196],
            [6.29841545924022, 6.03142253907156, 12.15216342060753],
            [7.24021678932661, 5.76470237599724, 13.60886759041269],
            [6.39821561035415, 9.19794888520013, 11.62631572771209],
            [7.25949158868288, 9.28995548037147, 10.52485212240314],
            [6.87769191503973, 9.99060982784414, 9.38786381152245],
            [7.53147740130160, 10.06954951103202, 8.53323791967030],
            [5.63345148006191, 10.60042767242617, 9.34587710794095],
            [5.34139757112859, 11.14355256597967, 8.45897899065916],
            [4.77238433114735, 10.51595232562087, 10.42355702127492],
            [3.80390395381351, 10.99114129290503, 10.38333181597284],
            [5.16224799383735, 9.81621832382147, 11.55479762025347],
            [4.49349146106097, 9.74714119072590, 12.39865154498646],
            [9.34681398977060, 8.61193797256253, 9.59704706790516],
            [8.88929509647994, 8.13508117851874, 8.72415344699606],
            [10.19006791173813, 8.01346166787290, 9.93753460896899],
            [9.69392238084811, 9.61392806213496, 9.32462911770203]
        ],
        'expected_geometry': 'tetrahedral',  # Placeholder for expected geometry, fill this as needed
        'atoms': [
            'Ti', 'Cl', 'Cl', 'Cl', 'Si', 'Si', 'O', 'O', 'N', 'C', 'C', 'C', 'H', 'C', 'H', 'C', 'H', 'C', 'H', 'C',
            'H', 'H', 'H', 'C', 'H', 'H', 'H', 'C', 'H', 'H', 'H', 'C', 'H', 'H', 'H', 'C', 'H', 'H', 'H', 'C', 'C',
            'C', 'H', 'C', 'H', 'C', 'H', 'C', 'H', 'C', 'H', 'C', 'H'
        ]
    }
}

# Test the function

# Test calculate_distance function
def test_calculate_distance():
    # Define test cases and expected results
    coord1 = (0, 0, 0)
    coord2 = (3, 4, 0)
    expected_distance = 5.0
    
    # Call the function
    distance = calculate_distance(coord1, coord2)
    
    # Check the result
    assert distance == expected_distance, f"Expected distance: {expected_distance}, Got: {distance}"

# Test get_covalent_radius function
def test_get_covalent_radius():
    # Define test cases and expected results
    atom_symbol = 'C'
    expected_radius = 0.75
    
    # Call the function
    radius = get_covalent_radius(atom_symbol)
    
    # Check the result
    assert radius == expected_radius, f"Expected radius: {expected_radius}, Got: {radius}"

# Test read_lines_around_keyword function
def test_read_lines_around_keyword():
    # Define test cases and expected results
    keyword = 'KUMBAX'
    expected_output = ('53\nCSD_code = KUMBAX | q = 0 | S = 0 | Stoichiometry = C18H26Cl3NO2Si2Ti | MND = 4\nTi        9.93116755586730    7.63893951636350   14.06242636466981\nCl       11.89023979210495    8.26809372013789   14.78070769423852\nCl       10.69692689636514    6.82343000073777   12.20142095795944\nCl        9.55873451491081    5.72308113381907   15.01791478924112\nSi        9.18745152381335   10.43497128550873   13.64374044262446\nSi        7.03097950471707    8.23227786974568   13.12728192739099\nO         9.25234511169237    8.57811694985959   16.15552678744744\nO         8.45025231359200    8.65374882645531   10.68451538856048\nN         8.60945712626762    8.80522489738401   13.57756428776594\nC         9.85087491681459   10.76068036339107   15.41221909381012\nC         9.83870919713564    9.80740991922912   16.43682184293685\nC        10.39251247437480   10.09666432651581   17.67839437209156\nH        10.41253609780200    9.35948418163370   18.46503923730392\nC        10.96117656051414   11.33722591444766   17.91502738832495\nH        11.39547361532039   11.54450104302724   18.88140230163064\nC        10.97579487791026   12.29498861999595   16.91968906531651\nH        11.41784729142005   13.26297555206184   17.09996252843956\nC        10.42691139657310   11.99459916454058   15.68575587411511\nH        10.44271249909679   12.73806353568821   14.90309322465675\nC         8.88630601479674    7.78193179057208   17.27677789361978\nH         9.76203721597875    7.31082085399500   17.73613244226454\nH         8.22654926977827    7.00014991893794   16.89577620018798\nH         8.35246360836880    8.38848596460022   18.01547796084803\nC         7.86230930553524   11.75613207407416   13.27664820409585\nH         7.02058338513370   11.68026452003058   13.95585135033263\nH         7.47980596382085   11.64790825644465   12.26862767364380\nH         8.27569543278478   12.75377215030569   13.37233517109020\nC        10.67138251410527   10.60365401154867   12.43434820171340\nH        11.09231062485643    9.62113243209324   12.20796664297703\nH        11.46443114936509   11.21349773044500   12.85897002696793\nH        10.36976537800108   11.06103867120100   11.49691128419148\nC         5.85036373919955    8.46269004470539   14.61967029583179\nH         5.72591679833245    9.51030157446582   14.87358907868362\nH         6.24602722812008    7.95505241646530   15.49541133981302\nH         4.86895324727909    8.04917781394758   14.41433969765355\nC         7.16822998136751    6.37097809722052   12.70704306543974\nH         8.04798094304137    6.17610398631367   12.09436096713196\nH         6.29841545924022    6.03142253907156   12.15216342060753\nH         7.24021678932661    5.76470237599724   13.60886759041269\nC         6.39821561035415    9.19794888520013   11.62631572771209\nC         7.25949158868288    9.28995548037147   10.52485212240314\nC         6.87769191503973    9.99060982784414    9.38786381152245\nH         7.53147740130160   10.06954951103202    8.53323791967030\nC         5.63345148006191   10.60042767242617    9.34587710794095\nH         5.34139757112859   11.14355256597967    8.45897899065916\nC         4.77238433114735   10.51595232562087   10.42355702127492\nH         3.80390395381351   10.99114129290503   10.38333181597284\nC         5.16224799383735    9.81621832382147   11.55479762025347\nH         4.49349146106097    9.74714119072590   12.39865154498646\nC         9.34681398977060    8.61193797256253    9.59704706790516\nH         8.88929509647994    8.13508117851874    8.72415344699606\nH        10.19006791173813    8.01346166787290    9.93753460896899\nH         9.69392238084811    9.61392806213496    9.32462911770203',
    53,
    '0')
    
    # Call the function
    output = read_lines_around_keyword(keyword)
    
    # Check the result
    assert output == expected_output, f"Expected output: {expected_output}, Got: {output}"

# Test read_xyz function
def test_read_xyz():
    # Define test cases and expected results
    keyword = 'KUMBAX'
    expected_output = (TEST_DATA['atoms'], TEST_DATA['expected_coordinates'])
    
    # Call the function
    output = read_xyz(keyword)
    
    # Check the result
    assert output == expected_output, f"Expected output: {expected_output}, Got: {output}"

# Test infer_bonds function
def test_infer_bonds():
    # Define test cases and expected results
    atoms = TEST_DATA['atoms']
    coordinates = TEST_DATA['expected_coordinates']
    expected_output = [(0, 1), (1, 2)]
    
    # Call the function
    output = infer_bonds(atoms, coordinates)
    
    # Check the result
    assert output == expected_output, f"Expected output: {expected_output}, Got: {output}"

# Test find_central_atom function
def test_find_central_atom():
    # Define test cases and expected results
    atoms = TEST_DATA['atoms']
    expected_output = ('Ti', 0)
    
    # Call the function
    output = find_central_atom(atoms)
    
    # Check the result
    assert output == expected_output, f"Expected output: {expected_output}, Got: {output}"

# Test find_ligands function
def test_find_ligands():
    # Define test cases and expected results
    atoms = TEST_DATA['atoms']
    coordinates = TEST_DATA['expected_coordinates']
    expected_output = ['C', 'C', 'C']
    
    # Call the function
    output = find_ligands(atoms, coordinates)
    
    # Check the result
    assert output == expected_output, f"Expected output: {expected_output}, Got: {output}"

# Test find_actual_ligand_count function
def test_find_actual_ligand_count():
    # Define test cases and expected results
    atoms = TEST_DATA['atoms']
    coordinates = TEST_DATA['expected_coordinates']
    bonds = [(0, 1), (0, 2), (0, 3)]
    central_atom_index = 0
    expected_output = 3
    
    # Call the function
    output = find_actual_ligand_count(atoms, coordinates, bonds, central_atom_index)
    
    # Check the result
    assert output == expected_output, f"Expected output: {expected_output}, Got: {output}"

# Test calculate_angle function
def test_calculate_angle():
    # Define test cases and expected results
    coord1 = (0, 0, 0)
    coord2 = (1, 0, 0)
    coord3 = (1, 1, 0)
    expected_angle = 90.0
    
    # Call the function
    angle = calculate_angle(coord1, coord2, coord3)
    
    # Check the result
    assert angle == expected_angle, f"Expected angle: {expected_angle}, Got: {angle}"

# Test determine_geometry function
def test_determine_geometry():
    # Define test cases and expected results
    angles = [90, 90, 90]
    coordination_number = 4
    expected_geometry = 'tetrahedral'
    
    # Call the function
    geometry = determine_geometry(angles, coordination_number)
    
    # Check the result
    assert geometry == expected_geometry, f"Expected geometry: {expected_geometry}, Got: {geometry}"

# Test calculate_angles_and_geometry function
def test_calculate_angles_and_geometry():
    # Define test cases and expected results
    atoms = ['C', 'C', 'C', 'C']
    coordinates = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)]
    expected_angles = [90, 90, 90]
    expected_geometry = 'tetrahedral'
    
    # Call the function
    angles, geometry = calculate_angles_and_geometry(atoms, coordinates)
    
    # Check the result
    assert angles == expected_angles, f"Expected angles: {expected_angles}, Got: {angles}"
    assert geometry == expected_geometry, f"Expected geometry: {expected_geometry}, Got: {geometry}"

# Test cn function
def test_cn():
    # Define test cases and expected results
    keyword = 'KUMBAX'
    expected_output = 4
    
    # Call the function
    output = cn(keyword)
    
    # Check the result
    assert output == expected_output, f"Expected output: {expected_output}, Got: {output}"

# Test charge function
def test_charge():
    # Define test cases and expected results
    keyword = 'KUMBAX'
    expected_output = 0
    
    # Call the function
    output = charge(keyword)
    
