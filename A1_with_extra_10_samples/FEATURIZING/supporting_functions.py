import pandas as pd
import seaborn as sns
import numpy as np
from rdkit import Chem
from rdkit.Chem import PeriodicTable
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator as rdf
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import keras
from collections import defaultdict
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from sklearn.model_selection import train_test_split
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFreeSASA, rdqueries
from rdkit.Chem import rdchem


BT = rdchem.BondType
pt = Chem.GetPeriodicTable()

BOND_ORDER = {
    BT.UNSPECIFIED:    float("nan"),  # unknown — sentinel, do NOT call it 0
    BT.SINGLE:         1.0,
    BT.DOUBLE:         2.0,
    BT.TRIPLE:         3.0,
    BT.QUADRUPLE:      4.0,   # Re–Re in [Re2Cl8]2-, Cr–Cr in [Cr2(OAc)4]
    BT.QUINTUPLE:      5.0,   # Cr–Cr
    BT.HEXTUPLE:       6.0,   # Mo–Mo, W–W
    BT.ONEANDAHALF:    1.5,
    BT.TWOANDAHALF:    2.5,
    BT.THREEANDAHALF:  3.5,
    BT.FOURANDAHALF:   4.5,
    BT.FIVEANDAHALF:   5.5,
    BT.AROMATIC:       1.5,   # delocalized — continuous proxy only
    BT.IONIC:          0.0,   # edge exists, no covalent order
    BT.HYDROGEN:       0.0,   # H-bond / agostic
    BT.THREECENTER:    0.5,   # 3c–2e (B–H–B, agostic) ≈ 0.5 per pair
    BT.DATIVEONE:      0.5,   # one-electron dative
    BT.DATIVE:         1.0,   # two-electron dative (classic coordination)
    BT.DATIVEL:        1.0,   # L → M
    BT.DATIVER:        1.0,   # M → L
    BT.OTHER:          float("nan"),
    BT.ZERO:           0.0,   # explicit "no order" coordination edge
}

def get_borrylation_candidates_from_smiles(smiles_):
    mol = Chem.MolFromSmiles(smiles_)
    #mol = Chem.AddHs(mol)
    mask_atoms = np.zeros(len(mol.GetAtoms()))
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6 and atom.GetTotalNumHs() > 0:
            mask_atoms[atom.GetIdx()] = 1
    return mask_atoms

def create_classes_from_smiles(smiles_, mask_atoms_, mapno_ = -2):
    mol = Chem.MolFromSmiles(smiles_)
    ranks = np.array(Chem.CanonicalRankAtoms(
        mol,
        breakTies=False,
        includeChirality=True,
        includeIsotopes=True,
        includeAtomMaps=False,
    ), dtype=np.int_)
    ranks += 2 #RANKS start from 0
    ranks = ranks*mask_atoms_ #Only consider hydrogenated carbons - will be supplied in mask. However, it might be that a reactin
    ranks = ranks - 1 #non-reacting: -1
    classes = defaultdict(list)
    reactive_atom_class = -1 #if we don't find a reacting atom class - return -1
    for atom_idx, r in enumerate(ranks):
        if r >= 0:
            if mol.GetAtomWithIdx(atom_idx).GetAtomMapNum() == mapno_:
                reactive_atom_class = r
            classes[r].append(atom_idx)
    if mapno_ != -2:
        return ranks, classes, reactive_atom_class
    else:
        return ranks, classes

def get_above_below_ds(mol_, positions_, atom_idx_, reagent_type_ = 'SUBSTRATE'):
    nbrs_H = []
    nbrs_not_H = []
    for nbr in mol_.GetAtomWithIdx(atom_idx_).GetNeighbors():
        nbr_id = nbr.GetIdx()
        if nbr.GetAtomicNum() == 1:
            nbrs_H.append(nbr_id)
        elif nbr.GetAtomicNum() != 1:
            nbrs_not_H.append(nbr_id)

    if reagent_type_ == 'SUBSTRATE':
        #the vectors to construct a plane are; CH-bond, C-(C,N,O,P...) bond, and the vector that is parallel to them
        positions_H = positions_[nbrs_H]
        vector_H = positions_H[0, :]
        if len(nbrs_not_H) != 0:
            positions_not_H = positions_[nbrs_not_H]
            position_orthogonal = positions_not_H[0]
        else:
            position_orthogonal = positions_H[1, :]
    else: #for the catalysts and bases... Vector_H will not actually contain H, but it might be renamed in the future
        
        if len(nbrs_not_H) != 0:
            if len(nbrs_not_H) > 1:
                positions_not_H = positions_[nbrs_not_H]
                vector_H = positions_not_H[0]
                position_orthogonal = positions_not_H[1]
            else:
                positions_not_H = positions_[nbrs_not_H]
                vector_H = positions_not_H[0]
                position_orthogonal = positions_not_H[0] #this is corrected later for a random vector
        else:
            vector_H = positions_H[0, :]
            position_orthogonal = positions_H[1, :]



    orthogonal_vector_1 = np.cross(vector_H, position_orthogonal)
    if np.sum(orthogonal_vector_1) == 0:
        orthogonal_vector_1 = np.cross(vector_H, np.array([1,1,1])) #for alkynes, the orthogonal vector will be 0 as C#C-H are on the same line. Hence, we just make a random orthogonal vector.
    orthogonal_vector_2 = np.cross(vector_H, orthogonal_vector_1)

    #ds come from place equation ax + by + cz + d = 0. If d > 0, the point is above the plane, < 0 - below, 0 - in the plane. 
    #atoms that belong to the plane will count into both above and below. This will prevent devision by 0 in case we decide to calculate molar fractions.
    d1 = positions_.dot(vector_H) 
    d2 = positions_.dot(orthogonal_vector_1) 
    d3 = positions_.dot(orthogonal_vector_2)

    return d1, d2, d3

def get_steric_description(mol_smiles_, atom_idx_, r_cutoff_ = 6, reagent_type_ = 'SUBSTRATE', 
                           no_above_below = False, conf_ = None):
    mol_no_Hs = Chem.MolFromSmiles(mol_smiles_)
    mol = Chem.AddHs(mol_no_Hs)
    if conf_ is None:
        AllChem.EmbedMolecule(mol, randomSeed = 42)
        conf = mol.GetConformer()
    else:
        conf = conf_
    atom_idx_ = int(atom_idx_)
    
    positions = conf.GetPositions()
    positions = positions - positions[atom_idx_, :]
    all_vdW_radii = np.array([pt.GetRvdw(atom.GetAtomicNum()) for atom in mol.GetAtoms()])
    all_vdW_volumes = np.pi*4/3*all_vdW_radii**3

    distances = np.sqrt((positions**2).dot(np.array([1,1,1])))
    distances_within_cutoff_mask = distances < (r_cutoff_ - all_vdW_radii/2)

    mol_copy = mol.__copy__() #to not to modify original molecule
    atoms_copy = mol_copy.GetAtoms()
    idxs = np.where(distances_within_cutoff_mask == 1)[0]
    for idx in idxs:
        atom = mol.GetAtomWithIdx(int(idx))
        atom.SetBoolProp(f'closer_than_{r_cutoff_:.1f}A', True)

    radii = rdFreeSASA.classifyAtoms(mol)

    # Query atoms carrying the selection property
    query_atom = rdqueries.HasPropQueryAtom(f'closer_than_{r_cutoff_:.1f}A')

    # Calculate SASA for the selected atoms
    sasa_within_rc = rdFreeSASA.CalcSASA(
        mol,
        radii=radii,
        query=query_atom
    )

    distances_borderline_mask = np.logical_not(distances_within_cutoff_mask) & (distances < r_cutoff_)
    effective_volumes = all_vdW_volumes*(distances_within_cutoff_mask + 0.5*distances_borderline_mask)
    v_tot =  np.pi *4/3*r_cutoff_**3

    free_volume_fraction = np.sum(effective_volumes)/v_tot

    if no_above_below:
        return sasa_within_rc, free_volume_fraction


    correction_factor = (np.pi*4/3*pt.GetRvdw(mol.GetAtomWithIdx(atom_idx_).GetAtomicNum())**3)/v_tot/2
    d1, d2, d3 = get_above_below_ds(mol, positions, atom_idx_, reagent_type_)

    vfrac_1_above = np.sum(effective_volumes * (d1 >= 0))/v_tot - correction_factor
    vfrac_1_below = np.sum(effective_volumes * (d1 <= 0))/v_tot - correction_factor

    vfrac_2_above = np.sum(effective_volumes * (d2 >= 0))/v_tot - correction_factor
    vfrac_2_below = np.sum(effective_volumes * (d2 <= 0))/v_tot - correction_factor

    vfrac_3_above = np.sum(effective_volumes * (d3 >= 0))/v_tot - correction_factor
    vfrac_3_below = np.sum(effective_volumes * (d3 <= 0))/v_tot - correction_factor
    
    


    output_dict = {
        f'{reagent_type_}_free_vol_frac_total_{r_cutoff_:.1f}A' : free_volume_fraction,

        f'{reagent_type_}_free_vol_frac_1_above_{r_cutoff_:.1f}A': vfrac_1_above,
        f'{reagent_type_}_free_vol_frac_1_below_{r_cutoff_:.1f}A': vfrac_1_below,

        f'{reagent_type_}_free_vol_frac_2_above_{r_cutoff_:.1f}A': vfrac_2_above,
        f'{reagent_type_}_free_vol_frac_2_below_{r_cutoff_:.1f}A': vfrac_2_below,

        f'{reagent_type_}_free_vol_frac_3_above_{r_cutoff_:.1f}A': vfrac_3_above,
        f'{reagent_type_}_free_vol_frac_3_below_{r_cutoff_:.1f}A': vfrac_3_below,

        f'{reagent_type_}_SASA_within_{r_cutoff_:.1f}A': sasa_within_rc
    }
    '''
    output_dict = {
            f'free_vol_frac_total_{r_cutoff_:.1f}A' : free_volume_fraction,
    
            f'free_vol_frac_1_above_{r_cutoff_:.1f}A': vfrac_1_above,
            f'free_vol_frac_1_below_{r_cutoff_:.1f}A': vfrac_1_below,
    
            f'free_vol_frac_2_above_{r_cutoff_:.1f}A': vfrac_2_above,
            f'free_vol_frac_2_below_{r_cutoff_:.1f}A': vfrac_2_below,
    
            f'free_vol_frac_3_above_{r_cutoff_:.1f}A': vfrac_3_above,
            f'free_vol_frac_3_below_{r_cutoff_:.1f}A': vfrac_3_below,
    
            f'SASA_within_{r_cutoff_:.1f}A': sasa_within_rc
        }'''
    return output_dict

def get_partial_charge(mol_, idx_):
    return mol_.GetAtomWithIdx(idx_).GetPropsAsDict()['_GasteigerCharge']

def get_hybridization(mol_, idx_):
    return mol_.GetAtomWithIdx(idx_).GetHybridization()

def get_connectivity_patterns(smiles_, idx, level_ = 2, reagent_type = 'SUBSTRATE'):
    visited = []
    visited_with_idx = []
    mol = Chem.MolFromSmiles(smiles_)
    mol= AllChem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed = 42)
    conf = mol.GetConformer()

    to_be_visited = [{'level':0, 'id': idx, 'atomic_no':mol.GetAtomWithIdx(idx).GetAtomicNum(), 'parent':0}]
    
    current = int(to_be_visited[0]['id'])
    current_parent = 0
    current_level = int(to_be_visited[0]['level'])

    important_atoms = ['H', 'C', 'O', 'N', 'P', 'S', 'F', 'Cl', 'Br', 'I', 'B', 'Si']

    levels = np.linspace(1,level_, level_)

    columns_ = []

    for level in levels:
        for imporant_atom in important_atoms:
            columns_.append(f'lvl{int(level)}_{imporant_atom}')

    encoded_atomic_counts_and_MPCs = pd.DataFrame(columns=columns_, data=np.zeros([1, len(columns_)]))
    while(len(to_be_visited)!= 0 and current_level < level+1):
        if current_level != 0:
            if mol.GetAtomWithIdx(current).GetSymbol() in important_atoms:
                bond_order = BOND_ORDER[mol.GetBondBetweenAtoms(current, current_parent).GetBondType()]
                encoded_atomic_counts_and_MPCs.loc[0, f'lvl{current_level}_{mol.GetAtomWithIdx(current).GetSymbol()}'] += bond_order
        
        visited.append(current)
        for nbr in mol.GetAtomWithIdx(current).GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx not in visited:     
                to_be_visited.append({'level':current_level+1, 'id': nbr_idx, 
                                      'atomic_no':mol.GetAtomWithIdx(nbr_idx).GetAtomicNum(),
                                      'parent': current})
        
        visited_with_idx.append(to_be_visited[0].copy())
        to_be_visited.pop(0)

        if len(to_be_visited)!= 0:
            current = int(to_be_visited[0]['id'])
            current_level = int(to_be_visited[0]['level'])
            current_parent = int(to_be_visited[0]['parent'])

    _ = AllChem.ComputeGasteigerCharges(mol)
    visited_with_idx = pd.DataFrame(visited_with_idx)
    visited_with_idx = visited_with_idx[visited_with_idx['atomic_no'] != 1]
    visited_with_idx['partial_charge'] = visited_with_idx['id'].apply(lambda x:get_partial_charge(mol, x))
    visited_with_idx['hybridization'] = visited_with_idx['id'].apply(lambda x:get_hybridization(mol, x))
    #visited_with_idx[['sasa_6A', 'free_vol_frac_6A']] = visited_with_idx['id'].apply(lambda x:get_steric_description(smiles_, x, 6, reagent_type, True), index=['sasa_6A', 'free_vol_frac_6A'])
    #visited_with_idx[['sasa_6A', 'free_vol_frac_6A']] = visited_with_idx['id'].apply(
    #    lambda x: pd.Series(get_steric_description(smiles_, x, 6, reagent_type, True), index=['sasa_6A', 'free_vol_frac_6A']))


    levels = np.linspace(0,level_, level_+1)
    MPCs = visited_with_idx.groupby('level').mean()['partial_charge']
    MaxPCs = visited_with_idx.groupby('level').max()['partial_charge']
    MinPCs = visited_with_idx.groupby('level').min()['partial_charge']
    STDPCs = visited_with_idx.groupby('level').std()['partial_charge']
    MHys = visited_with_idx.groupby('level').mean()['hybridization']
    #MSASA = visited_with_idx.groupby('level').mean()['sasa_6A']
    #MFVF = visited_with_idx.groupby('level').mean()['free_vol_frac_6A']
    lvls_present = list(visited_with_idx.groupby('level').mean()['partial_charge'].keys())

    for level in levels:
        if int(level) in lvls_present:
            encoded_atomic_counts_and_MPCs[f'lvl{int(level)}_Mean_Partial_Charge'] = MPCs[int(level)]
            encoded_atomic_counts_and_MPCs[f'lvl{int(level)}_Max_Partial_Charge'] = MaxPCs[int(level)]
            encoded_atomic_counts_and_MPCs[f'lvl{int(level)}_Min_Partial_Charge'] = MinPCs[int(level)]
            encoded_atomic_counts_and_MPCs[f'lvl{int(level)}_STD_Partial_Charge'] = STDPCs[int(level)]
            encoded_atomic_counts_and_MPCs[f'lvl{int(level)}_Mean_Hybridization'] = MHys[int(level)]
            #encoded_atomic_counts_and_MPCs[f'lvl{int(level)}_Mean_sasa_6A'] = MSASA[int(level)]
            #encoded_atomic_counts_and_MPCs[f'lvl{int(level)}_Mean_Free_Vol_Frac_6A'] = MFVF[int(level)]
        else:
            encoded_atomic_counts_and_MPCs[f'lvl{int(level)}_Mean_Partial_Charge'] = -99999
            encoded_atomic_counts_and_MPCs[f'lvl{int(level)}_Mean_Hybridization'] = -99999

    return encoded_atomic_counts_and_MPCs

def featurize_organic_molecule(smiles_, mapno_, extra_df_ = None):
    mol = Chem.MolFromSmiles(smiles_)
    ranks, classes, rc = create_classes_from_smiles(smiles_, get_borrylation_candidates_from_smiles(smiles_), mapno_)
    rows = []
    #_ = ComputeGasteigerCharges(mol)
    for c in list(classes.keys()):
        atoms_in_class = np.where(ranks == c)[0]
        atom_idx = int(atoms_in_class[0])
        atom = mol.GetAtomWithIdx(atom_idx)
        props = atom.GetPropsAsDict()


        #partial_charge_H = props['_GasteigerHCharge']
        N_in_class = len(atoms_in_class)
        num_Hs = atom.GetTotalNumHs(),
        is_in_ring = int(atom.IsInRing()),
        hybridization = int(atom.GetHybridization()),
        if c == rc:
            reactive = True
        else:
            reactive = False


        sterical_dict_6A = pd.DataFrame(get_steric_description(smiles_, atom_idx, r_cutoff_=6), index=[0])
        #sterical_dict_3A = pd.DataFrame(get_steric_description(smiles_, atom_idx, r_cutoff_=3), index=[0])
        '''graph_descriptors = get_connectivity_patterns(smiles_, atom_idx, level_=4)

        #df_row = sterical_dict_3A.join(sterical_dict_6A)
        df_row = sterical_dict_6A.join(graph_descriptors)'''


        df_row = get_connectivity_patterns(smiles_, atom_idx, level_=4).copy()
        #df_row['partial_charge_H'] = partial_charge_H
        df_row['N_in_class'] = N_in_class
        df_row['num_Hs'] = num_Hs
        df_row['is_in_ring'] = is_in_ring
        df_row['hybridization'] = hybridization
        df_row['reactive'] = reactive
        df_row = df_row.add_prefix('SUBSTRATE_')
        df_row = sterical_dict_6A.join(df_row)
        
        if extra_df_ is not None:
            df_row = pd.concat([df_row.reset_index(drop=True), extra_df_.reset_index(drop=True)], axis=1,)
        rows.append(df_row)
    
    df_res = pd.concat(rows)
    df_res.reset_index(inplace=True, drop = True)
    return df_res

def featurize_catalyst(smiles_, mapno_ = 1):
    mol_no_Hs = Chem.MolFromSmiles(smiles_)
    mol = AllChem.AddHs(mol_no_Hs)
    _ = ComputeGasteigerCharges(mol)
    cat_center_id = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() == mapno_:
            cat_center_id = atom.GetIdx()

    sterical_dict_6A = pd.DataFrame(get_steric_description(smiles_, cat_center_id, r_cutoff_=6, reagent_type_='CATALYST'), index=[0])
    graph_descriptors = get_connectivity_patterns(smiles_, cat_center_id, level_=3, reagent_type='CATALYST')
    df_row = sterical_dict_6A.join(graph_descriptors.add_prefix('CATALYST_'))
    df_row['CATALYST_atomic_no'] = mol.GetAtomWithIdx(cat_center_id).GetAtomicNum()
    return df_row
