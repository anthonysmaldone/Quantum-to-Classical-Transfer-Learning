import numpy as np
from rdkit import Chem
import pandas as pd
from tqdm import tqdm
from typing import Any
import numpy as np

def encode_smiles(smiles: str, maximum_smiles_len: int) -> Any:
    mol = Chem.MolFromSmiles(smiles) 
    
    tensor = np.zeros((maximum_smiles_len,57))

    atoms = mol.GetAtoms()
    atom_symbols = [atom.GetSymbol() for atom in atoms]

    atom_idx = 0
    i = 0
    
    while i < len(smiles):
        char = smiles[i]
        
        if char == '(':
            tensor[i][0] = 1
            i += 1
            continue
        elif char == ')':
            tensor[i][1] = 1
            i += 1
            continue
        elif char == '[':
            tensor[i][2] = 1
            i += 1
            continue
        elif char == ']':
            tensor[i][3] = 1
            i += 1
            continue
        elif char == ' ':
            tensor[i][4] = 1
            i += 1
            continue
        elif char == ':':
            tensor[i][5] = 1
            i += 1
            continue
        elif char == '=':
            tensor[i][6] = 1
            i += 1
            continue
        elif char == '#':
            tensor[i][7] = 1
            i += 1
            continue
        elif char == '\\':
            tensor[i][8] = 1
            i += 1
            continue
        elif char == '/':
            tensor[i][9] = 1
            i += 1
            continue
        elif char == '@':
            tensor[i][10] = 1
            i += 1
            continue
        elif char == '+':
            tensor[i][11] = 1
            i += 1
            continue
        elif char == '-':
            tensor[i][12] = 1
            i += 1
            continue
        elif char == '.':
            tensor[i][13] = 1
            i += 1
            continue
        
        elif char.isdigit():
            
            # Check for charges like "+2" or "-2"
            if (i > 0 and smiles[i-1] in ['+', '-']) or (i < len(smiles) - 1 and smiles[i+1] in ['+', '-']):
                
                # This is a charge numeral, skip the next character
                if char == '2':
                    tensor[i][14] = 1
                if char == '3':
                    tensor[i][15] = 1
                if char == '4':
                    tensor[i][16] = 1
                if char == '5':
                    tensor[i][17] = 1
                if char == '6':
                    tensor[i][18] = 1
                if char == '7':
                    tensor[i][19] = 1
                
                i += 1
                continue
            
            # Check if the character appears again in the string for ring
            elif smiles[i:].count(char) > 1:
                
                # This is the start of a ring
                tensor[i][20] = 1
                i += 1
                continue
            else:
                
                # This is the end of a ring
                tensor[i][21] = 1
                i += 1
                continue
        
        elif char in atom_symbols:
            if char.capitalize() == 'C':
                tensor[i][22] = 1
            elif char.capitalize() == 'H':
                tensor[i][23] = 1
            elif char.capitalize() == 'O':
                tensor[i][24] = 1
            elif char.capitalize() == 'N':
                tensor[i][25] = 1
            else:
                tensor[i][26] = 1
            
            if atoms[atom_idx].GetTotalNumHs() == 0:
                tensor[i][27] = 1
            elif atoms[atom_idx].GetTotalNumHs() == 1:
                tensor[i][28] = 1
            elif atoms[atom_idx].GetTotalNumHs() == 2:
                tensor[i][29] = 1
            else:
                tensor[i][30] = 1

            if atoms[atom_idx].GetFormalCharge() == -1:
                tensor[i][31] = 1
            elif atoms[atom_idx].GetFormalCharge() == 0:
                tensor[i][32] = 1
            else:
                tensor[i][33] = 1

            if atoms[atom_idx].GetTotalValence() == 1:
                tensor[i][34] = 1
            elif atoms[atom_idx].GetTotalValence() == 2:
                tensor[i][35] = 1
            elif atoms[atom_idx].GetTotalValence() == 3:
                tensor[i][36] = 1
            elif atoms[atom_idx].GetTotalValence() == 4:
                tensor[i][37] = 1
            elif atoms[atom_idx].GetTotalValence() == 5:
                tensor[i][38] = 1
            else:
                tensor[i][39] = 1

            if atoms[atom_idx].IsInRing():
                tensor[i][40] = 1
            
            if atoms[atom_idx].GetDegree() == 1:
                tensor[i][41] = 1
            elif atoms[atom_idx].GetDegree() == 2:
                tensor[i][42] = 1
            elif atoms[atom_idx].GetDegree() == 3:
                tensor[i][43] = 1
            elif atoms[atom_idx].GetDegree() == 4:
                tensor[i][44] = 1
            else:
                tensor[i][45] = 1
            
            if atoms[atom_idx].GetIsAromatic():
                tensor[i][46] = 1

            if atoms[atom_idx].GetChiralTag() == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:
                tensor[i][47] = 1
            elif atoms[atom_idx].GetChiralTag() == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW:
                tensor[i][48] = 1
            else:
                tensor[i][49] = 1

            if atoms[atom_idx].GetHybridization() == Chem.rdchem.HybridizationType.SP:
                tensor[i][50] = 1
            elif atoms[atom_idx].GetHybridization() == Chem.rdchem.HybridizationType.SP2:
                tensor[i][51] = 1
            elif atoms[atom_idx].GetHybridization() == Chem.rdchem.HybridizationType.SP3:
                tensor[i][52] = 1
            elif atoms[atom_idx].GetHybridization() == Chem.rdchem.HybridizationType.SP3D:
                tensor[i][53] = 1
            elif atoms[atom_idx].GetHybridization() == Chem.rdchem.HybridizationType.SP3D2:
                tensor[i][54] = 1
            elif atoms[atom_idx].GetHybridization() == Chem.rdchem.HybridizationType.UNSPECIFIED:
                tensor[i][55] = 1
            else:
                tensor[i][56] = 1

            atom_idx += 1
        i += 1

    return tensor

def load_csv(dataset):
    path = '../datasets/'+dataset+'.csv'
    data = pd.read_csv(path)

    smiles_list = data['smiles'].tolist()
    label_list = data['label'].tolist()


    bse_list = []

    for smile in tqdm(smiles_list,desc="Building BSE Matrices"):
        mol = Chem.MolFromSmiles(smile)
        if '[2H]' in smile:
            #print(f"Skipping compound with deuterium: {smile}")
            continue
        if not mol:
            #print(f"Skipping compound with incorrect valence: {smile}")
            continue
        bse_list.append(encode_smiles(smile, 400))

    processed_data = np.stack(bse_list, axis=0)

    print(processed_data.shape)

    np.savez('../datasets/'+dataset+'.npz', BSE=processed_data, label=label_list)
