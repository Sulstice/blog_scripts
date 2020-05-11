from peptide_builder import PeptideBuilder
from functional_group_enumerator import Cocktail
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit import DataStructs
import pandas as pd


import matplotlib
matplotlib.use('PS')

def compare_tanimoto_fingerprints_pairwise(smiles, fingerprints):

    """

    Arguments:

        smiles (List): List of smiles you would like to compare
        fingerprints (List): List of fingerprint RDKit objects for each smiles (should directly correlate)
    Returns:

        similarity_dataframe (Pandas Dataframe Object): a dataframe containing pairwise similarity.

    """


    query, target, similarity = [], [], []

    for index, fingerprint in enumerate(fingerprints):
        similarity_group = DataStructs.BulkTanimotoSimilarity(fingerprint, fingerprints[index+1:])
        for i in range(len(similarity_group)):
            query.append(combinations[index])
            target.append(combinations[index+1:][i])
            similarity.append(similarity_group[i])

    # build the dataframe and sort it
    similarity_data = {'query':query, 'target':target, 'similarity':similarity}
    similarity_dataframe = pd.DataFrame(data=similarity_data).sort_values('similarity', ascending=False)

    return similarity_dataframe

if __name__ == '__main__':

    peptide_backbone = PeptideBuilder(3)

    cocktail = Cocktail(peptide_backbone,ligand_library = ['Br', 'F', 'I'])
    combinations = cocktail.shake()
    print (combinations)

    # Render to molecules
    molecules = [Chem.MolFromSmiles(x) for x in combinations]

    # Render to fingerprints
    fingerprints = [FingerprintMols.FingerprintMol(x) for x in molecules]

    print(compare_tanimoto_fingerprints_pairwise(smiles=combinations, fingerprints=fingerprints))