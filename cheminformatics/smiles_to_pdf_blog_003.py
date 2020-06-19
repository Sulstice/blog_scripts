# Global Chem Imports
# -------------------
from global_chem import GlobalChem

# MolPDF Imports
# --------------
from molpdf import MolPDF, MolPDFParser

if __name__ == '__main__':

    # Initialize Global Chem
    global_chem = GlobalChem()

    # Retrieve all Functional Groups
    smiles_list = list(global_chem.functional_groups_smiles.values())

    # Initialize the document
    document = MolPDF(name='functional_groups.pdf')
    document.add_title('Functional Groups Global Chem')

    # Generate the document
    document.generate(smiles=smiles_list, include_failed_smiles=True)