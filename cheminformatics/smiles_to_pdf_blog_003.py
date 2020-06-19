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
    molecules = list(global_chem.functional_groups_smiles.values())

    document = MolPDF(name='functional_groups.pdf')
    document.add_title('Functional Groups Global Chem')
    document.add_spacer()

    # Generate the document
    document.generate(smiles=molecules, include_failed_smiles=True)

    # Read PDF
    document = MolPDFParser('functional_groups.pdf')
    molecules = document.extract_smiles()
