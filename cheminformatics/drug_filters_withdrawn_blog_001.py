from rdkit import Chem
from rdkit.Chem import Descriptors

import progressbar

if __name__ == '__main__':


    molecules = Chem.SDMolSupplier('withdrawn_database.sdf')

    results = {
        "Lipinski Rule of 5": 0,
        "Ghose Filter": 0,
        "Veber Filter": 0,
        "Rule of 3 Filter": 0,
        "REOS Filter": 0,
        "Drug-like Filter": 0,
        "Passes All Filters": 0,
    }

    print ("Molecule Database Length: " + str(len(molecules)))

    for i in progressbar.progressbar(range(len(molecules))):

        molecule = molecules[i]
        if molecule:

            lipinski = False
            rule_of_3 = False
            ghose_filter = False
            veber_filter = False
            reos_filter = False
            drug_like_filter = False

            molecular_weight = Descriptors.ExactMolWt(molecule)
            logp = Descriptors.MolLogP(molecule)
            h_bond_donor = Descriptors.NumHDonors(molecule)
            h_bond_acceptors = Descriptors.NumHAcceptors(molecule)
            rotatable_bonds = Descriptors.NumRotatableBonds(molecule)
            number_of_atoms = Chem.rdchem.Mol.GetNumAtoms(molecule)
            molar_refractivity = Chem.Crippen.MolMR(molecule)
            topological_surface_area_mapping = Chem.QED.properties(molecule).PSA
            formal_charge = Chem.rdmolops.GetFormalCharge(molecule)
            heavy_atoms = Chem.rdchem.Mol.GetNumHeavyAtoms(molecule)
            num_of_rings = Chem.rdMolDescriptors.CalcNumRings(molecule)

            # Lipinski
            if molecular_weight <= 500 and logp <= 5 and h_bond_donor <= 5 and h_bond_acceptors <= 5 and rotatable_bonds <= 5:
                lipinski = True
                results["Lipinski Rule of 5"] += 1

            # Ghose Filter
            if molecular_weight >= 160 and molecular_weight <= 480 and logp >= 0.4 and logp <= 5.6 and number_of_atoms >= 20 and number_of_atoms <= 70 and molar_refractivity >= 40 and molar_refractivity <= 130:
                ghose_filter = True
                results["Ghose Filter"] += 1

            # Veber Filter
            if rotatable_bonds <= 10 and topological_surface_area_mapping <= 140:
                veber_filter = True
                results["Veber Filter"] += 1

            # Rule of 3
            if molecular_weight <= 300 and logp <= 3 and h_bond_donor <= 3 and h_bond_acceptors <= 3 and rotatable_bonds <= 3:
                rule_of_3 = True
                results["Rule of 3 Filter"] += 1

            # REOS Filter
            if molecular_weight >= 200 and molecular_weight <= 500 and logp >= int(0 - 5) and logp <= 5 and h_bond_donor >= 0 and h_bond_donor <= 5 and h_bond_acceptors >= 0 and h_bond_acceptors <= 10 and formal_charge >= int(0-2) and formal_charge <= 2 and rotatable_bonds >= 0 and rotatable_bonds <= 8 and heavy_atoms >= 15 and heavy_atoms <= 50:
                reos_filter = True
                results["REOS Filter"] += 1

            #Drug Like Filter
            if molecular_weight < 400 and num_of_rings > 0 and rotatable_bonds < 5 and h_bond_donor <= 5 and h_bond_acceptors <= 10 and logp < 5:
                drug_like_filter = True
                results["Drug-like Filter"] += 1

            if lipinski and ghose_filter and veber_filter and rule_of_3 and reos_filter and drug_like_filter:
                results["Passes All Filters"] += 1

    print (results)