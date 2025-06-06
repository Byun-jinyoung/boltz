import sys
sys.path.append('src')

from rdkit import Chem
from rdkit.Chem import AllChem

from boltz.data.parse.schema import parse_polymer, ParsedRDKitBoundsConstraint
from boltz.data import const


def build_ala_mol():
    smiles = "N[C@@H](C)C(=O)O"
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)
    mol.GetConformer().SetProp("name", "Computed")
    names = ["N", "CA", "CB", "C", "O", "O2"]
    for atom, name in zip(mol.GetAtoms(), names):
        atom.SetProp("name", name)
    return Chem.RemoveHs(mol)


def test_parse_polymer_rdkit_bonds():
    mol = build_ala_mol()
    components = {"ALA": mol}
    seq = ["ALA"]
    chain = parse_polymer(
        sequence=seq,
        entity="0",
        chain_type=const.chain_type_ids["PROTEIN"],
        components=components,
        cyclic=False,
        add_rdkit_bonds=True,
    )
    res = chain.residues[0]
    assert res.rdkit_bounds_constraints is not None
    assert len(res.rdkit_bounds_constraints) > 0
    # ensure all constraints are bonds
    assert all(c.is_bond for c in res.rdkit_bounds_constraints)


def test_parse_polymer_no_bonds():
    mol = build_ala_mol()
    components = {"ALA": mol}
    seq = ["ALA"]
    chain = parse_polymer(
        sequence=seq,
        entity="0",
        chain_type=const.chain_type_ids["PROTEIN"],
        components=components,
        cyclic=False,
        add_rdkit_bonds=False,
    )
    res = chain.residues[0]
    assert res.rdkit_bounds_constraints is None
