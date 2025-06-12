import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
from mashumaro.mixins.dict import DataClassDictMixin
from rdkit.Chem import Mol

####################################################################################################
# SERIALIZABLE
####################################################################################################


class NumpySerializable:
    """Serializable datatype."""

    @classmethod
    def load(cls: "NumpySerializable", path: Path) -> "NumpySerializable":
        """Load the object from an NPZ file.

        Parameters
        ----------
        path : Path
            The path to the file.

        Returns
        -------
        Serializable
            The loaded object.

        """
        return cls(**np.load(path, allow_pickle=True))

    def dump(self, path: Path) -> None:
        """Dump the object to an NPZ file.

        Parameters
        ----------
        path : Path
            The path to the file.

        """
        np.savez_compressed(str(path), **asdict(self))


class JSONSerializable(DataClassDictMixin):
    """Serializable datatype."""

    @classmethod
    def load(cls: "JSONSerializable", path: Path) -> "JSONSerializable":
        """Load the object from a JSON file.

        Parameters
        ----------
        path : Path
            The path to the file.

        Returns
        -------
        Serializable
            The loaded object.

        """
        with path.open("r") as f:
            return cls.from_dict(json.load(f))

    def dump(self, path: Path) -> None:
        """Dump the object to a JSON file.

        Parameters
        ----------
        path : Path
            The path to the file.

        """
        with path.open("w") as f:
            json.dump(self.to_dict(), f)


####################################################################################################
# STRUCTURE
####################################################################################################

Atom = [
    ("name", np.dtype("4i1")),
    ("element", np.dtype("i1")),
    ("charge", np.dtype("i1")),
    ("coords", np.dtype("3f4")),
    ("conformer", np.dtype("3f4")),
    ("is_present", np.dtype("?")),
    ("chirality", np.dtype("i1")),
]

AtomV2 = [
    ("name", np.dtype("<U4")),
    ("coords", np.dtype("3f4")),
    ("is_present", np.dtype("?")),
    ("bfactor", np.dtype("f4")),
    ("plddt", np.dtype("f4")),
]

Bond = [
    ("atom_1", np.dtype("i4")),
    ("atom_2", np.dtype("i4")),
    ("type", np.dtype("i1")),
]

BondV2 = [
    ("chain_1", np.dtype("i4")),
    ("chain_2", np.dtype("i4")),
    ("res_1", np.dtype("i4")),
    ("res_2", np.dtype("i4")),
    ("atom_1", np.dtype("i4")),
    ("atom_2", np.dtype("i4")),
    ("type", np.dtype("i1")),
]

Residue = [
    ("name", np.dtype("<U5")),
    ("res_type", np.dtype("i1")),
    ("res_idx", np.dtype("i4")),
    ("atom_idx", np.dtype("i4")),
    ("atom_num", np.dtype("i4")),
    ("atom_center", np.dtype("i4")),
    ("atom_disto", np.dtype("i4")),
    ("is_standard", np.dtype("?")),
    ("is_present", np.dtype("?")),
]

Chain = [
    ("name", np.dtype("<U5")),
    ("mol_type", np.dtype("i1")),
    ("entity_id", np.dtype("i4")),
    ("sym_id", np.dtype("i4")),
    ("asym_id", np.dtype("i4")),
    ("atom_idx", np.dtype("i4")),
    ("atom_num", np.dtype("i4")),
    ("res_idx", np.dtype("i4")),
    ("res_num", np.dtype("i4")),
    ("cyclic_period", np.dtype("i4")),
]

Connection = [
    ("chain_1", np.dtype("i4")),
    ("chain_2", np.dtype("i4")),
    ("res_1", np.dtype("i4")),
    ("res_2", np.dtype("i4")),
    ("atom_1", np.dtype("i4")),
    ("atom_2", np.dtype("i4")),
]

Interface = [
    ("chain_1", np.dtype("i4")),
    ("chain_2", np.dtype("i4")),
]

Coords = [
    ("coords", np.dtype("3f4")),
]

Ensemble = [
    ("atom_coord_idx", np.dtype("i4")),
    ("atom_num", np.dtype("i4")),
]

# ==================== #
#  Code Modification   #
# ==================== #
MinDistance = [
    ("chain_1", np.dtype("i4")),
    ("chain_2", np.dtype("i4")),
    ("res_1", np.dtype("i4")),
    ("res_2", np.dtype("i4")),
    ("atom_1", np.dtype("i4")),
    ("atom_2", np.dtype("i4")),
    ("distance", np.dtype("f4")),
]

NMRDistance = [
    ("chain_1", np.dtype("i4")),
    ("chain_2", np.dtype("i4")),
    ("res_1", np.dtype("i4")),
    ("res_2", np.dtype("i4")),
    ("atom_1", np.dtype("i4")),
    ("atom_2", np.dtype("i4")),
    ("lower_bound", np.dtype("f4")),
    ("upper_bound", np.dtype("f4")),
    ("weight", np.dtype("f4")),
]


@dataclass(frozen=True)
class Structure(NumpySerializable):
    """Structure datatype."""

    atoms: np.ndarray
    bonds: np.ndarray
    residues: np.ndarray
    chains: np.ndarray
    connections: np.ndarray
    # code modification - add constraint fields
    min_distances: np.ndarray
    nmr_distances: np.ndarray
    interfaces: np.ndarray
    mask: np.ndarray

    @classmethod
    def load(cls: "Structure", path: Path) -> "Structure":
        """Load a structure from an NPZ file.

        Parameters
        ----------
        path : Path
            The path to the file.

        Returns
        -------
        Structure
            The loaded structure.

        """
        structure = np.load(path)
        return cls(
            atoms=structure["atoms"],
            bonds=structure["bonds"],
            residues=structure["residues"],
            chains=structure["chains"],
            connections=structure["connections"].astype(Connection),
            min_distances=structure.get("min_distances", np.array([], dtype=MinDistance)),
            nmr_distances=structure.get("nmr_distances", np.array([], dtype=NMRDistance)),
            interfaces=structure["interfaces"],
            mask=structure["mask"],
        )

    def remove_invalid_chains(self) -> "Structure":  # noqa: PLR0915
        """Remove invalid chains.

        Parameters
        ----------
        structure : Structure
            The structure to process.

        Returns
        -------
        Structure
            The structure with masked chains removed.

        """
        entity_counter = {}
        atom_idx, res_idx, chain_idx = 0, 0, 0
        atoms, residues, chains = [], [], []
        atom_map, res_map, chain_map = {}, {}, {}
        for i, chain in enumerate(self.chains):
            # Skip masked chains
            if not self.mask[i]:
                continue

            # Update entity counter
            entity_id = chain["entity_id"]
            if entity_id not in entity_counter:
                entity_counter[entity_id] = 0
            else:
                entity_counter[entity_id] += 1

            # Update the chain
            new_chain = chain.copy()
            new_chain["atom_idx"] = atom_idx
            new_chain["res_idx"] = res_idx
            new_chain["asym_id"] = chain_idx
            new_chain["sym_id"] = entity_counter[entity_id]
            chains.append(new_chain)
            chain_map[i] = chain_idx
            chain_idx += 1

            # Add the chain residues
            res_start = chain["res_idx"]
            res_end = chain["res_idx"] + chain["res_num"]
            for j, res in enumerate(self.residues[res_start:res_end]):
                # Update the residue
                new_res = res.copy()
                new_res["atom_idx"] = atom_idx
                new_res["atom_center"] = (
                    atom_idx + new_res["atom_center"] - res["atom_idx"]
                )
                new_res["atom_disto"] = (
                    atom_idx + new_res["atom_disto"] - res["atom_idx"]
                )
                residues.append(new_res)
                res_map[res_start + j] = res_idx
                res_idx += 1

                # Update the atoms
                start = res["atom_idx"]
                end = res["atom_idx"] + res["atom_num"]
                atoms.append(self.atoms[start:end])
                atom_map.update({k: atom_idx + k - start for k in range(start, end)})
                atom_idx += res["atom_num"]

        # Concatenate the tables
        atoms = np.concatenate(atoms, dtype=Atom)
        residues = np.array(residues, dtype=Residue)
        chains = np.array(chains, dtype=Chain)

        # Update bonds
        bonds = []
        for bond in self.bonds:
            atom_1 = bond["atom_1"]
            atom_2 = bond["atom_2"]
            if (atom_1 in atom_map) and (atom_2 in atom_map):
                new_bond = bond.copy()
                new_bond["atom_1"] = atom_map[atom_1]
                new_bond["atom_2"] = atom_map[atom_2]
                bonds.append(new_bond)

        # Update connections
        connections = []
        for connection in self.connections:
            chain_1 = connection["chain_1"]
            chain_2 = connection["chain_2"]
            res_1 = connection["res_1"]
            res_2 = connection["res_2"]
            atom_1 = connection["atom_1"]
            atom_2 = connection["atom_2"]
            if (atom_1 in atom_map) and (atom_2 in atom_map):
                new_connection = connection.copy()
                new_connection["chain_1"] = chain_map[chain_1]
                new_connection["chain_2"] = chain_map[chain_2]
                new_connection["res_1"] = res_map[res_1]
                new_connection["res_2"] = res_map[res_2]
                new_connection["atom_1"] = atom_map[atom_1]
                new_connection["atom_2"] = atom_map[atom_2]
                connections.append(new_connection)

        # Code modification - Update min_distances constraints

        min_distances = []
        for min_dist in self.min_distances:
            chain_1 = min_dist["chain_1"]
            chain_2 = min_dist["chain_2"]
            res_1 = min_dist["res_1"]
            res_2 = min_dist["res_2"]
            atom_1 = min_dist["atom_1"]
            atom_2 = min_dist["atom_2"]
            if (atom_1 in atom_map) and (atom_2 in atom_map):
                new_min_dist = min_dist.copy()
                new_min_dist["chain_1"] = chain_map[chain_1]
                new_min_dist["chain_2"] = chain_map[chain_2]
                new_min_dist["res_1"] = res_map[res_1]
                new_min_dist["res_2"] = res_map[res_2]
                new_min_dist["atom_1"] = atom_map[atom_1]
                new_min_dist["atom_2"] = atom_map[atom_2]
                min_distances.append(new_min_dist)

        # Code modification - Update nmr_distances constraints

        nmr_distances = []
        for nmr_dist in self.nmr_distances:
            chain_1 = nmr_dist["chain_1"]
            chain_2 = nmr_dist["chain_2"]
            res_1 = nmr_dist["res_1"]
            res_2 = nmr_dist["res_2"]
            atom_1 = nmr_dist["atom_1"]
            atom_2 = nmr_dist["atom_2"]
            if (atom_1 in atom_map) and (atom_2 in atom_map):
                new_nmr_dist = nmr_dist.copy()
                new_nmr_dist["chain_1"] = chain_map[chain_1]
                new_nmr_dist["chain_2"] = chain_map[chain_2]
                new_nmr_dist["res_1"] = res_map[res_1]
                new_nmr_dist["res_2"] = res_map[res_2]
                new_nmr_dist["atom_1"] = atom_map[atom_1]
                new_nmr_dist["atom_2"] = atom_map[atom_2]
                nmr_distances.append(new_nmr_dist)

        # Create arrays
        bonds = np.array(bonds, dtype=Bond)
        connections = np.array(connections, dtype=Connection)
        min_distances = np.array(min_distances, dtype=MinDistance)
        nmr_distances = np.array(nmr_distances, dtype=NMRDistance)
        interfaces = np.array([], dtype=Interface)
        mask = np.ones(len(chains), dtype=bool)

        return Structure(
            atoms=atoms,
            bonds=bonds,
            residues=residues,
            chains=chains,
            connections=connections,
            min_distances=min_distances,
            nmr_distances=nmr_distances,
            interfaces=interfaces,
            mask=mask,
        )


@dataclass(frozen=True)
class StructureV2(NumpySerializable):
    """Structure datatype."""

    atoms: np.ndarray
    bonds: np.ndarray
    residues: np.ndarray
    chains: np.ndarray
    interfaces: np.ndarray
    mask: np.ndarray
    coords: np.ndarray
    ensemble: np.ndarray
    pocket: Optional[np.ndarray] = None
    # Code modification - add constraint fields for V2 compatibility
    min_distances: Optional[np.ndarray] = None
    nmr_distances: Optional[np.ndarray] = None

    def remove_invalid_chains(self) -> "StructureV2":  # noqa: PLR0915
        """Remove invalid chains.

        Parameters
        ----------
        structure : Structure
            The structure to process.

        Returns
        -------
        Structure
            The structure with masked chains removed.

        """
        entity_counter = {}
        atom_idx, res_idx, chain_idx = 0, 0, 0
        atoms, residues, chains = [], [], []
        atom_map, res_map, chain_map = {}, {}, {}
        for i, chain in enumerate(self.chains):
            # Skip masked chains
            if not self.mask[i]:
                continue

            # Update entity counter
            entity_id = chain["entity_id"]
            if entity_id not in entity_counter:
                entity_counter[entity_id] = 0
            else:
                entity_counter[entity_id] += 1

            # Update the chain
            new_chain = chain.copy()
            new_chain["atom_idx"] = atom_idx
            new_chain["res_idx"] = res_idx
            new_chain["asym_id"] = chain_idx
            new_chain["sym_id"] = entity_counter[entity_id]
            chains.append(new_chain)
            chain_map[i] = chain_idx
            chain_idx += 1

            # Add the chain residues
            res_start = chain["res_idx"]
            res_end = chain["res_idx"] + chain["res_num"]
            for j, res in enumerate(self.residues[res_start:res_end]):
                # Update the residue
                new_res = res.copy()
                new_res["atom_idx"] = atom_idx
                new_res["atom_center"] = (
                    atom_idx + new_res["atom_center"] - res["atom_idx"]
                )
                new_res["atom_disto"] = (
                    atom_idx + new_res["atom_disto"] - res["atom_idx"]
                )
                residues.append(new_res)
                res_map[res_start + j] = res_idx
                res_idx += 1

                # Update the atoms
                start = res["atom_idx"]
                end = res["atom_idx"] + res["atom_num"]
                atoms.append(self.atoms[start:end])
                atom_map.update({k: atom_idx + k - start for k in range(start, end)})
                atom_idx += res["atom_num"]

        # Concatenate the tables
        atoms = np.concatenate(atoms, dtype=AtomV2)
        residues = np.array(residues, dtype=Residue)
        chains = np.array(chains, dtype=Chain)

        # Update bonds
        bonds = []
        for bond in self.bonds:
            chain_1 = bond["chain_1"]
            chain_2 = bond["chain_2"]
            res_1 = bond["res_1"]
            res_2 = bond["res_2"]
            atom_1 = bond["atom_1"]
            atom_2 = bond["atom_2"]
            if (atom_1 in atom_map) and (atom_2 in atom_map):
                new_bond = bond.copy()
                new_bond["chain_1"] = chain_map[chain_1]
                new_bond["chain_2"] = chain_map[chain_2]
                new_bond["res_1"] = res_map[res_1]
                new_bond["res_2"] = res_map[res_2]
                new_bond["atom_1"] = atom_map[atom_1]
                new_bond["atom_2"] = atom_map[atom_2]
                bonds.append(new_bond)

        # Code modification - Update min_distances constraints for V2
        min_distances = []
        if self.min_distances is not None:
            try:
                # Safely check if it's iterable and has length
                min_dist_array = np.asarray(self.min_distances)
                if min_dist_array.ndim > 0 and len(min_dist_array) > 0:
                    for min_dist in min_dist_array:
                        chain_1 = min_dist["chain_1"]
                        chain_2 = min_dist["chain_2"]
                        res_1 = min_dist["res_1"]
                        res_2 = min_dist["res_2"]
                        atom_1 = min_dist["atom_1"]
                        atom_2 = min_dist["atom_2"]
                        if (atom_1 in atom_map) and (atom_2 in atom_map):
                            new_min_dist = min_dist.copy()
                            new_min_dist["chain_1"] = chain_map[chain_1]
                            new_min_dist["chain_2"] = chain_map[chain_2]
                            new_min_dist["res_1"] = res_map[res_1]
                            new_min_dist["res_2"] = res_map[res_2]
                            new_min_dist["atom_1"] = atom_map[atom_1]
                            new_min_dist["atom_2"] = atom_map[atom_2]
                            min_distances.append(new_min_dist)
            except Exception:
                # Silently handle any array processing errors
                pass

        # Code modification - Update nmr_distances constraints for V2
        nmr_distances = []
        if self.nmr_distances is not None:
            try:
                # Safely check if it's iterable and has length
                nmr_dist_array = np.asarray(self.nmr_distances)
                if nmr_dist_array.ndim > 0 and len(nmr_dist_array) > 0:
                    for nmr_dist in nmr_dist_array:
                        chain_1 = nmr_dist["chain_1"]
                        chain_2 = nmr_dist["chain_2"]
                        res_1 = nmr_dist["res_1"]
                        res_2 = nmr_dist["res_2"]
                        atom_1 = nmr_dist["atom_1"]
                        atom_2 = nmr_dist["atom_2"]
                        if (atom_1 in atom_map) and (atom_2 in atom_map):
                            new_nmr_dist = nmr_dist.copy()
                            new_nmr_dist["chain_1"] = chain_map[chain_1]
                            new_nmr_dist["chain_2"] = chain_map[chain_2]
                            new_nmr_dist["res_1"] = res_map[res_1]
                            new_nmr_dist["res_2"] = res_map[res_2]
                            new_nmr_dist["atom_1"] = atom_map[atom_1]
                            new_nmr_dist["atom_2"] = atom_map[atom_2]
                            nmr_distances.append(new_nmr_dist)
            except Exception:
                # Silently handle any array processing errors
                pass

        # Create arrays
        bonds = np.array(bonds, dtype=BondV2)
        interfaces = np.array([], dtype=Interface)
        mask = np.ones(len(chains), dtype=bool)
        coords = [(x,) for x in atoms["coords"]]
        coords = np.array(coords, dtype=Coords)
        ensemble = np.array([(0, len(coords))], dtype=Ensemble)
        
        # Code modification - Create constraint arrays for V2
        min_distances_array = np.array(min_distances, dtype=MinDistance) if min_distances else None
        nmr_distances_array = np.array(nmr_distances, dtype=NMRDistance) if nmr_distances else None

        return StructureV2(
            atoms=atoms,
            bonds=bonds,
            residues=residues,
            chains=chains,
            interfaces=interfaces,
            mask=mask,
            coords=coords,
            ensemble=ensemble,
            min_distances=min_distances_array,
            nmr_distances=nmr_distances_array,
        )


####################################################################################################
# MSA
####################################################################################################


MSAResidue = [
    ("res_type", np.dtype("i1")),
]

MSADeletion = [
    ("res_idx", np.dtype("i2")),
    ("deletion", np.dtype("i2")),
]

MSASequence = [
    ("seq_idx", np.dtype("i2")),
    ("taxonomy", np.dtype("i4")),
    ("res_start", np.dtype("i4")),
    ("res_end", np.dtype("i4")),
    ("del_start", np.dtype("i4")),
    ("del_end", np.dtype("i4")),
]


@dataclass(frozen=True)
class MSA(NumpySerializable):
    """MSA datatype."""

    sequences: np.ndarray
    deletions: np.ndarray
    residues: np.ndarray


####################################################################################################
# RECORD
####################################################################################################


@dataclass(frozen=True)
class StructureInfo:
    """StructureInfo datatype."""

    resolution: Optional[float] = None
    method: Optional[str] = None
    deposited: Optional[str] = None
    released: Optional[str] = None
    revised: Optional[str] = None
    num_chains: Optional[int] = None
    num_interfaces: Optional[int] = None
    pH: Optional[float] = None
    temperature: Optional[float] = None


@dataclass(frozen=False)
class ChainInfo:
    """ChainInfo datatype."""

    chain_id: int
    chain_name: str
    mol_type: int
    cluster_id: Union[str, int]
    msa_id: Union[str, int]
    num_residues: int
    valid: bool = True
    entity_id: Optional[Union[str, int]] = None


@dataclass(frozen=True)
class InterfaceInfo:
    """InterfaceInfo datatype."""

    chain_1: int
    chain_2: int
    valid: bool = True


@dataclass(frozen=True)
class InferenceOptions:
    """InferenceOptions datatype."""

    pocket_constraints: Optional[list[tuple[int, list[tuple[int, int]], float]]] = None


@dataclass(frozen=True)
class MDInfo:
    """MDInfo datatype."""

    forcefield: Optional[list[str]]
    temperature: Optional[float]  # Kelvin
    pH: Optional[float]
    pressure: Optional[float]  # bar
    prod_ensemble: Optional[str]
    solvent: Optional[str]
    ion_concentration: Optional[float]  # mM
    time_step: Optional[float]  # fs
    sample_frame_time: Optional[float]  # ps
    sim_time: Optional[float]  # ns
    coarse_grained: Optional[bool] = False


@dataclass(frozen=True)
class TemplateInfo:
    """InterfaceInfo datatype."""

    name: str
    query_chain: str
    query_st: int
    query_en: int
    template_chain: str
    template_st: int
    template_en: int


@dataclass(frozen=True)
class AffinityInfo:
    """AffinityInfo datatype."""

    chain_id: int
    mw: float


@dataclass(frozen=True)
class Record(JSONSerializable):
    """Record datatype."""

    id: str
    structure: StructureInfo
    chains: list[ChainInfo]
    interfaces: list[InterfaceInfo]
    inference_options: Optional[InferenceOptions] = None
    templates: Optional[list[TemplateInfo]] = None
    md: Optional[MDInfo] = None
    affinity: Optional[AffinityInfo] = None


####################################################################################################
# RESIDUE CONSTRAINTS
####################################################################################################


RDKitBoundsConstraint = [
    ("atom_idxs", np.dtype("2i4")),
    ("is_bond", np.dtype("?")),
    ("is_angle", np.dtype("?")),
    ("upper_bound", np.dtype("f4")),
    ("lower_bound", np.dtype("f4")),
]

ChiralAtomConstraint = [
    ("atom_idxs", np.dtype("4i4")),
    ("is_reference", np.dtype("?")),
    ("is_r", np.dtype("?")),
]

StereoBondConstraint = [
    ("atom_idxs", np.dtype("4i4")),
    ("is_reference", np.dtype("?")),
    ("is_e", np.dtype("?")),
]

PlanarBondConstraint = [
    ("atom_idxs", np.dtype("6i4")),
]

PlanarRing5Constraint = [
    ("atom_idxs", np.dtype("5i4")),
]

PlanarRing6Constraint = [
    ("atom_idxs", np.dtype("6i4")),
]


@dataclass(frozen=True)
class ResidueConstraints(NumpySerializable):
    """ResidueConstraints datatype."""

    rdkit_bounds_constraints: np.ndarray
    chiral_atom_constraints: np.ndarray
    stereo_bond_constraints: np.ndarray
    planar_bond_constraints: np.ndarray
    planar_ring_5_constraints: np.ndarray
    planar_ring_6_constraints: np.ndarray


####################################################################################################
# TARGET
####################################################################################################


@dataclass(frozen=True)
class Target:
    """Target datatype."""

    record: Record
    structure: Union[Structure, StructureV2]  # Code modification - Support both V1 and V2
    sequences: Optional[dict[str, str]] = None
    residue_constraints: Optional[ResidueConstraints] = None
    templates: Optional[dict[str, StructureV2]] = None
    extra_mols: Optional[dict[str, Mol]] = None


@dataclass(frozen=True)
class Manifest(JSONSerializable):
    """Manifest datatype."""

    records: list[Record]

    @classmethod
    def load(cls: "JSONSerializable", path: Path) -> "JSONSerializable":
        """Load the object from a JSON file.

        Parameters
        ----------
        path : Path
            The path to the file.

        Returns
        -------
        Serializable
            The loaded object.

        Raises
        ------
        TypeError
            If the file is not a valid manifest file.

        """
        with path.open("r") as f:
            data = json.load(f)
            if isinstance(data, dict):
                manifest = cls.from_dict(data)
            elif isinstance(data, list):
                records = [Record.from_dict(r) for r in data]
                manifest = cls(records=records)
            else:
                msg = "Invalid manifest file."
                raise TypeError(msg)

        return manifest


####################################################################################################
# INPUT
####################################################################################################


@dataclass(frozen=True, slots=True)
class Input:
    """Input datatype."""

    structure: Structure
    msa: dict[str, MSA]
    record: Optional[Record] = None
    residue_constraints: Optional[ResidueConstraints] = None
    templates: Optional[dict[str, StructureV2]] = None
    extra_mols: Optional[dict[str, Mol]] = None


####################################################################################################
# TOKENS
####################################################################################################

Token = [
    ("token_idx", np.dtype("i4")),
    ("atom_idx", np.dtype("i4")),
    ("atom_num", np.dtype("i4")),
    ("res_idx", np.dtype("i4")),
    ("res_type", np.dtype("i1")),
    ("sym_id", np.dtype("i4")),
    ("asym_id", np.dtype("i4")),
    ("entity_id", np.dtype("i4")),
    ("mol_type", np.dtype("i1")),
    ("center_idx", np.dtype("i4")),
    ("disto_idx", np.dtype("i4")),
    ("center_coords", np.dtype("3f4")),
    ("disto_coords", np.dtype("3f4")),
    ("resolved_mask", np.dtype("?")),
    ("disto_mask", np.dtype("?")),
    ("cyclic_period", np.dtype("i4")),
]

TokenBond = [
    ("token_1", np.dtype("i4")),
    ("token_2", np.dtype("i4")),
]


TokenV2 = [
    ("token_idx", np.dtype("i4")),
    ("atom_idx", np.dtype("i4")),
    ("atom_num", np.dtype("i4")),
    ("res_idx", np.dtype("i4")),
    ("res_type", np.dtype("i4")),
    ("res_name", np.dtype("<U8")),
    ("sym_id", np.dtype("i4")),
    ("asym_id", np.dtype("i4")),
    ("entity_id", np.dtype("i4")),
    ("mol_type", np.dtype("i4")),  # the total bytes need to be divisible by 4
    ("center_idx", np.dtype("i4")),
    ("disto_idx", np.dtype("i4")),
    ("center_coords", np.dtype("3f4")),
    ("disto_coords", np.dtype("3f4")),
    ("resolved_mask", np.dtype("?")),
    ("disto_mask", np.dtype("?")),
    ("modified", np.dtype("?")),
    ("frame_rot", np.dtype("9f4")),
    ("frame_t", np.dtype("3f4")),
    ("frame_mask", np.dtype("i4")),
    ("cyclic_period", np.dtype("i4")),
    ("affinity_mask", np.dtype("?")),
]

TokenBondV2 = [
    ("token_1", np.dtype("i4")),
    ("token_2", np.dtype("i4")),
    ("type", np.dtype("i1")),
]


@dataclass(frozen=True)
class Tokenized:
    """Tokenized datatype."""

    tokens: np.ndarray
    bonds: np.ndarray
    structure: Structure
    msa: dict[str, MSA]
    record: Optional[Record] = None
    residue_constraints: Optional[ResidueConstraints] = None
    templates: Optional[dict[str, StructureV2]] = None
    template_tokens: Optional[dict[str, np.ndarray]] = None
    template_bonds: Optional[dict[str, np.ndarray]] = None
    extra_mols: Optional[dict[str, Mol]] = None
