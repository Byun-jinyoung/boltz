from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import BasePredictionWriter
import torch
from torch import Tensor

from boltz.data.types import (
    Interface,
    Record,
    Structure,
)
from boltz.data.write.mmcif import to_mmcif
from boltz.data.write.pdb import to_pdb


class BoltzWriter(BasePredictionWriter):
    """Custom writer for predictions."""

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        output_format: Literal["pdb", "mmcif"] = "mmcif",
        # Code Modification
        save_intermediate_coords: bool = False,
        intermediate_output_format: str = "pdb",
        intermediate_save_every: int = 10,
    ) -> None:
        """Initialize the writer.

        Parameters
        ----------
        data_dir : str
            The directory containing input data.
        output_dir : str
            The directory to save the predictions.
        output_format : Literal["pdb", "mmcif"], optional
            The output format for final structures, by default "mmcif".
        save_intermediate_coords : bool, optional
            Whether to save intermediate coordinates, by default False.
        intermediate_output_format : str, optional
            Format for intermediate coordinates ("pdb", "npz", "both"), by default "pdb".
        intermediate_save_every : int, optional
            Save every N timesteps for intermediate coordinates, by default 10.

        """
        super().__init__(write_interval="batch")
        if output_format not in ["pdb", "mmcif"]:
            msg = f"Invalid output format: {output_format}"
            raise ValueError(msg)

        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_format = output_format
        # Code Modification
        self.save_intermediate_coords = save_intermediate_coords
        self.intermediate_output_format = intermediate_output_format
        self.intermediate_save_every = intermediate_save_every
        self.failed = 0

        # Create the output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.save_intermediate_coords:
            (self.output_dir / "trajectories").mkdir(parents=True, exist_ok=True)

    def write_on_batch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
        prediction: dict[str, Tensor],
        batch_indices: list[int],  # noqa: ARG002
        batch: dict[str, Tensor],
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int,  # noqa: ARG002
    ) -> None:
        """Write the predictions to disk."""
        if prediction["exception"]:
            self.failed += 1
            return

        # Get the records
        records: list[Record] = batch["record"]

        # Get the predictions
        coords = prediction["coords"]
        coords = coords.unsqueeze(0)

        pad_masks = prediction["masks"]

        # Get ranking
        argsort = torch.argsort(prediction["confidence_score"], descending=True)
        idx_to_rank = {idx.item(): rank for rank, idx in enumerate(argsort)}

        # Iterate over the records
        for record, coord, pad_mask in zip(records, coords, pad_masks):
            # Load the structure
            path = self.data_dir / f"{record.id}.npz"
            structure: Structure = Structure.load(path)

            # Compute chain map with masked removed, to be used later
            chain_map = {}
            for i, mask in enumerate(structure.mask):
                if mask:
                    chain_map[len(chain_map)] = i

            # Remove masked chains completely
            structure = structure.remove_invalid_chains()

            for model_idx in range(coord.shape[0]):
                # Save intermediate coordinates if available (code modification)
                if self.save_intermediate_coords and "intermediate_trajectory" in prediction:
                    trajectory = prediction["intermediate_trajectory"]
                    self.save_intermediate_coordinates(trajectory, record, model_idx, structure)

                # Get model coord
                model_coord = coord[model_idx]
                # Unpad
                coord_unpad = model_coord[pad_mask.bool()]
                coord_unpad = coord_unpad.cpu().numpy()

                # New atom table
                atoms = structure.atoms
                atoms["coords"] = coord_unpad
                atoms["is_present"] = True

                # Mew residue table
                residues = structure.residues
                residues["is_present"] = True

                # Update the structure
                interfaces = np.array([], dtype=Interface)
                new_structure: Structure = replace(
                    structure,
                    atoms=atoms,
                    residues=residues,
                    interfaces=interfaces,
                )

                # Update chain info
                chain_info = []
                for chain in new_structure.chains:
                    old_chain_idx = chain_map[chain["asym_id"]]
                    old_chain_info = record.chains[old_chain_idx]
                    new_chain_info = replace(
                        old_chain_info,
                        chain_id=int(chain["asym_id"]),
                        valid=True,
                    )
                    chain_info.append(new_chain_info)

                # Save the structure
                struct_dir = self.output_dir / record.id
                struct_dir.mkdir(exist_ok=True)

                # Get plddt's
                plddts = None
                if "plddt" in prediction:
                    plddts = prediction["plddt"][model_idx]

                # Create path name
                outname = f"{record.id}_model_{idx_to_rank[model_idx]}"

                # Save the structure
                if self.output_format == "pdb":
                    path = struct_dir / f"{outname}.pdb"
                    with path.open("w") as f:
                        f.write(to_pdb(new_structure, plddts=plddts))
                elif self.output_format == "mmcif":
                    path = struct_dir / f"{outname}.cif"
                    with path.open("w") as f:
                        f.write(to_mmcif(new_structure, plddts=plddts))
                else:
                    path = struct_dir / f"{outname}.npz"
                    np.savez_compressed(path, **asdict(new_structure))

                # Save confidence summary
                if "plddt" in prediction:
                    path = (
                        struct_dir
                        / f"confidence_{record.id}_model_{idx_to_rank[model_idx]}.json"
                    )
                    confidence_summary_dict = {}
                    for key in [
                        "confidence_score",
                        "ptm",
                        "iptm",
                        "ligand_iptm",
                        "protein_iptm",
                        "complex_plddt",
                        "complex_iplddt",
                        "complex_pde",
                        "complex_ipde",
                    ]:
                        confidence_summary_dict[key] = prediction[key][model_idx].item()
                    confidence_summary_dict["chains_ptm"] = {
                        idx: prediction["pair_chains_iptm"][idx][idx][model_idx].item()
                        for idx in prediction["pair_chains_iptm"]
                    }
                    confidence_summary_dict["pair_chains_iptm"] = {
                        idx1: {
                            idx2: prediction["pair_chains_iptm"][idx1][idx2][
                                model_idx
                            ].item()
                            for idx2 in prediction["pair_chains_iptm"][idx1]
                        }
                        for idx1 in prediction["pair_chains_iptm"]
                    }
                    with path.open("w") as f:
                        f.write(
                            json.dumps(
                                confidence_summary_dict,
                                indent=4,
                            )
                        )

                    # Save plddt
                    plddt = prediction["plddt"][model_idx]
                    path = (
                        struct_dir
                        / f"plddt_{record.id}_model_{idx_to_rank[model_idx]}.npz"
                    )
                    np.savez_compressed(path, plddt=plddt.cpu().numpy())

                # Save pae
                if "pae" in prediction:
                    pae = prediction["pae"][model_idx]
                    path = (
                        struct_dir
                        / f"pae_{record.id}_model_{idx_to_rank[model_idx]}.npz"
                    )
                    np.savez_compressed(path, pae=pae.cpu().numpy())

                # Save pde
                if "pde" in prediction:
                    pde = prediction["pde"][model_idx]
                    path = (
                        struct_dir
                        / f"pde_{record.id}_model_{idx_to_rank[model_idx]}.npz"
                    )
                    np.savez_compressed(path, pde=pde.cpu().numpy())

    def on_predict_epoch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
    ) -> None:
        """Print the number of failed examples."""
        # Print number of failed examples
        print(f"Number of failed examples: {self.failed}")  # noqa: T201

    def save_trajectory_as_pdb(
        self,
        trajectory: dict,
        record_id: str,
        model_idx: int,
        structure: Structure,
    ) -> None:
        """
        Save trajectory coordinates as PDB files with accurate atom and residue information.
        
        Parameters
        ----------
        trajectory : dict
            Dictionary containing intermediate coordinates and metadata.
        record_id : str
            Record identifier for file naming.
        model_idx : int
            Model index for file naming.
        structure : Structure
            Structure object for atom and residue information.
        """
        traj_dir = self.output_dir / "trajectories" / record_id / f"model_{model_idx}"
        traj_dir.mkdir(parents=True, exist_ok=True)
        
        # Save trajectory metadata
        metadata_file = traj_dir / "trajectory_metadata.json"
        with metadata_file.open("w") as f:
            json.dump(trajectory['metadata'], f, indent=2)
        
        # Get atom to residue mapping
        atom_to_res = {}
        atom_to_chain = {}
        res_to_chain = {}
        atom_idx = 0
        
        # Build mappings from structure data
        for chain_idx, chain in enumerate(structure.chains):
            chain_id = chr(65 + chain_idx % 26)  # A, B, C, ...
            
            res_start = chain["res_idx"]
            res_end = chain["res_idx"] + chain["res_num"]
            
            for res_idx, residue in enumerate(structure.residues[res_start:res_end]):
                res_id = res_start + res_idx
                res_to_chain[res_id] = chain_id
                
                atom_start = residue["atom_idx"]
                atom_end = residue["atom_idx"] + residue["atom_num"]
                
                for a_idx in range(atom_start, atom_end):
                    atom_to_res[atom_idx] = res_id
                    atom_to_chain[atom_idx] = chain_id
                    atom_idx += 1
        
        # Get periodic table for element mapping (from rdkit if available)
        try:
            from rdkit import Chem
            periodic_table = Chem.GetPeriodicTable()
        except ImportError:
            periodic_table = None
        
        # Save coordinates at selected timesteps
        for i, timestep in enumerate(trajectory['timesteps']):
            if i % self.intermediate_save_every == 0 and trajectory['denoised_coords'][i] is not None:
                coords = trajectory['denoised_coords'][i][0]  # First sample
                sigma = trajectory['sigmas'][i]
                
                # Create PDB content
                pdb_content = f"REMARK Timestep {timestep}, Sigma {sigma:.6f}\n"
                pdb_content += f"REMARK Intermediate coordinate from diffusion reverse process\n"
                
                atom_index = 1
                for j, coord in enumerate(coords.cpu().numpy()):
                    if j < len(structure.atoms):
                        # Get actual atom information from structure
                        atom = structure.atoms[j]
                        atom_name_bytes = atom["name"]
                        
                        # Convert atom name from bytes to string
                        atom_name = "".join([chr(c + 32) for c in atom_name_bytes if c != 0])
                        
                        # Format atom name according to PDB standard:
                        # - Right-justify single-letter atom names (e.g. " C  ")
                        # - For most standard atoms, the first character is the element symbol (e.g. " CA " for alpha carbon)
                        # - For hydrogen, use a specific format (e.g. " H  " or " HA ")
                        if len(atom_name.strip()) == 1:
                            atom_name = f" {atom_name.strip()}  "
                        elif len(atom_name.strip()) == 2:
                            # If it's not hydrogen (which starts with H), right-justify
                            if atom_name.strip()[0] == 'H':
                                atom_name = f" {atom_name.strip()} "
                            else:
                                atom_name = f" {atom_name.strip()} "
                        elif len(atom_name.strip()) == 3:
                            atom_name = f" {atom_name.strip()}"
                        elif len(atom_name.strip()) == 4:
                            atom_name = atom_name.strip()
                        
                        # Get element
                        element = ""
                        if periodic_table and atom["element"].item() > 0:
                            element = periodic_table.GetElementSymbol(atom["element"].item()).upper()
                        else:
                            # Fallback: extract element from atom name
                            if atom_name.strip()[0].isalpha():
                                element = atom_name.strip()[0].upper()
                            else:
                                element = "C"  # Default to carbon if unknown
                        
                        # Get residue information
                        res_id = atom_to_res.get(j, j//4)  # Fallback: approximate
                        if res_id < len(structure.residues):
                            residue = structure.residues[res_id]
                            res_name = str(residue["name"][:3])
                        else:
                            res_name = "UNK"  # Unknown residue
                        
                        # Get chain ID
                        chain_id = atom_to_chain.get(j, "A")  # Fallback: chain A
                        
                        # Get residue index (1-based)
                        residue_index = (res_id % 10000) + 1  # Avoid overflow
                        
                        # Insertion code (blank by default)
                        insertion_code = " "
                    else:
                        # Fallback for atoms beyond structure atom count
                        atom_name = " CA "
                        res_name = "UNK"
                        chain_id = "X"
                        residue_index = (j % 10000) + 1
                        element = "C"
                        insertion_code = " "
                    
                    # Standard PDB format:
                    # COLUMNS        DATA TYPE       CONTENTS
                    # 1 -  6        Record name     "ATOM  "
                    # 7 - 11        Integer         Atom serial number
                    # 13 - 16       Atom            Atom name
                    # 17            Character       Alternate location indicator (or space)
                    # 18 - 20       Residue name    Residue name
                    # 22            Character       Chain identifier
                    # 23 - 26       Integer         Residue sequence number
                    # 27            AChar           Code for insertion of residues (or space)
                    # 31 - 38       Real(8.3)       Orthogonal coordinates for X
                    # 39 - 46       Real(8.3)       Orthogonal coordinates for Y
                    # 47 - 54       Real(8.3)       Orthogonal coordinates for Z
                    # 55 - 60       Real(6.2)       Occupancy
                    # 61 - 66       Real(6.2)       Temperature factor
                    # 77 - 78       LString(2)      Element symbol, right-justified
                    # 79 - 80       LString(2)      Charge on the atom
                    
                    # Format according to PDB standard columns
                    pdb_content += (
                        f"ATOM  {atom_index:5d} {atom_name:<4}"  # Columns 1-16
                        f" "  # Column 17 (alt location)
                        f"{res_name:3s} "  # Columns 18-21 (residue name + space)
                        f"{chain_id:1s}"  # Column 22 (chain ID)
                        f"{residue_index:4d}"  # Columns 23-26 (residue number)
                        f"{insertion_code:1s}   "  # Column 27 + 3 spaces
                        f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"  # Columns 31-54 (x,y,z)
                        f"  1.00 20.00          "  # Columns 55-76 (occupancy, temp factor, segment)
                        f"{element:>2s}"  # Columns 77-78 (element, right-justified)
                        f"  \n"  # Columns 79-80 (charge) + newline
                    )
                    atom_index += 1
                
                # Add TER records between chains
                last_chain = None
                last_res_name = None
                last_residue_index = None
                
                # Loop again to add TER records at chain breaks
                for j in range(len(coords)):
                    if j < len(structure.atoms):
                        chain_id = atom_to_chain.get(j, "A")
                        res_id = atom_to_res.get(j, j//4)
                        
                        if res_id < len(structure.residues):
                            residue = structure.residues[res_id]
                            res_name = str(residue["name"][:3])
                        else:
                            res_name = "UNK"
                            
                        residue_index = (res_id % 10000) + 1
                        
                        # Add TER record at chain breaks
                        if chain_id != last_chain and last_chain is not None:
                            # Format TER record according to PDB standard
                            # COLUMNS       DATA TYPE      FIELD         DEFINITION
                            # 1 -  6       Record name    "TER   "
                            # 7 - 11       Integer        serial        Serial number
                            # 18 - 20      Residue name   resName       Residue name
                            # 22           Character      chainID       Chain identifier
                            # 23 - 26      Integer        resSeq        Residue sequence number
                            # 27           AChar          iCode         Insertion code
                            pdb_content += (
                                f"TER   {atom_index:5d}      "
                                f"{last_res_name:3s} {last_chain:1s}{last_residue_index:4d} \n"
                            )
                            atom_index += 1
                            
                        last_chain = chain_id
                        last_res_name = res_name
                        last_residue_index = residue_index
                
                # Add final TER record
                if last_chain is not None:
                    pdb_content += (
                        f"TER   {atom_index:5d}      "
                        f"{last_res_name:3s} {last_chain:1s}{last_residue_index:4d} \n"
                    )
                    atom_index += 1
                
                pdb_content += "END\n"
                
                # Save PDB file
                pdb_file = traj_dir / f"timestep_{timestep:03d}_sigma_{sigma:.6f}.pdb"
                with pdb_file.open("w") as f:
                    f.write(pdb_content)

    def save_trajectory_as_npz(
        self,
        trajectory: dict,
        record_id: str,
        model_idx: int,
    ) -> None:
        """
        Save trajectory coordinates as NPZ files.
        
        Parameters
        ----------
        trajectory : dict
            Dictionary containing intermediate coordinates and metadata.
        record_id : str
            Record identifier for file naming.
        model_idx : int
            Model index for file naming.
        """
        traj_dir = self.output_dir / "trajectories" / record_id / f"model_{model_idx}"
        traj_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for NPZ saving
        save_data = {
            'timesteps': np.array(trajectory['timesteps']),
            'sigmas': np.array(trajectory['sigmas']),
            'metadata': trajectory['metadata']
        }
        
        # Convert tensors to numpy arrays at selected timesteps
        selected_indices = range(0, len(trajectory['timesteps']), self.intermediate_save_every)
        
        noisy_coords_list = []
        denoised_coords_list = []
        final_coords_list = []
        selected_timesteps = []
        selected_sigmas = []
        
        for i in selected_indices:
            if i < len(trajectory['timesteps']) and trajectory['denoised_coords'][i] is not None:
                selected_timesteps.append(trajectory['timesteps'][i])
                selected_sigmas.append(trajectory['sigmas'][i])
                noisy_coords_list.append(trajectory['noisy_coords'][i].cpu().numpy())
                denoised_coords_list.append(trajectory['denoised_coords'][i].cpu().numpy())
                final_coords_list.append(trajectory['final_coords'][i].cpu().numpy())
        
        if noisy_coords_list:
            save_data.update({
                'selected_timesteps': np.array(selected_timesteps),
                'selected_sigmas': np.array(selected_sigmas),
                'noisy_coords': np.stack(noisy_coords_list),
                'denoised_coords': np.stack(denoised_coords_list),
                'final_coords': np.stack(final_coords_list),
            })
            
            # Save NPZ file
            npz_file = traj_dir / f"trajectory_data.npz"
            np.savez_compressed(npz_file, **save_data)

    def save_intermediate_coordinates(
        self,
        trajectory: dict,
        record: Record,
        model_idx: int,
        structure: Structure,
    ) -> None:
        """
        Save intermediate coordinates in the specified format(s).
        
        Parameters
        ----------
        trajectory : dict
            Dictionary containing intermediate coordinates and metadata.
        record : Record
            Record object for identification.
        model_idx : int
            Model index for file naming.
        structure : Structure
            Structure object for atom information.
        """
        if not self.save_intermediate_coords:
            return
            
        if self.intermediate_output_format in ["pdb", "both"]:
            self.save_trajectory_as_pdb(trajectory, record.id, model_idx, structure)
            
        if self.intermediate_output_format in ["npz", "both"]:
            self.save_trajectory_as_npz(trajectory, record.id, model_idx)
            
        # Save trajectory analysis summary
        self.save_trajectory_analysis(trajectory, record.id, model_idx)

    def save_trajectory_analysis(
        self,
        trajectory: dict,
        record_id: str,
        model_idx: int,
    ) -> None:
        """
        Save trajectory analysis summary.
        
        Parameters
        ----------
        trajectory : dict
            Dictionary containing intermediate coordinates and metadata.
        record_id : str
            Record identifier for file naming.
        model_idx : int
            Model index for file naming.
        """
        traj_dir = self.output_dir / "trajectories" / record_id / f"model_{model_idx}"
        traj_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate trajectory statistics
        analysis = {
            "record_id": record_id,
            "model_idx": model_idx,
            "num_timesteps": len(trajectory['timesteps']),
            "initial_sigma": trajectory['metadata']['init_sigma'],
            "final_sigma": trajectory['sigmas'][-1] if trajectory['sigmas'] else 0.0,
            "coordinate_shape": trajectory['metadata']['shape'],
            "sampling_steps": trajectory['metadata']['num_sampling_steps'],
            "multiplicity": trajectory['metadata']['multiplicity'],
        }
        
        # Calculate RMSD evolution if possible
        if len(trajectory['denoised_coords']) > 2:
            rmsds = []
            coords_list = [coord for coord in trajectory['denoised_coords'][1:] if coord is not None]
            
            for i in range(1, len(coords_list)):
                if coords_list[i-1] is not None and coords_list[i] is not None:
                    prev_coords = coords_list[i-1][0]  # First sample
                    curr_coords = coords_list[i][0]
                    rmsd = torch.sqrt(((prev_coords - curr_coords) ** 2).mean()).item()
                    rmsds.append(rmsd)
            
            if rmsds:
                analysis.update({
                    "mean_step_rmsd": float(np.mean(rmsds)),
                    "std_step_rmsd": float(np.std(rmsds)),
                    "max_step_rmsd": float(np.max(rmsds)),
                    "min_step_rmsd": float(np.min(rmsds)),
                })
                
                # Calculate overall structural change
                if coords_list:
                    first_coords = coords_list[0][0]
                    last_coords = coords_list[-1][0]
                    overall_rmsd = torch.sqrt(((first_coords - last_coords) ** 2).mean()).item()
                    analysis["overall_rmsd"] = float(overall_rmsd)
        
        # Save analysis
        analysis_file = traj_dir / "trajectory_analysis.json"
        with analysis_file.open("w") as f:
            json.dump(analysis, f, indent=2)
