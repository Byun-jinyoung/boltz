import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
from Bio import PDB
from Bio.PDB import PDBParser, MMCIFParser
from Bio.SeqUtils import seq1
import warnings

from boltz.data.types import MinDistance
from boltz.data.parse.struct2seq import StructureSequenceMapper


def calculate_sequence_identity(
    aligned_seq1: str, aligned_seq2: str
    ) -> float:
    """
    Calculate sequence identity from aligned sequences.
    
    Parameters
    ----------
    aligned_seq1 : str
        First aligned sequence
    aligned_seq2 : str
        Second aligned sequence
        
    Returns
    -------
    float
        Sequence identity (0.0 to 1.0)
    """
    if len(aligned_seq1) != len(aligned_seq2):
        return 0.0
        
    matches = sum(1 for a, b in zip(aligned_seq1, aligned_seq2) 
        if a == b and a != '-' and b != '-'
    )
    aligned_length = sum(1 for a, b in zip(aligned_seq1, aligned_seq2) 
        if a != '-' and b != '-'
    )
    return matches / aligned_length if aligned_length > 0 else 0.0


class TemplateConstraintGenerator:
    """
    Template-based distance constraint generator for protein structure prediction.
    
    This class generates multiple distance constraints based on template structures
    to prevent non-physical conformations when using single atom pair constraints.
    """
    
    def __init__(
        self, 
        distance_threshold: float = 20.0,
        cb_distance_cutoff: float = 50.0,
        min_sequence_identity: float = 0.6,
        gap_penalty: float = -2.0
    ):
        """
        Initialize the template constraint generator.
        
        Parameters
        ----------
        distance_threshold : float
            Maximum distance to consider for constraints (Angstroms)
        cb_distance_cutoff : float
            Maximum Cb-Cb distance for constraint generation (Angstroms)
        min_sequence_identity : float
            Minimum sequence identity for reliable alignment
        gap_penalty : float
            Gap penalty for sequence alignment
        """
        self.distance_threshold = distance_threshold
        self.cb_distance_cutoff = cb_distance_cutoff
        self.min_sequence_identity = min_sequence_identity
        self.mapper = StructureSequenceMapper(gap_penalty=gap_penalty)
        
        print(f"  INFO: cb_distance_cutoff: {cb_distance_cutoff} (Angstroms)")
    
    def _extract_cb_coordinates(
        self, 
        structure_file: str, 
        chain_id: str
    ) -> Dict[int, np.ndarray]:
        """
        Extract Cb coordinates from template structure.
        
        Parameters
        ----------
        structure_file : str
            Path to template structure file
        chain_id : str
            Chain identifier
            
        Returns
        -------
        Dict[int, np.ndarray]
            Dictionary mapping residue index to Cb coordinates
        """
        try:
            structure = self.mapper._get_structure_parser(structure_file)
            cb_coords = {}
            
            for model in structure:
                for chain in model:
                    if chain.id == chain_id:
                        residue_idx = 0
                        for residue in chain:
                            if residue.id[0] == " ":  # Standard residue                                
                                res_name = residue.resname.strip() # Get residue name
                                
                                cb_atom = None
                                if (res_name == "GLY") or (res_name == "PRO"):                                     
                                    pass
                                else:                                    
                                    if 'CB' in residue: # For other residues, try CB first
                                        cb_atom = residue['CB']                                    
                                    else:
                                        warnings.warn(f"Missing both CB and CA atoms in {res_name} residue at position {residue_idx}")
                                
                                if cb_atom is not None:
                                    cb_coords[residue_idx] = np.array(cb_atom.get_coord())
                                # else:
                                #     warnings.warn(f"Missing CB atom in {res_name}{residue_idx} residue")
                                    
                                residue_idx += 1
                        break
                break
                        
            return cb_coords
            
        except Exception as e:
            warnings.warn(f"Failed to extract Cb coordinates: {e}")
            return {}
    
    
    def _compute_distance_map(        
        self,
        cb_coords: Dict[int, np.ndarray]
    ) -> Dict[Tuple[int, int], float]:
        """
        Compute Cb-Cb distance map from coordinates.
        
        Parameters
        ----------
        cb_coords : Dict[int, np.ndarray]
            Dictionary mapping residue index to Cb coordinates
            
        Returns
        -------
        Dict[Tuple[int, int], float]
            Distance map with residue pair tuples as keys
        """
        distance_map = {}
        residue_indices = list(cb_coords.keys())
        
        for i, idx1 in enumerate(residue_indices):
            for idx2 in residue_indices[i+1:]:
                if idx1 in cb_coords and idx2 in cb_coords:
                    coord1 = cb_coords[idx1]
                    coord2 = cb_coords[idx2]
                    distance = np.linalg.norm(coord1 - coord2)
                    
                    # apply distance filtering - only include distances below cutoff
                    if 11 <= distance <= self.cb_distance_cutoff:
                        distance_map[(idx1, idx2)] = distance
                        
        return distance_map
    
    def generate_template_constraints(
        self,
        query_sequence: str,
        template_structure: str,
        template_chain_id: str,
        query_chain_id: str = "A",
        constraint_type: str = "nmr_distance",
        distance_buffer: float = 0.1,
        base_weight: float = 1.0,
        sequence_identity_weight: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate template-based distance constraints.
        
        Parameters
        ----------
        query_sequence : str
            Query protein sequence
        template_structure : str
            Path to template structure file
        template_chain_id : str
            Template chain identifier
        query_chain_id : str
            Query chain identifier (default: "A")
        constraint_type : str
            Type of constraint to generate ("min_distance" or "nmr_distance", default: "nmr_distance")
        distance_buffer : float
            Buffer percentage for NMR bounds (default: 0.1 = 10%)
        base_weight : float
            Base weight for NMR constraints (default: 1.0)
        sequence_identity_weight : bool
            Whether to scale weight by sequence identity (default: True)
            
        Returns
        -------
        List[Dict[str, Any]]
            List of constraint dictionaries compatible with boltz schema
        """
        try:
            # Map sequences
            aligned_struct, aligned_given, mapping, stats = self.mapper.map_sequences(
                template_structure, template_chain_id, query_sequence
            )
            
            # Calculate sequence identity
            seq_identity = calculate_sequence_identity(aligned_struct, aligned_given)
            print(f"  INFO: Sequence identity: {seq_identity:.3f}")
            
            if seq_identity < self.min_sequence_identity:
                warnings.warn(
                    f"Low sequence identity ({seq_identity:.3f}) "
                    f"may lead to unreliable constraints"
                )
            
            # Extract CB coordinates from template
            cb_coords = self._extract_cb_coordinates(template_structure, template_chain_id)
            if not cb_coords:
                warnings.warn("No CB coordinates extracted from template")
                return []
            
            # Compute distance map
            distance_map = self._compute_distance_map(cb_coords)
            if not distance_map:
                warnings.warn("No valid distances found in template")
                return []
            
            # Generate constraints based on mapping
            constraints = []
            mapping_dict = dict(mapping)
            
            for (template_idx1, template_idx2), distance in distance_map.items():
                # Find corresponding query indices
                query_idx1 = None
                query_idx2 = None
                
                for query_idx, template_idx in mapping_dict.items():
                    if template_idx == template_idx1:
                        query_idx1 = query_idx
                    elif template_idx == template_idx2:
                        query_idx2 = query_idx
                
                if query_idx1 is not None and query_idx2 is not None:
                    # Generate constraint based on type
                    if constraint_type == "min_distance":
                        constraint = {
                            "min_distance": {
                                "atom1": [query_chain_id, query_idx1 + 1, "CB"],  # 1-indexed
                                "atom2": [query_chain_id, query_idx2 + 1, "CB"],  # 1-indexed
                                "distance": float(distance)
                            }
                        }
                    elif constraint_type == "nmr_distance":
                        # Calculate bounds with buffer
                        lower_bound = max(0.0, distance * (1 - distance_buffer))
                        upper_bound = distance * (1 + distance_buffer)
                        
                        # Calculate weight
                        weight = base_weight
                        if sequence_identity_weight:
                            weight *= seq_identity
                        
                        constraint = {
                            "nmr_distance": {
                                "atom1": [query_chain_id, query_idx1 + 1, "CB"],  # 1-indexed
                                "atom2": [query_chain_id, query_idx2 + 1, "CB"],  # 1-indexed
                                "lower_bound": float(lower_bound),
                                "upper_bound": float(upper_bound),
                                "weight": float(weight)
                            }
                        }
                    else:
                        raise ValueError(f"Unknown constraint type: {constraint_type}")
                    
                    constraints.append(constraint)
            
            print(f"  INFO: Generated {len(constraints)} template constraints")
            return constraints
            
        except Exception as e:
            warnings.warn(f"Failed to generate template constraints: {e}")
            return []
    
    def generate_constraints_for_boltz_schema(
        self,
        schema_data: Dict[str, Any],
        template_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate constraints and integrate into boltz schema.
        
        Parameters
        ----------
        schema_data : Dict[str, Any]
            Original boltz schema data
        template_info : Dict[str, Any]
            Template information
            
        Returns
        -------
        Dict[str, Any]
            Schema data with added template constraints
        """
        try:
            # Extract template constraints
            constraints = self.generate_template_constraints(**template_info)
            
            # Add to schema
            if "constraints" not in schema_data:
                schema_data["constraints"] = []
            
            schema_data["constraints"].extend(constraints)
            
            return schema_data
            
        except Exception as e:
            warnings.warn(f"Failed to integrate template constraints: {e}")
            return schema_data


def apply_template_constraints(
    schema_data: Dict[str, Any],
    template_structure: str,
    template_chain_id: str,
    target_chain_id: str = "A",
    constraint_type: str = "nmr_distance",
    distance_buffer: float = 0.1,
    base_weight: float = 1.0,
    sequence_identity_weight: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to apply template constraints to schema.
    
    Parameters
    ----------
    schema_data : Dict[str, Any]
        Boltz schema data
    template_structure : str
        Path to template structure
    template_chain_id : str
        Template chain ID
    target_chain_id : str
        Target chain ID
    constraint_type : str
        Constraint type
    distance_buffer : float
        Distance buffer for NMR constraints
    base_weight : float
        Base weight for constraints
    sequence_identity_weight : bool
        Whether to use sequence identity weighting
        
    Returns
    -------
    Dict[str, Any]
        Updated schema data
    """
    # Find target sequence
    target_sequence = None
    for item in schema_data.get("sequences", []):
        entity_type = next(iter(item.keys())).lower()
        if entity_type == "protein":
            chain_ids = item[entity_type]["id"]
            if isinstance(chain_ids, str):
                chain_ids = [chain_ids]
            if target_chain_id in chain_ids:
                target_sequence = item[entity_type]["sequence"]
                break
    
    if not target_sequence:
        warnings.warn(f"Could not find sequence for chain {target_chain_id}")
        return schema_data
    
    # Generate constraints
    generator = TemplateConstraintGenerator(**kwargs)
    template_info = {
        "query_sequence": target_sequence,
        "template_structure": template_structure,
        "template_chain_id": template_chain_id,
        "query_chain_id": target_chain_id,
        "constraint_type": constraint_type,
        "distance_buffer": distance_buffer,
        "base_weight": base_weight,
        "sequence_identity_weight": sequence_identity_weight
    }
    
    return generator.generate_constraints_for_boltz_schema(schema_data, template_info)