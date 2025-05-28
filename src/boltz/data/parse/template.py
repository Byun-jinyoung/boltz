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
        cb_distance_cutoff: float = 100.0,
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
                                # Try Cb first, fall back to Ca for Glycine
                                cb_atom = None
                                if 'CB' in residue:
                                    cb_atom = residue['CB']
                                elif 'CA' in residue:  # Glycine case
                                    cb_atom = residue['CA']
                                
                                if cb_atom is not None:
                                    cb_coords[residue_idx] = np.array(cb_atom.get_coord())
                                
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
                    
                    if distance <= self.cb_distance_cutoff:
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
            List of constraint dictionaries compatible with Boltz schema
        """
        try:
            # Validate constraint type
            if constraint_type not in ["min_distance", "nmr_distance"]:
                raise ValueError(f"Invalid constraint_type: {constraint_type}. Must be 'min_distance' or 'nmr_distance'")
            
            # Perform sequence alignment
            aligned_template, aligned_query, mapping, stats = self.mapper.map_sequences(
                template_structure, template_chain_id, query_sequence
            )
            
            # Check sequence identity
            #seq_identity = self._calculate_sequence_identity(aligned_template, aligned_query)
            seq_identity = calculate_sequence_identity(aligned_template, aligned_query)
            if seq_identity < self.min_sequence_identity:
                warnings.warn(
                    f"Low sequence identity ({seq_identity:.2f}). " f"Constraints may not be reliable."
                )
            
            # Extract Cb coordinates from template
            cb_coords = self._extract_cb_coordinates(template_structure, template_chain_id)
            if not cb_coords:
                raise ValueError("No Cb coordinates extracted from template. Is it only backbone structure?")
            
            # Compute distance map
            distance_map = self._compute_distance_map(cb_coords)
            
            # Generate constraints based on alignment mapping
            constraints = []
            
            # Create mapping dictionaries for efficient lookup
            template_to_query = {t_idx: q_idx for q_idx, t_idx in mapping}
            
            for (temp_i, temp_j), distance in distance_map.items():
                # Find corresponding query residues for template residue pair
                if temp_i in template_to_query and temp_j in template_to_query:
                    query_i = template_to_query[temp_i]
                    query_j = template_to_query[temp_j]
                    
                    # Avoid very close residues (sequence separation)
                    if abs(query_i - query_j) > 5:  # Skip nearby residues
                        # Use CA for Glycine, CB for other residues
                        atom1_name = "CA" if query_sequence[query_i] == "G" else "CB"
                        atom2_name = "CA" if query_sequence[query_j] == "G" else "CB"
                        
                        # Generate constraint based on type
                        if constraint_type == "min_distance":
                            # Original min_distance format
                            constraint = {
                                constraint_type: {
                                    "atom1": [query_chain_id, query_i + 1, atom1_name],  # 1-indexed
                                    "atom2": [query_chain_id, query_j + 1, atom2_name],  # 1-indexed
                                    "distance": round(float(distance), 3)
                                }
                            }
                        elif constraint_type == "nmr_distance":
                            # Calculate bounds with buffer
                            lower_bound = distance * (1.0 - distance_buffer)
                            upper_bound = distance * (1.0 + distance_buffer)
                            
                            # Calculate weight based on sequence identity and distance
                            weight = base_weight
                            # if sequence_identity_weight: (forge later)
                            #     # Scale weight by sequence identity (higher identity = higher weight)
                            #     weight *= seq_identity
                            #     # Scale weight by inverse distance (closer pairs = higher weight)
                            #     # Normalize by typical protein distance range (5-50 Å)
                            #     distance_factor = max(0.1, (50.0 - min(distance, 50.0)) / 45.0)                                
                            #     weight *= distance_factor
                            print(lower_bound, upper_bound, weight)                            
                            # NMR distance format with bounds and weight
                            constraint = {
                                constraint_type: {
                                    "atom1": [query_chain_id, query_i + 1, atom1_name],  # 1-indexed
                                    "atom2": [query_chain_id, query_j + 1, atom2_name],  # 1-indexed
                                    "lower_bound": round(float(lower_bound), 3),
                                    "upper_bound": round(float(upper_bound), 3),
                                    "weight": round(float(weight), 3)
                                }
                            }
                        
                        constraints.append(constraint)
            
            # Remove duplicate constraints
            unique_constraints = []
            seen_pairs = set()
            
            for constraint in constraints:
                if constraint_type in constraint:
                    atom1_info = tuple(constraint[constraint_type]["atom1"])
                    atom2_info = tuple(constraint[constraint_type]["atom2"])
                    
                    # Ensure consistent ordering
                    if atom1_info > atom2_info:
                        atom1_info, atom2_info = atom2_info, atom1_info
                        constraint[constraint_type]["atom1"] = list(atom1_info)
                        constraint[constraint_type]["atom2"] = list(atom2_info)
                    
                    pair_key = (atom1_info, atom2_info)
                    if pair_key not in seen_pairs:
                        seen_pairs.add(pair_key)
                        unique_constraints.append(constraint)
            
            print(f"  INFO: generated {len(unique_constraints)} template-based {constraint_type} constraints")
            print(f"  INFO: sequence identity: {seq_identity:.3f}")
            print(f"  INFO: alignment length: {stats.get('aligned_length', 0)}")
            if constraint_type == "nmr_distance":
                print(f"  INFO: distance buffer: {distance_buffer:.1%}")
                print(f"  INFO: base weight: {base_weight}")
            
            return unique_constraints
            
        except Exception as e:
            warnings.warn(f"Failed to generate template constraints: {e}")
            return []
    
    def generate_constraints_for_boltz_schema(
        self,
        schema_data: Dict[str, Any],
        template_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Add template-based constraints to Boltz schema data.
        
        Parameters
        ----------
        schema_data : Dict[str, Any]
            Original Boltz schema data
        template_info : Dict[str, Any]
            Template information containing:
            - structure_path: path to template structure
            - chain_id: template chain identifier
            - target_chain_id: target chain identifier
            - constraint_type: type of constraint ("min_distance" or "nmr_distance", optional)
            - distance_buffer: buffer percentage for NMR bounds (optional)
            - base_weight: base weight for NMR constraints (optional)
            - sequence_identity_weight: whether to scale weight by sequence identity (optional)
            
        Returns
        -------
        Dict[str, Any]
            Modified schema data with template constraints added
        """
        # Find target protein sequence
        target_sequence = None
        target_chain_id = template_info.get("target_chain_id", "A")
        
        for seq_entry in schema_data.get("sequences", []):
            if "protein" in seq_entry:
                protein_data = seq_entry["protein"]
                if protein_data["id"] == target_chain_id:
                    target_sequence = protein_data["sequence"]
                    break
        
        if target_sequence is None:
            warnings.warn(f"Target protein chain {target_chain_id} not found")
            return schema_data
        
        # Extract constraint generation parameters
        constraint_kwargs = {
            "constraint_type": template_info.get("constraint_type", "nmr_distance"),
            "distance_buffer": template_info.get("distance_buffer", 0.1),
            "base_weight": template_info.get("base_weight", 1.0),
            "sequence_identity_weight": template_info.get("sequence_identity_weight", True)
        }
        
        # Generate template constraints
        template_constraints = self.generate_template_constraints(
            query_sequence=target_sequence,
            template_structure=template_info["structure_path"],
            template_chain_id=template_info["chain_id"],
            query_chain_id=target_chain_id,
            **constraint_kwargs
        )
        
        # Add constraints to schema
        if template_constraints:
            if "constraints" not in schema_data:
                schema_data["constraints"] = []
            
            schema_data["constraints"].extend(template_constraints)
        
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
    Convenience function to apply template-based constraints to schema data.
    
    Parameters
    ----------
    schema_data : Dict[str, Any]
        Original Boltz schema data
    template_structure : str
        Path to template structure file
    template_chain_id : str
        Template chain identifier
    target_chain_id : str
        Target chain identifier (default: "A")
    constraint_type : str
        Type of constraint to generate ("min_distance" or "nmr_distance", default: "nmr_distance")
    distance_buffer : float
        Buffer percentage for NMR bounds (default: 0.1 = 10%)
    base_weight : float
        Base weight for NMR constraints (default: 1.0)
    sequence_identity_weight : bool
        Whether to scale weight by sequence identity (default: True)
    **kwargs
        Additional parameters for TemplateConstraintGenerator
        
    Returns
    -------
    Dict[str, Any]
        Modified schema data with template constraints
    """
    generator = TemplateConstraintGenerator(**kwargs)
    
    template_info = {
        "structure_path": template_structure,
        "chain_id": template_chain_id,
        "target_chain_id": target_chain_id,
        "constraint_type": constraint_type,
        "distance_buffer": distance_buffer,
        "base_weight": base_weight,
        "sequence_identity_weight": sequence_identity_weight
    }
    
    return generator.generate_constraints_for_boltz_schema(schema_data, template_info) 