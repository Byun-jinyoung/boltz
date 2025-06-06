import Bio
from Bio import PDB
from Bio.PDB import PDBParser, MMCIFParser, Select
from Bio.SeqUtils import seq1
from typing import List, Tuple
from pathlib import Path
import numpy as np

# Needleman-Wunsch Alignment Algorithm
class NeedlemanWunsch:
    def __init__(self, match_score: int = 1, mismatch_penalty: int = -1, gap_penalty: int = -2):
        self.match_score = match_score
        self.mismatch_penalty = mismatch_penalty
        self.gap_penalty = gap_penalty

    def score(self, a: str, b: str) -> int:
        return self.match_score if a == b else self.mismatch_penalty

    def align(self, seq1: str, seq2: str) -> Tuple[str, str, List[Tuple[int, int]]]:
        n, m = len(seq1), len(seq2)
        dp = np.zeros((n + 1, m + 1), dtype=int)

        for i in range(n + 1):
            dp[i][0] = i * self.gap_penalty
        for j in range(m + 1):
            dp[0][j] = j * self.gap_penalty

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                match = dp[i - 1][j - 1] + self.score(seq1[i - 1], seq2[j - 1])
                delete = dp[i - 1][j] + self.gap_penalty
                insert = dp[i][j - 1] + self.gap_penalty
                dp[i][j] = max(match, delete, insert)

        aligned_seq1, aligned_seq2 = [], []
        mapping = []
        i, j = n, m
        while i > 0 or j > 0:
            current_score = dp[i][j]
            if i > 0 and j > 0 and current_score == dp[i - 1][j - 1] + self.score(seq1[i - 1], seq2[j - 1]):
                aligned_seq1.append(seq1[i - 1])
                aligned_seq2.append(seq2[j - 1])
                mapping.append((i - 1, j - 1))
                i -= 1
                j -= 1
            elif i > 0 and current_score == dp[i - 1][j] + self.gap_penalty:
                aligned_seq1.append(seq1[i - 1])
                aligned_seq2.append('-')
                i -= 1
            else:
                aligned_seq1.append('-')
                aligned_seq2.append(seq2[j - 1])
                j -= 1

        aligned_seq1 = ''.join(reversed(aligned_seq1))
        aligned_seq2 = ''.join(reversed(aligned_seq2))
        mapping = list(reversed(mapping))

        return aligned_seq1, aligned_seq2, mapping



def remove_duplicate_residues(chain):
    seen_residues = set()
    residues_to_remove = []
    for residue in chain:
        residue_id = (residue.id[1], residue.id[2])  # (residue_number, altloc)
        if residue_id in seen_residues:
            residues_to_remove.append(residue)
        else:
            seen_residues.add(residue_id)
    for residue in residues_to_remove:
        chain.detach_child(residue.id)

class AltLocSelector(PDB.Select):
    """Selects only residues with the default altloc (' ')."""
    def accept_residue(self, residue):
        # Filter out alternate locations
        if residue.id[2] != " ":
            return 0
        return 1


class StructureSequenceMapper:
    def __init__(self, gap_penalty: float = -2, match_score: float = 1, mismatch_penalty: float = -1):
        self.needleman_wunsch = NeedlemanWunsch(match_score, mismatch_penalty, gap_penalty)

    def _get_structure_parser(self, file_path: str) -> PDB.Structure.Structure:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Structure file {file_path} not found")

        if file_path.suffix.lower() in ['.pdb', '.ent']:
            parser = PDBParser(QUIET=True, PERMISSIVE=True)
        elif file_path.suffix.lower() in ['.cif', '.mmcif']:
            parser = MMCIFParser(QUIET=True)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        structure = parser.get_structure("structure", str(file_path))

        # Remove duplicate residues
        for model in structure:
            for chain in model:
                remove_duplicate_residues(chain)
        return structure

    def extract_sequence_from_structure(self, structure_file: str, chain_id: str) -> Tuple[str, List[PDB.Residue.Residue]]:
        structure = self._get_structure_parser(structure_file)
        sequence = ""
        residues = []
        seen_positions = set()

        for model in structure:
            for chain in model:
                if chain.id == chain_id:
                    for residue in chain:
                        res_id = residue.id[1]
                        if (residue.id[0] == " " and 
                            residue.get_resname() in PDB.Polypeptide.aa3 and 
                            res_id not in seen_positions):
                            residues.append(residue)
                            sequence += seq1(residue.get_resname())
                            seen_positions.add(res_id)
                    if sequence:
                        return sequence, residues
        raise ValueError(f"Chain {chain_id} not found or contains no valid residues in {structure_file}")


    def map_sequences(self, structure_file: str, chain_id: str, given_sequence: str) -> Tuple[str, str, List[Tuple[int, int]], dict]:
        try:
            structure_sequence, structure_residues = self.extract_sequence_from_structure(structure_file, chain_id)
            
            # Perform Needleman-Wunsch alignment
            aligned_struct, aligned_given, mapping = self.needleman_wunsch.align(structure_sequence, given_sequence)

            # Check aligned lengths
            assert len(aligned_struct) == len(aligned_given), "Aligned sequences have mismatched lengths!"
            
            # Convert to residue-based mapping
            final_mapping = [
                #(given_idx + 1, structure_residues[struct_idx].id[1]) # template 구조에 있는 residue number 그대로 사용
                (given_idx, struct_idx)  # Renumber structure residues to start at 0
                for struct_idx, given_idx in mapping
            ]
            # Initialize final mapping
            final_mapping = []
            struct_idx, given_idx = 0, 0

            # Process aligned sequences to include gaps
            for a_struct, a_given in zip(aligned_struct, aligned_given):
                #print(a_struct, a_given)
                if a_struct != '-' and a_given != '-':
                    final_mapping.append((given_idx, struct_idx))
                
                struct_idx += 1
                given_idx += 1
            
            print(final_mapping)            

            stats = {
                "aligned_length": len(mapping), 
                "structure_length": len(structure_sequence), 
                "given_length": len(given_sequence)
            }

            return aligned_struct, aligned_given, final_mapping, stats

        except Exception as e:
            print(e)
            print('fuck')
            return [], {"error": str(e)}
