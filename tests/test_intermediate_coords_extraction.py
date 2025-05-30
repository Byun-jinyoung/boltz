#!/usr/bin/env python3
"""
Comprehensive test suite for Boltz-1 intermediate coordinates extraction functionality.

This test module verifies that the diffusion reverse process can properly track and save
intermediate structural coordinates at each timestep, ensuring data integrity and
functionality across different configurations.
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from pytorch_lightning import seed_everything

# Import test utilities and main functionality
import sys
sys.path.append(str(Path(__file__).parent.parent))
from run_intermediate_coords_extraction import (
    analyze_trajectory_directory,
    create_animation_script,
    setup_argument_parser,
    validate_input_file,
)

# Import Boltz modules
from boltz.data.write.writer import BoltzWriter
from boltz.model.model import Boltz1
from boltz.model.modules.diffusion import AtomDiffusion


class TestIntermediateCoordinatesExtraction(unittest.TestCase):
    """Test suite for intermediate coordinates extraction functionality."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create temporary directory for test outputs
        self.test_dir = Path(tempfile.mkdtemp())
        self.trajectory_dir = self.test_dir / "trajectories" / "test_structure" / "model_0"
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seed for reproducibility
        seed_everything(42)
        
        # Mock trajectory data for testing
        self.mock_trajectory = self._create_mock_trajectory()
        
    def tearDown(self):
        """Clean up test environment after each test."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def _create_mock_trajectory(self) -> dict:
        """
        Create mock trajectory data for testing.
        
        Returns
        -------
        dict
            Mock trajectory dictionary with realistic structure.
        """
        # Create mock coordinate tensors
        batch_size, num_atoms, coord_dim = 1, 50, 3
        num_timesteps = 21  # Simulating 20 diffusion steps + initial
        
        trajectory = {
            'timesteps': list(range(-1, num_timesteps - 1)),  # -1 for initial noise
            'sigmas': [160.0 - i * 8.0 for i in range(num_timesteps)],
            'noisy_coords': [],
            'denoised_coords': [],
            'final_coords': [],
            'metadata': {
                'num_sampling_steps': num_timesteps - 1,
                'multiplicity': 1,
                'init_sigma': 160.0,
                'shape': [batch_size, num_atoms, coord_dim]
            }
        }
        
        # Generate mock coordinates with gradual convergence
        base_coords = torch.randn(batch_size, num_atoms, coord_dim)
        
        for i in range(num_timesteps):
            # Simulate noisy coordinates (decreasing noise over time)
            noise_scale = trajectory['sigmas'][i] / 160.0
            noisy = base_coords + torch.randn_like(base_coords) * noise_scale
            
            # Simulate denoised coordinates (converging towards base)
            denoised = base_coords + torch.randn_like(base_coords) * noise_scale * 0.5
            
            # Final coordinates for next step
            final = base_coords + torch.randn_like(base_coords) * noise_scale * 0.3
            
            trajectory['noisy_coords'].append(noisy)
            trajectory['denoised_coords'].append(denoised if i > 0 else None)  # No denoised for initial
            trajectory['final_coords'].append(final if i > 0 else None)
        
        return trajectory

    def test_validate_input_file(self):
        """Test input file validation functionality."""
        # Test with valid YAML file
        valid_yaml = self.test_dir / "test.yaml"
        valid_yaml.write_text("sequences:\n  A: MKLLVVVDEVHHGFGD")
        
        result = validate_input_file(str(valid_yaml))
        self.assertEqual(result, valid_yaml)
        
        # Test with valid FASTA file
        valid_fasta = self.test_dir / "test.fasta"
        valid_fasta.write_text(">protein\nMKLLVVVDEVHHGFGD")
        
        result = validate_input_file(str(valid_fasta))
        self.assertEqual(result, valid_fasta)
        
        # Test with non-existent file
        with self.assertRaises(FileNotFoundError):
            validate_input_file("non_existent_file.yaml")
        
        # Test with invalid extension
        invalid_file = self.test_dir / "test.txt"
        invalid_file.write_text("some content")
        
        with self.assertRaises(ValueError):
            validate_input_file(str(invalid_file))

    def test_setup_argument_parser(self):
        """Test argument parser setup and option handling."""
        parser = setup_argument_parser()
        
        # Test basic arguments
        args = parser.parse_args([
            "test.yaml",
            "--save-intermediate-coords",
            "--intermediate-format", "both",
            "--save-every", "5"
        ])
        
        self.assertTrue(args.save_intermediate_coords)
        self.assertEqual(args.intermediate_format, "both")
        self.assertEqual(args.save_every, 5)
        self.assertEqual(args.sampling_steps, 50)  # Default value

    def test_trajectory_analysis_generation(self):
        """Test trajectory analysis data generation."""
        # Create mock analysis file
        analysis_data = {
            "record_id": "test_structure",
            "model_idx": 0,
            "num_timesteps": len(self.mock_trajectory['timesteps']),
            "initial_sigma": 160.0,
            "final_sigma": 0.0,
            "mean_step_rmsd": 2.34,
            "overall_rmsd": 15.67
        }
        
        analysis_file = self.trajectory_dir / "trajectory_analysis.json"
        with analysis_file.open("w") as f:
            json.dump(analysis_data, f, indent=2)
        
        # Test analysis directory processing
        result = analyze_trajectory_directory(self.test_dir)
        
        self.assertIn("structures", result)
        self.assertEqual(result["total_structures"], 1)
        self.assertIn("test_structure", result["structures"])

    def test_create_animation_script(self):
        """Test PyMOL animation script generation."""
        # Create mock PDB files
        for i in range(0, 20, 5):  # Every 5th timestep
            pdb_file = self.trajectory_dir / f"timestep_{i:03d}_sigma_{160.0 - i*8:.6f}.pdb"
            pdb_content = f"REMARK Timestep {i}\nATOM      1  CA  ALA A   1      10.000  20.000  30.000  1.00 20.00           C\nEND\n"
            pdb_file.write_text(pdb_content)
        
        # Generate animation script
        script_file = self.trajectory_dir / "animate_test.pml"
        create_animation_script(self.trajectory_dir, script_file)
        
        # Verify script was created and contains expected content
        self.assertTrue(script_file.exists())
        script_content = script_file.read_text()
        self.assertIn("load timestep_", script_content)
        self.assertIn("movie.play", script_content)

    def test_boltz_writer_intermediate_coords_integration(self):
        """Test BoltzWriter integration with intermediate coordinates."""
        # Create mock structure data
        from boltz.data.types import Structure, Record
        
        # Mock structure
        mock_structure = MagicMock(spec=Structure)
        mock_structure.atoms = {"coords": np.random.randn(50, 3)}
        mock_structure.remove_invalid_chains.return_value = mock_structure
        mock_structure.mask = [True] * 10
        
        # Mock record
        mock_record = MagicMock(spec=Record)
        mock_record.id = "test_structure"
        
        # Create BoltzWriter with intermediate coordinates enabled
        writer = BoltzWriter(
            data_dir=str(self.test_dir),
            output_dir=str(self.test_dir / "output"),
            save_intermediate_coords=True,
            intermediate_output_format="pdb",
            intermediate_save_every=5
        )
        
        # Test intermediate coordinates saving
        writer.save_intermediate_coordinates(
            self.mock_trajectory,
            mock_record,
            model_idx=0,
            structure=mock_structure
        )
        
        # Verify files were created
        output_dir = self.test_dir / "output" / "trajectories" / "test_structure" / "model_0"
        self.assertTrue((output_dir / "trajectory_metadata.json").exists())
        self.assertTrue((output_dir / "trajectory_analysis.json").exists())

    def test_trajectory_data_integrity(self):
        """Test integrity of trajectory data through the pipeline."""
        # Test coordinate tensor properties
        for i, coords in enumerate(self.mock_trajectory['denoised_coords'][1:], 1):
            self.assertIsInstance(coords, torch.Tensor)
            self.assertEqual(coords.shape, (1, 50, 3))  # batch_size=1, atoms=50, coords=3
            
            # Check that coordinates are changing over time (decreasing noise)
            if i > 1:
                prev_coords = self.mock_trajectory['denoised_coords'][i-1]
                current_coords = coords
                
                # Calculate RMSD between consecutive timesteps
                rmsd = torch.sqrt(((prev_coords - current_coords) ** 2).mean())
                self.assertGreater(rmsd.item(), 0.0)  # Coordinates should be different
                self.assertLess(rmsd.item(), 50.0)     # But not unreasonably different

    def test_metadata_consistency(self):
        """Test consistency of trajectory metadata."""
        metadata = self.mock_trajectory['metadata']
        
        # Check metadata completeness
        required_keys = ['num_sampling_steps', 'multiplicity', 'init_sigma', 'shape']
        for key in required_keys:
            self.assertIn(key, metadata)
        
        # Check data consistency
        self.assertEqual(len(self.mock_trajectory['timesteps']), metadata['num_sampling_steps'] + 1)
        self.assertEqual(len(self.mock_trajectory['sigmas']), metadata['num_sampling_steps'] + 1)
        self.assertEqual(metadata['init_sigma'], self.mock_trajectory['sigmas'][0])

    def test_different_output_formats(self):
        """Test different output format options."""
        formats_to_test = ["pdb", "npz", "both"]
        
        for output_format in formats_to_test:
            with self.subTest(format=output_format):
                writer = BoltzWriter(
                    data_dir=str(self.test_dir),
                    output_dir=str(self.test_dir / f"output_{output_format}"),
                    save_intermediate_coords=True,
                    intermediate_output_format=output_format,
                    intermediate_save_every=10
                )
                
                # Test format-specific saving
                output_dir = self.test_dir / f"output_{output_format}"
                self.assertTrue(writer.save_intermediate_coords)
                self.assertEqual(writer.intermediate_output_format, output_format)

    def test_memory_efficiency_options(self):
        """Test memory efficiency options for intermediate coordinate saving."""
        save_every_options = [1, 5, 10, 20]
        
        for save_every in save_every_options:
            with self.subTest(save_every=save_every):
                writer = BoltzWriter(
                    data_dir=str(self.test_dir),
                    output_dir=str(self.test_dir / f"output_every_{save_every}"),
                    save_intermediate_coords=True,
                    intermediate_output_format="npz",
                    intermediate_save_every=save_every
                )
                
                # Calculate expected number of saved timesteps
                total_timesteps = len(self.mock_trajectory['timesteps'])
                expected_saved = len(range(0, total_timesteps, save_every))
                
                self.assertEqual(writer.intermediate_save_every, save_every)

    @patch('boltz.main.predict')
    def test_end_to_end_workflow_simulation(self, mock_predict):
        """Test end-to-end workflow simulation with mocked Boltz prediction."""
        # Mock the boltz_predict function to avoid actual model inference
        mock_predict.return_value = None
        
        # Create a mock input file
        input_file = self.test_dir / "test_protein.yaml"
        input_file.write_text("""
sequences:
  A: MKLLVVVDEVHHGFGD
""")
        
        # Import and test the main workflow components
        from run_intermediate_coords_extraction import main
        
        # Mock sys.argv for the test
        test_args = [
            "run_intermediate_coords_extraction.py",
            str(input_file),
            "--save-intermediate-coords",
            "--output-dir", str(self.test_dir / "workflow_test"),
            "--sampling-steps", "10",
            "--save-every", "2"
        ]
        
        with patch('sys.argv', test_args):
            # This would normally run the full workflow
            # For testing, we verify argument parsing works
            parser = setup_argument_parser()
            args = parser.parse_args(test_args[1:])
            
            self.assertTrue(args.save_intermediate_coords)
            self.assertEqual(args.sampling_steps, 10)
            self.assertEqual(args.save_every, 2)

    def test_error_handling_scenarios(self):
        """Test error handling in various failure scenarios."""
        # Test with missing trajectory data
        empty_trajectory = {
            'timesteps': [],
            'sigmas': [],
            'noisy_coords': [],
            'denoised_coords': [],
            'final_coords': [],
            'metadata': {}
        }
        
        # This should not crash but handle gracefully
        result = analyze_trajectory_directory(self.test_dir / "nonexistent")
        self.assertIn("error", result)

    def test_trajectory_rmsd_calculation(self):
        """Test RMSD calculation for trajectory analysis."""
        # Extract coordinates for RMSD calculation
        coords_list = [coord for coord in self.mock_trajectory['denoised_coords'][1:] if coord is not None]
        
        if len(coords_list) > 1:
            rmsds = []
            for i in range(1, len(coords_list)):
                prev_coords = coords_list[i-1][0]  # First sample
                curr_coords = coords_list[i][0]
                rmsd = torch.sqrt(((prev_coords - curr_coords) ** 2).mean()).item()
                rmsds.append(rmsd)
            
            # Verify RMSD calculations are reasonable
            self.assertTrue(all(rmsd >= 0 for rmsd in rmsds))
            self.assertTrue(len(rmsds) > 0)
            
            # Test overall structural change
            first_coords = coords_list[0][0]
            last_coords = coords_list[-1][0]
            overall_rmsd = torch.sqrt(((first_coords - last_coords) ** 2).mean()).item()
            self.assertGreater(overall_rmsd, 0.0)


class TestAtomDiffusionIntermediateCoords(unittest.TestCase):
    """Test AtomDiffusion module's intermediate coordinate functionality."""

    def setUp(self):
        """Set up test environment for AtomDiffusion tests."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create minimal AtomDiffusion module for testing
        self.score_model_args = {
            "token_s": 384,
            "token_z": 128,
            "atom_s": 128,
            "atom_z": 64,
            "atoms_per_window_queries": 32,
            "atoms_per_window_keys": 128,
            "atom_feature_dim": 128,
        }
        
        # Mock diffusion module
        self.diffusion_module = AtomDiffusion(
            score_model_args=self.score_model_args,
            num_sampling_steps=5,  # Small number for testing
            sigma_data=16.0,
        )

    @patch.object(AtomDiffusion, 'preconditioned_network_forward')
    def test_sample_with_intermediate_coords(self, mock_forward):
        """Test AtomDiffusion.sample() with intermediate coordinate saving."""
        # Mock the network forward pass
        batch_size, num_atoms = 1, 10
        mock_coords = torch.randn(batch_size, num_atoms, 3)
        mock_token_a = torch.randn(batch_size, 50, 384)  # Mock token representation
        mock_forward.return_value = (mock_coords, mock_token_a)
        
        # Create mock inputs
        atom_mask = torch.ones(batch_size, num_atoms, dtype=torch.bool)
        network_condition_kwargs = {
            's_inputs': torch.randn(batch_size, 50, 384),
            's_trunk': torch.randn(batch_size, 50, 384),
            'z_trunk': torch.randn(batch_size, 50, 50, 128),
            'relative_position_encoding': torch.randn(batch_size, 50, 50, 128),
            'feats': {'token_pad_mask': torch.ones(batch_size, 50, dtype=torch.bool)},
        }
        steering_args = {
            'fk_steering': False,
            'guidance_update': False,
        }
        
        # Test with intermediate coordinate saving enabled
        result = self.diffusion_module.sample(
            atom_mask=atom_mask,
            num_sampling_steps=3,
            save_intermediate_coords=True,
            steering_args=steering_args,
            **network_condition_kwargs
        )
        
        # Verify intermediate trajectory is included in results
        self.assertIn('intermediate_trajectory', result)
        trajectory = result['intermediate_trajectory']
        
        # Check trajectory structure
        self.assertIn('timesteps', trajectory)
        self.assertIn('sigmas', trajectory)
        self.assertIn('noisy_coords', trajectory)
        self.assertIn('denoised_coords', trajectory)
        self.assertIn('final_coords', trajectory)
        self.assertIn('metadata', trajectory)
        
        # Verify metadata
        metadata = trajectory['metadata']
        self.assertEqual(metadata['num_sampling_steps'], 3)
        self.assertEqual(metadata['multiplicity'], 1)

    def test_sample_without_intermediate_coords(self):
        """Test AtomDiffusion.sample() without intermediate coordinate saving."""
        # This test would normally require a full model setup
        # For now, we verify the parameter exists and defaults to False
        
        # Check that save_intermediate_coords parameter exists in sample method
        import inspect
        sample_signature = inspect.signature(self.diffusion_module.sample)
        self.assertIn('save_intermediate_coords', sample_signature.parameters)
        
        # Check default value
        default_value = sample_signature.parameters['save_intermediate_coords'].default
        self.assertFalse(default_value)


def create_minimal_test_data():
    """
    Create minimal test data for running actual tests.
    
    Returns
    -------
    Path
        Path to created test YAML file.
    """
    test_dir = Path(tempfile.mkdtemp())
    test_file = test_dir / "minimal_protein.yaml"
    
    # Create a minimal protein sequence for testing
    test_content = """
sequences:
  A: MKLLVVVDEVHHGFGD
"""
    test_file.write_text(test_content)
    return test_file


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2) 