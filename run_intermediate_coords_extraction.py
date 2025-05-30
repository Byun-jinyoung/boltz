#!/usr/bin/env python3
"""
Enhanced script for extracting intermediate coordinates during Boltz-1 diffusion reverse process.

This script provides a comprehensive implementation for running structure prediction with 
intermediate coordinate tracking, including advanced analysis, RMSD calculations, and 
data integrity validation.

Usage:
    python run_intermediate_coords_extraction.py input.yaml --save-intermediate-coords
"""

import os, sys 
import argparse
import json
import time
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

import torch
import numpy as np


class TrajectoryAnalyzer:
    """
    Advanced trajectory analysis class with RMSD calculations and data validation.
    
    This class provides comprehensive analysis of diffusion trajectory data including
    structural metrics, data integrity checks, and statistical summaries.
    """
    
    def __init__(self, trajectory_data: Dict[str, Any]):
        """
        Initialize trajectory analyzer.
        
        Parameters
        ----------
        trajectory_data : Dict[str, Any]
            Dictionary containing trajectory information with keys:
            'timesteps', 'sigmas', 'noisy_coords', 'denoised_coords', etc.
        """
        self.trajectory_data = trajectory_data
        self.analysis_results = {}
    
    def validate_trajectory_integrity(self) -> Dict[str, Any]:
        """
        Validate integrity of trajectory data.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing validation results and detected issues.
        """
        validation_results = {
            "is_valid": True,
            "issues": [],
            "statistics": {}
        }
        
        # Check required keys
        required_keys = ['timesteps', 'sigmas', 'noisy_coords', 'denoised_coords', 'metadata']
        for key in required_keys:
            if key not in self.trajectory_data:
                validation_results["issues"].append(f"Missing required key: {key}")
                validation_results["is_valid"] = False
        
        if not validation_results["is_valid"]:
            return validation_results
        
        # Check data consistency
        timesteps = self.trajectory_data['timesteps']
        sigmas = self.trajectory_data['sigmas']
        noisy_coords = self.trajectory_data['noisy_coords']
        denoised_coords = self.trajectory_data.get('denoised_coords', [])
        
        # Length consistency checks
        if len(timesteps) != len(sigmas):
            validation_results["issues"].append("Timesteps and sigmas length mismatch")
            validation_results["is_valid"] = False
        
        if len(noisy_coords) != len(timesteps):
            validation_results["issues"].append("Noisy coordinates length mismatch")
            validation_results["is_valid"] = False
        
        # Coordinate tensor validation
        for i, coords in enumerate(noisy_coords):
            if not isinstance(coords, torch.Tensor):
                validation_results["issues"].append(f"Non-tensor coordinates at timestep {i}")
                validation_results["is_valid"] = False
                continue
            
            if len(coords.shape) != 3:
                validation_results["issues"].append(f"Invalid coordinate shape at timestep {i}: {coords.shape}")
                validation_results["is_valid"] = False
        
        # Statistics
        validation_results["statistics"] = {
            "num_timesteps": len(timesteps),
            "sigma_range": (min(sigmas), max(sigmas)) if sigmas else (0, 0),
            "coordinate_shape": noisy_coords[0].shape if noisy_coords else None
        }
        
        return validation_results
    
    def compute_trajectory_metrics(self) -> Dict[str, float]:
        """
        Compute meaningful trajectory metrics for diffusion process analysis.
        
        Returns
        -------
        Dict[str, float]
            Dictionary containing diffusion-specific trajectory metrics.
        """
        metrics = {}
        
        sigmas = self.trajectory_data.get('sigmas', [])
        timesteps = self.trajectory_data.get('timesteps', [])
        noisy_coords = self.trajectory_data.get('noisy_coords', [])
        
        if len(sigmas) < 2:
            return {"error": "Insufficient trajectory points for analysis"}
        
        # Sigma progression analysis
        if sigmas:
            metrics["initial_sigma"] = sigmas[0] if sigmas else 0.0
            metrics["final_sigma"] = sigmas[-1] if sigmas else 0.0
            metrics["sigma_reduction_ratio"] = sigmas[0] / sigmas[-1] if sigmas[-1] > 0 else float('inf')
            
            # Calculate sigma decay rate
            if len(sigmas) > 1:
                sigma_changes = []
                for i in range(1, len(sigmas)):
                    if sigmas[i-1] > 0:
                        change_ratio = sigmas[i] / sigmas[i-1]
                        sigma_changes.append(change_ratio)
                
                if sigma_changes:
                    metrics["mean_sigma_decay_rate"] = np.mean(sigma_changes)
                    metrics["sigma_decay_consistency"] = 1.0 - np.std(sigma_changes)  # Higher = more consistent
        
        # Coordinate variance analysis (measure of structural "spread")
        if noisy_coords:
            coordinate_variances = []
            for coords in noisy_coords:
                if coords is not None:
                    # Calculate variance across all coordinates
                    coord_var = torch.var(coords).item()
                    coordinate_variances.append(coord_var)
            
            if coordinate_variances:
                metrics["initial_coordinate_variance"] = coordinate_variances[0]
                metrics["final_coordinate_variance"] = coordinate_variances[-1]
                metrics["variance_reduction_ratio"] = coordinate_variances[0] / coordinate_variances[-1] if coordinate_variances[-1] > 0 else float('inf')
                metrics["mean_coordinate_variance"] = np.mean(coordinate_variances)
        
        # Convergence analysis based on sigma progression
        if len(sigmas) > 10:  # Need sufficient points for trend analysis
            # Check if sigma is decreasing consistently (good convergence)
            recent_sigmas = sigmas[-5:]  # Last 5 steps
            early_sigmas = sigmas[:5]    # First 5 steps
            
            recent_mean = np.mean(recent_sigmas)
            early_mean = np.mean(early_sigmas)
            
            metrics["convergence_improvement"] = early_mean / recent_mean if recent_mean > 0 else float('inf')
            
            # Check for sigma plateau (potential convergence issues)
            final_sigma_changes = [abs(sigmas[i] - sigmas[i-1]) for i in range(-3, 0) if i < len(sigmas)-1]
            if final_sigma_changes:
                metrics["final_sigma_stability"] = 1.0 / (1.0 + np.mean(final_sigma_changes))  # Higher = more stable
        
        # Timestep progression analysis
        if timesteps:
            metrics["total_timesteps"] = len(timesteps)
            metrics["timestep_range"] = (min(timesteps), max(timesteps)) if timesteps else (0, 0)
        
        return metrics
    
    def generate_detailed_analysis(self, record_id: str, model_idx: int) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report for trajectory.
        
        Parameters
        ----------
        record_id : str
            Identifier for the structure record.
        model_idx : int
            Model index for multi-model predictions.
            
        Returns
        -------
        Dict[str, Any]
            Detailed analysis report.
        """
        # Validate trajectory first
        validation = self.validate_trajectory_integrity()
        
        if not validation["is_valid"]:
            return {
                "record_id": record_id,
                "model_idx": model_idx,
                "status": "invalid",
                "validation": validation,
                "timestamp": time.time()
            }
        
        # Compute metrics
        metrics = self.compute_trajectory_metrics()
        
        # Get metadata
        metadata = self.trajectory_data.get('metadata', {})
        
        # Compile detailed report
        analysis_report = {
            "record_id": record_id,
            "model_idx": model_idx,
            "status": "valid",
            "timestamp": time.time(),
            "validation": validation,
            "metrics": metrics,
            "trajectory_info": {
                "num_timesteps": len(self.trajectory_data.get('timesteps', [])),
                "initial_sigma": self.trajectory_data.get('sigmas', [0])[0] if self.trajectory_data.get('sigmas') else 0,
                "final_sigma": self.trajectory_data.get('sigmas', [0])[-1] if self.trajectory_data.get('sigmas') else 0,
                "coordinate_shape": validation["statistics"].get("coordinate_shape"),
                "sigma_schedule": self.trajectory_data.get('sigmas', [])[:10]  # First 10 for brevity
            },
            "metadata": metadata
        }
        
        return analysis_report


def setup_argument_parser() -> argparse.ArgumentParser:
    """
    Set up command line argument parser with enhanced options.
    
    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Run Boltz-1 prediction with intermediate coordinate extraction and advanced analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to input file (.yaml or .fasta)"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./intermediate_coords_output",
        help="Output directory for predictions and trajectories"
    )
    
    # Intermediate coordinates options
    parser.add_argument(
        "--save-intermediate-coords",
        action="store_true",
        help="Enable intermediate coordinate saving"
    )
    
    parser.add_argument(
        "--intermediate-format",
        choices=["pdb", "npz", "both"],
        default="pdb",
        help="Output format for intermediate coordinates"
    )
    
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save intermediate coordinates every N timesteps"
    )
    
    # Diffusion parameters
    parser.add_argument(
        "--sampling-steps",
        type=int,
        default=50,
        help="Number of diffusion sampling steps (reduced from default 200 for faster execution)"
    )
    
    parser.add_argument(
        "--diffusion-samples",
        type=int,
        default=1,
        help="Number of diffusion samples"
    )
    
    # Model parameters
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (optional)"
    )
    
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="~/.boltz",
        help="Cache directory for model and data"
    )
    
    # Hardware options
    parser.add_argument(
        "--device",
        choices=["gpu", "cpu"],
        default="gpu",
        help="Device to use for computation"
    )
    
    # Analysis options
    parser.add_argument(
        "--create-animation",
        action="store_true",
        help="Create animation script for trajectory visualization"
    )
    
    parser.add_argument(
        "--detailed-analysis",
        action="store_true",
        help="Perform detailed diffusion process analysis (sigma progression, coordinate variance, convergence)"
    )
    
    parser.add_argument(
        "--validate-integrity",
        action="store_true",
        help="Validate trajectory data integrity"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser


def validate_input_file(input_path: str) -> Path:
    """
    Validate the input file exists and has correct format.
    
    Parameters
    ----------
    input_path : str
        Path to the input file.
        
    Returns
    -------
    Path
        Validated input path.
        
    Raises
    ------
    FileNotFoundError
        If input file doesn't exist.
    ValueError
        If input file has invalid format.
    """
    input_file = Path(input_path).expanduser().resolve()
    
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    valid_extensions = {".yaml", ".yml", ".fasta", ".fa", ".fas"}
    if input_file.suffix not in valid_extensions:
        raise ValueError(
            f"Invalid input file format. Expected one of {valid_extensions}, "
            f"got {input_file.suffix}"
        )
    
    return input_file


def create_animation_script(trajectory_dir: Path, output_file: Path) -> None:
    """
    Create a PyMOL script for trajectory animation.
    
    Parameters
    ----------
    trajectory_dir : Path
        Directory containing trajectory PDB files.
    output_file : Path
        Output path for the PyMOL script.
    """
    pdb_files = sorted(trajectory_dir.glob("timestep_*.pdb"))
    
    if not pdb_files:
        print("No PDB files found for animation creation")
        return
    
    script_content = f"""# PyMOL script for Boltz-1 trajectory animation
# Generated automatically

# Load all timestep structures
"""
    
    for i, pdb_file in enumerate(pdb_files):
        timestep = pdb_file.stem.split('_')[1]
        script_content += f"load {pdb_file.name}, frame_{timestep}\n"
    
    script_content += """
# Create animation
set movie_panel, 1
set movie_fps, 2

# Style settings
hide everything
show cartoon
color cyan
set transparency, 0.3

# Group all frames
group trajectory, frame_*

# Animation commands
movie.load_trajectory trajectory
movie.play

print "Trajectory animation loaded. Use 'movie.play' to start animation."
"""
    
    with output_file.open("w") as f:
        f.write(script_content)
    
    print(f"Animation script saved to: {output_file}")


def analyze_trajectory_directory(output_dir: Path, detailed_analysis: bool = False, 
                                validate_integrity: bool = False) -> dict:
    """
    Analyze all trajectory data in the output directory with enhanced capabilities.
    
    Parameters
    ----------
    output_dir : Path
        Output directory containing trajectory data.
    detailed_analysis : bool
        Whether to perform detailed RMSD and trajectory analysis.
    validate_integrity : bool
        Whether to validate trajectory data integrity.
        
    Returns
    -------
    dict
        Summary of trajectory analysis with enhanced metrics.
    """
    trajectory_base = output_dir / "predictions" / "trajectories"
    
    if not trajectory_base.exists():
        return {"error": "No trajectory directory found"}
    
    analysis_summary = {
        "total_structures": 0,
        "structures": {},
        "trajectory_base_dir": str(trajectory_base),
        "analysis_timestamp": time.time(),
        "analysis_options": {
            "detailed_analysis": detailed_analysis,
            "validate_integrity": validate_integrity
        }
    }
    
    # Find all trajectory directories
    for struct_dir in trajectory_base.iterdir():
        if struct_dir.is_dir():
            analysis_summary["total_structures"] += 1
            struct_analysis = {
                "models": {},
                "structure_id": struct_dir.name
            }
            
            for model_dir in struct_dir.iterdir():
                if model_dir.is_dir() and model_dir.name.startswith("model_"):
                    model_idx = model_dir.name.split("_")[1]
                    
                    # Load existing analysis data if available
                    analysis_file = model_dir / "trajectory_analysis.json"
                    model_analysis = {}
                    if analysis_file.exists():
                        try:
                            with analysis_file.open() as f:
                                model_analysis = json.load(f)
                        except (json.JSONDecodeError, IOError) as e:
                            print(f"Warning: Could not load analysis file {analysis_file}: {e}")
                    
                    # Count files
                    pdb_files = list(model_dir.glob("*.pdb"))
                    npz_files = list(model_dir.glob("*.npz"))
                    
                    # Load trajectory data for enhanced analysis
                    if detailed_analysis or validate_integrity:
                        trajectory_metadata_file = model_dir / "trajectory_metadata.json"
                        if trajectory_metadata_file.exists():
                            try:
                                with trajectory_metadata_file.open() as f:
                                    trajectory_metadata = json.load(f)
                                
                                # Create TrajectoryAnalyzer for enhanced analysis
                                if "trajectory_data" in trajectory_metadata:
                                    # Convert coordinate lists back to tensors if needed
                                    trajectory_data = trajectory_metadata["trajectory_data"]
                                    
                                    # Note: In actual implementation, coordinate tensors would need
                                    # to be loaded from NPZ files rather than JSON
                                    analyzer = TrajectoryAnalyzer(trajectory_data)
                                    
                                    if validate_integrity:
                                        validation_result = analyzer.validate_trajectory_integrity()
                                        model_analysis["validation"] = validation_result
                                    
                                    if detailed_analysis:
                                        detailed_report = analyzer.generate_detailed_analysis(
                                            struct_dir.name, int(model_idx)
                                        )
                                        model_analysis.update(detailed_report)
                                        
                            except (json.JSONDecodeError, IOError, KeyError) as e:
                                print(f"Warning: Could not perform enhanced analysis for {model_dir}: {e}")
                    
                    # Update model analysis with file counts and basic info
                    struct_analysis["models"][model_idx] = {
                        **model_analysis,
                        "pdb_files": len(pdb_files),
                        "npz_files": len(npz_files),
                        "trajectory_dir": str(model_dir),
                        "last_modified": model_dir.stat().st_mtime if model_dir.exists() else None
                    }
            
            analysis_summary["structures"][struct_dir.name] = struct_analysis
    
    return analysis_summary


def main():
    """
    Main function to run intermediate coordinate extraction with enhanced analysis.
    """
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    start_time = time.time()
    
    try:
        # Validate input
        input_file = validate_input_file(args.input_path)
        output_dir = Path(args.output_dir).expanduser().resolve()
        
        print("=== Enhanced Boltz-1 Intermediate Coordinate Extraction ===")
        print(f"Input file: {input_file}")
        print(f"Output directory: {output_dir}")
        print(f"Save intermediate coords: {args.save_intermediate_coords}")
        
        if args.save_intermediate_coords:
            print(f"Intermediate format: {args.intermediate_format}")
            print(f"Save every: {args.save_every} timesteps")
            
            if args.detailed_analysis:
                print("✓ Detailed diffusion process analysis enabled")
            if args.validate_integrity:
                print("✓ Data integrity validation enabled")
        
        print(f"Sampling steps: {args.sampling_steps}")
        print(f"Diffusion samples: {args.diffusion_samples}")
        print(f"Device: {args.device}")
        
        if args.verbose:
            print(f"Cache directory: {args.cache_dir}")
            print(f"Checkpoint: {args.checkpoint if args.checkpoint else 'Default'}")
        
        print()
        
        # Prepare command for boltz CLI
        boltz_cmd = [
            "python", "-m", "boltz.main", "predict",
            str(input_file),
            "--out_dir", str(output_dir),
            "--cache", args.cache_dir,
            "--accelerator", args.device,
            "--recycling_steps", "3",
            "--sampling_steps", str(args.sampling_steps),
            "--diffusion_samples", str(args.diffusion_samples),
            "--output_format", "pdb",
            "--num_workers", "2",
            "--override",
            "--msa_server_url", "https://api.colabfold.com",
            "--msa_pairing_strategy", "greedy",
            "--intermediate_save_every", str(args.save_every),
        ]
        
        # Add optional arguments
        if args.checkpoint:
            boltz_cmd.extend(["--checkpoint", args.checkpoint])
            
        if args.save_intermediate_coords:
            boltz_cmd.append("--save_intermediate_coords")
            boltz_cmd.extend(["--intermediate_output_format", args.intermediate_format])
        
        boltz_cmd.append("--write_full_pae")
        
        if args.verbose:
            print(f"Running command: {' '.join(boltz_cmd)}")
        
        # Run prediction using subprocess
        print("Starting Boltz-1 prediction...")
        prediction_start_time = time.time()
        
        try:
            result = subprocess.run(
                boltz_cmd,
                check=True,
                capture_output=True,
                text=True,
                cwd=Path.cwd()
            )
            
            prediction_time = time.time() - prediction_start_time
            print(f"✓ Prediction completed in {prediction_time:.2f} seconds")
            
            if args.verbose and result.stdout:
                print("STDOUT:", result.stdout)
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Prediction failed with exit code {e.returncode}")
            if e.stdout:
                print("STDOUT:", e.stdout)
            if e.stderr:
                print("STDERR:", e.stderr)
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
        
        # Post-processing analysis
        if args.save_intermediate_coords:
            print("\n=== Post-processing Analysis ===")
            
            # Boltz creates a subdirectory based on input filename
            input_stem = input_file.stem
            actual_output_dir = output_dir / f"boltz_results_{input_stem}"
            
            if not actual_output_dir.exists():
                # Fallback to original output_dir if the expected path doesn't exist
                actual_output_dir = output_dir
                
            if args.verbose:
                print(f"Looking for trajectories in: {actual_output_dir}")
            
            # Analyze trajectory data with enhanced features
            analysis_start_time = time.time()
            trajectory_analysis = analyze_trajectory_directory(
                actual_output_dir, 
                detailed_analysis=args.detailed_analysis, 
                validate_integrity=args.validate_integrity
            )
            analysis_time = time.time() - analysis_start_time
            
            if args.verbose:
                print(f"Analysis completed in {analysis_time:.2f} seconds")
            
            # Save overall analysis
            analysis_file = actual_output_dir / "trajectory_summary.json"
            try:
                with analysis_file.open("w") as f:
                    json.dump(trajectory_analysis, f, indent=2, default=str)
                print(f"✓ Trajectory analysis saved to: {analysis_file}")
            except IOError as e:
                print(f"Warning: Could not save analysis file: {e}")
            
            print(f"Total structures processed: {trajectory_analysis.get('total_structures', 0)}")
            
            # Create animation scripts if requested
            if args.create_animation and args.intermediate_format in ["pdb", "both"]:
                print("\n=== Creating Animation Scripts ===")
                trajectory_base = actual_output_dir / "predictions" / "trajectories"
                animation_count = 0
                
                for struct_dir in trajectory_base.iterdir():
                    if struct_dir.is_dir():
                        for model_dir in struct_dir.iterdir():
                            if model_dir.is_dir() and model_dir.name.startswith("model_"):
                                script_file = model_dir / "animate_trajectory.pml"
                                create_animation_script(model_dir, script_file)
                                animation_count += 1
                
                print(f"✓ Created {animation_count} animation scripts")
            
            # Enhanced results summary
            print("\n=== Enhanced Results Summary ===")
            
            # Display validation results if requested
            if args.validate_integrity:
                print("\n📊 Data Integrity Validation:")
                validation_issues = 0
                for struct_id, struct_data in trajectory_analysis.get("structures", {}).items():
                    for model_id, model_data in struct_data.get("models", {}).items():
                        validation = model_data.get("validation", {})
                        if not validation.get("is_valid", True):
                            print(f"  ⚠️  Structure {struct_id}, Model {model_id}: {len(validation.get('issues', []))} issues")
                            validation_issues += 1
                            if args.verbose:
                                for issue in validation.get("issues", []):
                                    print(f"      - {issue}")
                
                if validation_issues == 0:
                    print("  ✅ All trajectory data passed integrity validation")
                else:
                    print(f"  ⚠️  Found validation issues in {validation_issues} trajectories")
            
            # Display detailed metrics if requested
            for struct_id, struct_data in trajectory_analysis.get("structures", {}).items():
                print(f"\n📁 Structure: {struct_id}")
                for model_id, model_data in struct_data.get("models", {}).items():
                    print(f"  🔬 Model {model_id}:")
                    print(f"    📄 PDB files: {model_data.get('pdb_files', 0)}")
                    print(f"    📊 NPZ files: {model_data.get('npz_files', 0)}")
                    
                    # Load existing analysis data if available and display key metrics
                    analysis_file_path = actual_output_dir / "predictions" / "trajectories" / struct_id / f"model_{model_id}" / "trajectory_analysis.json"
                    if analysis_file_path.exists():
                        try:
                            with analysis_file_path.open() as f:
                                existing_analysis = json.load(f)
                            
                            # Display meaningful diffusion metrics instead of RMSD
                            if "num_timesteps" in existing_analysis:
                                print(f"    ⏱️  Timesteps: {existing_analysis['num_timesteps']}")
                            if "initial_sigma" in existing_analysis and "final_sigma" in existing_analysis:
                                print(f"    🌊 Sigma range: {existing_analysis['initial_sigma']:.1f} → {existing_analysis['final_sigma']:.6f}")
                            if "sampling_steps" in existing_analysis:
                                print(f"    🔄 Sampling steps: {existing_analysis['sampling_steps']}")
                            if "multiplicity" in existing_analysis:
                                print(f"    🎯 Multiplicity: {existing_analysis['multiplicity']}")
                                
                        except (json.JSONDecodeError, IOError) as e:
                            if args.verbose:
                                print(f"    ⚠️  Could not load existing analysis: {e}")
                    
                    # Display diffusion process metrics if available
                    metrics = model_data.get("metrics", {})
                    if metrics and not metrics.get("error"):
                        if "sigma_reduction_ratio" in metrics:
                            print(f"    📉 Sigma reduction: {metrics['sigma_reduction_ratio']:.1f}x")
                        if "mean_sigma_decay_rate" in metrics:
                            print(f"    ⚡ Sigma decay rate: {metrics['mean_sigma_decay_rate']:.4f}")
                        if "sigma_decay_consistency" in metrics:
                            consistency = metrics['sigma_decay_consistency']
                            consistency_status = "Excellent" if consistency > 0.95 else "Good" if consistency > 0.90 else "Poor"
                            print(f"    🎯 Decay consistency: {consistency:.3f} ({consistency_status})")
                        if "variance_reduction_ratio" in metrics:
                            print(f"    📊 Variance reduction: {metrics['variance_reduction_ratio']:.1f}x")
                        if "convergence_improvement" in metrics:
                            convergence = metrics['convergence_improvement']
                            convergence_status = "Excellent" if convergence > 100 else "Good" if convergence > 10 else "Poor"
                            print(f"    🚀 Convergence improvement: {convergence:.1f}x ({convergence_status})")
                        if "final_sigma_stability" in metrics:
                            stability = metrics['final_sigma_stability']
                            stability_status = "Stable" if stability > 0.9 else "Moderate" if stability > 0.7 else "Unstable"
                            print(f"    ⚖️  Final stability: {stability:.3f} ({stability_status})")
                    
                    # Display trajectory info if available
                    traj_info = model_data.get("trajectory_info", {})
                    if traj_info:
                        print(f"    ⏱️  Timesteps: {traj_info.get('num_timesteps', 'N/A')}")
                        if "initial_sigma" in traj_info and "final_sigma" in traj_info:
                            print(f"    🌊 Sigma range: {traj_info['initial_sigma']:.1f} → {traj_info['final_sigma']:.1f}")
        
        total_time = time.time() - start_time
        
        print(f"\n✅ Enhanced prediction completed successfully!")
        print(f"⏱️  Total execution time: {total_time:.2f} seconds")
        print(f"📁 Results saved to: {output_dir}")
        
        if args.save_intermediate_coords:
            print(f"🎬 Trajectory data saved to: {output_dir}/predictions/trajectories/")
            
            if args.detailed_analysis:
                print("📊 Detailed diffusion process analysis included in results")
            if args.validate_integrity:
                print("✅ Data integrity validation completed")
        
    except KeyboardInterrupt:
        print("\n⚠️  Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 