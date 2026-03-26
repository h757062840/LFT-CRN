"""
Machine Learning-Driven Nudged Elastic Band (ML-NEB) Optimization Script

This script performs structural relaxation of reaction pathways to locate transition states 
using Machine Learning Interatomic Potentials (MLIPs) coupled with the Atomic Simulation 
Environment (ASE). It implements a multi-stage optimization strategy using dynamic spring 
constants and alternating optimizers (FIRE/BFGS) to ensure convergence.

This module is designed to be executed as a sub-process by the main active learning driver 
(makeneb.py) via Slurm batch jobs.

Usage Example:
    python MLneb.py \
        --work_directory /path/to/project_root \
        --iteration 2 \
        --directory /path/to/reaction_folder \
        --checkpoint_path /path/to/best_checkpoint.pt \
        --fix_atoms 24 \
        --n_images 8 \
        --k 1.0 \
        --fmax_list 0.8 0.4 0.2 0.08 \
        --utils_dir /path/to/utils \
        --is_filename ../IS/CONTCAR-ase-1 \
        --fs_filename ./CONTCAR-ase-1
"""

import os
import shutil
import subprocess
import argparse
import copy
import torch
import numpy as np
from ase import io
from ase.io import read, write
from ase.constraints import FixAtoms  
from fairchem.core import OCPCalculator
from ase.optimize import FIRE, BFGS
from ase.mep import DyNEB

def tag_atoms(slab):
    """
    Assign tags to atoms. Tag 1 for metals, Tag 2 for non-metals (adsorbates).
    """
    def is_metal(atom):
        # Common non-metal elements in surface adsorption studies
        if atom.symbol not in ['H', 'C', 'N', 'O', 'S', 'Cl', 'Br', 'I', 'F']: 
            return True
        else:
            return False
        
    tags = [1 if is_metal(i) else 2 for i in slab]  
    slab.set_tags(tags)
    return slab    
    
def run_ML(work_directory, iteration, directory, checkpoint_path, fix_atoms, n_images, k, fmax_list, utils_dir, is_filename, fs_filename, auto_n_images=False):
    """
    Main driver function for Machine Learning-driven Nudged Elastic Band (ML-NEB) calculation.
    """
    # Load and optimize Initial State (IS) and Final State (FS)
    initial = tag_atoms(read(is_filename))
    final = tag_atoms(read(fs_filename))
    
    initial = optimize_structure(initial, checkpoint_path=checkpoint_path, fmax=0.03, steps=300, fix_atoms=fix_atoms)
    final = optimize_structure(final, checkpoint_path=checkpoint_path, fmax=0.03, steps=300, fix_atoms=fix_atoms)
    
    IS_file = os.path.join(directory, 'IS.vasp')
    FS_file = os.path.join(directory, 'FS.vasp')
    write(IS_file, initial)
    write(FS_file, final)
    
    # Determine the number of intermediate images
    if auto_n_images:
        n_images = n_images_auto(is_filename, fs_filename, utils_dir)
    
    print(f'Number of intermediate images: {n_images}')
    
    # Perform pre-NEB setup using IDPP interpolation for the first iteration
    if iteration == 1:
        neb0_folder = "neb0"
        os.makedirs(neb0_folder, exist_ok=True)
        os.chdir(neb0_folder)
        subprocess.run(["cp", f"{utils_dir}/idpp.py", "."], check=True)
        subprocess.run(["python", "idpp.py", os.path.join(directory, 'IS.vasp'), os.path.join(directory, 'FS.vasp'), f'{n_images}'], check=True)
        os.chdir("..")

    # Read intermediate images from the previous iteration
    image_list = [tag_atoms(read(f"neb{iteration-1}/{str(i).zfill(2)}/POSCAR")) for i in range(1, n_images+1)]   
    images = [initial] + image_list + [final]
    
    # Set constraints for the bottom layers
    z_positions = initial.positions[:, 2]
    fixed_indices = z_positions.argsort()[:fix_atoms]
    print(f"Running ML NEB for {directory} (Iteration {iteration}) with checkpoint {checkpoint_path}")

    # Set calculators and constraints for all intermediate images
    for image in images[1:-1]:
        image.set_calculator(OCPCalculator(
            checkpoint_path=checkpoint_path,
            cutoff=12,
            cpu=not torch.cuda.is_available()
        ))
        image.set_constraint(FixAtoms(indices=fixed_indices))
        
    # Initialize and run the dynamic NEB
    neb = DyNEB(images, k=k, climb=False, dynamic_relaxation=False, scale_fmax=0)
    final_xyz_file, convergence_status = optimize_neb(
        neb=neb, 
        fmax_list=fmax_list, 
        best_checkpoint_path=checkpoint_path, 
        fix_atoms=fix_atoms, 
        iteration=iteration, 
        generated_xyz_files=[]
    )    

    # Verify convergence status
    max_images = 10
    expected_convergence_status = f"iter{iteration}_conv1" if iteration == 1 else f"iter{iteration}_conv3"
    expected_conv_num = int(expected_convergence_status.split('_')[1].replace('conv', ''))    
    
    if convergence_status is None:
        print("Warning: convergence_status is None. Treating as not converged.")
        current_conv_num = 0
    else:
        current_conv_num = int(convergence_status.split('_')[1].replace('conv', ''))
    
    # Process results if target convergence is met
    if current_conv_num >= expected_conv_num:
        print(f"Convergence status {convergence_status} meets or exceeds target {expected_convergence_status}. Proceeding to post-processing.")
        post_process(work_directory, directory, xyz_file=final_xyz_file, initial=images[0], final=images[n_images+1], iteration=iteration, convergence_status=convergence_status, checkpoint_path=checkpoint_path, fix_atoms=fix_atoms)
    else:
        # Iteratively increase the number of images and re-run if convergence fails
        while n_images <= max_images:
            if convergence_status is None:
                current_conv_num = 0
            else:
                current_conv_num = int(convergence_status.split('_')[1].replace('conv', ''))
            
            if current_conv_num >= expected_conv_num:
                print(f"Convergence status {convergence_status} meets or exceeds target {expected_convergence_status}. Stopping image addition.")
                post_process(work_directory, directory, xyz_file=final_xyz_file, initial=images[0], final=images[n_images+1], iteration=iteration, convergence_status=convergence_status, checkpoint_path=checkpoint_path, fix_atoms=fix_atoms)
                break
            
            if n_images >= max_images:
                print("Maximum number of NEB images reached. Proceeding to post-processing with current state.")
                post_process(work_directory, directory, xyz_file=final_xyz_file, initial=images[0], final=images[n_images+1], iteration=iteration, convergence_status=convergence_status, checkpoint_path=checkpoint_path, fix_atoms=fix_atoms)
                break
                
            n_images += 1
            neb_folder = f"neb{iteration-1}"
            os.chdir(neb_folder)
            subprocess.run(["cp", f"{utils_dir}/idpp.py", "."], check=True)
            subprocess.run(["python", "idpp.py", os.path.join(directory, 'IS.vasp'), os.path.join(directory, 'FS.vasp'), f'{n_images}'], check=True)
            os.chdir("..")
            
            images = [tag_atoms(read(f"neb{iteration-1}/{str(i).zfill(2)}/POSCAR")) for i in range(0, n_images+2)]   
            print(f"Current convergence ({convergence_status}) below target ({expected_convergence_status}). Increasing images to: {n_images}")
            
            for image in images[1:-1]:
                image = tag_atoms(image)
                image.set_calculator(OCPCalculator(
                    checkpoint_path=checkpoint_path,
                    cutoff=12,
                    cpu=not torch.cuda.is_available()
                ))
                image.set_constraint(FixAtoms(indices=fixed_indices))
            
            # Re-run NEB calculation with CI-NEB enabled
            neb = DyNEB(images, k=k, climb=True, dynamic_relaxation=False, scale_fmax=0)
            final_xyz_file, convergence_status = optimize_neb(
                neb=neb, 
                fmax_list=fmax_list, 
                best_checkpoint_path=checkpoint_path, 
                fix_atoms=fix_atoms, 
                iteration=iteration, 
                generated_xyz_files=[]
            )
            
def optimize_structure(structure, checkpoint_path, fmax=0.03, steps=300, fix_atoms=32):
    """
    Perform structural relaxation using the BFGS optimizer.
    """
    structure.set_calculator(OCPCalculator(checkpoint_path=checkpoint_path, cutoff=12, cpu=not torch.cuda.is_available()))
    structure.set_constraint(FixAtoms(indices=structure.positions[:, 2].argsort()[:fix_atoms]))    
    
    optimizer = BFGS(structure)
    print(f"Optimizing structure to fmax = {fmax} eV/A...")    
    optimizer.run(fmax=fmax, steps=steps)
    
    return structure  

def n_images_auto(initial_file, final_file, utils_dir, threshold=0.8):
    """
    Automatically determine the number of NEB images based on the Cartesian distance between IS and FS.
    """
    dist_script = "dist.pl"
    shutil.copy(os.path.join(utils_dir, dist_script), ".")

    try:
        result = subprocess.run(
            [dist_script, initial_file, final_file],
            capture_output=True,
            text=True,
            check=True,
        )
        distance_sum = float(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"Error executing dist.pl: {e}")
        raise
        
    print(f'Integrated path distance between IS and FS: {distance_sum:.3f} Angstroms')
    
    n_images = int(distance_sum / threshold)
    n_images = max(n_images, 1)

    return n_images
    
def save_converged_results(neb, iteration, conv_level, generated_xyz_files):
    """Save the current NEB band to an XYZ trajectory file."""
    xyz_filename = f"iter{iteration}_conv{conv_level}.xyz"
    try:
        write(xyz_filename, neb.images)
        generated_xyz_files.append(xyz_filename)
        return xyz_filename
    except IOError as e:
        print(f"Error writing XYZ file: {e}")
        return None

def decrease_spring_force(neb, k_decrement=0.4, min_k=0.2):
    """Reduce the spring constant if structural divergence is detected."""
    if neb.k[0] > min_k:
        neb.k = [x - k_decrement for x in neb.k]
        print(f"Adjusted spring force to: {neb.k[0]:.2f} eV/A^2")
        return True
    return False

def restore_images(neb, last_converged_images, best_checkpoint_path, fix_atoms):
    """Restore the NEB band to the last stable configuration."""
    images = [image.copy() for image in last_converged_images]
    fixed_indices = neb.images[0].positions[:, 2].argsort()[:fix_atoms]
    for image in images[1:-1]:
        image.set_calculator(OCPCalculator(checkpoint_path=best_checkpoint_path, cutoff=12, cpu=not torch.cuda.is_available()))
        image.set_constraint(FixAtoms(indices=fixed_indices))
        image = tag_atoms(image)
    neb.images = images

def optimize_neb(neb, fmax_list, best_checkpoint_path, fix_atoms, iteration, generated_xyz_files):
    """
    Execute a multi-stage NEB relaxation across defined fmax thresholds.
    Alternates between FIRE and BFGS optimizers to ensure stability.
    """
    last_converged_images = copy.deepcopy(neb.images)
    final_xyz_file = None
    convergence_status = None

    for conv_level, fmax in enumerate(fmax_list, start=1):
        neb.climb = conv_level > 1
        print(f"CI-NEB Active: {neb.climb}")

        optimizer = FIRE(neb)
        print(f"Stage {conv_level} optimization initiated (Optimizer: FIRE, fmax: {fmax} eV/A, k: {neb.k[0]:.2f} eV/A^2)")
        conv = optimizer.run(fmax=fmax, steps=80)

        if not conv:
            print(f"Stage {conv_level} failed to converge with FIRE. Switching to BFGS.")
            restore_images(neb, last_converged_images, best_checkpoint_path, fix_atoms)
            optimizer = BFGS(neb)
            conv = optimizer.run(fmax=fmax, steps=80)

        while not conv and decrease_spring_force(neb):
            print(f"Stage {conv_level} failed to converge. Restoring structure and reducing spring constant.")
            restore_images(neb, last_converged_images, best_checkpoint_path, fix_atoms)
            optimizer = FIRE(neb)
            conv = optimizer.run(fmax=fmax, steps=50)

        if conv:
            xyz_filename = save_converged_results(neb, iteration, conv_level, generated_xyz_files)
            if xyz_filename:
                last_converged_images = copy.deepcopy(neb.images)
                convergence_status = f"iter{iteration}_conv{conv_level}"
                final_xyz_file = xyz_filename
        else:
            print(f"Stage {conv_level} optimization failed. Minimum spring force threshold reached.")
            break

    # Clean up intermediate XYZ trajectory files
    if generated_xyz_files and len(generated_xyz_files) > 1:
        for xyz_file in generated_xyz_files[:-1]:
            if os.path.exists(xyz_file):
                os.remove(xyz_file)

    return final_xyz_file, convergence_status

def post_process(work_directory, directory, xyz_file, initial, final, iteration, convergence_status, checkpoint_path, fix_atoms):
    """
    Extract the transition state and calculate the activation barrier from the relaxed path.
    """
    images = io.read(xyz_file, index=':')
    z_positions = initial.positions[:, 2]
    fixed_indices = z_positions.argsort()[:fix_atoms] 
     
    for image in images:
        image.set_calculator(OCPCalculator(checkpoint_path=checkpoint_path, cutoff=12, cpu=not torch.cuda.is_available())) 
        image.set_constraint(FixAtoms(indices=fixed_indices))            
    energies = [image.get_potential_energy() for image in images]
    
    for i, energy in enumerate(energies):
        print(f"Image {i}: Energy = {energy:.4f} eV")

    energy_barrier = max(energies) - energies[0]
    print(f"Activation Energy (Ea): {energy_barrier:.4f} eV")
    
    max_energy = max(energies)
    max_energy_index = energies.index(max_energy)
    ts_image = images[max_energy_index]
    
    # Export each frame to a VASP POSCAR file
    current_dir = os.getcwd()
    for i, image in enumerate(images):
        contcar_file = os.path.join(current_dir, f'iter{iteration}_POSCAR{str(i).zfill(2)}')
        write(contcar_file, image)
    
    # Export the transition state geometry
    structure_dir = f"{work_directory}/ML_conf_iter{iteration}"
    os.makedirs(structure_dir, exist_ok=True)
    ts_file_name = f"{directory.split('/')[-3]}_{directory.split('/')[-1]}.vasp"
    ts_file = os.path.join(structure_dir, ts_file_name)
    write(ts_file, ts_image)
    print(f"Transition state geometry written to {ts_file}")
    
    # Append barrier data to the tracking file
    ea_file = os.path.join(work_directory, 'Ea.txt')
    with open(ea_file, 'a') as f:
        f.write(f"{ts_file_name}  {iteration} {convergence_status} {energy_barrier:.4f}\n")
    print(f"Reaction energetics appended to {ea_file}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute MLIP-driven NEB optimization via ASE")
    
    parser.add_argument("--work_directory", type=str, required=True, help="Base working directory")
    parser.add_argument("--iteration", type=int, required=True, help="Current active learning iteration index")
    parser.add_argument("--directory", type=str, required=True, help="Target reaction directory containing the IS and FS")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the MLIP checkpoint file (.pt)")
    parser.add_argument("--fix_atoms", type=int, required=True, help="Number of bottom-layer atoms to freeze during relaxation")
    parser.add_argument("--n_images", type=int, required=True, help="Number of intermediate images for the NEB band")
    parser.add_argument("--k", type=float, required=True, help="Initial spring constant (eV/A^2)")
    parser.add_argument("--fmax_list", type=float, nargs='+', required=True, help="Sequential max force thresholds (eV/A) for multi-stage relaxation")    
    parser.add_argument("--utils_dir", type=str, required=True, help="Path to the utility directory containing auxiliary scripts")    
    
    # New arguments for IS and FS filenames
    parser.add_argument("--is_filename", type=str, default="../IS/CONTCAR-ase-1", help="Filepath to the Initial State geometry")
    parser.add_argument("--fs_filename", type=str, default="./CONTCAR-ase-1", help="Filepath to the Final State geometry")
    
    parser.add_argument("--auto_n_images", action="store_true", help="Dynamically determine the number of images based on path length")
    
    args = parser.parse_args()
    
    run_ML(
        work_directory=args.work_directory,
        iteration=args.iteration,
        directory=args.directory,
        checkpoint_path=args.checkpoint_path,
        fix_atoms=args.fix_atoms,
        n_images=args.n_images,
        k=args.k,
        fmax_list=args.fmax_list,
        utils_dir=args.utils_dir,
        is_filename=args.is_filename,
        fs_filename=args.fs_filename,
        auto_n_images=args.auto_n_images
    )