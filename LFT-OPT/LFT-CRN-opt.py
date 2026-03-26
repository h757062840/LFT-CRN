import os
import shutil
import time
import subprocess
import argparse
import re
import yaml
import numpy as np
import torch
import dpdata
from ase import io
from ase.io import read
from ase.constraints import FixAtoms  
from fairchem.core import OCPCalculator
from ase.optimize import BFGS
from ase.db import connect

def modify_yaml_parameter(yml_file_path="finetune1.yml", param_path='optim.eval_every', new_value=6):
    """
    Modify a specific parameter value in a YAML file.
    """
    with open(yml_file_path, 'r') as file:
        config = yaml.safe_load(file)

    keys = param_path.split(".")
    temp = config
    for key in keys[:-1]:
        temp = temp[key]
    temp[keys[-1]] = new_value

    with open(yml_file_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
        
def wait_for_job_completion(job_id):
    """Wait for a single Slurm job to complete."""
    previous_status = None 
    while True:
        result = subprocess.run(['squeue', '--job', job_id], stdout=subprocess.PIPE)
        current_status = job_id in result.stdout.decode()
        
        if not current_status:
            print(f"Job {job_id} has completed.")
            break
        
        if current_status != previous_status:
            print(f"Waiting for job {job_id} to complete...")
        
        previous_status = current_status 
        time.sleep(5)

def wait_for_jobs_completion(job_ids):
    """Wait for multiple Slurm jobs to complete."""
    previous_status = {job_id: None for job_id in job_ids}
    while job_ids:
        for job_id in job_ids[:]:
            result = subprocess.run(['squeue', '--job', job_id], stdout=subprocess.PIPE)
            current_status = job_id in result.stdout.decode()
            
            if not current_status:
                job_ids.remove(job_id) 
                print(f"Job {job_id} has completed.")
            
            if current_status != previous_status[job_id]:
                print(f"Waiting for job {job_id} to complete...")
                
            previous_status[job_id] = current_status 
        
        if job_ids:
            time.sleep(10)
    print("All jobs have completed.")    

def write_record(record_file, iteration, step, directory):
    """Log the progress to enable restarting from the point of failure."""
    with open(record_file, 'a') as f:
        f.write(f"{iteration} {step} {directory}\n")

def read_last_record(record_file):
    """Read the last logged step from the record file."""
    if not os.path.exists(record_file):
        return None 

    with open(record_file, 'r') as f:
        lines = f.readlines()
        if not lines:
            return None 
        last_line = lines[-1].strip()
        iteration, step, directory = last_line.split()
        return int(iteration), step, directory 

def finetune_model(work_directory, iteration, utils_dir, record_file):
    print(f"Finetuning model for iteration {iteration}...")
    finetune_dir = f'{work_directory}/finetune_opt_{iteration}'
    os.makedirs(finetune_dir, exist_ok=True)
    shutil.copy(os.path.join(utils_dir, 'sub.oc'), f'{finetune_dir}/sub.oc')
    shutil.copy(os.path.join(utils_dir, 'finetune1.yml'), f'{finetune_dir}/finetune1.yml')   
    shutil.copy(os.path.join(utils_dir, 'main.py'), f'{finetune_dir}/main.py')       
    
    # Determine the number of data points in the output_database.db
    db_path = os.path.join(work_directory, 'output_database.db')
    with connect(db_path) as db:
        n_data = db.count()
    print(f'Data number: {n_data}')
    
    os.chdir(finetune_dir)
    if iteration > 1:
        change_model(work_directory, iteration)
        
    param_to_change = 'optim.eval_every'
    eval_every = int(2 * n_data / 6)
    modify_yaml_parameter(yml_file_path="finetune1.yml", param_path=param_to_change, new_value=eval_every)    
    print(f"Changed parameter {param_to_change} to {eval_every}")
    
    result = subprocess.run(['sbatch', 'sub.oc'], stdout=subprocess.PIPE)
    job_id = result.stdout.decode().strip().split()[-1]
    wait_for_job_completion(job_id)
    write_record(record_file, iteration, step="finetune", directory=work_directory)
    
def change_model(work_directory, iteration):
    """Update the submission script to use the latest model checkpoint."""
    prev_iteration_dir = f"{work_directory}/finetune_opt_{iteration - 1}"
    checkpoints_dir = os.path.join(prev_iteration_dir, 'checkpoints')
    checkpoint_dirs = sorted(os.listdir(checkpoints_dir))
    last_checkpoint_dir = checkpoint_dirs[-1]
    
    best_checkpoint_path = os.path.join(checkpoints_dir, last_checkpoint_dir, 'best_checkpoint.pt')
    print(f"Using checkpoint from: {best_checkpoint_path}")
    
    sub_oc_file = 'sub.oc'
    with open(sub_oc_file, 'r') as file:
        content = file.read()
        
    pattern = r'--checkpoint\s+([^\s]+)' 
    new_content = re.sub(pattern, f'--checkpoint {best_checkpoint_path}', content)
    
    with open(sub_oc_file, 'w') as file:
        file.write(new_content)

def move_tag2_atoms(slab):
    """Shift atoms with tag=2 if they are close to the upper bound of tag=1 atoms."""
    tag1_atoms = [atom for atom in slab if atom.tag == 1]
    tag2_atoms = [atom for atom in slab if atom.tag == 2]

    if not tag1_atoms or not tag2_atoms:
        return slab

    max_z_tag1 = max(atom.position[2] for atom in tag1_atoms)
    tag2_positions = np.array([atom.position for atom in tag2_atoms])
    tag2_centroid_z = np.mean(tag2_positions[:, 2])

    if tag2_centroid_z > max_z_tag1 + 4:
        delta_z = (max_z_tag1 + 4) - tag2_centroid_z
        for atom in tag2_atoms:
            atom.position[2] += delta_z

    return slab        

def run_ML(work_directory, iteration, directory, checkpoint_path, fix_atoms, record_file):
    os.chdir(directory)
    print(f"Running ML optimization for {directory} (iteration {iteration}) with checkpoint {checkpoint_path}")
    input_file = 'POSCAR' if iteration == 1 else f'opt{iteration-1}/CONTCAR'
    
    slab = io.read(input_file)
    initial_slab = slab.copy()  
    
    tags = [1 if atom.symbol not in ['H', 'C', 'N', 'O', 'S', 'Cl', 'Br', 'I', 'F'] else 2 for atom in slab]
    slab.set_tags(tags)

    sorted_indices = sorted(range(len(slab)), key=lambda i: slab.positions[i][2])
    fixed_indices = sorted_indices[:fix_atoms]
    constraints = FixAtoms(indices=fixed_indices)
    slab.set_constraint(constraints)

    calculator = OCPCalculator(checkpoint_path=checkpoint_path, cutoff=12, cpu=not torch.cuda.is_available())
    slab.set_calculator(calculator)
    
    fmax = 0.1 if iteration == 1 else 0.03
    steps = 300
    dyn = BFGS(slab)
    try:
        dyn.run(fmax=fmax, steps=steps)
    except RuntimeError as e:
        print(f"Optimization failed: {e}")
        output_file = f'CONTCAR-ase-{iteration}'
        io.write(output_file, initial_slab)
        return
        
    optimized_energy = slab.get_potential_energy()
    with open(f'{work_directory}/E_pre.txt', 'a') as f:
        f.write(f"{iteration} {directory} {optimized_energy:.3f}\n")
        
    # slab = move_tag2_atoms(slab)
    output_file = f'CONTCAR-ase-{iteration}'
    structure_dir = os.path.join(work_directory, f'ML_conf_iter{iteration}')
    os.makedirs(structure_dir, exist_ok=True)
    io.write(output_file, slab)
    
    output_file2 = f'{structure_dir}/{directory.split("/")[-3]}_{directory.split("/")[-2]}_{directory.split("/")[-1]}.vasp'    
    io.write(output_file2, slab)   
    write_record(record_file, iteration, step="ML_opt", directory=directory)

def extract_energy_from_outcar(outcar_path):
    with open(outcar_path, 'r') as outcar_file:
        lines = outcar_file.readlines()
    
    for line in reversed(lines):
        match = re.search(r'energy\(sigma->0\)\s*=\s*(-?\d+\.\d+)', line)
        if match:
            return float(match.group(1))
    return None
    
def modify_incar_parameter(incar_file, parameter, value):
    value = str(value)
    with open(incar_file, 'r') as f:
        incar_lines = f.readlines()

    modified = False
    with open(incar_file, 'w') as f:
        for line in incar_lines:
            stripped_line = line.strip().upper()
            if stripped_line.replace(' ', '').startswith(f'{parameter.upper()}='):
                equal_sign_index = stripped_line.find('=')
                param_name = stripped_line[:equal_sign_index].strip()
                if param_name == parameter.upper():
                    f.write(f'{parameter.upper()} = {value}\n')
                    modified = True
                else:
                    f.write(line)
            else:
                f.write(line)
        
        if not modified:
            if incar_lines and not incar_lines[-1].endswith('\n'):
                f.write('\n') 
            f.write(f'{parameter.upper()} = {value}\n')

def assign_mag_incar(poscar_path, incar_path):
    """Assign MAGMOM parameter in INCAR based on magnetic elements in POSCAR."""
    magnetic_elements = {'Fe', 'Co', 'Ni', 'Mn'}
    non_magnetic_elements = {'C', 'H', 'O'}
    
    atoms = read(poscar_path)
    symbols = atoms.get_chemical_symbols()
    if not symbols:
        raise ValueError("No elements found in POSCAR.")
    
    current_symbol = symbols[0]
    count = 1
    elements = []
    counts = []
    for s in symbols[1:]:
        if s == current_symbol:
            count += 1
        else:
            elements.append(current_symbol)
            counts.append(count)
            current_symbol = s
            count = 1
    elements.append(current_symbol)
    counts.append(count)
    
    magmom_parts = []
    for elem, cnt in zip(elements, counts):
        if elem in magnetic_elements:
            magmom_parts.append(f"{cnt}*3")
        elif elem in non_magnetic_elements:
            magmom_parts.append(f"{cnt}*1")
        else:
            magmom_parts.append(f"{cnt}*0")
    magmom_line = "\n" + "MAGMOM = " + " ".join(magmom_parts) + "\n"
    
    with open(incar_path, 'r') as f:
        incar_lines = f.readlines()
    
    incar_lines = [line for line in incar_lines if not line.strip().startswith("MAGMOM")]
    incar_lines.append(magmom_line)
    
    with open(incar_path, 'w') as f:
        f.writelines(incar_lines)
    
    print(f"Updated INCAR at {incar_path} with: {magmom_line.strip()}")
            
def run_DFT(work_directory, iteration, directories, utils_dir, batch_size, record_file, if_final=False):
    print(f"Running DFT optimization for iteration {iteration}...")
    structure_dir = os.path.join(work_directory, f'DFT_conf_iter{iteration}')
    os.makedirs(structure_dir, exist_ok=True)
    energy_file_path = os.path.join(work_directory, 'E_DFT.txt')
    
    def split_list(lst, n):
        k, m = divmod(len(lst), n)
        return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]
    
    batches = split_list(directories, batch_size)
    job_ids = []
    
    for batch_idx, current_batch in enumerate(batches):
        if not current_batch:
            continue
        print(f"Processing batch {batch_idx + 1}/{len(batches)} with {len(current_batch)} directories...")
        
        for directory in current_batch:
            opt_dir = os.path.join(directory, f'opt{iteration}')
            os.makedirs(opt_dir, exist_ok=True)
            
            shutil.copy(f'{directory}/CONTCAR-ase-{iteration}', f'{opt_dir}/POSCAR')
            incar_path = os.path.join(opt_dir, 'INCAR')
            shutil.copy(os.path.join(utils_dir, 'INCAR_SCF'), incar_path)   
            
            magnetic_elements = {'Fe', 'Co', 'Ni', 'Mn'}
            if any(el in directory for el in magnetic_elements):
                modify_incar_parameter(incar_path, "ISPIN", 2)
                modify_incar_parameter(incar_path, "EDIFF", '1E-4')
                assign_mag_incar(f'{opt_dir}/POSCAR', incar_path)
            
            os.chdir(opt_dir)
            os.system('echo -e "103\n" | vaspkit > /dev/null')
            os.system('echo -e "402\n1\n3\n0 0.45\n1\nall\n" | vaspkit > /dev/null')
            shutil.copy(f'{utils_dir}/KPOINTS', opt_dir)
            shutil.copy('POSCAR_FIX.vasp', 'POSCAR')
            os.chdir(work_directory) 

        batch_sub_dir = os.path.join(structure_dir, f'batch_{batch_idx}')
        os.makedirs(batch_sub_dir, exist_ok=True)
        batch_sub_path = os.path.join(batch_sub_dir, 'sub.vasp')        
        
        with open(batch_sub_path, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("#SBATCH -N 1\n")
            f.write("#SBATCH -p intel96,standard\n")
            f.write("#SBATCH --exclusive\n\n")
            f.write("#SBATCH --exclude=master\n")
            f.write("module load VASP/6.4.1-gzbuild-intel_8300\n\n")
            
            for directory in current_batch:
                opt_dir = os.path.abspath(os.path.join(directory, f'opt{iteration}'))
                f.write(f"echo 'Processing {opt_dir}'\n")
                f.write(f"cd {opt_dir}\n")
                f.write("start_time=$(date +%s)\n")
                f.write('echo "Start time: $(date)"\n') 
                f.write("mpirun vasp_std > fp.log 2>vasp.err\n\n")
                f.write("end_time=$(date +%s)\n")
                f.write('echo "End time: $(date)"\n') 
                f.write("execution_time=$((end_time - start_time))\n")
                f.write('echo "Execution time: ${execution_time} seconds"\n') 
                f.write(f'echo "Completed processing {opt_dir}"\n\n')
            
            f.write('echo "All directories in this batch have been processed."\n')
        
        os.chdir(batch_sub_dir)
        result = subprocess.run(['sbatch', 'sub.vasp'], stdout=subprocess.PIPE)
        job_id = result.stdout.decode().strip().split()[-1]
        job_ids.append(job_id)
        print(f"Submitted batch {batch_idx+1} job {job_id}")
        os.chdir(work_directory)
    
    wait_for_jobs_completion(job_ids)
    
    with open(energy_file_path, 'a') as energy_file:
        for directory in directories:
            opt_dir = os.path.join(directory, f'opt{iteration}')
            contcar_path = os.path.join(opt_dir, 'CONTCAR')
            if os.path.exists(contcar_path):
                dest_name = f"{os.path.basename(os.path.dirname(directory))}.vasp"
                shutil.copy(contcar_path, os.path.join(structure_dir, dest_name))
            
            energy = extract_energy_from_outcar(os.path.join(opt_dir, 'OUTCAR'))
            if energy is not None:
                energy_file.write(f"{iteration} {directory}: {energy}\n")
            write_record(record_file, iteration, "DFT_opt", directory)
    
    print(f"All DFT optimizations completed for iteration {iteration}.")

def convert_data(iteration, directory, db_path, record_file):
    db_dir = os.path.dirname(db_path)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
    try:
        data = dpdata.LabeledSystem(f"{directory}/opt{iteration}/OUTCAR")
    except Exception as e:
        print(f"Error loading OUTCAR file: {e}")
        return
        
    db = connect(db_path)
    for i, structure in enumerate(data):
        ase_structures = structure.to_ase_structure()
        if isinstance(ase_structures, list):
            for j, ase_atoms in enumerate(ase_structures):
                db.write(ase_atoms, key_value_pairs={"step": i, "frame": j})
        else:
            db.write(ase_structures, key_value_pairs={"step": i})
            
    print(f"Iteration {iteration}: Successfully converted {directory} data to {db_path}")
    write_record(record_file, iteration, step="convert_data", directory=directory)  
    
def get_latest_model(work_directory, iteration):
    finetune_dir = f'{work_directory}/finetune_opt_{iteration}'
    checkpoints_dir = os.path.join(finetune_dir, 'checkpoints')
    if not os.path.exists(checkpoints_dir):
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")
        
    checkpoint_dirs = sorted(os.listdir(checkpoints_dir))
    if len(checkpoint_dirs) == 0:
        raise FileNotFoundError(f"No checkpoint directories found in: {checkpoints_dir}")
        
    last_checkpoint_dir = checkpoint_dirs[-1] 
    best_checkpoint_path = os.path.join(checkpoints_dir, last_checkpoint_dir, 'best_checkpoint.pt')
    if not os.path.exists(best_checkpoint_path):
        raise FileNotFoundError(f"best_checkpoint.pt not found in: {best_checkpoint_path}")
    return best_checkpoint_path 

def main():
    parser = argparse.ArgumentParser(description="Active Learning Optimization Workflow")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file")
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    directories = config.get("directories", [])
    max_iteration = config.get("max_iteration", 4)
    work_directory = os.path.abspath(config.get("work_directory", os.getcwd()))
    utils_dir = config.get("utils_dir", "")
    db_path = os.path.join(work_directory, config.get("db_path", "output_database.db"))
    record_file = os.path.join(work_directory, config.get("record_file", "record.txt"))
    fix_atoms = config.get("fix_atoms", 24)
    batch_size = config.get("batch_size", 8)
    origin_checkpoint_path = config.get("origin_checkpoint_path", "")
    final_DFT = config.get("final_DFT", True)

    iteration = 1
    last_record = read_last_record(record_file)
    
    if last_record is None:
        iteration = 1
        last_step = None
        last_directory = None
        print("No record available, starting from the first iteration!")
    else:
        iteration, last_step, last_directory = last_record
        print(f"Record available at iteration {iteration}, step {last_step} and directory {last_directory}!")
        if last_step == 'finetune':
            iteration += 1  
            last_step = None  
    
    for i in range(iteration, max_iteration + 1):
        if i == 1:
            best_checkpoint_path = origin_checkpoint_path
        else:
            previous_iteration = i - 1
            print(f"Loading checkpoint from iteration {previous_iteration}")
            best_checkpoint_path = get_latest_model(work_directory, previous_iteration)     
            
        if i == max_iteration:
            if last_step == 'ML_opt' and last_directory:
                directories_to_process = directories[directories.index(last_directory) + 1:]
                for directory in directories_to_process:
                    run_ML(work_directory, i, directory, checkpoint_path=best_checkpoint_path, fix_atoms=fix_atoms, record_file=record_file)
                print(f"Performing ML optimization for the final iteration {i}")
                
                run_DFT(work_directory, i, directories, utils_dir, batch_size, record_file=record_file, if_final=final_DFT)  
                for directory in directories:
                    convert_data(iteration='final', directory=directory, db_path=db_path, record_file=record_file)
                finetune_model(work_directory, i, utils_dir, record_file=record_file)
                break
                
            elif last_step == 'DFT_opt' and last_directory:
                directories_to_process = directories[directories.index(last_directory) + 1:]      
                run_DFT(work_directory, i, directories_to_process, utils_dir, batch_size, record_file=record_file, if_final=final_DFT)
                for directory in directories:
                    convert_data(iteration='final', directory=directory, db_path=db_path, record_file=record_file)
                finetune_model(work_directory, i, utils_dir, record_file=record_file)
                        
            else:
                for directory in directories:
                    run_ML(work_directory, i, directory, checkpoint_path=best_checkpoint_path, fix_atoms=fix_atoms, record_file=record_file)
                run_DFT(work_directory, i, directories, utils_dir, batch_size, record_file=record_file, if_final=final_DFT)
                for directory in directories:
                    convert_data(iteration='final', directory=directory, db_path=db_path, record_file=record_file)
                finetune_model(work_directory, i, utils_dir, record_file=record_file)
            break    
            
        if last_step == 'ML_opt' and last_directory:
            directories_to_process = directories[directories.index(last_directory) + 1:]
            for directory in directories_to_process:
                run_ML(work_directory, i, directory, checkpoint_path=best_checkpoint_path, fix_atoms=fix_atoms, record_file=record_file)
            run_DFT(work_directory, i, directories, utils_dir, batch_size, record_file=record_file, if_final=False)
            for directory in directories:
                convert_data(i, directory, db_path=db_path, record_file=record_file)
            print(f"Starting finetune process for iteration {i}")
            os.chdir(work_directory)
            finetune_model(work_directory, i, utils_dir, record_file=record_file)
            
        elif last_step == 'DFT_opt' and last_directory:
            directories_to_process = directories[directories.index(last_directory) + 1:]      
            run_DFT(work_directory, i, directories_to_process, utils_dir, batch_size, record_file=record_file, if_final=False)
            for directory in directories:
                convert_data(i, directory, db_path=db_path, record_file=record_file)
            print(f"Starting finetune process for iteration {i}")
            os.chdir(work_directory)
            finetune_model(work_directory, i, utils_dir, record_file=record_file)
            
        elif last_step == 'convert_data' and last_directory:
            directories_to_process = directories[directories.index(last_directory) + 1:]               
            for directory in directories_to_process:
                convert_data(i, directory, db_path=db_path, record_file=record_file)
            print(f"Starting finetune process for iteration {i}")
            os.chdir(work_directory)
            finetune_model(work_directory, i, utils_dir, record_file=record_file)            
        else:
            for directory in directories:
                run_ML(work_directory, i, directory, checkpoint_path=best_checkpoint_path, fix_atoms=fix_atoms, record_file=record_file)
            run_DFT(work_directory, i, directories, utils_dir, batch_size, record_file=record_file, if_final=False)
            for directory in directories:            
                convert_data(i, directory, db_path=db_path, record_file=record_file)
            print(f"Starting finetune process for iteration {i}")
            os.chdir(work_directory)
            finetune_model(work_directory, i, utils_dir, record_file=record_file)
        last_step = None  
        
    print(f"Total {max_iteration} iterations have finished.")    

if __name__ == "__main__":
    main()