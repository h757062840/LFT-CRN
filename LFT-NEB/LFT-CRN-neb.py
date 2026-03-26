import os
import shutil
import time
import subprocess
import argparse
import re
import yaml
import dpdata
from ase import io
from ase.io import read
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
            else:
                if current_status != previous_status[job_id]:
                    print(f"Waiting for job {job_id} to complete...")
                
            previous_status[job_id] = current_status
        
        if job_ids:
            time.sleep(10)

def write_record(record_file, iteration, step, directory):
    """Log progress to enable restarting."""
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
    
    db_path = os.path.join(work_directory, 'output_database.db')
    with connect(db_path) as db:
        n_data = db.count()
    print(f'Data number: {n_data}')

    eval_every = int(3 * n_data / 6) if n_data > 6 else 1
    
    finetune_dir = f'{work_directory}/finetune_neb_{iteration}'
    os.makedirs(finetune_dir, exist_ok=True)
    
    shutil.copy(os.path.join(utils_dir, 'sub.oc'), f'{finetune_dir}/sub.oc')
    shutil.copy(os.path.join(utils_dir, 'finetune1.yml'), f'{finetune_dir}/finetune1.yml')   
    shutil.copy(os.path.join(utils_dir, 'main.py'), f'{finetune_dir}/main.py')       
    
    os.chdir(finetune_dir)
    
    if iteration > 1:
        change_model(work_directory, iteration)
    
    modify_yaml_parameter(yml_file_path="finetune1.yml", param_path='optim.eval_every', new_value=eval_every)    
    
    result = subprocess.run(['sbatch', 'sub.oc'], stdout=subprocess.PIPE)
    job_id = result.stdout.decode().strip().split()[-1]
    wait_for_jobs_completion([job_id]) # Unified waiting function
    write_record(record_file, iteration, step="finetune", directory=work_directory)
    
def change_model(work_directory, iteration):
    """Update fine-tuning script to use the best checkpoint from previous iteration."""
    prev_iteration_dir = f"{work_directory}/finetune_neb_{iteration - 1}"
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

def extract_energy_from_outcar(outcar_path):
    """Extract final energy from VASP OUTCAR."""
    with open(outcar_path, 'r') as outcar_file:
        lines = outcar_file.readlines()
    
    for line in reversed(lines):
        match = re.search(r'energy\(sigma->0\)\s*=\s*(-?\d+\.\d+)', line)
        if match:
            return float(match.group(1))
    return None  

def get_previous_images_num(directory, iteration):
    """Determine number of NEB images from previous iteration's directory structure."""
    neb_folder = os.path.join(directory, f'neb{iteration-1}')
    
    if not os.path.exists(neb_folder):
        raise FileNotFoundError(f"The folder {neb_folder} does not exist.")
    
    subdirectories = [name for name in os.listdir(neb_folder) if os.path.isdir(os.path.join(neb_folder, name))]
    
    # Subtract initial and final state folders (00 and N+1)
    n_images = len(subdirectories) - 2
    return n_images
    
def run_ML(work_directory, iteration, directories, checkpoint_path, fix_atoms, record_file, utils_dir, n_images, k, fmax_list, batch_size):
    """
    Submit ML-NEB calculation batches via Slurm.
    Relies on external 'MLneb.py' inside utils_dir.
    """
    print(f"Running ML NEB for iteration {iteration}...")
    if not directories:
        print("No directories to process. Skipping ML NEB calculation.")
        return    
    structure_dir = f"{work_directory}/ML_conf_iter{iteration}"
    os.makedirs(structure_dir, exist_ok=True)
    total_dirs = len(directories)
    
    job_ids = []
    all_processed_dirs = []

    fmax_str = ' '.join(map(str, fmax_list))

    for batch_idx, batch_start in enumerate(range(0, total_dirs, batch_size), 1):
        current_batch = directories[batch_start:batch_start + batch_size]
        batch_folder = f"{structure_dir}/batch_{batch_idx}"
        os.makedirs(batch_folder, exist_ok=True)
        all_processed_dirs.extend(current_batch)
        total_tasks = len(current_batch)

        print(f"Building batch {batch_idx} ({total_tasks} directories)")
        
        subneb_path = f"{batch_folder}/sub.neb"
        batch_log = f"{batch_folder}/batch.log"
        with open(subneb_path, 'w') as f:
            # SLURM header - adjust based on cluster config
            f.write("#!/bin/bash\n")
            f.write("#SBATCH -N 1\n")
            f.write("#SBATCH -p intel96,standard\n")
            f.write("#SBATCH --exclusive\n\n")

            # Batch initialization
            f.write(f'echo "=== BATCH START: $(date +\'%Y-%m-%d %H:%M:%S\') ==="\n')
            f.write(f'echo "Total tasks: {total_tasks}" >> {batch_log}\n')
            
            # Task commands
            for task_num, directory in enumerate(current_batch, 1):
                abs_dir = os.path.abspath(directory)
                # Determine image number: static for iter 1, dynamic based on previous DFT folders for iter > 1
                curr_n_images = get_previous_images_num(directory, iteration) if iteration != 1 else n_images
                
                f.write(f'echo "[$(date +\'%Y-%m-%d %H:%M:%S\')] START TASK {task_num}/{total_tasks}"\n')
                
                # Command execution, calling MLneb.py from utils
                cmd = (
                    f"(cd {abs_dir} && "
                    f"echo '===== TASK START =====' >> MLneb.log && "
                    f"python {utils_dir}/MLneb.py "
                    f"--checkpoint_path {checkpoint_path} "
                    f"--fix_atoms {fix_atoms} "
                    f"--n_images {curr_n_images} "
                    f"--k {k} "
                    f"--fmax_list {fmax_str} "
                    f" && "
                    f"echo '===== TASK COMPLETED =====' >> MLneb.log || "
                    f"echo '!!!!! TASK FAILED !!!!!' >> MLneb.log) "
                    f"2>&1 | tee -a {abs_dir}/MLneb.log\n"
                )
                f.write(cmd + "\n")
                f.write(f'echo "[$(date +\'%Y-%m-%d %H:%M:%S\')] END TASK {task_num}/{total_tasks}" >> {batch_log}\n')
        
        # Submit job
        os.chdir(batch_folder)
        result = subprocess.run(['sbatch', subneb_path], stdout=subprocess.PIPE)
        job_id = result.stdout.decode().strip().split()[-1]
        job_ids.append(job_id)
        print(f"Batch {batch_idx} submitted (Job ID: {job_id})")

    wait_for_jobs_completion(job_ids)
    
    # Verify and record completion
    print("\nVerifying ML NEB completion:")
    for directory in all_processed_dirs:
        log_file = f"{directory}/MLneb.log"
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                content = f.read()
                if "COMPLETED" in content:
                    write_record(record_file, iteration, "ML_neb", directory)
                    print(f"✓ {directory}")
                    continue
        print(f"✗ {directory} (Incomplete or failed)")
    
    os.chdir(work_directory)

def run_DFT(work_directory, iteration, directories, utils_dir, batch_size, record_file, n_images, if_final=False):
    """
    Prepare VASP NEB directories and submit batches via Slurm.
    Setup assumes specific path structure for Initial State (IS) OUTCARs.
    """
    if not directories:
        print("No directories to process. Skipping DFT NEB calculation.")
        return    
    print(f"Running DFT NEB for iteration {iteration}...")
    structure_dir = os.path.join(work_directory, f'DFT_conf_iter{iteration}')
    os.makedirs(structure_dir, exist_ok=True)

    total_dirs = len(directories)
    batches = [directories[i:i + batch_size] for i in range(0, total_dirs, batch_size)]
    
    print(f"Total directories: {total_dirs}, divided into {len(batches)} batches.")

    all_job_ids = []

    for batch_idx, current_batch in enumerate(batches):
        print(f"Processing batch {batch_idx + 1}/{len(batches)}...")

        batch_structure_dir = os.path.join(structure_dir, f'batch_{batch_idx}')
        os.makedirs(batch_structure_dir, exist_ok=True)

        batch_script_path = os.path.join(batch_structure_dir, 'sub.vasp')
        with open(batch_script_path, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("#SBATCH -N 1\n")
            f.write("#SBATCH -p intel96,standard\n")
            f.write("#SBATCH --exclusive\n")
            f.write("module load VASP/6.4.1-gzbuild-intel_8300\n\n")
            
            for directory in current_batch:
                # Dynamic image number determination
                curr_n_images = get_previous_images_num(directory, iteration) if iteration != 1 else n_images
                neb_dir = f'{directory}/neb{iteration}'
                os.makedirs(neb_dir, exist_ok=True)
                
                print(f'Setting up DFT for {directory} with {curr_n_images} images.')
                
                # Setup image folders (00 to N+1)
                for i in range(curr_n_images + 2):
                    folder_name = os.path.join(neb_dir, f"{i:02}")
                    os.makedirs(folder_name, exist_ok=True)
                    
                    if i == 0:
                        # Pathing assumes specific project structure for Initial State
                        is_outcar = os.path.abspath(os.path.join(directory, "../../IS/opt3/OUTCAR"))
                        is_poscar = os.path.abspath(os.path.join(directory, "../../IS/opt3/POSCAR"))
                        shutil.copy(is_outcar, os.path.join(folder_name, "OUTCAR"))
                        shutil.copy(is_poscar, os.path.join(folder_name, "POSCAR"))
                    elif i == curr_n_images + 1:
                        fs_outcar = os.path.abspath(os.path.join(directory, "../opt3/OUTCAR"))
                        fs_poscar = os.path.abspath(os.path.join(directory, "../opt3/POSCAR"))
                        shutil.copy(fs_outcar, os.path.join(folder_name, "OUTCAR"))
                        shutil.copy(fs_poscar, os.path.join(folder_name, "POSCAR"))
                    else:
                        # Copy generated ML POSCARs
                        ml_poscar = os.path.abspath(os.path.join(directory, f"iter{iteration}_POSCAR{str(i).zfill(2)}"))
                        shutil.copy(ml_poscar, os.path.join(folder_name, "POSCAR"))
                        
                        # VASP standard inputs
                        shutil.copy(f"{utils_dir}/KPOINTS", os.path.join(folder_name, "KPOINTS"))
                        shutil.copy(f"{utils_dir}/INCAR_NEB", os.path.join(folder_name, "INCAR"))
                        
                        # Generate POTCAR using VASPKIT
                        # Note: Chdir implies setup must happen sequentially before writing batch script,
                        # refactored for batch setup efficiency.
                
                # Standardized setup routine (separated from Chdir logic)
                for i in range(1, curr_n_images + 1):
                    img_path = os.path.abspath(os.path.join(neb_dir, f"{i:02}"))
                    cmd_setup = f"cd {img_path} && echo -e '103\n' | vaspkit > /dev/null"
                    subprocess.run(cmd_setup, shell=True)

                f.write(f"echo 'Starting VASP NEB in {os.path.abspath(neb_dir)}'\n")
                f.write(f"cd {os.path.abspath(neb_dir)}; mpirun vasp_std > fp.log 2>vasp.err\n\n")        

        os.chdir(batch_structure_dir)
        result = subprocess.run(['sbatch', 'sub.vasp'], stdout=subprocess.PIPE)
        job_id = result.stdout.decode().strip().split()[-1]
        all_job_ids.append(job_id)
        print(f"Batch {batch_idx} submitted (Job ID: {job_id})")

    wait_for_jobs_completion(all_job_ids)
    
    # Record completion
    for directory in directories:
        write_record(record_file, iteration, step="DFT_neb", directory=directory)
    
    os.chdir(work_directory)
    print(f"All DFT NEBs completed for iteration {iteration}.")

def convert_data(iteration, directory, db_path, record_file):
    """Convert VASP NEB OUTCARs to ASE Database utilizing dpdata."""
    print(f"Converting data for {directory} (iteration {iteration})")
    
    db_dir = os.path.dirname(db_path)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
    
    db = connect(db_path)
    try:
        n_images = get_previous_images_num(directory, iteration)
    except Exception as e:
        print(f"Skipping conversion for {directory}: {e}")
        return

    # Iterate over image folders (excluding 00 and N+1 typically)
    for i in range(1, n_images + 1): 
        i_str = str(i).zfill(2)
        outcar_path = os.path.join(directory, f"neb{iteration}", i_str, "OUTCAR")     
        
        if not os.path.exists(outcar_path):
            print(f"Warning: {outcar_path} does not exist. Skipping.")
            continue   
            
        try:
            data = dpdata.LabeledSystem(outcar_path)
        except Exception as e:
            print(f"Error loading OUTCAR {outcar_path}: {e}")
            continue
        
        for j, structure in enumerate(data):
            ase_structures = structure.to_ase_structure()
            if isinstance(ase_structures, list):
                for k, ase_atoms in enumerate(ase_structures):
                    db.write(ase_atoms, key_value_pairs={"iteration": iteration, "image": i, "frame": j, "subframe": k})
            else:
                db.write(ase_structures, key_value_pairs={"iteration": iteration, "image": i, "frame": j})   
                
    print(f"Data for {directory} added to {db_path}")
    write_record(record_file, iteration, step="convert_data", directory=directory) 
    
def get_latest_model(work_directory, iteration):
    """Retrieve path to the best checkpoint from the previous fine-tuning iteration."""
    finetune_dir = f'{work_directory}/finetune_neb_{iteration}'
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
    parser = argparse.ArgumentParser(description="Active Learning NEB Workflow")
    parser.add_argument("--config", type=str, default="config_neb.yaml", help="Path to NEB config file")
    args = parser.parse_args()

    # Load External Configuration
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    directories = config['directories']
    n_images = config['neb_params']['default_n_images']
    k = config['neb_params']['spring_constant']
    max_iteration = config['workflow']['max_iteration']
    batch_size = config['workflow']['batch_size']
    final_DFT = config['workflow']['final_DFT']
    fix_atoms = config['structure']['fix_atoms']
    
    work_directory = os.path.abspath(config['paths']['work_directory'])
    utils_dir = config['paths']['utils_dir']
    db_path = os.path.join(work_directory, config['paths']['db_name'])
    record_file = os.path.join(work_directory, config['paths']['record_file'])
    origin_checkpoint_path = config['paths']['origin_checkpoint_path']

    # Fmax schedules
    fmax_schedule = config['fmax_schedules']

    # Restart Logic
    last_record = read_last_record(record_file)
    iteration = 1
    last_step = None
    last_directory = None
    
    if last_record is None:
        print("No record available, starting from first iteration!")
    else:
        iteration, last_step, last_directory = last_record
        print(f"Record found: Iteration {iteration}, Step {last_step}, Directory {last_directory}")
        if last_step == 'finetune':
            iteration += 1
            last_step = None

    # Main Loop
    for i in range(iteration, max_iteration + 1):
        # Determine Checkpoint and Fmax list
        if i == 1:
            best_checkpoint_path = origin_checkpoint_path
            current_fmax_list = fmax_schedule['stage_1']
        else:
            previous_iteration = i - 1
            print(f"Loading checkpoint from iteration {previous_iteration}")
            best_checkpoint_path = get_latest_model(work_directory, previous_iteration)
            current_fmax_list = fmax_schedule['stage_2']

        # Determine if current batch requires special handling based on last record
        dirs_to_ml = directories
        dirs_to_dft = directories
        dirs_to_convert = directories

        if i == iteration and last_directory:
            idx = directories.index(last_directory)
            if last_step == 'ML_neb':
                dirs_to_ml = directories[idx + 1:]
            elif last_step == 'DFT_neb':
                dirs_to_ml = [] # ML done
                dirs_to_dft = directories[idx + 1:]
            elif last_step == 'convert_data':
                dirs_to_ml = [] # ML done
                dirs_to_dft = [] # DFT done
                dirs_to_convert = directories[idx + 1:]

        # 1. ML NEB
        if dirs_to_ml:
            run_ML(work_directory, i, dirs_to_ml, checkpoint_path=best_checkpoint_path, 
                   fix_atoms=fix_atoms, record_file=record_file, utils_dir=utils_dir, 
                   n_images=n_images, k=k, fmax_list=current_fmax_list, batch_size=batch_size)
        
        # 2. DFT NEB
        # Determine if final high-accuracy DFT is required
        use_final_settings = final_DFT if i == max_iteration else False
        if dirs_to_dft:
            run_DFT(work_directory, i, dirs_to_dft, utils_dir, batch_size, 
                    record_file=record_file, n_images=n_images, if_final=use_final_settings)
        
        # 3. Convert Data
        if dirs_to_convert:
            for directory in dirs_to_convert:
                convert_data(i, directory, db_path=db_path, record_file=record_file)

        # 4. Fine-tune Model (Skip on actual final iteration of workflow)
        if i < max_iteration:
            print(f"Starting Finetune process for iteration {i}")
            os.chdir(work_directory)
            finetune_model(work_directory, i, utils_dir, record_file=record_file)
        
        # Reset last step for next iteration loop
        last_step = None  

    print(f"Active Learning workflow finished after {max_iteration} iterations.")

if __name__ == "__main__":
    main()