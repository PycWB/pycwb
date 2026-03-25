import os
import subprocess
import click
import shutil


class Slurm:
    def __init__(self, working_dir='.', conda_env=None, additional_init="", job_per_worker=10,
                 n_proc=1, memory="6GB", disk="4GB"):
        self.working_dir = os.path.abspath(working_dir)
        self.conda_env = conda_env
        self.additional_init = additional_init
        self.n_proc = n_proc
        self.memory = memory
        self.disk = disk
        self.slurm_dir = os.path.join(self.working_dir, 'slurm')
        self.slurm_script = None
        self.job_per_worker = job_per_worker

    def create(self, job_segments, submit=False):
        if os.path.exists(self.slurm_dir):
            if not click.confirm("Are you sure you want to clean the existing slurm directory?", default=False):
                print("Cleaning aborted.")
                return
            shutil.rmtree(self.slurm_dir, ignore_errors=True)
    
        self.generate_job_script(job_segments)
        self.generate_merge_script()
        if submit:
            self.submit()

    def generate_job_script(self, job_segments):
        working_dir = self.working_dir
        slurm_dir = self.slurm_dir
        job_per_worker = self.job_per_worker
        n_proc = self.n_proc
        memory = self.memory
        conda_env = self.conda_env

        n_workers = (len(job_segments) + job_per_worker - 1) // job_per_worker
        os.makedirs(slurm_dir, exist_ok=True)

        # create run.sh
        with open(f"{slurm_dir}/run.sh", 'w') as f:
            f.write(f"""#!/bin/bash
#SBATCH --job-name={os.path.basename(working_dir)}
#SBATCH --output=log/output_%A_%a.out
#SBATCH --error=log/error_%A_%a.err
#SBATCH --array=0-{n_workers-1}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={n_proc}
#SBATCH --time=72:00:00
#SBATCH --constraint=cal
#SBATCH --mem={memory}

total={len(job_segments)} # Total number of job segments
jobs_per_worker={job_per_worker} # Number of jobs per worker
n_proc={n_proc}                  # Number of processes per worker

# Compute the start and end indices for this task
task_id=${{SLURM_ARRAY_TASK_ID}}
start=$((task_id * jobs_per_worker + 1))
end=$(((task_id + 1) * jobs_per_worker))

# Cap the end index to not exceed the total
if [ $end -gt $total ]; then
    end=$total
fi

echo "Task ID: $task_id processing jobs $start to $end using $n_proc processes."

conda activate {conda_env}
{self.additional_init}
pycwb batch-runner {working_dir}/config/user_parameters.yaml --work-dir={working_dir} --jobs=$start-$end --n-proc=1 --n-workers={self.n_proc}
                """)

        os.chmod(f"{slurm_dir}/run.sh", 0o755)
        self.slurm_script = os.path.join(slurm_dir, 'run.sh')
        print(f'SLURM script: {self.slurm_script}')

    def generate_merge_script(self):
        working_dir = self.working_dir
        slurm_dir = self.slurm_dir

        os.makedirs(slurm_dir, exist_ok=True)
        pass

    def submit(self):
        if not self.slurm_script or not os.path.exists(self.slurm_script):
            raise RuntimeError("SLURM script not found. Run generate_job_script() first.")

        result = subprocess.run(['sbatch', self.slurm_script], check=True,
                                capture_output=True, text=True)
        print(result.stdout.strip())
        if result.stderr:
            print(result.stderr.strip())