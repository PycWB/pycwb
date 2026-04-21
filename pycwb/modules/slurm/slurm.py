import os
import re
import subprocess
import click
import shutil


class Slurm:
    def __init__(self, working_dir='.', conda_env=None, additional_init="", job_per_worker=10,
                 n_proc=1, memory="6GB", disk="4GB",
                 time="72:00:00", constraint=None, partition=None, n_retries=5):
        self.working_dir = os.path.abspath(working_dir)
        self.conda_env = conda_env
        self.additional_init = additional_init
        self.n_proc = n_proc
        self.memory = memory
        self.disk = disk
        self.time = time or "72:00:00"
        self.constraint = constraint
        self.partition = partition
        self.n_retries = n_retries
        self.slurm_dir = os.path.join(self.working_dir, 'slurm')
        self.slurm_script = None
        self.merge_script = None
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

        optional_lines = []
        if self.constraint:
            optional_lines.append(f"#SBATCH --constraint={self.constraint}")
        if self.partition:
            optional_lines.append(f"#SBATCH --partition={self.partition}")
        optional_sbatch = ('\n' + '\n'.join(optional_lines)) if optional_lines else ''

        # create run.sh
        with open(f"{slurm_dir}/run.sh", 'w') as f:
            f.write(f"""#!/bin/bash
#SBATCH --job-name={os.path.basename(working_dir)}
#SBATCH --output=log/output_%A_%a.out
#SBATCH --error=log/error_%A_%a.err
#SBATCH --array=0-{n_workers-1}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={n_proc}
#SBATCH --time={self.time}
#SBATCH --mem={memory}
#SBATCH --requeue{optional_sbatch}

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

MAX_RETRIES={self.n_retries}
attempt=0
while [ $attempt -lt $MAX_RETRIES ]; do
    pycwb batch-runner {working_dir}/config/user_parameters.yaml --work-dir={working_dir} --jobs=$start-$end --n-proc=1 --n-workers={self.n_proc} && break
    attempt=$((attempt + 1))
    echo "Attempt $attempt failed, retrying in 30s..."
    sleep 30
done
if [ $attempt -eq $MAX_RETRIES ]; then
    echo "All $MAX_RETRIES attempts failed for jobs $start-$end"
    exit 1
fi
""")

        os.chmod(f"{slurm_dir}/run.sh", 0o755)
        self.slurm_script = os.path.join(slurm_dir, 'run.sh')
        print(f'SLURM job script: {self.slurm_script}')

    def generate_merge_script(self):
        working_dir = self.working_dir
        slurm_dir = self.slurm_dir

        os.makedirs(slurm_dir, exist_ok=True)

        optional_lines = []
        if self.constraint:
            optional_lines.append(f"#SBATCH --constraint={self.constraint}")
        if self.partition:
            optional_lines.append(f"#SBATCH --partition={self.partition}")
        optional_sbatch = ('\n' + '\n'.join(optional_lines)) if optional_lines else ''

        with open(f"{slurm_dir}/merge.sh", 'w') as f:
            f.write(f"""#!/bin/bash
#SBATCH --job-name={os.path.basename(working_dir)}_merge
#SBATCH --output=log/merge.out
#SBATCH --error=log/merge.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem={self.memory}{optional_sbatch}

conda activate {self.conda_env}
{self.additional_init}
pycwb merge --work-dir={working_dir}
""")

        os.chmod(f"{slurm_dir}/merge.sh", 0o755)
        self.merge_script = os.path.join(slurm_dir, 'merge.sh')
        print(f'SLURM merge script: {self.merge_script}')

    def submit(self):
        if not self.slurm_script or not os.path.exists(self.slurm_script):
            raise RuntimeError("SLURM script not found. Run generate_job_script() first.")
        if not self.merge_script or not os.path.exists(self.merge_script):
            raise RuntimeError("SLURM merge script not found. Run generate_merge_script() first.")

        # Submit the batch array job
        result = subprocess.run(['sbatch', self.slurm_script], check=True,
                                capture_output=True, text=True)
        print(result.stdout.strip())

        # Parse job ID from "Submitted batch job 12345"
        match = re.search(r'Submitted batch job (\d+)', result.stdout)
        if not match:
            raise RuntimeError(f"Could not parse job ID from sbatch output: {result.stdout!r}")
        job_id = match.group(1)

        # Submit merge job to run only after all array tasks succeed
        merge_result = subprocess.run(
            ['sbatch', f'--dependency=afterok:{job_id}', '--kill-on-invalid-dep=yes',
             self.merge_script],
            check=True, capture_output=True, text=True
        )
        print(merge_result.stdout.strip())
        if merge_result.stderr:
            print(merge_result.stderr.strip())

        print(f"Batch array job ID: {job_id}")
        print(f"Merge job submitted with dependency afterok:{job_id}")