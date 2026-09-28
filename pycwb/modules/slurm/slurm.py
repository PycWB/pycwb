import os
import re
import subprocess
import click
import shutil
from pycwb.workflow.execution.settings import byte_size


class Slurm:
    def __init__(self, working_dir='.', conda_env=None, additional_init="", job_per_worker=10,
                 n_proc=1, memory="6GB", disk="4GB",
                 time="72:00:00", constraint=None, partition=None, n_retries=5, conda_init=None, job_groups=None, account=None, qos=None,
                 array_max_parallel=None, merge_memory=None, summary_memory=None):
        self.working_dir = os.path.abspath(working_dir)
        self.conda_env = conda_env
        if not conda_init:
            conda_init = 'source /cvmfs/software.igwn.org/conda/etc/profile.d/conda.sh'
        self.conda_init = conda_init
        self.additional_init = additional_init or ""
        self.n_proc = n_proc if n_proc is not None else 1
        self.memory = memory if memory is not None else "6GB"
        self.disk = disk if disk is not None else "4GB"
        self.time = time or "72:00:00"
        self.constraint = constraint
        self.partition = partition
        self.n_retries = n_retries if n_retries is not None else 5
        self.slurm_dir = os.path.join(self.working_dir, 'slurm')
        self.slurm_script = None
        self.merge_script = None
        self.simulation_summary_script = None
        self.job_per_worker = job_per_worker if job_per_worker is not None else 10
        self.job_groups = job_groups
        for name, value in (("account", account), ("qos", qos)):
            if value and not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
                raise ValueError(f"Invalid SLURM {name}: {value!r}")
        if array_max_parallel is not None and (type(array_max_parallel) is not int or array_max_parallel < 1):
            raise ValueError("array_max_parallel must be a positive integer")
        self.account, self.qos = account, qos
        self.array_max_parallel = array_max_parallel
        self.merge_memory = merge_memory or self.memory
        self.summary_memory = summary_memory or self.memory

    def create(self, job_segments, submit=False):
        if os.path.exists(self.slurm_dir):
            if not click.confirm("Are you sure you want to clean the existing slurm directory?", default=False):
                print("Cleaning aborted.")
                return
            shutil.rmtree(self.slurm_dir, ignore_errors=True)

        has_simulations = any(seg.injections for seg in job_segments)

        self.generate_job_script(job_segments)
        self.generate_merge_script()
        if has_simulations:
            self.generate_simulation_summary_script()
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
        if self.job_groups is not None:
            n_workers = len(self.job_groups)
        os.makedirs(slurm_dir, exist_ok=True)

        optional_lines = ["#SBATCH --nodes=1"]
        if self.account:
            optional_lines.append(f"#SBATCH --account={self.account}")
        if self.qos:
            optional_lines.append(f"#SBATCH --qos={self.qos}")
        if self.constraint:
            optional_lines.append(f"#SBATCH --constraint={self.constraint}")
        if self.partition:
            optional_lines.append(f"#SBATCH --partition={self.partition}")
        optional_sbatch = ('\n' + '\n'.join(optional_lines)) if optional_lines else ''

        selection = """start=$((task_id * jobs_per_worker + 1))
end=$(((task_id + 1) * jobs_per_worker))
if [ $end -gt $total ]; then
    end=$total
fi
jobs=$start-$end"""
        if self.job_groups is not None:
            selectors = [",".join(str(segment.index) for segment in group) for group in self.job_groups]
            # Numeric IDs are safe shell literals. Embed the immutable mapping so
            # retries do not depend on a mutable external scheduling file.
            selection = ("job_groups=(" + " ".join(selectors) + ")\njobs=${job_groups[$task_id]}\n"
                         "printf -v batch_id 'b%06d' \"$task_id\"")

        allocation_args = ""
        if self.job_groups is not None:
            allocation_args = (f"--allocated-cores={self.n_proc} "
                               f"--memory-limit={byte_size(self.memory)}B")
        job_argument = "--batch-id=$batch_id" if self.job_groups is not None else "--jobs=$jobs"
        # create run.sh
        with open(f"{slurm_dir}/run.sh", 'w') as f:
            f.write(f"""#!/bin/bash
#SBATCH --job-name={os.path.basename(working_dir)}
#SBATCH --output=log/output_%A_%a.out
#SBATCH --error=log/error_%A_%a.err
#SBATCH --array=0-{n_workers-1}{f"%{self.array_max_parallel}" if self.array_max_parallel else ""}
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
{selection}

echo "Task ID: $task_id processing jobs $jobs using $n_proc processes."

{self.conda_init}
{f'conda activate {conda_env}' if conda_env else ''}
{self.additional_init}

MAX_RETRIES={self.n_retries}
attempt=0
while [ $attempt -lt $MAX_RETRIES ]; do
pycwb batch-runner {working_dir}/config/user_parameters.yaml --work-dir={working_dir} {job_argument} {allocation_args} --n-proc=1 --n-workers={self.n_proc} && break
    attempt=$((attempt + 1))
    echo "Attempt $attempt failed, retrying in 30s..."
    sleep 30
done
if [ $attempt -eq $MAX_RETRIES ]; then
    echo "All $MAX_RETRIES attempts failed for jobs $jobs"
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

        optional_lines = ["#SBATCH --nodes=1"]
        if self.account:
            optional_lines.append(f"#SBATCH --account={self.account}")
        if self.qos:
            optional_lines.append(f"#SBATCH --qos={self.qos}")
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
#SBATCH --mem={self.merge_memory}{optional_sbatch}

{self.conda_init}
{f'conda activate {self.conda_env}' if self.conda_env else ''}
{self.additional_init}
pycwb merge --work-dir={working_dir}
""")

        os.chmod(f"{slurm_dir}/merge.sh", 0o755)
        self.merge_script = os.path.join(slurm_dir, 'merge.sh')
        print(f'SLURM merge script: {self.merge_script}')

    def generate_simulation_summary_script(self):
        """Generate simulation_summary.sh — submitted independently (no dependency
        on the analysis array job) since it only needs the config and job-segment
        metadata."""
        working_dir = self.working_dir
        slurm_dir = self.slurm_dir

        os.makedirs(slurm_dir, exist_ok=True)

        optional_lines = ["#SBATCH --nodes=1"]
        if self.account:
            optional_lines.append(f"#SBATCH --account={self.account}")
        if self.qos:
            optional_lines.append(f"#SBATCH --qos={self.qos}")
        if self.constraint:
            optional_lines.append(f"#SBATCH --constraint={self.constraint}")
        if self.partition:
            optional_lines.append(f"#SBATCH --partition={self.partition}")
        optional_sbatch = ('\n' + '\n'.join(optional_lines)) if optional_lines else ''

        with open(f"{slurm_dir}/simulation_summary.sh", 'w') as f:
            f.write(f"""#!/bin/bash
#SBATCH --job-name={os.path.basename(working_dir)}_sim_summary
#SBATCH --output=log/simulation_summary.out
#SBATCH --error=log/simulation_summary.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem={self.summary_memory}{optional_sbatch}

{self.conda_init}
{f'conda activate {self.conda_env}' if self.conda_env else ''}
{self.additional_init}
pycwb simulation-summary {working_dir}/config/user_parameters.yaml --work-dir={working_dir}
""")

        os.chmod(f"{slurm_dir}/simulation_summary.sh", 0o755)
        self.simulation_summary_script = os.path.join(slurm_dir, 'simulation_summary.sh')
        print(f'SLURM simulation summary script: {self.simulation_summary_script}')

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

        # Submit simulation summary job independently — it only needs the config and
        # job-segment metadata, so it can run in parallel with the analysis array job.
        if self.simulation_summary_script and os.path.exists(self.simulation_summary_script):
            sim_result = subprocess.run(
                ['sbatch', self.simulation_summary_script],
                check=True, capture_output=True, text=True
            )
            print(sim_result.stdout.strip())
            if sim_result.stderr:
                print(sim_result.stderr.strip())
            sim_match = re.search(r'Submitted batch job (\d+)', sim_result.stdout)
            if sim_match:
                print(f"Simulation summary job submitted: {sim_match.group(1)}")

        print(f"Batch array job ID: {job_id}")
        print(f"Merge job submitted with dependency afterok:{job_id}")
