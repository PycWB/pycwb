import os
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
            if os.path.exists(self.dag_dir):
                if not click.confirm("Are you sure you want to clean the existing slurm directory?", default=False):
                    print("Cleaning aborted.")
                    return
                shutil.rmtree(self.dag_dir, ignore_errors=True)
        
            self.generate_job_script()
            self.generate_merge_script()
            if submit:
                self.submit()

        def generate_job_script(self):
            working_dir = self.working_dir
            slurm_dir = self.slurm_dir

            os.makedirs(slurm_dir, exist_ok=True)

            # create run.sh
            pass

        def generate_merge_script(self):
            working_dir = self.working_dir
            slurm_dir = self.slurm_dir

            os.makedirs(slurm_dir, exist_ok=True)
            pass

        def submit(self):
            pass