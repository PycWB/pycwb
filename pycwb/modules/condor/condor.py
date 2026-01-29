import os
import shutil
import click


class HTCondor:
    def __init__(self, working_dir='.', conda_env=None, additional_init="", 
                 accounting_group=None, job_per_worker=10, container_image=None,
                 should_transfer_files=False,
                 n_proc=1, memory="6GB", disk="4GB", conda_init=None):
        self.working_dir = os.path.abspath(working_dir)
        self.conda_env = conda_env
        self.additional_init = additional_init
        self.n_proc = n_proc
        self.memory = memory
        self.disk = disk
        self.dag_dir = os.path.join(self.working_dir, 'condor')
        self.dag_file = None
        self.container_image = container_image
        self.should_transfer_files = should_transfer_files

        if container_image:
            self.should_transfer_files = True

        if not accounting_group:
            raise ValueError("Accounting group is required for condor batch submission")

        self.accounting_group = accounting_group
        self.job_per_worker = job_per_worker
        if not conda_init:
            if not container_image:
                self.conda_init = 'source /cvmfs/software.igwn.org/conda/etc/profile.d/conda.sh'
            else:
                self.conda_init = ''
        

    def create(self, job_segments, submit=False):
        if os.path.exists(self.dag_dir):
            if not click.confirm("Are you sure you want to clean the existing condor directory?", default=False):
                print("Cleaning aborted.")
                return
            shutil.rmtree(self.dag_dir, ignore_errors=True)
    
        self.generate_job_script()
        self.generate_merge_script()
        self.generate_condor_dag(job_segments)
        if submit:
            self.submit_condor_dag()

    def generate_job_script(self):
        working_dir = self.working_dir
        dag_dir = self.dag_dir
        should_transfer_files = self.should_transfer_files

        os.makedirs(dag_dir, exist_ok=True)

        # create run.sh
        with open(f"{dag_dir}/run.sh", 'w') as f:
            f.write(f"""#!/bin/bash
{self.conda_init}
{f'conda activate {self.conda_env}' if self.conda_env else ''}
{self.additional_init if self.additional_init else ''}
{ 'mkdir -p job_status trigger output log' if should_transfer_files else ''}
pycwb batch-runner {working_dir}/config/user_parameters.yaml --work-dir={working_dir} --jobs=$1 --n-proc={self.n_proc}
            """)

        # add execute permission to run.sh
        os.chmod(f"{dag_dir}/run.sh", 0o755)

    def generate_merge_script(self):
        working_dir = self.working_dir
        dag_dir = self.dag_dir
        should_transfer_files = self.should_transfer_files

        os.makedirs(dag_dir, exist_ok=True)

        # create merge.sh
        with open(f"{dag_dir}/merge.sh", 'w') as f:
            f.write(f"""#!/bin/bash
{self.conda_init}
{f'conda activate {self.conda_env}' if self.conda_env else ''}
{self.additional_init if self.additional_init else ''}
{ 'mkdir -p log' if should_transfer_files else ''}
pycwb merge-catalog --work-dir={working_dir}
            """)

        # add execute permission to merge.sh
        os.chmod(f"{dag_dir}/merge.sh", 0o755)

    def generate_condor_dag(self, job_segments):
        import getpass
        import htcondor2 as htcondor
        from htcondor2 import dags
        
        working_dir = self.working_dir
        dag_dir = self.dag_dir
        accounting_group = self.accounting_group
        job_per_worker = self.job_per_worker
        container_image = self.container_image
        should_transfer_files = self.should_transfer_files

        n_workers = (len(job_segments) + job_per_worker - 1) // job_per_worker
        jobs = [{
            'jobs': f"{i * job_per_worker + 1}-{min((i + 1) * job_per_worker, len(job_segments))}"
        } for i in range(n_workers)]

        os.makedirs(dag_dir, exist_ok=True)

        # create the submit description for the batch job
        batch_job_config = {
            "universe": "vanilla",
            "initialdir": working_dir,
            "executable": "run.sh",
            "arguments": f"$(jobs)",  # Passing jobs as an argument
            "should_transfer_files": "no",
            "output": "log/batch-$(jobs).out",
            "error": "log/batch-$(jobs).err",
            "log": "log/batch-$(jobs).log",
            "accounting_group": accounting_group,
            "accounting_group_user": getpass.getuser(),
            "on_exit_hold": "(ExitCode != 0)",
            "request_cpus": f"{self.n_proc}",
            "request_memory": self.memory,
            "request_disk": self.disk,
            "use_oauth_services": "scitokens",
            "environment": "BEARER_TOKEN_FILE=$$(CondorScratchDir)/.condor_creds/scitokens.use",
        }

        merge_job_config = {
            "universe": "vanilla",
            "initialdir": working_dir,
            "executable": "merge.sh",
            "should_transfer_files": "no",
            "log": "log/merge.log",
            "output": "log/merge.out",
            "error": "log/merge.err",
            "accounting_group": accounting_group,
            "accounting_group_user": getpass.getuser(),
            "request_cpus": f"{self.n_proc}",
            "request_memory": self.memory,
            "request_disk": self.disk,
        }

        if container_image:
            batch_job_config['universe'] = 'container'
            batch_job_config['container_image'] = container_image
            merge_job_config['universe'] = 'container'
            merge_job_config['container_image'] = container_image

        if should_transfer_files:
            batch_job_config['transfer_input_files'] =  f"{working_dir}/job_status, {working_dir}/config, {working_dir}/input, {working_dir}/wdmXTalk, {working_dir}/catalog"
            batch_job_config['transfer_output_files'] = "catalog, job_status, trigger, output, log"
            batch_job_config['should_transfer_files'] = "yes"
            batch_job_config['when_to_transfer_output'] = "ON_EXIT_OR_EVICT"
            merge_job_config['transfer_input_files'] = f"{working_dir}/catalog"
            merge_job_config['transfer_output_files'] = "catalog, log"
            merge_job_config['should_transfer_files'] = "yes"
            merge_job_config['when_to_transfer_output'] = "ON_EXIT_OR_EVICT"

        batch_job = htcondor.Submit(batch_job_config)
        merge_job = htcondor.Submit(merge_job_config)

        dag = dags.DAG()

        batch_layer = dag.layer(
            name='pycwb_batch',
            submit_description=batch_job,
            vars=jobs,
        )

        merge_layer = batch_layer.child_layer(
            name = 'merge',
            submit_description = merge_job,
        )

        # make the magic happen!
        dag_file = dags.write_dag(dag, dag_dir, dag_file_name=f'{os.path.basename(working_dir)}.dag')

        print(f'DAG directory: {dag_dir}')
        print(f'DAG description file: {dag_file}')

        self.dag_file = dag_file

    def submit_condor_dag(self):
        import htcondor2 as htcondor

        working_dir = self.working_dir
        dag_dir = self.dag_dir

        dag_submit = htcondor.Submit.from_dag(str(self.dag_file), {'force': 1, 'import_env': 1})
        print('------------------------')
        print(dag_submit)
        print('------------------------')
        os.chdir(dag_dir)

        schedd = htcondor.Schedd()
        cluster_id = schedd.submit(dag_submit).cluster()

        print(f"DAGMan job cluster is {cluster_id}")

        os.chdir(working_dir)
