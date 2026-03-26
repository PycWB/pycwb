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

        if should_transfer_files:
            working_dir = '.'

        os.makedirs(dag_dir, exist_ok=True)

        # create run.sh
        with open(f"{dag_dir}/run.sh", 'w') as f:
            f.write(f"""#!/bin/bash
{self.conda_init}
{f'conda activate {self.conda_env}' if self.conda_env else ''}
{self.additional_init if self.additional_init else ''}
{ '''mkdir -p catalog job_status trigger output log
# HTCondor flattens individually-listed files to the execute root; restore expected layout.
[ -f catalog.parquet ] && mv catalog.parquet catalog/
for f in catalog_*.parquet progress_*.parquet; do [ -f "$f" ] && mv "$f" catalog/; done''' if should_transfer_files else ''}
pycwb batch-runner {working_dir}/config/user_parameters.yaml --work-dir={working_dir} --jobs=$1 --n-proc={self.n_proc}
            """)

        # add execute permission to run.sh
        os.chmod(f"{dag_dir}/run.sh", 0o755)

    def generate_merge_script(self):
        working_dir = self.working_dir
        dag_dir = self.dag_dir
        should_transfer_files = self.should_transfer_files
        if should_transfer_files:
                    working_dir = '.'

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
            "use_oauth_services": "scitokens",
            "environment": "BEARER_TOKEN_FILE=$$(CondorScratchDir)/.condor_creds/scitokens.use",
        }

        if container_image:
            batch_job_config['universe'] = 'container'
            batch_job_config['container_image'] = container_image
            merge_job_config['universe'] = 'container'
            merge_job_config['container_image'] = container_image

        if should_transfer_files:
            # Transfer only this job's own catalog and progress fragments.
            # catalog_$(jobs).parquet  — created by prepare_job_runs before submission;
            #                            the job opens it and appends triggers to it.
            # progress_$(jobs).parquet — read by get_completed_lags for per-lag resume;
            #                            empty stubs are pre-created below so HTCondor can
            #                            list them even on the very first run.
            # Never transfer the whole catalog/ dir: on output transfer HTCondor would
            # overwrite other jobs' fragments with the stale copies in this scratch dir.
            batch_job_config['transfer_input_files'] = (
                f"{working_dir}/job_status, {working_dir}/config, "
                f"{working_dir}/input, {working_dir}/wdmXTalk, "
                f"{working_dir}/catalog/catalog.parquet, "
                f"{working_dir}/catalog/catalog_$(jobs).parquet, "
                f"{working_dir}/catalog/progress_$(jobs).parquet, "
                f"$(framefiles)"
            )
            batch_job_config['transfer_output_files'] = (
                "catalog/catalog_$(jobs).parquet, catalog/progress_$(jobs).parquet, "
                "job_status, trigger, output, log"
            )
            batch_job_config['should_transfer_files'] = "yes"
            batch_job_config['when_to_transfer_output'] = "ON_EXIT_OR_EVICT"
            merge_job_config['transfer_input_files'] = f"{working_dir}/catalog"
            merge_job_config['transfer_output_files'] = "catalog, log"
            merge_job_config['should_transfer_files'] = "yes"
            merge_job_config['when_to_transfer_output'] = "ON_EXIT_OR_EVICT"

        batch_job = htcondor.Submit(batch_job_config)
        merge_job = htcondor.Submit(merge_job_config)

        dag = dags.DAG()

        n_workers = (len(job_segments) + job_per_worker - 1) // job_per_worker
        jobs = []
        for i in range(n_workers):
            job_start = i * job_per_worker
            job_end = min((i + 1) * job_per_worker, len(job_segments))
            
            # Collect frame files for this batch of jobs
            framefiles = set()
            for seg in job_segments[job_start:job_end]:
                if seg.frames:
                    for frame in seg.frames:
                        framefiles.add(frame.path)
            
            jobs.append({
                'jobs': f"{job_start + 1}-{job_end}",
                'framefiles': ','.join(sorted(framefiles)) if framefiles else ''
            })

        if should_transfer_files:
            # Pre-create per-job catalog and progress files on the submit node so HTCondor
            # can always find them in transfer_input_files, even on the first run.
            from pycwb.modules.catalog.catalog import Catalog
            from dacite import from_dict
            from pycwb.types.job import WaveSegment
            from pycwb.modules.catalog import read_catalog_metadata
            from pycwb.utils.parser import parse_id_string
            from pycwb.config import Config

            catalog_meta = read_catalog_metadata(
                os.path.join(working_dir, 'catalog', Catalog.DEFAULT_FILENAME)
            )
            config_obj = Config()
            config_obj.load_from_dict(catalog_meta['config'])
            all_segments = [from_dict(WaveSegment, s) for s in catalog_meta['jobs']]

            catalog_dir = os.path.join(working_dir, 'catalog')
            os.makedirs(catalog_dir, exist_ok=True)
            for job in jobs:
                job_ids = parse_id_string(job['jobs'])
                selected = [all_segments[i - 1] for i in job_ids]

                catalog_frag = os.path.join(catalog_dir, f"catalog_{job['jobs']}.parquet")
                if not os.path.exists(catalog_frag):
                    Catalog.create(catalog_frag, config_obj, selected)

                progress_path = os.path.join(catalog_dir, f"progress_{job['jobs']}.parquet")
                if not os.path.exists(progress_path):
                    open(progress_path, 'w').close()

        batch_layer = dag.layer(
            name='pycwb_batch',
            submit_description=batch_job,
            vars=jobs,
            retries=5,
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
