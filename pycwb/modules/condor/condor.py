import os
import subprocess


def generate_job_script(user_parameter_file, conda_env, working_dir, threads=0):
    """Generate the submission file for the search.

    Parameters
    ----------
    user_parameter_file: str
        path to user parameters file
    conda_env: str
        name of the conda environment
    working_dir: str
        path to the working directory

    Returns
    -------
    None
    """
    working_dir = os.path.abspath(working_dir)
    f = open(f"{working_dir}/submit.sh", "w")
    script = f"""#!/bin/bash
source /cvmfs/oasis.opensciencegrid.org/ligo/sw/conda/etc/profile.d/conda.sh
conda activate {conda_env}
export HOME_WAT_FILTERS={os.environ.get('HOME_WAT_FILTERS')}

cd {working_dir}

pycwb_search {user_parameter_file} --work-dir {working_dir} --overwrite -n {threads} | tee run.log
"""
    f.write(script)
    f.close()

    return script


def generate_condor_sub(workingDir, accounting_group="ligo.sim.o4.burst.allsky.cwboffline", threads=6):
    import getpass
    f = open(f"{workingDir}/submit.sub", "w")
    sub = f"""# Unix submit description file
universe = vanilla

executable              = submit.sh

log                     = log/main.log
output                  = log/outfile.txt
error                   = log/errors.txt

should_transfer_files   = Yes

accounting_group = {accounting_group}
accounting_group_user = {getpass.getuser()}

request_cpus   = {threads}
request_memory = 8G
request_disk   = 2G

queue 1
"""
    f.write(sub)
    f.close()

    return sub


def submit(working_dir):
    pwd = os.getcwd()
    os.chdir(working_dir)
    try:
        process = subprocess.run(
            ["condor_submit", f"submit.sub"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        # output results
        if process.returncode == 0:
            print(process.stderr)
        else:
            print(process.stderr)
    except Exception as e:
        print(e)
    finally:
        # leave working directory
        os.chdir(pwd)
        print(f" -- Working directory: {pwd}")