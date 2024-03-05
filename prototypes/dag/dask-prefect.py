from time import sleep

import dask.distributed
from prefect import flow, task
from prefect_dask import DaskTaskRunner, get_dask_client


@task(name="Print Hello")
def print_hello(name):
    msg = f"Hello {name}!"
    sleep(20)
    print(msg)
    return msg


@flow
def dask_pipeline():
    # df = read_data.submit("1988", "2022")
    # df_yearly_average = process_data.submit(df)
    # return df_yearly_average
    print_hello.map(["Alice", "Bob", "Charlie", "David", "Eve"])


if __name__ == "__main__":
    cluster = dask.distributed.LocalCluster(n_workers=2, processes=True, threads_per_worker=1)
    client = dask.distributed.Client(cluster)
    print(client.scheduler.address)
    dask_pipeline.with_options(
        task_runner=DaskTaskRunner(address=client.scheduler.address), log_prints=True, retries=0
    ).serve(name="Dask Pipeline")
    # dask_pipeline()