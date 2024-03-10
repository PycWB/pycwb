from prefect import task
from .builtin import generate_noise_for_job_seg_task, save_trigger, generate_injection_task, \
    read_file_from_job_segment, merge_frame_task, data_conditioning_task, \
    coherence_task, supercluster_and_likelihood_task, load_xtalk_catalog, reconstruct_waveform, plot_triggers


@task
def read_and_condition_data_sequence(config, job_seg):
    if job_seg.frames:
        frame_data = read_file_from_job_segment.fn(config, job_seg, job_seg.frames)
        data = merge_frame_task.fn(job_seg, frame_data, config.segEdge)
    else:
        data = None

    if job_seg.noise:
        data = generate_noise_for_job_seg_task.fn(job_seg, config, data=data)
    if job_seg.injections:
        data = generate_injection_task.fn(config, job_seg, data)

    conditioned_data = []
    for ifo in range(len(job_seg.ifos)):
        conditioned_data.append(data_conditioning_task.fn(config, data, ifo))

    return conditioned_data


@task
def process_data_sequence(config, data):
    xtalk_catalog = load_xtalk_catalog.submit(config.MRAcatalog)

    fragment_clusters_multi_res = []
    for res in range(config.nRES):
        fragment_clusters_multi_res.append(coherence_task.fn(config, data, res))

    triggers_data = supercluster_and_likelihood_task.fn(config, fragment_clusters_multi_res,
                                                        data, xtalk_catalog)

    return triggers_data


@task
def post_production_sequence(working_dir, config, job_seg, triggers_data, plot):
    trigger_folders = []
    for trigger_data in triggers_data:
        trigger_folder = save_trigger.fn(working_dir, config, job_seg, trigger_data)
        # TODO: allow disable reconstruct_waveform
        reconstruct_waveform.map(trigger_folder, config, job_seg, trigger_data, plot=plot)
        trigger_folders.append(trigger_folder)

    if plot:
        for trigger_folder, trigger_data in zip(trigger_folders, triggers_data):
            plot_triggers.map(trigger_folder, trigger_data)