from diagrams import Diagram
from diagrams.c4 import Person, Container, Database, System, SystemBoundary, Relationship

graph_attr = {
    "splines": "spline",
}

with Diagram("pycWB Search", direction="TB", graph_attr=graph_attr):
    config = Container(
        name="Configuration File", description="A YAML file."
    )

    with SystemBoundary("Job Initialization"):
        job_creation = Container(
            name="Job Creation",
            technology="create_job_segment_from_config",
            description="Creates jobs from the configuration file.",
        )

        create_catalog_file = Container(
            name="catalog.create_catalog",
            technology="Python",
            description="Creates a catalog file from the configuration file.",
        )

        create_web_viewer = Container(
            name="Create Web Viewer",
            technology="web_viewer.create_web_viewer",
            description="Creates a web viewer from the configuration file.",
        )

        job_processing_loop = Container(
            name="Job Processing Loop",
            technology="Loop for each job segment",
            description="Processes jobs from the configuration file.",
        )

    with SystemBoundary("Analyze Job Segment"):
        get_input = Container(
            name="Read frame files",
            technology="read_data.read_from_job_segment",
            description="Read frame files from the job segment if provided.",
        )

        inject = Container(
            name="Injection",
            technology="read_data.read_from_simulation",
            description="Injects a signal into the data. Generate noise if specified.",
        )

        data_conditioning = Container(
            name="Data Conditioning",
            technology="data_conditioning.data_conditioning",
            description="Apply regression and whitening to the input data.",
        )

        coherence = Container(
            name="Coherence",
            technology="coherence.coherence",
            description="Calculate coherence between the input data.",
        )

        create_network = Container(
            name="Create Network",
            technology="Network",
            description="Create a network",
        )

        super_cluster = Container(
            name="Super Cluster",
            technology="super_cluster.supercluster",
            description="Supercluster analysis",
        )

        likelihood = Container(
            name="Likelihood",
            technology="super_cluster.supercluster",
            description="Calculate likelihood",
        )

    with SystemBoundary("Post Production"):
        save_post_production = Container(
            name="Save Event and Cluster",
            technology="save_pkl_file",
            description="Save post production data.",
        )

        wave_reconstruction = Container(
            name="Wave Reconstruction",
            technology="reconstruction.get_network_MRA_wave",
            description="Reconstruct the waveforms.",
        )

        likelihood_map = Container(
            name="Likelihood Map",
            technology="plot.cluster_statistics.plot_statistics",
            description="Create a likelihood map.",
        )

        get_glitchness = Container(
            name="Get Glitchness",
            technology="autoencoder.get_glitchness",
            description="Get glitchness.",
        )


    webapp = System(name="Event Viewer", description="", external=True)

    # mainframe = System(
    #     name="pycWB search",
    #     description="",
    #     external=True,
    # )

    config >> Relationship("Create job from config") >> job_creation
    job_creation >> Relationship("Create Catalog File") >> create_catalog_file
    job_creation >> Relationship("Job Processing Loop") >> job_processing_loop
    job_creation >> Relationship("Create Web Viewer") >> create_web_viewer
    job_processing_loop >> Relationship("Trigger Job Analysis") >> get_input
    get_input >> Relationship("Inject") >> inject
    inject >> Relationship("Data Conditioning") >> data_conditioning
    data_conditioning >> Relationship("Coherence") >> coherence
    data_conditioning >> Relationship("Create Network") >> create_network >> super_cluster
    coherence >> Relationship("Super Cluster") >> super_cluster
    super_cluster >> Relationship("Likelihood") >> likelihood
    likelihood >> Relationship("Save Event and Cluster") >> save_post_production
    likelihood >> Relationship("Wave Reconstruction") >> wave_reconstruction
    likelihood >> Relationship("Likelihood Map") >> likelihood_map
    wave_reconstruction >> Relationship("Get Glitchness") >> get_glitchness
