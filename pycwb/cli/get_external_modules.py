import logging


def init_parser(parser):
    # Add the arguments
    parser.add_argument('config_file',
                        metavar='file_path',
                        type=str,
                        help='the path to the config file')
    
    parser.add_argument('-u',
                        '--update',
                        action='store_true',
                        help='update the existing external modules')


def command(args):
    from pycwb.modules.external_module_manager.manager import load_config, check_module_existence, pull_external_module
    from pycwb.modules.logger import logger_init

    logger_init()

    logger = logging.getLogger(__name__)

    # Load the configuration file
    target_dir, modules = load_config(args.config_file)
    logger.info(f"Target directory: {target_dir}")

    for module in modules:
        module_name = module.get("name")
        module_path = module.get("module_path")
        repo_url = module.get("repo_url")
        version = module.get("version")

        # Check if the module exists
        if check_module_existence(module_name, target_dir):
            logger.info(f"Module {module_name} already exists in {target_dir}")
            if args.update:
                logger.info(f"Updating module {module_name}...")
                pull_external_module(module_name, module_path, repo_url, target_dir, version)
            else:
                logger.info(f"Skipping update for {module_name}")
        else:
            logger.info(f"Module {module_name} does not exist. Pulling from {repo_url}...")
            pull_external_module(module_name, module_path, repo_url, target_dir, version)

    