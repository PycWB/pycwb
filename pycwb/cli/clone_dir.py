"""
CLI command for cloning an existing directory to a new directory.

This commands clones the config and input subdirectories as a default,
allowing users to reproduce the same search or minimally modify the config files
for a new search. 
"""

import logging

logger = logging.getLogger(__name__)

def init_parser(parser):
    parser.add_argument(
        'indir', 
        type=str, 
        help='Source directory to clone')
    parser.add_argument(
        'outdir',
        type=str, 
        help='Destination directory to clone to')
    parser.add_argument(
        '--catalog', 
        action='store_true', 
        help='Whether to clone the catalog files')
    parser.add_argument(
        '--force',
        '-f',
        action='store_true', 
        help='Whether to overwrite the destination directory if it already exists')

def command(args):
    import os
    import shutil
    from pathlib import Path
    
    logging.basicConfig(level=logging.INFO)

    src_dir = Path(args.indir).resolve()

    if args.outdir:
        dest_dir = Path(args.outdir).resolve()
    else:
        # if destination directory is not provided, use the basename of the source directory
        dest_dir = src_dir.name
        dest_dir = Path(dest_dir).resolve()
    
    # check if source directory exists
    if not os.path.exists(src_dir):
        logger.error(f"Source directory {src_dir} does not exist.")
        return
    
    logger.info(f"Cloning directory {src_dir} to {dest_dir}")
    
    # create destination directory
    if dest_dir.exists():
        if args.force:
            shutil.rmtree(dest_dir)
        else:
            logger.error(f"Destination directory {dest_dir} already exists.")
            return
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Clone the config and input directories by default
    subdirs = ['config', 'input']
    for subdir in subdirs:
        src_subdir = src_dir / subdir
        dest_subdir = dest_dir / subdir
        
        if src_subdir.exists():
            # os.system(f"cp -r {src_subdir} {dest_subdir}")
            shutil.copytree(src_subdir, dest_subdir, dirs_exist_ok=True)
            logger.info(f"Clone {src_subdir} -> {dest_subdir}")
        else:
            logger.warning(f"{src_subdir} does not exist, skipping.")
    
    # If --catalog is provided, create a symbolic link to the catalog files
    if args.catalog:
        src_catalog = src_dir / 'catalog'
        dest_catalog = dest_dir / 'catalog'
        dest_catalog.mkdir(parents=True, exist_ok=True)
        
        if src_catalog.exists():
            for catalog_file in src_catalog.iterdir():    
                src_catalog_file = src_catalog / catalog_file.name
                dest_catalog_file = dest_catalog / catalog_file.name
                
                os.symlink(src_catalog_file, dest_catalog_file)
                logger.info(f"Create symbolic link {src_catalog_file} -> {dest_catalog_file}")
        else:
            logger.warning(f"{src_catalog} does not exist, skipping.")