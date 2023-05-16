import logging, os, pycwb

logger = logging.getLogger(__name__)


def create_web_viewer(outputDir):
    logger.info("Copying web_viewer files to output folder")
    import shutil
    web_viewer_path = os.path.dirname(os.path.abspath(pycwb.__file__)) + '/vendor/web_viewer'
    for file in os.listdir(web_viewer_path):
        shutil.copy(f'{web_viewer_path}/{file}', outputDir)