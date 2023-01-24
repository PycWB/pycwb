import logging

logger = logging.getLogger(__name__)


# adapt from config.cc EXPORT
def update_global_var(gROOT, type, var, cmd):
    global_var = gROOT.GetGlobal(var, True)
    if not global_var:
        cmd = f"{type} {cmd}"

    # print(cmd)
    gROOT.ProcessLine(cmd + ';')


def cwb_load_macro(gROOT, config, file_name):
    gROOT.LoadMacro(config.cwb_macros + "/" + file_name)
    logger.info(f"Loaded macro from {file_name}")


def copy_char_array(ROOT, var, value):
    cmd = f"char {var}[{len(value)}] = '{value}'"
    ROOT.gInterpreter.ProcessLine(cmd)