# adapt from config.cc EXPORT
def update_global_var(gROOT, type, var, cmd):
    global_var = gROOT.GetGlobal(var, True)
    if not global_var:
        cmd = f"{type} {cmd}"

    print(cmd)
    gROOT.ProcessLine(cmd + ';')
