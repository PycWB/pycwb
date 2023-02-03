import ROOT


def init_network(run_id, config: dict):
    net = ROOT.network()
    for ifo in config['ifo']:
        det = ROOT.detector(ifo)

        det.rate = config["inRate"] if not config['fResample'] else config['fResample']
        net.add(det)

    # set network skymaps
    net.setSkyMaps(int(config['healpix']))
    net.setAntenna()

    # restore network parameters
    net.constraint(config['delta'], config['gamma'])
    net.setDelay(config['refIFO'])
    net.Edge = config['segEdge']
    net.netCC = config['netCC']
    net.netRHO = config['netRHO']
    net.EFEC = config['EFEC']
    net.precision = config['precision']
    net.nSky = config['nSky']
    net.setRunID(run_id)
    net.setAcore(config['Acore'])
    net.optim = config['optim']
    net.pattern = config['pattern']

    # set sky mask
    # decleare cpp struct with healpix
    tmp_cfg = ROOT.CWB.config()
    tmp_cfg.healpix = config['healpix']
    tmp_cfg.Theta1 = config['Theta1']
    tmp_cfg.Theta2 = config['Theta2']
    tmp_cfg.Phi1 = config['Phi1']
    tmp_cfg.Phi2 = config['Phi2']

    if len(config['skyMaskFile']) > 0:
        ROOT.SetSkyMask(net, tmp_cfg, config['skyMaskFile'], 'e')

    if len(config['skyMaskCCFile']) > 0:
        ROOT.SetSkyMask(net, tmp_cfg, config['skyMaskCCFile'], 'c')

    return net
