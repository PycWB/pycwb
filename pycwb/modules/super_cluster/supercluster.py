import ROOT
import logging
import numpy as np
from pycwb.config import Config
from pycwb.constants import MIN_SKYRES_HEALPIX
from pycwb.modules.coherence.network import update_sky_map, update_sky_mask
from pycwb.modules.netcluster import append_cluster
logger = logging.getLogger(__name__)


def supercluster(config: Config, net: ROOT.network,
                 wdm_list: list[ROOT.WDM(np.double)],
                 cluster_list: list[ROOT.netcluster],
                 sparse_table_list: list):

    # decrease skymap resolution to improve subNetCut performances
    skyres = MIN_SKYRES_HEALPIX if config.healpix > MIN_SKYRES_HEALPIX else 0
    if skyres > 0:
        update_sky_map(config, net, skyres)
        update_sky_mask(config, net, skyres)

    hot = []
    for n in range(config.nIFO):
        hot.append(net.getifo(n).getHoT())
    # set low-rate TD filters
    for wdm in net.wdmList:
        wdm.setTDFilter(config.TDSize, 1)
    # read sparse map to detector
    for n in range(config.nIFO):
        det = net.getifo(n)
        det.sclear()
        for sparse_table in sparse_table_list:
            det.vSS.push_back(sparse_table[n])

    cluster = cluster_list[0]
    cluster.clear()

    for c in cluster_list:
        append_cluster(cluster, c, -2)

    nevt = 0
    nnn = 0
    mmm = 0
    for j in range(int(net.nLag)):
        # cycle = cfg.simulation ? ifactor : Long_t(NET.wc_List[j].shift);
        cycle = int(net.wc_List[j].shift)
        cycle_name = f"lag={cycle}"

        logger.info("-> Processing %s ...", cycle_name)
        logger.info("   --------------------------------------------------")
        logger.info("    coher clusters|pixels      : %6d|%d", cluster.esize(0), cluster.psize(0))

        if config.l_high == config.l_low:
            net.pair = False
        if net.pattern != 0:
            net.pair = False

        cluster.supercluster('L',net.e2or,config.TFgap,False)
        logger.info("    super clusters|pixels      : %6d|%d", cluster.esize(0), cluster.psize(0))

        # defragmentation for pattern != 0
        if net.pattern != 0:
            cluster.defragment(config.Tgap, config.Fgap)
            logger.info("   defrag clusters|pixels      : %6d|%d", cluster.esize(0), cluster.psize(0))

        # copy selected clusters to network
        pwc = net.getwc(j)
        pwc.cpf(cluster, False)

        # apply subNetCut() only for pattern=0 || cfg.subnet>0 || cfg.subcut>0 || cfg.subnorm>0 || cfg.subrho>=0
        if net.pattern == 0 or config.subnet > 0 or config.subcut > 0 or config.subnorm > 0 or config.subrho >= 0:
            if config.subacor > 0:
                # set Acore for subNetCuts
                net.acor = config.subacor
            if config.subrho > 0:
                # set netRHO for subNetCuts
                net.netRHO = config.subrho
            net.setDelayIndex(hot[0].rate())
            pwc.setcore(False)
            psel = 0
            while True:
                count = pwc.loadTDampSSE(net, 'a', config.BATCH, config.LOUD)
                # FIXME: O4 code have addtional config.subnorm
                psel += net.subNetCut(j, config.subnet, config.subcut, config.subnorm, ROOT.nullptr)
                # ptot = cluster.psize(1) + cluster.psize(-1)
                # pfrac = ptot / psel if ptot > 0 else 0
                if count < 10000:
                    break
            logger.info("   subnet clusters|pixels      : %6d|%d", net.events(), pwc.psize(-1))
            if config.subacor > 0:
                # restore Acore
                net.acor = config.Acore
            if config.subrho > 0:
                # restore netRHO
                net.netRHO = config.netRHO

        if net.pattern == 0:
            pwc.defragment(config.Tgap, config.Fgap)
            logger.info("   defrag clusters|pixels      : %6d|%d", cluster.esize(0), cluster.psize(0))

        nevt += net.events()
        nnn += pwc.psize(-1)
        mmm += pwc.psize(1) + pwc.psize(-1)

    logger.info("Supercluster done")
    if mmm:
        logger.info("total  clusters|pixels|frac : %6d|%d|%f", nevt, nnn, nnn / mmm)
    else:
        logger.info("total  clusters             : %6d", nevt)

    return cluster, pwc