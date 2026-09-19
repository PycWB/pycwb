from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from pycwb.types.detector import Detector,gmst_accurate,compute_sky_delay_and_patterns
from pycwb.constants.release_detector_geometry import VECTORS
from pycwb.utils.network import max_delay


@pytest.mark.parametrize('name',['H1','L1'])
def test_literal_vectors_and_release_antenna_oracle(name):
    data=np.load(Path(__file__).with_name('reference')/'release_detector_geometry.npz')
    detector=Detector(name,geometry_model='cwb_6.4.6.9')
    r,x,y=map(np.array,VECTORS[name])
    np.testing.assert_array_equal(detector.vertex_vec_earth_centered,r)
    np.testing.assert_array_equal(detector.x_vec,x)
    np.testing.assert_array_equal(detector.y_vec,y)
    gps=1387221740.;ra=np.radians(data['phi'])+gmst_accurate(gps);dec=np.pi/2-np.radians(data['theta'])
    fp,fx=detector.atenna_pattern(ra,dec,0.,gps)
    np.testing.assert_allclose(np.stack([fp,fx],axis=1),data[name],rtol=0,atol=2e-15)
    assert detector.geometry_model=='cwb_6.4.6.9'


def test_default_lal_and_unsupported_release_instruments():
    for name in ['H1','L1','V1']:
        np.testing.assert_array_equal(Detector(name).response,Detector(name,geometry_model='lal').response)
    with pytest.raises(ValueError,match='H1 and L1'):
        Detector('V1',geometry_model='cwb_6.4.6.9')
    with pytest.raises(ValueError,match='Unknown'):
        Detector('H1',geometry_model='unknown')


def test_max_delay_uses_selected_vertices():
    expected=np.linalg.norm(np.array(VECTORS['H1'][0])-VECTORS['L1'][0])/299792458.
    assert max_delay(['L1','H1'],geometry_model='cwb_6.4.6.9')==expected


def test_subnet_and_likelihood_use_selected_model():
    from pycwb.modules.super_cluster_native.super_cluster import setup_supercluster
    from pycwb.modules.likelihoodWP.pixel_data import build_sky_delay_and_antenna_patterns
    from pycwb.types.time_series import TimeSeries
    config=SimpleNamespace(ifo=['L1','H1'],refIFO='L1',rateANA=8192,upTDF=4,TDRate=32768,
                           TDSize=12,max_delay=.0101,healpix=2,MIN_SKYRES_HEALPIX=1,
                           detector_geometry='cwb_6.4.6.9')
    gps=1387221740.;context=setup_supercluster(config,gps)
    strains=[TimeSeries(data=np.zeros(128),dt=1/8192,t0=gps)]*2
    full=build_sky_delay_and_antenna_patterns(2,strains,config)
    for actual,key in zip(full,['ml_likelihood','FP_likelihood','FX_likelihood']):
        np.testing.assert_array_equal(actual,context[key])
    subnet=compute_sky_delay_and_patterns(config.ifo,config.refIFO,8192,max(12,int(.0101*8192)+1),gps,healpix_order=1,geometry_model=config.detector_geometry)
    np.testing.assert_array_equal(context['ml'],subnet[0]*4)
    np.testing.assert_array_equal(context['FP'],subnet[1])
    np.testing.assert_array_equal(context['FX'],subnet[2])


def test_injection_projection_uses_selected_geometry():
    from pycwb.modules.injection.strain import project_to_detector
    from pycwb.types.time_series import TimeSeries
    gps=1387221800.;hp=TimeSeries(data=np.sin(np.arange(64)),dt=1/8192,t0=-.1)
    hc=TimeSeries(data=np.cos(np.arange(64)),dt=1/8192,t0=-.1)
    actual=project_to_detector(hp,hc,.4,-.2,.3,['L1','H1'],gps,geometry_model='cwb_6.4.6.9')
    for name,strain in zip(['L1','H1'],actual):
        detector=Detector(name,geometry_model='cwb_6.4.6.9')
        expected=detector.project_wave(TimeSeries(data=hp.data,dt=hp.dt,t0=hp.t0+gps),
            TimeSeries(data=hc.data,dt=hc.dt,t0=hc.t0+gps),.4,-.2,.3,reference_time=gps)
        np.testing.assert_array_equal(strain.data,expected.data)
        assert strain.t0==expected.t0
