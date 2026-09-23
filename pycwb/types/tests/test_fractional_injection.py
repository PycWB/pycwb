import numpy as np
from pycwb.types.time_series import TimeSeries

def test_fractional_shift_against_analytic_hf_wave():
 rate=16384.;dt=1/rate;n=16384;t=(np.arange(n)-n//2)*dt
 for f in [70,1304,3000,4090]:
  source=np.exp(-.5*(t/.015)**2)*np.cos(2*np.pi*f*t)
  for shift in [-.49,-.2,.2,.49]:
   dest=TimeSeries(np.zeros(n),dt=dt,t0=0.)
   result=dest.inject(TimeSeries(source,dt=dt,t0=shift*dt))
   expected=np.exp(-.5*((t-shift*dt)/.015)**2)*np.cos(2*np.pi*f*(t-shift*dt))
   assert np.linalg.norm(result.data-expected)/np.linalg.norm(expected)<1e-12
   assert not dest.data.any()

def test_integer_alignment_and_edges():
 for offset in [-5,0,5,10,20]:
  source=np.arange(10,dtype=float);dest=TimeSeries(np.zeros(15),dt=.25,t0=10.)
  result=dest.inject(TimeSeries(source,dt=.25,t0=10.+offset*.25),copy=False)
  expected=np.zeros(15)
  for i,v in enumerate(source):
   if 0<=i+offset<15:expected[i+offset]+=v
  np.testing.assert_array_equal(result.data,expected)
  assert result is dest
