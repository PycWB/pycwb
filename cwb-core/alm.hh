/*
# Copyright (C) 2019 Gabriele Vedovato
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/


/*! \file alm.hh
 *  Class for storing spherical harmonic coefficients.
 *
 *  alm wat interface to healpix alm.h (aligned with healpix 3.31)
 */

#ifndef ALM_HH
#define ALM_HH

#include "alm.h"
#include "xcomplex.h"		// starting from healpix 3.30 the complex class is the std::complex !!!
#include "alm_powspec_tools.h"
#include <complex>
#include "TMath.h"

using namespace std;

namespace wat {


/*! Base class for calculating the storage layout of spherical harmonic
    coefficients. */                                                   
class Alm_Base                                                         
  {
  protected:
    int lmax, mmax, tval;

  public:
    /*! Constructs an Alm_Base object with given \a lmax and \a mmax. */
    Alm_Base (int lmax_=0, int mmax_=0)                                 
      : lmax(lmax_), mmax(mmax_), tval(2*lmax+1) {}                     

    /*! Returns the total number of coefficients for maximum quantum numbers
        \a l and \a m. */                                                   
    int Num_Alms (int l, int m)
      {
      planck_assert(m<=l,"mmax must not be larger than lmax");
      return ((m+1)*(m+2))/2 + (m+1)*(l-m);
      }

    /*! Changes the object's maximum quantum numbers to \a lmax and \a mmax. */
    void Set (int lmax_, int mmax_)                                            
      {
      lmax=lmax_;
      mmax=mmax_;
      tval=2*lmax+1;
      }

    /*! Returns the maximum \a l */
    int Lmax() const { return lmax; }
    /*! Returns the maximum \a m */  
    int Mmax() const { return mmax; }

    /*! Returns an array index for a given m, from which the index of a_lm
        can be obtained by adding l. */                                   
    int index_l0 (int m) const                                            
      { return ((m*(tval-m))>>1); }                                       

    /*! Returns the array index of the specified coefficient. */
    int index (int l, int m) const                              
      { return index_l0(m) + l; }

    /*! Returns \a true, if both objects have the same \a lmax and \a mmax,
        else  \a false. */
    bool conformable (const Alm_Base &other) const
      { return ((lmax==other.lmax) && (mmax==other.mmax)); }

    /*! Swaps the contents of two Alm_Base objects. */
    void swap (Alm_Base &other);
  };


/*! Class for storing spherical harmonic coefficients. */
template<typename T> class Alm_Template: public Alm_Base
  {
  private:
    arr<T> alm;

  public:
    /*! Constructs an Alm_Template object with given \a lmax and \a mmax. */
    Alm_Template (int lmax_=0, int mmax_=0)
      : Alm_Base(lmax_,mmax_), alm (Num_Alms(lmax,mmax)) {}

    /*! Deletes the old coefficients and allocates storage according to
        \a lmax and \a mmax. */
    void Set (int lmax_, int mmax_)
      {
      Alm_Base::Set(lmax_, mmax_);
      alm.alloc(Num_Alms(lmax,mmax));
      }

    /*! Deallocates the old coefficients and uses the content of \a data
        for storage. \a data is deallocated during the call. */
    void Set (arr<T> &data, int lmax_, int mmax_)
      {
      planck_assert (Num_Alms(lmax_,mmax_)==data.size(),"wrong array size");
      Alm_Base::Set(lmax_, mmax_);
      alm.transfer(data);
      }

    /*! Sets all coefficients to zero. */
    void SetToZero ()
      { alm.fill (0); }

    /*! Multiplies all coefficients by \a factor. */
    template<typename T2> void Scale (const T2 &factor)
      { for (tsize m=0; m<alm.size(); ++m) alm[m]*=factor; }
    /*! \a a(l,m) *= \a factor[l] for all \a l,m. */
    template<typename T2> void ScaleL (const arr<T2> &factor)
      {
      planck_assert(factor.size()>tsize(lmax),
        "alm.ScaleL: factor array too short");
      for (int m=0; m<=mmax; ++m)
        for (int l=m; l<=lmax; ++l)
          operator()(l,m)*=factor[l];
      }
      /*! \a a(l,m) *= \a factor[m] for all \a l,m. */
    template<typename T2> void ScaleM (const arr<T2> &factor)
      {
      planck_assert(factor.size()>tsize(mmax),
        "alm.ScaleM: factor array too short");
      for (int m=0; m<=mmax; ++m)
        for (int l=m; l<=lmax; ++l)
          operator()(l,m)*=factor[m];
      }
    /*! Adds \a num to a_00. */
    template<typename T2> void Add (const T2 &num)
      { alm[0]+=num; }

    /*! Returns a reference to the specified coefficient. */
    T &operator() (int l, int m)
      { return alm[index(l,m)]; }
    /*! Returns a constant reference to the specified coefficient. */
    const T &operator() (int l, int m) const
      { return alm[index(l,m)]; }

    /*! Returns a pointer for a given m, from which the address of a_lm
        can be obtained by adding l. */
    T *mstart (int m)
      { return &alm[index_l0(m)]; }
    /*! Returns a pointer for a given m, from which the address of a_lm
        can be obtained by adding l. */
    const T *mstart (int m) const
      { return &alm[index_l0(m)]; }

    /*! Returns a constant reference to the a_lm data. */
    const arr<T> &Alms () const { return alm; }

    /*! Swaps the contents of two Alm_Template objects. */
    void swap (Alm_Template &other)
      {
      Alm_Base::swap(other);
      alm.swap(other.alm);
      }

    /*! Adds all coefficients from \a other to the own coefficients. */
    void Add (const Alm_Template &other)
      {
      planck_assert (conformable(other), "A_lm are not conformable");
      for (tsize m=0; m<alm.size(); ++m)
        alm[m] += other.alm[m];
      }
  };

  class Alm: public Alm_Template<complex<double> >
    {
    public:
      Alm (int lmax_=0, int mmax_=0)
        : Alm_Template<complex<double> >(lmax_,mmax_) {}

      //: Copy constructors
      Alm(const Alm& alm) {*this = alm;}                      

      Alm(const ::Alm<xcomplex<double> >& alm) {*this = alm;}                      

      // applies gaussian smoothing to alm
      void smoothWithGauss(double fwhm) {
        double deg2rad = PI/180.;
        fwhm*=deg2rad; 

        ::Alm<xcomplex<double> > alm(Lmax(),Mmax());
        for(int l=0;l<=Lmax();l++) {
          int limit = TMath::Min(l,Mmax());
          for (int m=0; m<=limit; m++)
            alm(l,m)=complex<double>((*this)(l,m).real(),(*this)(l,m).imag());
        }
        //::Alm<xcomplex<double> > alm = *this;
        ::smoothWithGauss(alm, fwhm);
        *this = alm;
      }


      // Rotates alm through the Euler angles psi, theta and phi.
      // The Euler angle convention  is right handed, rotations are active.
      // - psi is the first rotation about the z-axis (vertical)
      // - then theta about the ORIGINAL (unrotated) y-axis
      // - then phi  about the ORIGINAL (unrotated) z-axis (vertical)
      //   relates Alm */
      void rotate(double psi, double theta, double phi) {
        ::Alm<xcomplex<double> > alm(Lmax(),Mmax());
        for(int l=0;l<=Lmax();l++) {
          int limit = TMath::Min(l,Mmax());
          for (int m=0; m<=limit; m++)
            alm(l,m)=complex<double>((*this)(l,m).real(),(*this)(l,m).imag());
        }
        rotate_alm(alm, psi, theta, phi);
        *this = alm;
      }
    void operator=(const ::Alm<xcomplex<double> >& alm) {
      Set(alm.Lmax(),alm.Mmax());
      for(int l=0;l<=Lmax();l++) {
        int limit = TMath::Min(l,Mmax());
        for (int m=0; m<=limit; m++)
          (*this)(l,m) = complex<double>( alm(l,m).real(), alm(l,m).imag());
      }
    }
/*
    ::Alm<xcomplex<double> >& operator=(const Alm& a) {
      ::Alm<xcomplex<double> > alm(Lmax(),Mmax());
      for(int l=0;l<=Lmax();l++) {
        int limit = TMath::Min(l,Mmax());
        for (int m=0; m<=limit; m++)
          alm(l,m)=complex<double>((*this)(l,m).real(),(*this)(l,m).imag());
      }
      return alm;
    }
*/

  };

} // end namespace

#endif
