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


// --------------------------------------------------------------------------
//                          constants.hh  -  description
//                             -------------------
//    begin                : Sat May 23 2012
//    copyright            : (C) 2012 by Gabriele Vedovato
//    email                : vedovato@lnl.infn.it
//
// Note : constants are imported from LALConstants.h
// Provides standard numerical constants for WAT
// --------------------------------------------------------------------------

#ifndef WATCONSTANTS_H
#define WATCONSTANTS_H

#include "TString.h"

// --------------------------------------------------------------------------
// Exact physical constants
// --------------------------------------------------------------------------

#define WAT_C_SI      299792458                                   // Speed of light in vacuo, m s^-1 
#define WAT_EPSILON0_SI  8.8541878176203898505365630317107503e-12 // Permittivity of free space, C^2 N^-1 m^-2 
#define WAT_MU0_SI    1.2566370614359172953850573533118012e-6     // Permeability of free space, N A^-2 
#define WAT_GEARTH_SI 9.80665                                     // Standard gravity, m s^-2 
#define WAT_PATM_SI   101325                                      // Standard atmosphere, Pa 
#define WAT_YRJUL_SI  31557600                                    // Julian year, s 
#define WAT_LYR_SI    9.4607304725808e15                          // (Julian) Lightyear, m 

// --------------------------------------------------------------------------
// Physical constants
// --------------------------------------------------------------------------

#define WAT_G_SI      6.67259e-11    	// Gravitational constant, N m^2 kg^-2 
#define WAT_H_SI      6.6260755e-34  	// Planck constant, J s 
#define WAT_HBAR_SI   1.05457266e-34 	// Reduced Planck constant, J s 
#define WAT_MPL_SI    2.17671e-8     	// Planck mass, kg 
#define WAT_LPL_SI    1.61605e-35    	// Planck length, m 
#define WAT_TPL_SI    5.39056e-44    	// Planck time, s 
#define WAT_K_SI      1.380658e-23   	// Boltzmann constant, J K^-1 
#define WAT_R_SI      8.314511       	// Ideal gas constant, J K^-1 
#define WAT_MOL       6.0221367e23   	// Avogadro constant, dimensionless 
#define WAT_BWIEN_SI  2.897756e-3    	// Wien displacement law constant, m K 
#define WAT_SIGMA_SI  5.67051e-8     	// Stefan-Boltzmann constant, W m^-2 K^-4 
#define WAT_AMU_SI    1.6605402e-27  	// Atomic mass unit, kg 
#define WAT_MP_SI     1.6726231e-27  	// Proton mass, kg 
#define WAT_ME_SI     9.1093897e-31  	// Electron mass, kg 
#define WAT_QE_SI     1.60217733e-19 	// Electron charge, C 
#define WAT_ALPHA     7.297354677e-3 	// Fine structure constant, dimensionless 
#define WAT_RE_SI     2.81794092e-15 	// Classical electron radius, m 
#define WAT_LAMBDAE_SI 3.86159323e-13 	// Electron Compton wavelength, m 
#define WAT_AB_SI     5.29177249e-11 	// Bohr radius, m 
#define WAT_MUB_SI    9.27401543e-24 	// Bohr magneton, J T^-1 
#define WAT_MUN_SI    5.05078658e-27 	// Nuclear magneton, J T^-1 

// --------------------------------------------------------------------------
// Astrophysical parameters
// --------------------------------------------------------------------------

#define WAT_REARTH_SI 6.378140e6      	// Earth equatorial radius, m 
#define WAT_AWGS84_SI 6.378137e6      	// Semimajor axis of WGS-84 Reference Ellipsoid, m 
#define WAT_BWGS84_SI 6.356752314e6   	// Semiminor axis of WGS-84 Reference Ellipsoid, m 
#define WAT_MEARTH_SI 5.97370e24      	// Earth mass, kg 
#define WAT_IEARTH    0.409092804     	// Earth inclination (2000), radians 
#define WAT_EEARTH    0.0167          	// Earth orbital eccentricity 
#define WAT_RSUN_SI   6.960e8         	// Solar equatorial radius, m 
#define WAT_MSUN_SI   1.98892e30      	// Solar mass, kg 
#define WAT_MRSUN_SI  1.47662504e3    	// Geometrized solar mass, m 
#define WAT_MTSUN_SI  4.92549095e-6   	// Geometrized solar mass, s 
#define WAT_LSUN_SI   3.846e26        	// Solar luminosity, W 
#define WAT_AU_SI     1.4959787066e11 	// Astronomical unit, m 
#define WAT_PC_SI     3.0856775807e16 	// Parsec, m 
#define WAT_YRTROP_SI 31556925.2      	// Tropical year (1994), s 
#define WAT_YRSID_SI  31558149.8      	// Sidereal year (1994), s 
#define WAT_DAYSID_SI 86164.09053     	// Mean sidereal day, s 

// --------------------------------------------------------------------------
// Cosmological parameters
// --------------------------------------------------------------------------

#define WAT_H0FAC_SI  3.2407792903e-18 	// Hubble constant prefactor, s^-1 
#define WAT_H0_SI     2e-18            	// Approximate Hubble constant, s^-1 
#define WAT_RHOCFAC_SI 1.68860e-9      	// Critical density prefactor, J m^-3 
#define WAT_RHOC_SI   7e-10            	// Approximate critical density, J m^-3 
#define WAT_TCBR_SI   2.726            	// Cosmic background radiation temperature, K 
#define WAT_VCBR_SI   3.695e5          	// Solar velocity with respect to CBR, m s^-1 
#define WAT_RHOCBR_SI 4.177e-14        	// Energy density of CBR, J m^-3 
#define WAT_NCBR_SI   4.109e8          	// Number density of CBR photons, m^-3 
#define WAT_SCBR_SI   3.993e-14        	// Entropy density of CBR, J K^-1 m^-3 

namespace watconstants {

  // --------------------------------------------------------------------------
  // Exact physical constants
  // --------------------------------------------------------------------------

  inline double SpeedOfLightInVacuo()                     {return WAT_C_SI;}          //    m s^-1
  inline double PermittivityOfFreeSpace()                 {return WAT_EPSILON0_SI;}   //    C^2 N^-1 m^-2
  inline double PermeabilityOfFreeSpace()                 {return WAT_MU0_SI;}        //    N A^-2
  inline double StandardGravity()                         {return WAT_GEARTH_SI;}     //    m s^-2
  inline double StandardAtmosphere()                      {return WAT_PATM_SI;}       //    Pa

  inline TString SpeedOfLightInVacuoUnit()                {return "m s^-1";}          //    m s^-1
  inline TString PermittivityOfFreeSpaceUnit()            {return "C^2 N^-1 m^-2";}   //    C^2 N^-1 m^-2
  inline TString PermeabilityOfFreeSpaceUnit()            {return "N A^-2";}          //    N A^-2
  inline TString StandardGravityUnit()                    {return "m s^-2";}          //    m s^-2
  inline TString StandardAtmosphereUnit()                 {return "Pa";}              //    Pa

  
  // --------------------------------------------------------------------------
  // Physical constants
  // --------------------------------------------------------------------------

  inline double GravitationalConstant()                   {return WAT_G_SI;}        //    N m^2 kg^-2
  inline double PlanckConstant()                          {return WAT_H_SI;}        //    J s
  inline double ReducedPlanckConstant()                   {return WAT_HBAR_SI;}     //    J s
  inline double PlanckMass()                              {return WAT_MPL_SI;}      //    kg
  inline double PlanckLength()                            {return WAT_LPL_SI;}      //    m
  inline double PlanckTime()                              {return WAT_TPL_SI;}      //    s
  inline double BoltzmannConstant()                       {return WAT_K_SI;}        //    J K^-1
  inline double IdealGasConstant()                        {return WAT_R_SI;}        //     J K^-1
  inline double AvogadroConstant()                        {return WAT_MOL;}         //
  inline double WienDisplacementLawConstant()             {return WAT_BWIEN_SI;}    //    m K
  inline double StefanBoltzmannConstant()                 {return WAT_SIGMA_SI;}    //    W m^-2 K^-4
  inline double AtomicMassUnit()                          {return WAT_AMU_SI;}      //    kg
  inline double ProtonMass()                              {return WAT_MP_SI;}       //    kg
  inline double ElectronMass()                            {return WAT_ME_SI;}       //    kg
  inline double ElectronCharge()                          {return WAT_QE_SI;}       //    C
  inline double FineStructureConstant()                   {return WAT_ALPHA;}       //
  inline double ClassicalElectronRadius()                 {return WAT_RE_SI;}       //    m
  inline double ElectronComptonWavelength()               {return WAT_LAMBDAE_SI;}  //    m
  inline double BohrRadius()                              {return WAT_AB_SI;}       //    m
  inline double BohrMagneton()                            {return WAT_MUB_SI;}      //    J T^-1
  inline double NuclearMagneton()                         {return WAT_MUN_SI;}      //    J T^-1

  inline TString GravitationalConstantUnit()              {return "N m^2 kg^-2";}   //    N m^2 kg^-2
  inline TString PlanckConstantUnit()                     {return "J s";}           //    J s
  inline TString ReducedPlanckConstantUnit()              {return "J s";}           //    J s
  inline TString PlanckMassUnit()                         {return "kg";}            //    kg
  inline TString PlanckLengthUnit()                       {return "m";}             //    m
  inline TString PlanckTimeUnit()                         {return "s";}             //    s
  inline TString BoltzmannConstantUnit()                  {return "J K^-1";}        //    J K^-1
  inline TString IdealGasConstantUnit()                   {return "J K^-1";}        //    J K^-1
  inline TString AvogadroConstantUnit()                   {return "";}              //
  inline TString WienDisplacementLawConstantUnit()        {return "m K";}           //    m K
  inline TString StefanBoltzmannConstantUnit()            {return "W m^-2 K^-4";}   //    W m^-2 K^-4
  inline TString AtomicMassUnitUnit()                     {return "kg";}            //    kg
  inline TString ProtonMassUnit()                         {return "kg";}            //    kg
  inline TString ElectronMassUnit()                       {return "kg";}            //    kg
  inline TString ElectronChargeUnit()                     {return "C";}             //    C
  inline TString FineStructureConstantUnit()              {return "";}              //
  inline TString ClassicalElectronRadiusUnit()            {return "m";}             //    m
  inline TString ElectronComptonWavelengthUnit()          {return "m";}             //    m
  inline TString BohrRadiusUnit()                         {return "m";}             //    m
  inline TString BohrMagnetonUnit()                       {return "J T^-1";}        //    J T^-1
  inline TString NuclearMagnetonUnit()                    {return "J T^-1";}        //    J T^-1

    
  // --------------------------------------------------------------------------
  // Astrophysical parameters
  // --------------------------------------------------------------------------

  inline double GalacticCenterLongitude()  {return (17.+45./60.+37.14/3600.)*360./24.;}  // R.A. 17h45m37.14s 
  inline double GalacticCenterLatitude()   {return -28.9361;}

  inline TString GalacticCenterLongitudeUnit()                 {return "radians";}    //    radians
  inline TString GalacticCenterLatitudeUnit()                  {return "radians";}    //    radians

  inline double EarthEquatorialRadius()                   {return WAT_REARTH_SI;}     //    m
  inline double SemimajorAxisOfWGS84ReferenceEllipsoid()  {return WAT_AWGS84_SI;}     //    m
  inline double SemiminorAxisOfWGS84ReferenceEllipsoid()  {return WAT_BWGS84_SI;}     //    m
  inline double EarthMass()                               {return WAT_MEARTH_SI;}     //    kg
  inline double EarthInclination2000()                    {return WAT_IEARTH;}        //    radians
  inline double EarthOrbitalEccentricity()                {return WAT_EEARTH;}
  inline double SolarEquatorialRadius()                   {return WAT_RSUN_SI;}       //    m

  inline TString EarthEquatorialRadiusUnit()                   {return "m";}          //    m
  inline TString SemimajorAxisOfWGS84ReferenceEllipsoidUnit()  {return "m";}          //    m
  inline TString SemiminorAxisOfWGS84ReferenceEllipsoidUnit()  {return "m";}          //    m
  inline TString EarthMassUnit()                               {return "kg";}         //    kg
  inline TString EarthInclination2000Unit()                    {return "radians";}    //    radians
  inline TString EarthOrbitalEccentricityUnit()                {return "";}
  inline TString SolarEquatorialRadiusUnit()                   {return "m";}          //    m

  inline double SolarMass()                               {return WAT_MSUN_SI;}       //    kg
  inline double GeometrizedSolarMass()                    {return WAT_MRSUN_SI;}      //    m
  inline double SolarLuminosity()                         {return WAT_LSUN_SI;}       //    W
  inline double AstronomicalUnit()                        {return WAT_AU_SI;}         //    m
  inline double Parsec()                                  {return WAT_PC_SI;}         //    m

  inline TString SolarMassUnit()                               {return "kg";}         //    kg
  inline TString GeometrizedSolarMassUnit()                    {return "m";}          //    m
  inline TString SolarLuminosityUnit()                         {return "W";}          //    W
  inline TString AstronomicalUnitUnit()                        {return "m";}          //    m
  inline TString ParsecUnit()                                  {return "m";}          //    m

  inline double TropicalYear1994()                        {return WAT_YRTROP_SI;}     //    s
  inline double SiderealYear1994()                        {return WAT_YRSID_SI;}      //    s
  inline double MeanSiderealDay()                         {return WAT_DAYSID_SI;}     //    s
  inline double Lightyear()                               {return WAT_LYR_SI;}        //    m

  inline TString TropicalYear1994Unit()                        {return "s";}          //    s
  inline TString SiderealYear1994Unit()                        {return "s";}          //    s
  inline TString MeanSiderealDayUnit()                         {return "s";}          //    s
  inline TString LightyearUnit()                               {return "m";}          //    m

  // --------------------------------------------------------------------------
  // Cosmological parameters
  // --------------------------------------------------------------------------

  inline double HubbleConstantPrefactor()               {return WAT_H0FAC_SI;}        //    s^-1
  inline double HubbleApproximateConstant()             {return WAT_H0_SI;}           //    s^-1
  inline double CriticalDensityPrefactor()              {return WAT_RHOCFAC_SI;}      //    J m^-3
  inline double CriticalApproximateDensity()            {return WAT_RHOC_SI;}         //    J m^-3
  inline double CosmicBackgroundRadiationTemperature()  {return WAT_TCBR_SI;}         //    K
  inline double SolarVelocityWithRespectToCBR()         {return WAT_VCBR_SI;}         //    m s^-1
  inline double EnergyDensityOfCBR()                    {return WAT_RHOCBR_SI;}       //    J m^-3
  inline double NumberDensityOfCBRPhotons()             {return WAT_NCBR_SI;}         //    m^-3
  inline double EntropyDensityOfCBR()                   {return WAT_SCBR_SI;}         //    J K^-1 m^-3

  inline TString HubbleConstantPrefactorUnit()               {return "s^-1";}         //    s^-1
  inline TString HubbleApproximateConstantUnit()             {return "s^-1";}         //    s^-1
  inline TString CriticalDensityPrefactorUnit()              {return "J m^-3";}       //    J m^-3
  inline TString CriticalApproximateDensityUnit()            {return "J m^-3";}       //    J m^-3
  inline TString CosmicBackgroundRadiationTemperatureUnit()  {return "K";}            //    K
  inline TString SolarVelocityWithRespectToCBRUnit()         {return "m s^-1";}       //    m s^-1
  inline TString EnergyDensityOfCBRUnit()                    {return "J m^-3";}       //    J m^-3
  inline TString NumberDensityOfCBRPhotonsUnit()             {return "m^-3";}         //    m^-3
  inline TString EntropyDensityOfCBRUnit()                   {return "J K^-1 m^-3";}  //    J K^-1 m^-3
  
}    

#endif
