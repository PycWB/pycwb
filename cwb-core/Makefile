
#------------------------------------------------------
# Package:     Wavelet Analysis Tool
# File name:   Makefile
#------------------------------------------------------
# This Makefile requires GNU Make.
#------------------------------------------------------
#eval `/home/klimenko/ligotools/bin/use_ligotools`

ifdef _USE_ICC
ASM	    = icc
CPP	    = icc
LD	    = icc
else
ASM	    = gcc
CPP	    = gcc
LD	    = gcc
endif
AR	    = ar r

ifndef XIFO
XIFO        = 4
endif

UNAME       = $(shell uname -s)
BTIME	    = $(shell date -u) 
WTIME	    = $(shell date -u +"%Y-%m-%d %I-%M"-%S) 
KERNEL_NAME = $(shell uname -s)
NODE_NAME   = $(shell uname -n)
KERNEL_REL  = $(shell uname -r)
KERNEL_VER  = $(shell uname -v)
MACH_NAME   = $(shell uname -m)
PROC_NAME   = $(shell uname -p)

# Set directory names
#----------------------------
# ROOT system
ROOT        = $(ROOTSYS)
ROOTINC     = $(ROOTSYS)/include
# Data monitoring tool
#DMT         = $(GDS)
# Virgo Frame library
#FR          = $(HOME_FRLIB)
# where to place HTML documentation for access via WWW
HTML_DST    = $(HOME)/public_html/WAT

ifdef _USE_HEALPIX
  ifdef _USE_CONDA
    HEALPIX_INC = $(HOME_HEALPIX)/include/healpix_cxx
  else
    HEALPIX_INC = $(HOME_HEALPIX)/src/cxx/Healpix_cxx
    HEALPIX_SUP_INC = $(HOME_HEALPIX)/src/cxx/cxxsupport
  endif
endif

# Recommended option for Ultra SPARC CPU is 'ultrasparc'
#CPU_OPT = -mcpu=ultrasparc
# Recommended option for Alpha CPU is 'ev56' or 'ev6'
#CPU_OPT= -mcpu=ev6
# Recommended option for Intel is 'pentium' or 'pentiumpro'
#CPU_OPT += -mcpu=pentium

CPP_OPT = -Wno-deprecated -fPIC -mavx -fopenmp

CPPFLAGS    = -O2 -Wall $(CPU_OPT) $(CPP_OPT)
CPPFLAGS   += -m64
CPPFLAGS   += -mavx 
CPPFLAGS   += -march=native 
#CPPFLAGS  += -funroll-loops

CPPSHFLAGS  = -O2 -fPIC $(CPU_OPT) $(CPP_OPT)
CPPSHFLAGS +=  -fexceptions
ifdef _USE_ICC
CPPSHFLAGS +=  -diag-disable=2196
endif
ifdef _USE_CPP
CPPSHFLAGS += -std=c++${_USE_CPP}
endif

SRC_LIST  =	injection.cc wavecomplex.cc Wavelet.cc WaveDWT.cc Haar.cc Biorthogonal.cc \
	        Daubechies.cc Symlet.cc Meyer.cc SymmArray.cc SymmArraySSE.cc SymmObjArray.cc WDM.cc \
                wavearray.cc wseries.cc watplot.cc cluster.cc wavecor.cc wavefft.cc waverdc.cc lossy.cc \
                wavelinefilter.cc netpixel.cc netcluster.cc skymap.cc detector.cc network.cc \
                netevent.cc regression.cc time.cc monster.cc sseries.cc 

MISC_SRC =

ifdef FR
MISC_SRC += readfrfile.cc readframe.cc rdfr.cc
endif

ifdef DMT
MISC_SRC += LineFilter.cc 
endif

# extra header files
EXTRA_H    = 

WAT_MACROS = ReadDataSample.C Spectrum.C Plot.C AddSignals.C WSpectrum.C\
             Clean.C WLNtuple.C Histogram.C WTSpectrum.C

ODIR      = obj
SHODIR	  = shared
LIBODIR	  = lib

# ------------------------------------------------------
WAVE_SRC  = $(SRC_LIST)
WAVE_SRC  += $(MISC_SRC)

WAVE_H    = $(WAVE_SRC:.cc=.hh)
WAVE_H   += watfun.hh

WAVE_OBJ  = $(WAVE_SRC:%.cc=$(ODIR)/%.o)

WAVE_SHO  = $(WAVE_SRC:%.cc=$(SHODIR)/%.sho)
#WAVE_SHO += $(FR)/lib/libFrame.so
WAVE_SHO += $(SHODIR)/wave_dict.sho

WSH_H    += $(WAVE_SRC:.cc=.hh)
WSH_H    += watversion.hh
WSH_H    += watfun.hh
WSH_H    += constants.hh
ifdef _USE_HEALPIX
WSH_H    += alm.hh
endif
WSH_H    += wavelet_LinkDef.h

CPPSHFLAGS += -D_USE_ROOT -I$(ROOTINC) -DXIFO=${XIFO} -I.  
CPPFLAGS += -D_USE_ROOT -I$(ROOTINC) -DXIFO=${XIFO} -I.  
CINTOPT =  -D_USE_ROOT -I$(ROOTINC) -DXIFO=${XIFO} -I. 
LDFLAGS   =
LIBOPT    = -L. -lwavelet-${XIFO}x

ifdef _USE_HEALPIX
CPPSHFLAGS += -D_USE_HEALPIX -I$(HEALPIX_INC) -I$(HEALPIX_SUP_INC)
CPPFLAGS   += -D_USE_HEALPIX -I$(HEALPIX_INC) -I$(HEALPIX_SUP_INC)
CINTOPT    += -D_USE_HEALPIX -I$(HEALPIX_INC) -I$(HEALPIX_SUP_INC)
endif

ifdef _USE_ICC
CPPSHFLAGS += -D_USE_ICC
CPPFLAGS   += -D_USE_ICC
CINTOPT    += -D_USE_ICC
endif
ifdef _USE_CPP
CPPSHFLAGS += -D_USE_CPP
CPPFLAGS   += -D_USE_CPP
CINTOPT    += -D_USE_CPP
endif
ifdef _USE_ROOT6
CPPSHFLAGS += -D_USE_ROOT6
CPPFLAGS   += -D_USE_ROOT6
CINTOPT    += -D_USE_ROOT6
endif
ifdef _USE_LAL
CPPSHFLAGS += -D_USE_LAL
CPPFLAGS   += -D_USE_LAL
CINTOPT    += -D_USE_LAL
endif
ifdef _USE_EBBH
CPPSHFLAGS += -D_USE_EBBH
CPPFLAGS   += -D_USE_EBBH
CINTOPT    += -D_USE_EBBH
endif

ifeq ($(UNAME),Darwin)
WDIR            = ${PWD}
SHLDFLAGS+= -fno-common -dynamiclib -undefined dynamic_lookup
else
WDIR            = .
SHLDFLAGS+= -fno-common -shared
endif


#SHLDFLAGS+= --warn-once

ifdef DMT
CPPFLAGS   += -D_USE_DMT -I$(DMT)/include
CPPSHFLAGS += -D_USE_DMT -I$(DMT)/include
CINTOPT    += -D_USE_DMT -I$(DMT)/include
LIBOPT     += -L$(DMT)/lib -lzlib
endif

ifdef FR
CPPFLAGS  += -D_USE_FR -I$(FR)/include
CPPSHFLAGS  += -D_USE_FR -I$(FR)/include
CINTOPT  += -D_USE_FR -I$(FR)/include 
LIBOPT     += -L$(FR)/$(UNAME) -lFrame
#WAVE_SHO += $(FR)/lib/libFrame.so
endif
# ------------------------------------------------------

all : dir whash wtag wbranch wurl watversion shared

dir:
	if [ ! -d $(ODIR) ]; then mkdir -p $(ODIR) ; fi;
	if [ ! -d $(SHODIR) ]; then mkdir -p $(SHODIR) ; fi;
	if [ ! -d $(LIBODIR) ]; then mkdir -p $(LIBODIR) ; fi;
	#if [ ! -d archive ]; then mkdir archive ; fi;
	#if [ ! -d html ]; then mkdir html ; fi;
	#if [ ! -d macro ]; then mkdir macro ; fi;
	#if [ ! -d lib ]; then mkdir lib ; fi;	
	#if [ ! -d include ]; then mkdir macro ; fi;
clean:
	\rm -f lib/wavelet-*.so
	\rm -f lib/libwavelet-*.a
	if [ -d $(LIBODIR) ]; then cd $(LIBODIR) ; \rm -f *.so ; fi;
	if [ -d $(LIBODIR) ]; then rmdir $(LIBODIR) ; fi;
	if [ -d $(SHODIR) ]; then cd $(SHODIR) ; \rm -f *.sho ; fi;
	if [ -d $(SHODIR) ]; then rmdir $(SHODIR) ; fi;
	if [ -d $(ODIR) ]; then cd $(ODIR); \rm -f *.o ; fi;
	if [ -d $(ODIR) ]; then rmdir $(ODIR) ; fi;
	\rm -f watversion.hh
	\rm -f wathash
	\rm -f wattag
	\rm -f watbranch
	\rm -f waturl
	\rm -f wave_dict.cc wave_dict.h

cleanlib:
	\rm -f lib/libwavelet-*.a
	cd $(ODIR); \rm -f *.o

main :  $(ODIR)/main.o $(WAVE_OBJ)
	$(LD) $(LDFLAGS) $(ODIR)/main.o -o main $(LIBOPT)

$(ODIR)/main.o :  main.cc $(WAVE_H) lib/libwavelet.a
	$(CPP) -c $(CPPFLAGS) main.cc -o $(ODIR)/main.o

#$(ODIR)/wavefft.o : wavefft.cc 
#	$(CPP) -c $(CPPFLAGS) $< -o $@

#$(SHODIR)/wavefft.sho : wavefft.cc
#	$(CPP) -c $(CPPSHFLAGS) $< -o $@

$(ODIR)/%.o : %.cc $(WAVE_H)
	$(CPP) -c $(CPPFLAGS) $< -o $@

$(SHODIR)/%.sho : %.cc $(WAVE_H)
	$(CPP) -c $(CPPSHFLAGS) $< -o $@

t%: $(ODIR)/t%.o lib/libwavelet-$(XIFO)x.a
	$(LD) $(LDFLAGS) $< -o $@ $(LIBOPT)

lib: lib/libwavelet-${XIFO}x.a
	\cd lib;ln -sf libwavelet-${XIFO}x.a libwavelet.a

lib/libwavelet-${XIFO}x.a: $(WAVE_OBJ)
ifeq ($(UNAME),Darwin)
	$(ASM) -c watasm.S -o $(ODIR)/watasm.o
	$(AR) $@ $(ODIR)/watasm.o $(WAVE_OBJ)
else
	$(AR) $@ $(WDIR)/watasm_elf64.o $(WAVE_OBJ)
endif

shared: lib/wavelet-${XIFO}x.so
	\cd lib;ln -sf wavelet-${XIFO}x.so wavelet.so
	\cd lib;ln -sf wavelet.so libwavelet.so

lib/wavelet-${XIFO}x.so: $(WAVE_SHO)
	\rm -rf lib/wavelet-${XIFO}x.so
ifeq ($(UNAME),Darwin)
	$(ASM) -c watasm.S -o $(SHODIR)/watasm.o
	$(LD) $(SHLDFLAGS) -O2 -g -o $(WDIR)/lib/wavelet-${XIFO}x.so $(SHODIR)/watasm.o $(WAVE_SHO)
else
	$(LD) $(SHLDFLAGS) -O2 -g -o $(WDIR)/lib/wavelet-${XIFO}x.so $(WDIR)/watasm_elf64.o $(WAVE_SHO)
endif

wave_dict.cc: $(WSH_H)
	$(ROOTSYS)/bin/rootcint -f wave_dict.cc -c -p $(CINTOPT) $(WSH_H)  

# generate HTML documentation
html: html-doc install-html

html-doc: $(WAVE_SRC) $(WAVE_H)  
	root -b -n -q -l macro/html_doc.C

install:
	if [ ! -d ${prefix}/macro ]; then mkdir ${prefix}/macro ;else \rm -f ${prefix}/macro/* ; fi;
	if [ ! -d ${prefix}/lib ]; then mkdir ${prefix}/lib ;else \rm -f ${prefix}/lib/* ; fi;	
	if [ ! -d ${prefix}/include ]; then mkdir ${prefix}/include ; else \rm -f ${prefix}/include/* ;  fi;
	cp -r macro/*  ${prefix}/macro/ 
	cp -r include/* ${prefix}/include/
	cp -r lib/*  ${prefix}/lib/

# copy html documentation to place where it will be accessed via WWW
install-html:
	if [ -n $(HTML_DST) -a -d $(HTML_DST) ] ; \
        then cp -r html/* $(HTML_DST); fi

# create archives 
tar: archive/wat.tar archive/macro.tar

archive/wat.tar: $(WAVE_SRC) $(MISC_SRC) $(WAVE_H) Makefile
	if [ -f $@ ]; then mv $@ $@_old ; fi;
	tar cf $@ $(SRC_LIST) $(MISC_SRC) $(WSH_H) $(EXTRA_H) Makefile INSTALL

archive/macro.tar: macro/*.C
	if [ -f $@ ]; then mv $@ $@_old ; fi;
	tar cf $@ macro/*.C
	chmod a+r $@

# create PostScript Document for compact printout (2 pages on each sheet)

PS_HEADER='$$n|<< Wavelet Analysis Tool -- $$W>>|Page $$% of $$='

ps: 
	enscript --header=$(PS_HEADER) --line-numbers --pretty-print=cpp -o - \
	$(foreach f, $(basename $(SRC_LIST) $(MISC_SRC)),$f.hh $f.cc)\
	$(EXTRA_H) \
	wavelet_LinkDef.h Makefile INSTALL | \
        psnup -2 > archive/wavetool.ps

watversion: 
	echo 	'#ifndef WATVERSION_HH'	  			>  watversion.hh
	echo 	'#define WATVERSION_HH'	 			>> watversion.hh
	echo                             			>> watversion.hh
	echo    '/* WAT Version */'				>> watversion.hh
	echo    '#define WAT_VERSION "6.4.1.0"'			>> watversion.hh
	echo    '/* WAT Version Major Number  */'		>> watversion.hh
	echo    '#define WAT_VERSION_MAJOR 6'			>> watversion.hh
	echo    '/* WAT Version Minor Number  */'		>> watversion.hh
	echo    '#define WAT_VERSION_MINOR 4'			>> watversion.hh
	echo    '/* WAT Version Micro Number  */'		>> watversion.hh
	echo    '#define WAT_VERSION_MICRO 1'			>> watversion.hh
	echo    '/* WAT Version Devel Number  */'		>> watversion.hh
	echo    '#define WAT_VERSION_DEVEL 0'			>> watversion.hh
	echo                             			>> watversion.hh
	echo    'const char *const watVersion       = WAT_VERSION;'		>> watversion.hh
	echo    'const int         watVersionMajor  = WAT_VERSION_MAJOR;'	>> watversion.hh
	echo    'const int         watVersionMinor  = WAT_VERSION_MINOR;'	>> watversion.hh
	echo    'const int         watVersionMicro  = WAT_VERSION_MICRO;'	>> watversion.hh
	echo    'const int         watVersionDevel  = WAT_VERSION_DEVEL;'	>> watversion.hh
	echo                             			>> watversion.hh
	echo    "inline char* watversion(char c='s')"		>> watversion.hh
	echo    '{'				      		>> watversion.hh
	echo    "  if(c=='s') "					>> watversion.hh
	echo    '    return (char*)watVersion;'			>> watversion.hh
	echo    "  if(c=='r') "					>> watversion.hh
	echo 	'    return (char*)"'`git -C ${HOME_WAT} rev-parse HEAD`'";' >> watversion.hh
	echo    "  if(c=='R') "					>> watversion.hh
	echo 	'    return (char*)"'`git -C ${HOME_WAT} rev-parse --short HEAD`'";' >> watversion.hh
	echo    "  if(c=='b') "					>> watversion.hh
	echo    '    return (char*)"'`git -C $(HOME_WAT) branch | grep \* | cut -d ' ' -f2` '";' >> watversion.hh
	echo    "  if(c=='x') "					>> watversion.hh
	echo    '    return (char*)"'`git -C $(HOME_WAT) show | head | grep Date`'";' >> watversion.hh
	echo    "  if(c=='u') "					>> watversion.hh
	echo    '    return (char*)"'`git -C ${HOME_WAT} config --get remote.origin.url`'";' >> watversion.hh
	echo    "  if(c=='g') "					>> watversion.hh
	echo    '    return (char*)"'`git -C ${HOME_WAT} tag -l --points-at HEAD`'";' >> watversion.hh
	echo    "  if(c=='k') "					>> watversion.hh
	echo    '    return (char*)"$(KERNEL_NAME)";'		>> watversion.hh
	echo    "  if(c=='n') "					>> watversion.hh
	echo    '    return (char*)"$(NODE_NAME)";'		>> watversion.hh
	echo    "  if(c=='q') "					>> watversion.hh
	echo    '    return (char*)"$(KERNEL_REL)";'		>> watversion.hh
	echo    "  if(c=='v') "					>> watversion.hh
	echo    '    return (char*)"$(KERNEL_VER)";'		>> watversion.hh
	echo    "  if(c=='m') "					>> watversion.hh
	echo    '    return (char*)"$(MACH_NAME)";'		>> watversion.hh
	echo    "  if(c=='p') "					>> watversion.hh
	echo    '    return (char*)"$(PROC_NAME)";'		>> watversion.hh
	echo    "  if(c=='t') "					>> watversion.hh
	echo    '    return (char*)"$(BTIME)";'			>> watversion.hh
	echo    "  if(c=='T') "					>> watversion.hh
	echo    '    return (char*)"$(WTIME)";'			>> watversion.hh
	echo    "  if(c=='i') "					>> watversion.hh
	echo    '    return (char*)"$(XIFO)";'			>> watversion.hh
	echo    '  else'					>> watversion.hh
	echo    '    return (char*)watVersion;'			>> watversion.hh
	echo    '}'      					>> watversion.hh
	echo                             			>> watversion.hh
	echo 	'#endif'	 				>> watversion.hh

whash:
	echo 	`git -C ${HOME_WAT} rev-parse HEAD`                 >  wathash

wtag:
	echo    `git -C ${HOME_WAT} tag -l --points-at HEAD`        >  wattag

wbranch:
	echo    `git -C ${HOME_WAT} rev-parse --abbrev-ref HEAD`    >  watbranch

wurl:
	echo    `git -C ${HOME_WAT} config --get remote.origin.url` >  waturl

