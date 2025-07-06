.. modification_cwb:

Notes on modifications of cWB core code
=======================================

This file contains notes on modifications of the cWB core code.  It is
not intended to be a complete list of all modifications, but rather a
list of the most important ones.


Modifications on CMakeLists.txt
-------------------------------

Added support for macOS

- support `dylib` extension for macOS
- added ASM flag and .S source file so that the code compiles on macOS
- removed unnecessary source file

Fixed Warning
-------------

Commented out the following line in `WDM.cc`

.. code-block:: c

    template<class DataType_t> double* WDM<DataType_t>::Cos[MAXBETA];
    template<class DataType_t> double* WDM<DataType_t>::Cos2[MAXBETA];
    template<class DataType_t> double* WDM<DataType_t>::SinCos[MAXBETA];
    template<class DataType_t> double  WDM<DataType_t>::CosSize[MAXBETA];
    template<class DataType_t> double  WDM<DataType_t>::Cos2Size[MAXBETA];
    template<class DataType_t> double  WDM<DataType_t>::SinCosSize[MAXBETA];
    template<class DataType_t>    int  WDM<DataType_t>::objCounter = 0;

and add inline keyword to the following lines in `WDM.hh`

.. code-block:: c

   static inline double *Cos[MAXBETA], *Cos2[MAXBETA], *SinCos[MAXBETA];
   static inline double CosSize[MAXBETA], Cos2Size[MAXBETA], SinCosSize[MAXBETA];
   static inline int objCounter;

to avoid some warnings.

Removed Some Printing
---------------------

Removed printing of the following lines in `WDM.cc`

.. code-block:: c

    printf("Filter length = %d,  norm = %.16f\n", N, 1.-residual);

Replace the following in `monster.cc`

.. code-block:: c

    // comment out to avoid printing
    printf("layers[%d] = %d\n", i, layers[i] = (int)tmp);
    // add
    layers[i] = (int)tmp;


to let the output managed by the python code.

Solved part of the memory leak
--------------------------------

In `network.cc`, the following line was added to the destructor of `Network` class:

.. code-block:: c

    // Delete detector pointers
    for (detector* d : ifoList) {
        delete d;
    }

    // Delete WDM pointers
    for (WDM<double>* wdm : wdmList) {
        if (wdm) {
        delete wdm;    // delete WDM object
        }
    }

    // Delete wc_List
    for (size_t i = 0; i < wc_List.size(); ++i) {
        wc_List[i].clear();
    }
    re
