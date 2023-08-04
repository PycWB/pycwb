/*
# Copyright (C) 2019 Sergey Klimenko
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


#include <string>
#include "FrameL.h"
#include "wavearray.hh"

template<class X> wavearray<X>* rdfrm(double tlen, char *cname,  char *fname, double tskip);
wavearray<float>* rdfrmF(double ten, char *cnme,  char *name, double skip);
wavearray<double>* rdfrmD(double len, char *came,  char *fame, double tkip);
using namespace std;
