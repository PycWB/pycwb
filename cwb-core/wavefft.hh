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


/********************************************************/
/* Wavelet Analysis Tool                                */
/* file wavefft.h                                           */
/********************************************************/

#ifndef _WAVEFFT_H
#define _WAVEFFT_H
void wavefft(double a[], double b[], int ntot, int n, int nspan, int isn);
#endif
