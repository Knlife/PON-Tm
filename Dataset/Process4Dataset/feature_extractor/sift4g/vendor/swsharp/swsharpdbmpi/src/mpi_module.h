/*
swsharp - CUDA parallelized Smith Waterman with applying Hirschberg's and 
Ukkonen's algorithm and dynamic cell pruning.
Copyright (C) 2013 Matija Korpar, contributor Mile Šikić

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Contact the author by mkorpar@gmail.com.
*/

#ifndef __MPI_MODULEH__
#define __MPI_MODULEH__

#include "swsharp/swsharp.h"

#ifdef __cplusplus 
extern "C" {
#endif

extern void sendMpiData(DbAlignment*** dbAlignments, int* dbAlignmentsLens, 
    int dbAlignmentsLen, int node);

extern void recieveMpiData(DbAlignment**** dbAlignments, int** dbAlignmentsLens, 
    int* dbAlignmentsLen, Chain** queries, Chain** database, Scorer* scorer, 
    int node);
    
#ifdef __cplusplus 
}
#endif
#endif // __MPI_MODULEH__
