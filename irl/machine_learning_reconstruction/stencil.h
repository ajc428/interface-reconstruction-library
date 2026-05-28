// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2026 Andrew Cahaly <andrew.cahaly@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef IRL_MACHINE_LEARNING_RECONSTRUCTION_STENCIL_H_
#define IRL_MACHINE_LEARNING_RECONSTRUCTION_STENCIL_H_

#include <cstring>
#include <iostream>
#include <cassert>
#include <vector>
#include "irl/geometry/general/pt.h"
#include "irl/geometry/polyhedrons/rectangular_cuboid.h"

namespace IRL 
{
   class stencil
   {
   private:
      std::vector<double> x;
      std::vector<double> y;
      std::vector<double> z;
      int NX;
      int NY;
      int NZ;
      double dx;
      double dy;
      double dz;
      int ic;
      int jc;
      int kc;

   public:
      stencil()
      {
         NX = 3;
         NY = 3;
         NZ = 3;
         dx = 1;
         dy = 1;
         dz = 1;
         ic = 1;
         jc = 1;
         kc = 1;
         x.resize(NX);
         y.resize(NY);
         z.resize(NZ);
         for (int i = 0; i < NX; ++i)
         {
            x[i] = 0;
         }
         for (int j = 0; j < NY; ++j)
         {
            y[j] = 0;
         }
         for (int k = 0; k < NZ; ++k)
         {
            z[k] = 0;
         }
      };

      stencil(int nx, int ny, int nz, int sx, int sy, int sz, double lx, double ly, double lz)
      {
         NX = nx;
         NY = ny;
         NZ = nz;
         dx = sx;
         dy = sy;
         dz = sz;
         ic = NX/2;
         jc = NY/2;
         kc = NZ/2;
         x.resize(NX);
         y.resize(NY);
         z.resize(NZ);
         for (int i = 0; i < NX; ++i)
         {
            x[i] = lx+i*dx;
         }
         for (int j = 0; j < NY; ++j)
         {
            y[j] = ly+j*dy;
         }
         for (int k = 0; k < NZ; ++k)
         {
            z[k] = lz+k*dz;
         }
      };

      void setNX(int x) {NX=x; ic = NX/2;};
      void setNY(int x) {NY=x; jc = NY/2;};
      void setNZ(int x) {NZ=x; kc = NZ/2;};
      void setDx(double x) {dx=x;};
      void setDy(double x) {dy=x;};
      void setDz(double x) {dz=x;};
      int getNX() {return NX;};
      int getNY() {return NY;};
      int getNZ() {return NZ;};
      int get_ic() {return ic;};
      int get_jc() {return jc;};
      int get_kc() {return kc;};
      double getDx() {return dx;};
      double getDy() {return dy;};
      double getDz() {return dz;};

      double get_xm(int i) {return x[i]+dx/2.0;};
      double get_ym(int j) {return y[j]+dy/2.0;};
      double get_zm(int k) {return z[k]+dz/2.0;};

      IRL::RectangularCuboid getCell(int i, int j, int k)
      {
         assert(i>=0);
         assert(i<NX);
         assert(j>=0);
         assert(j<NY);
         assert(k>=0);
         assert(k<NZ);
         return IRL::RectangularCuboid::fromBoundingPts(IRL::Pt(x[i], y[j], z[k]),IRL::Pt(x[i]+dx, y[j]+dy, z[k]+dz));
      };
   };
}

#endif