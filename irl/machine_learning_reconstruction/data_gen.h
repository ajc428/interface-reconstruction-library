// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2023 Andrew Cahaly <andrew.cahaly@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef IRL_MACHINE_LEARNING_RECONSTRUCTION_DATA_GEN_H_
#define IRL_MACHINE_LEARNING_RECONSTRUCTION_DATA_GEN_H_

#include "irl/machine_learning_reconstruction/moments_gen.h"
#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <random>

namespace IRL 
{
    class data_gen
    {
    private:
        int NX;     
        int NY;
        int NZ;
        int Ndata;  

        IRL::moments_gen *gen;

        inline int get_idx(int i, int j, int k, int n) const 
        {
            return 7 * (i * NY * NZ + j * NZ + k) + n;
        }

    public:
        data_gen(int num, int nx, int ny, int nz, int sx, int sy, int sz, double lx, double ly, double lz)
        {
            srand((unsigned) time(NULL));
            Ndata = num;
            NX = nx;
            NY = ny;
            NZ = nz;
            gen = new IRL::moments_gen(nx, ny, nz, sx, sy, sz, lx, ly, lz);
        };

        ~data_gen()
        {
            delete gen;
        };

        void generate(double rota_l, double rota_h, double rotb_l, double rotb_h, double rotc_l, double rotc_h, double coa_l, double coa_h, double cob_l, double cob_h, double ox_l, double ox_h, double oy_l, double oy_h, double oz_l, double oz_h, bool disturb)
        {
            bool flip;
            std::ofstream output;
            std::string data_name = "moments.txt";
            output.open(data_name, std::ios_base::app);
            std::ofstream normals;
            std::string normals_name = "normals.txt";
            normals.open(normals_name, std::ios_base::app);
            std::ofstream coefficients;
            std::string name = "coefficients.txt";
            coefficients.open(name, std::ios_base::app);

            std::random_device rd;  
            std::mt19937_64 a_eng(rd());
            std::uniform_int_distribution<int> dis(0, 7);
            std::uniform_real_distribution<double> noise(-0.1, 0.1);
            for (int n = 0; n < Ndata; ++n) 
            {
                std::cout << n << std::endl;
                IRL::Paraboloid paraboloid = gen->new_random_paraboloid(rota_l, rota_h, rotb_l, rotb_h, rotc_l, rotc_h, coa_l, coa_h, cob_l, cob_h, ox_l, ox_h, oy_l, oy_h, oz_l, oz_h);
                auto angles = gen->getAngles();

                coefficients << paraboloid.getDatum().x() << "," << paraboloid.getDatum().y() << "," << paraboloid.getDatum().z()
                << "," << angles[0] << "," << angles[1] << "," << angles[2]
                << "," << paraboloid.getAlignedParaboloid().a() << "," << paraboloid.getAlignedParaboloid().b() << "\n";
                
                auto moments = gen->get_moments(paraboloid, 1, true, flip);
                
                int direction = 0;
                int direction2 = 0;
                
                reflectMoments(moments, direction, direction2);

                int p = dis(a_eng);
                bool full = false;

                for (int i = 0; i < moments.size(); ++i)
                {
                    if (p == 0 || !disturb)
                    {
                        output << moments[i] << ",";
                    }
                    else
                    {
                        if (i % 7 == 0)
                        {
                            full = false;
                            output << moments[i] << ",";
                            if (moments[i] > IRL::global_constants::VF_HIGH || moments[i] < IRL::global_constants::VF_LOW)
                            {
                                full = true;
                            }
                        }
                        else
                        {
                            double c = noise(a_eng);
                            if (moments[i] + c > 0.5 && !full)
                            {
                                moments[i] = 0.5;
                                output << 0.5 << ",";
                            }
                            else if (moments[i] + c < -0.5 && !full)
                            {
                                moments[i] = -0.5;
                                output << -0.5 << ",";
                            }
                            else if (!full)
                            {
                                output << moments[i] + c << ",";
                                moments[i] = moments[i] + c;
                            }
                            else
                            {
                                output << moments[i] << ",";
                            }
                        }
                    }
                }
                output << "\n";

                auto cell = gen->getStencil()->getCell(gen->getStencil()->get_ic(),gen->getStencil()->get_jc(),gen->getStencil()->get_kc());
                auto surface_and_moments = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParaboloidParametrizedSurfaceOutput>>(cell, paraboloid);
                auto surface = surface_and_moments.getSurface();
                auto normal = surface.getAverageNormalNonAligned();

                double tmp;
                switch (direction)
                {
                    case 1:
                        normal[0] = -normal[0];
                    break;
                    case 2:
                        normal[1] = -normal[1];
                    break;
                    case 3:
                        normal[2] = -normal[2];
                    break;
                    case 4:
                        normal[0] = -normal[0];
                        normal[1] = -normal[1];
                    break;
                    case 5:
                        normal[0] = -normal[0];
                        normal[2] = -normal[2];
                    break;
                    case 6:
                        normal[1] = -normal[1];
                        normal[2] = -normal[2];
                    break;
                    case 7:
                        normal[0] = -normal[0];
                        normal[1] = -normal[1];
                        normal[2] = -normal[2];
                    break;
                }
                switch (direction2)
                {
                    case 1:
                        tmp=normal[0]; 
                        normal[0]=normal[1]; 
                        normal[1]=tmp;
                    break;
                    case 2:
                        tmp=normal[1]; 
                        normal[1]=normal[2]; 
                        normal[2]=tmp;
                    break;
                    case 3:
                        tmp=normal[0]; 
                        normal[0]=normal[2]; 
                        normal[2]=tmp;
                    break;
                    case 4:
                        tmp=normal[0]; 
                        normal[0]=normal[1]; 
                        normal[1]=tmp;
                        tmp=normal[1]; 
                        normal[1]=normal[2]; 
                        normal[2]=tmp;
                    break;
                    case 5:
                        tmp=normal[0]; 
                        normal[0]=normal[1]; 
                        normal[1]=tmp;
                        tmp=normal[0]; 
                        normal[0]=normal[2]; 
                        normal[2]=tmp;
                    break;
                }
                if (!flip)
                {
                    normal[0] = -normal[0];
                    normal[1] = -normal[1];
                    normal[2] = -normal[2];
                }

                normal.normalize();
                normals << normal[0] << "," << normal[1] << "," << normal[2] << "\n";                
            }  
            output.close();  
            normals.close(); 
            coefficients.close(); 
        }; 




        //************************************************************
        //**************ROTATE****************************************
        //************************************************************

        void reflectMomentsX(std::vector<double>& moments) 
        {
            for (int k = 0; k < NZ; ++k) 
            {
                for (int j = 0; j < NY; ++j) 
                {
                    for (int i = 0; i < NX / 2; ++i) 
                    {
                        int i_sym = NX - 1 - i;
                        for (int n = 0; n <= 6; ++n) 
                        {
                            double temp = moments[get_idx(i, j, k, n)];
                            if (n == 1 || n == 4) 
                            {
                                moments[get_idx(i, j, k, n)] = -moments[get_idx(i_sym, j, k, n)];
                                moments[get_idx(i_sym, j, k, n)] = -temp;
                            } 
                            else 
                            {
                                moments[get_idx(i, j, k, n)] = moments[get_idx(i_sym, j, k, n)];
                                moments[get_idx(i_sym, j, k, n)] = temp;
                            }
                        }
                    }
                    if (NX % 2 != 0) 
                    {
                        int i_mid = NX / 2;
                        moments[get_idx(i_mid, j, k, 1)] = -moments[get_idx(i_mid, j, k, 1)];
                        moments[get_idx(i_mid, j, k, 4)] = -moments[get_idx(i_mid, j, k, 4)];
                    }
                }
            }
        };

        void reflectMomentsY(std::vector<double>& moments) 
        {
            for (int k = 0; k < NZ; ++k) 
            {
                for (int i = 0; i < NX; ++i) 
                {
                    for (int j = 0; j < NY / 2; ++j) 
                    {
                        int j_sym = NY - 1 - j;
                        for (int n = 0; n <= 6; ++n) 
                        {
                            double temp = moments[get_idx(i, j, k, n)];
                            if (n == 2 || n == 5) 
                            {
                                moments[get_idx(i, j, k, n)] = -moments[get_idx(i, j_sym, k, n)];
                                moments[get_idx(i, j_sym, k, n)] = -temp;
                            } 
                            else 
                            {
                                moments[get_idx(i, j, k, n)] = moments[get_idx(i, j_sym, k, n)];
                                moments[get_idx(i, j_sym, k, n)] = temp;
                            }
                        }
                    }
                    if (NY % 2 != 0) 
                    {
                        int j_mid = NY / 2;
                        moments[get_idx(i, j_mid, k, 2)] = -moments[get_idx(i, j_mid, k, 2)];
                        moments[get_idx(i, j_mid, k, 5)] = -moments[get_idx(i, j_mid, k, 5)];
                    }
                }
            }
        };

        void reflectMomentsZ(std::vector<double>& moments) 
        {
            for (int j = 0; j < NY; ++j) 
            {
                for (int i = 0; i < NX; ++i) 
                {
                    for (int k = 0; k < NZ / 2; ++k) 
                    {
                        int k_sym = NZ - 1 - k;
                        for (int n = 0; n <= 6; ++n) 
                        {
                            double temp = moments[get_idx(i, j, k, n)];
                            if (n == 3 || n == 6) 
                            {
                                moments[get_idx(i, j, k, n)] = -moments[get_idx(i, j, k_sym, n)];
                                moments[get_idx(i, j, k_sym, n)] = -temp;
                            } 
                            else 
                            {
                                moments[get_idx(i, j, k, n)] = moments[get_idx(i, j, k_sym, n)];
                                moments[get_idx(i, j, k_sym, n)] = temp;
                            }
                        }
                    }
                    if (NZ % 2 != 0) 
                    {
                        int k_mid = NZ / 2;
                        moments[get_idx(i, j, k_mid, 3)] = -moments[get_idx(i, j, k_mid, 3)];
                        moments[get_idx(i, j, k_mid, 6)] = -moments[get_idx(i, j, k_mid, 6)];
                    }
                }
            }
        };

        void reflectMomentsXY(std::vector<double>& moments) 
        {
            for (int k = 0; k < NZ; ++k) 
            {
                for (int i = 0; i < NX; ++i) 
                {
                    for (int j = 0; j < i; ++j) 
                    {
                        for (int n = 0; n <= 6; ++n) 
                        {
                            if (n == 1 || n == 4) 
                            { 
                                double temp_x = moments[get_idx(i, j, k, n)];
                                double temp_y = moments[get_idx(i, j, k, n+1)];
                                
                                moments[get_idx(i, j, k, n)]   = moments[get_idx(j, i, k, n+1)]; 
                                moments[get_idx(i, j, k, n+1)] = moments[get_idx(j, i, k, n)];   
                                
                                moments[get_idx(j, i, k, n)]   = temp_y; 
                                moments[get_idx(j, i, k, n+1)] = temp_x; 
                            } 
                            else if (n == 0 || n == 3 || n == 6) 
                            { 
                                double temp = moments[get_idx(i, j, k, n)];
                                moments[get_idx(i, j, k, n)] = moments[get_idx(j, i, k, n)];
                                moments[get_idx(j, i, k, n)] = temp;
                            }
                        }
                    }
                    for (int n = 0; n <= 6; ++n) 
                    {
                        if (n == 1 || n == 4) 
                        {
                            double temp = moments[get_idx(i, i, k, n)];
                            moments[get_idx(i, i, k, n)] = moments[get_idx(i, i, k, n + 1)];
                            moments[get_idx(i, i, k, n + 1)] = temp;
                        }
                    }
                }
            }
        };

        void reflectMomentsYZ(std::vector<double>& moments) 
        {
            for (int i = 0; i < NX; ++i) 
            {
                for (int j = 0; j < NY; ++j) 
                {
                    for (int k = 0; k < j; ++k) 
                    {
                        for (int n = 0; n <= 6; ++n) 
                        {
                            if (n == 2 || n == 5) 
                            { 
                                double temp_y = moments[get_idx(i, j, k, n)];
                                double temp_z = moments[get_idx(i, j, k, n+1)];
                                
                                moments[get_idx(i, j, k, n)]   = moments[get_idx(i, k, j, n+1)]; 
                                moments[get_idx(i, j, k, n+1)] = moments[get_idx(i, k, j, n)];   
                                
                                moments[get_idx(i, k, j, n)]   = temp_z; 
                                moments[get_idx(i, k, j, n+1)] = temp_y; 
                            } 
                            else if (n == 0 || n == 1 || n == 4) 
                            { 
                                double temp = moments[get_idx(i, j, k, n)];
                                moments[get_idx(i, j, k, n)] = moments[get_idx(i, k, j, n)];
                                moments[get_idx(i, k, j, n)] = temp;
                            }
                        }
                    }
                    for (int n = 0; n <= 6; ++n) 
                    {
                        if (n == 2 || n == 5) 
                        {
                            double temp = moments[get_idx(i, j, j, n)];
                            moments[get_idx(i, j, j, n)] = moments[get_idx(i, j, j, n + 1)];
                            moments[get_idx(i, j, j, n + 1)] = temp;
                        }
                    }
                }
            }
        };

        void reflectMomentsXZ(std::vector<double>& moments) 
        {
            for (int j = 0; j < NY; ++j) 
            {
                for (int i = 0; i < NX; ++i) 
                {
                    for (int k = 0; k < i; ++k) 
                    {
                        for (int n = 0; n <= 6; ++n) 
                        {
                            if (n == 1 || n == 4) 
                            { 
                                double temp_x = moments[get_idx(i, j, k, n)];
                                double temp_z = moments[get_idx(i, j, k, n+2)];
                                
                                moments[get_idx(i, j, k, n)]   = moments[get_idx(k, j, i, n+2)]; 
                                moments[get_idx(i, j, k, n+2)] = moments[get_idx(k, j, i, n)];   
                                
                                moments[get_idx(k, j, i, n)]   = temp_z; 
                                moments[get_idx(k, j, i, n+2)] = temp_x; 
                            } 
                            else if (n == 0 || n == 2 || n == 5) 
                            { 
                                double temp = moments[get_idx(i, j, k, n)];
                                moments[get_idx(i, j, k, n)] = moments[get_idx(k, j, i, n)];
                                moments[get_idx(k, j, i, n)] = temp;
                            }
                        }
                    }
                    for (int n = 0; n <= 6; ++n) 
                    {
                        if (n == 1 || n == 4) 
                        {
                            double temp = moments[get_idx(i, j, i, n)];
                            moments[get_idx(i, j, i, n)] = moments[get_idx(i, j, i, n + 2)];
                            moments[get_idx(i, j, i, n + 2)] = temp;
                        }
                    }
                }
            }
        };

        void reflectMoments(std::vector<double>& moments, int& dir, int& dir2) 
        {
            IRL::Pt center = get_global_centroid(moments);
            double temp;

            if (std::abs(center[0]) <= IRL::global_constants::VF_LOW) center[0] = 0;
            if (std::abs(center[1]) <= IRL::global_constants::VF_LOW) center[1] = 0;
            if (std::abs(center[2]) <= IRL::global_constants::VF_LOW) center[2] = 0;

            if (center[0] < 0 && center[1] >= 0 && center[2] >= 0) 
            {
                dir = 1;
                reflectMomentsX(moments);
                center[0] = -center[0];
            } 
            else if (center[0] >= 0 && center[1] < 0 && center[2] >= 0) 
            {
                dir = 2;
                reflectMomentsY(moments);
                center[1] = -center[1];
            } 
            else if (center[0] >= 0 && center[1] >= 0 && center[2] < 0) 
            {
                dir = 3;
                reflectMomentsZ(moments);
                center[2] = -center[2];
            } 
            else if (center[0] < 0 && center[1] < 0 && center[2] >= 0) 
            {
                dir = 4;
                reflectMomentsX(moments);
                reflectMomentsY(moments);
                center[0] = -center[0];
                center[1] = -center[1];
            } 
            else if (center[0] < 0 && center[1] >= 0 && center[2] < 0) 
            {
                dir = 5;
                reflectMomentsX(moments);
                reflectMomentsZ(moments);
                center[0] = -center[0];
                center[2] = -center[2];
            } 
            else if (center[0] >= 0 && center[1] < 0 && center[2] < 0) 
            {
                dir = 6;
                reflectMomentsY(moments);
                reflectMomentsZ(moments);
                center[1] = -center[1];
                center[2] = -center[2];
            } 
            else if (center[0] < 0 && center[1] < 0 && center[2] < 0) 
            {
                dir = 7;
                reflectMomentsX(moments);
                reflectMomentsY(moments);
                reflectMomentsZ(moments);
                center[0] = -center[0];
                center[1] = -center[1];
                center[2] = -center[2];
            }

            if (std::abs(center[0] - center[1]) <= IRL::global_constants::VF_LOW && (center[0] - center[2]) > IRL::global_constants::VF_LOW) 
            {
                dir2 = 0;
            } 
            else if (std::abs(center[1] - center[2]) <= IRL::global_constants::VF_LOW && (center[0] - center[1]) > IRL::global_constants::VF_LOW) 
            {
                dir2 = 0;
            } 
            else if (std::abs(center[0] - center[1]) <= IRL::global_constants::VF_LOW && (center[2] - center[0]) > IRL::global_constants::VF_LOW) 
            {
                dir2 = 3;
                this->reflectMomentsXZ(moments);
                temp = center[0];
                center[0] = center[2];
                center[2] = temp;
            } 
            else if (std::abs(center[0] - center[2]) <= IRL::global_constants::VF_LOW && (center[1] - center[0]) > IRL::global_constants::VF_LOW) 
            {
                dir2 = 1;
                this->reflectMomentsXY(moments);
                temp = center[0];
                center[0] = center[1];
                center[1] = temp;
            } 
            else if (std::abs(center[0] - center[2]) <= IRL::global_constants::VF_LOW && (center[0] - center[1]) > IRL::global_constants::VF_LOW) 
            {
                dir2 = 2;
                this->reflectMomentsYZ(moments);
                temp = center[1];
                center[1] = center[2];
                center[2] = temp;
            } 
            else if (std::abs(center[1] - center[2]) <= IRL::global_constants::VF_LOW && (center[1] - center[0]) > IRL::global_constants::VF_LOW) 
            {
                dir2 = 3;
                this->reflectMomentsXZ(moments);
                temp = center[0];
                center[0] = center[2];
                center[2] = temp;
            } 
            else if (center[1] > center[0] && center[0] >= center[2]) 
            {
                dir2 = 1;
                this->reflectMomentsXY(moments);
                temp = center[0];
                center[0] = center[1];
                center[1] = temp;
            } 
            else if (center[2] > center[1] && center[0] >= center[2]) 
            {
                dir2 = 2;
                this->reflectMomentsYZ(moments);
                temp = center[1];
                center[1] = center[2];
                center[2] = temp;
            } 
            else if (center[2] > center[1] && center[1] >= center[0]) 
            {
                dir2 = 3;
                this->reflectMomentsXZ(moments);
                temp = center[0];
                center[0] = center[2];
                center[2] = temp;
            } 
            else if (center[1] > center[0]) 
            {
                dir2 = 4;
                this->reflectMomentsXY(moments);
                this->reflectMomentsYZ(moments);
                temp = center[0];
                center[0] = center[1];
                center[1] = temp;
                temp = center[1];
                center[1] = center[2];
                center[2] = temp;
            } 
            else if (center[2] > center[1]) 
            {
                dir2 = 5;
                this->reflectMomentsXY(moments);
                this->reflectMomentsXZ(moments);
                temp = center[0];
                center[0] = center[1];
                center[1] = temp;
                temp = center[0];
                center[0] = center[2];
                center[2] = temp;
            }
        };

        IRL::Pt get_global_centroid(std::vector<double>& moments)
        {
            double m000 = 0;
            double m100 = 0;
            double m010 = 0;
            double m001 = 0;
            for(int i = 0; i < NX; ++i)
            {
                for(int j = 0; j < NY; ++j)
                {
                    for(int k = 0; k < NZ; ++k)
                    {
                        m000 = m000 + moments[get_idx(i, j, k, 0)];
                        m100 = m100 + (moments[get_idx(i, j, k, 1)] + gen->getStencil()->getDx()*(i - (NX - 1) / 2.0)) * moments[get_idx(i, j, k, 0)];
                        m010 = m010 + (moments[get_idx(i, j, k, 2)] + gen->getStencil()->getDy()*(j - (NY - 1) / 2.0)) * moments[get_idx(i, j, k, 0)];
                        m001 = m001 + (moments[get_idx(i, j, k, 3)] + gen->getStencil()->getDz()*(k - (NZ - 1) / 2.0)) * moments[get_idx(i, j, k, 0)];
                    }
                }
            }
            IRL::Pt centers;
            if (m000 > IRL::global_constants::VF_LOW) 
            {
                centers[0] = m100 / m000;
                centers[1] = m010 / m000;
                centers[2] = m001 / m000;
            } 
            else 
            {
                centers[0] = 0; centers[1] = 0; centers[2] = 0;
            }

            return centers;
        };
    };
}

#endif