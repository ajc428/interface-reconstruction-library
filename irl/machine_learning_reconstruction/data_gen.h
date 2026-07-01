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

        void generate(double rota_l, double rota_h, double rotb_l, double rotb_h, double rotc_l, double rotc_h, double coa_l, double coa_h, double cob_l, double cob_h, double ox_l, double ox_h, double oy_l, double oy_h, double oz_l, double oz_h, bool disturb, std::string nam)
        {
            bool flip;
            std::ofstream output;
            std::string data_name = "moments"+nam+".txt";
            output.open(data_name, std::ios_base::app);
            std::ofstream normals;
            std::string normals_name = "normals"+nam+".txt";
            normals.open(normals_name, std::ios_base::app);
            std::ofstream coefficients;
            std::string name = "coefficients"+nam+".txt";
            coefficients.open(name, std::ios_base::app);

            std::random_device rd;  
            std::mt19937_64 a_eng(rd());
            std::uniform_int_distribution<int> dis(0, 7);
            //std::uniform_real_distribution<double> noise(-0.025, 0.025);
            std::normal_distribution<double> noise(0.0,0.0125);
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
                
                IRL::Pt center = get_global_centroid(moments);
                reflectMoments(moments, direction, direction2, center);

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

        void generate_sheet(double coa_l, double coa_h, double cob_l, double cob_h, double t_l, double t_h, bool disturb, std::string nam)
        {
            bool flip;
            std::ofstream output;
            std::string data_name = "moments"+nam+".txt";
            output.open(data_name, std::ios_base::app);
            std::ofstream normals;
            std::string normals_name = "normals"+nam+".txt";
            normals.open(normals_name, std::ios_base::app);
            std::ofstream coefficients1;
            std::string name1 = "coefficients1"+nam+".txt";
            coefficients1.open(name1, std::ios_base::app);
            std::ofstream coefficients2;
            std::string name2 = "coefficients2"+nam+".txt";
            coefficients2.open(name2, std::ios_base::app);

            std::random_device rd;  
            std::mt19937_64 a_eng(rd());
            std::uniform_int_distribution<int> dis(0, 7);
            //std::uniform_real_distribution<double> noise(-0.025, 0.025);
            std::normal_distribution<double> noise(0.0,0.0125);

            std::uniform_real_distribution<double> thick(t_l, t_h);
            std::uniform_real_distribution<double> ang1(0, 2*M_PI);
            std::uniform_real_distribution<double> ang2(-M_PI/2.0, M_PI/2.0);
            std::uniform_real_distribution<double> translation(-1, 1);
            std::uniform_real_distribution<double> vec2(0.7, 1.0);
            std::uniform_real_distribution<double> dist_a(coa_l, coa_h);
            std::uniform_real_distribution<double> dist_b(cob_l, cob_h);
            std::uniform_int_distribution<int> dist_sign(1, 2);
            for (int n = 0; n < Ndata; ++n) 
            {
                std::cout << n << std::endl;

                bool valid_pair = false;
                IRL::Paraboloid paraboloid1;
                IRL::Paraboloid paraboloid2;
                IRL::Normal dir1;
                IRL::Normal dir2;

                int sign1 = dist_sign(a_eng);
                int sign2 = dist_sign(a_eng);
                int sign3 = dist_sign(a_eng);
                int sign4 = dist_sign(a_eng);
                double thickness = thick(a_eng);
                double angle1 = ang1(a_eng);
                double angle2 = ang2(a_eng);
                dir1 = IRL::Normal(cos(angle1)*cos(angle2),sin(angle1)*cos(angle2),sin(angle2));
                dir1.normalize();

                int max_it = 0;
                while (!valid_pair)
                {
                    if (max_it > 100)
                    {
                        sign1 = dist_sign(a_eng);
                        sign2 = dist_sign(a_eng);
                        sign3 = dist_sign(a_eng);
                        sign4 = dist_sign(a_eng);
                        max_it = 0;
                    }
                    else
                    {
                        ++max_it;
                    }
                    double o_x = translation(a_eng);
                    double o_y = translation(a_eng);
                    double o_z = translation(a_eng);
                    IRL::Pt datum = IRL::Pt(o_x,o_y,o_z);
                    IRL::Pt datum1 = datum - (thickness/2.0)*dir1;
                    IRL::Pt datum2 = datum + (thickness/2.0)*dir1;

                    double dot = vec2(a_eng);
                    double rot = ang1(a_eng);
                    double r = sqrt(1 - dot*dot);
                    IRL::Normal a;
                    if (abs(dir1[0]) >= 0.97)
                    {
                        a = IRL::Normal(0,1,0);
                    }
                    else
                    {
                        a = IRL::Normal(1,0,0);
                    }
                    IRL::Normal t1 = IRL::crossProduct(dir1,a);
                    t1.normalize();
                    IRL::Normal t2 = IRL::crossProduct(dir1,t1);
                    t2.normalize();
                    IRL::ReferenceFrame frame1 = IRL::ReferenceFrame(t1,t2,dir1);
                    dir2 = r * cos(rot) * t1 + r * sin(rot) * t2 + dot * dir1;
                    if (abs(dir2[0]) >= 0.97)
                    {
                        a = IRL::Normal(0,1,0);
                    }
                    else
                    {
                        a = IRL::Normal(1,0,0);
                    }
                    dir2.normalize();
                    t1 = IRL::crossProduct(dir2,a);
                    t1.normalize();
                    t2 = IRL::crossProduct(dir2,t1);
                    t2.normalize();
                    IRL::ReferenceFrame frame2 = IRL::ReferenceFrame(t1,t2,dir2);

                    double a1 = (2*sign1-3)*dist_a(a_eng);
                    double b1 = (2*sign2-3)*dist_b(a_eng);
                    double a2 = (2*sign3-3)*dist_a(a_eng);
                    double b2 = (2*sign4-3)*dist_b(a_eng);

                    if (a1 < 0 && b1 < 0 && a2 > 0 && b2 > 0)
                    {
                        datum1 = datum + (thickness/2.0)*dir1;
                        datum2 = datum - (thickness/2.0)*dir1;
                    }

                    // datum1 = IRL::Pt(0.153338,0.303501,0.0682626);
                    // frame1 = IRL::ReferenceFrame(IRL::Normal(-0,0.993111,0.117177),IRL::Normal(-0.998757,0.00584061,-0.0495011),IRL::Normal(-0.0498445,-0.117031,0.991877));
                    // datum2 = IRL::Pt(0.144769,0.283383,0.238773);
                    // frame2 = IRL::ReferenceFrame(IRL::Normal(-0,0.999144,0.0413719),IRL::Normal(-0.990775,0.00560653,-0.135399),IRL::Normal(-0.135515,-0.0409902,0.989927));
                    paraboloid1 = IRL::Paraboloid(datum1,frame1,a1,b1);
                    paraboloid2 = IRL::Paraboloid(datum2,frame2,a2,b2);

                    const auto cube = gen->getStencil()->getCell(gen->getStencil()->get_ic(),gen->getStencil()->get_jc(),gen->getStencil()->get_kc());
                    const auto vol1 = IRL::getVolumeMoments<IRL::Volume>(cube, paraboloid1);
                    const auto vol2 = IRL::getVolumeMoments<IRL::Volume>(cube, paraboloid2);

                    //if (!intersect_in_domain(paraboloid1, paraboloid2, -(gen->getStencil()->getNX()*gen->getStencil()->getDx())/2.0, (gen->getStencil()->getNX()*gen->getStencil()->getDx())/2.0) && (vol1 > IRL::global_constants::VF_LOW && vol1 < IRL::global_constants::VF_HIGH && vol2 > IRL::global_constants::VF_LOW  && vol2 < IRL::global_constants::VF_HIGH)) 
                    if (!intersect_in_domain(paraboloid1, paraboloid2, -(gen->getStencil()->getNX()*gen->getStencil()->getDx())/2.0, (gen->getStencil()->getNX()*gen->getStencil()->getDx())/2.0) && (vol1 > IRL::global_constants::VF_LOW && vol1 < IRL::global_constants::VF_HIGH)) 
                    {
                        valid_pair = true;
                    }
                }

                coefficients1 << paraboloid1.getDatum().x() << "," << paraboloid1.getDatum().y() << "," << paraboloid1.getDatum().z()
                << "," << paraboloid1.getReferenceFrame()[0][0] << "," << paraboloid1.getReferenceFrame()[0][1] << "," << paraboloid1.getReferenceFrame()[0][2]
                << "," << paraboloid1.getReferenceFrame()[1][0] << "," << paraboloid1.getReferenceFrame()[1][1] << "," << paraboloid1.getReferenceFrame()[1][2]
                << "," << paraboloid1.getReferenceFrame()[2][0] << "," << paraboloid1.getReferenceFrame()[2][1] << "," << paraboloid1.getReferenceFrame()[2][2]
                << "," << paraboloid1.getAlignedParaboloid().a() << "," << paraboloid1.getAlignedParaboloid().b() << "," << thickness << "\n";
                coefficients2 << paraboloid2.getDatum().x() << "," << paraboloid2.getDatum().y() << "," << paraboloid2.getDatum().z()
                << "," << paraboloid2.getReferenceFrame()[0][0] << "," << paraboloid2.getReferenceFrame()[0][1] << "," << paraboloid2.getReferenceFrame()[0][2]
                << "," << paraboloid2.getReferenceFrame()[1][0] << "," << paraboloid2.getReferenceFrame()[1][1] << "," << paraboloid2.getReferenceFrame()[1][2]
                << "," << paraboloid2.getReferenceFrame()[2][0] << "," << paraboloid2.getReferenceFrame()[2][1] << "," << paraboloid2.getReferenceFrame()[2][2]
                << "," << paraboloid2.getAlignedParaboloid().a() << "," << paraboloid2.getAlignedParaboloid().b() << "," << thickness << "\n";
                
                auto moments1 = gen->get_moments(paraboloid1, 1, false, flip);
                auto moments2 = gen->get_moments(paraboloid2, 1, false, flip);
                std::vector<double> moments;
                std::vector<IRL::Pt> points;
                moments.resize(moments1.size());
                
                int direction = 0;
                int direction2 = 0;

                double total_v1 = 0.0;
                double total_v2 = 0.0;

                for (int i = 0; i < NX; ++i) 
                {
                    for (int j = 0; j < NY; ++j) 
                    {
                        for (int k = 0; k < NZ; ++k) 
                        {
                            total_v1 += moments1[get_idx(i, j, k, 0)];
                            total_v2 += moments2[get_idx(i, j, k, 0)];
                        }
                    }
                }

                if (total_v1 >= total_v2)
                {
                    for (int i = 0; i < NX; ++i)
                    {
                        for (int j = 0; j < NY; ++j)
                        {
                            for (int k = 0; k < NZ; ++k)
                            {
                                if (moments1[get_idx(i, j, k, 0)] > IRL::global_constants::VF_HIGH && moments2[get_idx(i, j, k, 0)] > IRL::global_constants::VF_LOW && moments2[get_idx(i, j, k, 0)] < IRL::global_constants::VF_HIGH)
                                {
                                    moments[get_idx(i, j, k, 0)] = 1 - moments2[get_idx(i, j, k, 0)];
                                    moments[get_idx(i, j, k, 1)] = moments2[get_idx(i, j, k, 4)];
                                    moments[get_idx(i, j, k, 2)] = moments2[get_idx(i, j, k, 5)];
                                    moments[get_idx(i, j, k, 3)] = moments2[get_idx(i, j, k, 6)];
                                    moments[get_idx(i, j, k, 4)] = moments2[get_idx(i, j, k, 1)];
                                    moments[get_idx(i, j, k, 5)] = moments2[get_idx(i, j, k, 2)];
                                    moments[get_idx(i, j, k, 6)] = moments2[get_idx(i, j, k, 3)];
                                }
                                else
                                {
                                    moments[get_idx(i, j, k, 0)] = moments1[get_idx(i, j, k, 0)] - moments2[get_idx(i, j, k, 0)];
                                    for (int m = 1; m < 4; ++m)
                                    {
                                        if (moments[get_idx(i, j, k, 0)] > IRL::global_constants::VF_LOW)
                                        {
                                            moments[get_idx(i, j, k, m)] = (moments1[get_idx(i, j, k, 0)]*moments1[get_idx(i, j, k, m)] - moments2[get_idx(i, j, k, 0)]*moments2[get_idx(i, j, k, m)])/moments[get_idx(i, j, k, 0)];
                                        }
                                        else
                                        {
                                            moments[get_idx(i, j, k, m)] = 0;
                                        }
                                    }
                                    for (int m = 4; m < 7; ++m)
                                    {
                                        if (1-moments[get_idx(i, j, k, 0)] > IRL::global_constants::VF_LOW)
                                        {
                                            moments[get_idx(i, j, k, m)] = ((1-moments1[get_idx(i, j, k, 0)])*moments1[get_idx(i, j, k, m)] - (1-moments2[get_idx(i, j, k, 0)])*moments2[get_idx(i, j, k, m)])/(1-moments[get_idx(i, j, k, 0)]);
                                        }
                                        else
                                        {
                                            moments[get_idx(i, j, k, m)] = 0;
                                        }
                                    }
                                }
                                if (moments[get_idx(i, j, k, 0)] > IRL::global_constants::VF_LOW) 
                                {
                                    points.push_back(IRL::Pt(moments[get_idx(i, j, k, 1)],moments[get_idx(i, j, k, 2)],moments[get_idx(i, j, k, 3)])+IRL::Pt(gen->getStencil()->get_xm(i),gen->getStencil()->get_ym(j),gen->getStencil()->get_zm(k)));
                                }
                            }
                        }
                    }
                }
                else
                {
                    for (int i = 0; i < NX; ++i)
                    {
                        for (int j = 0; j < NY; ++j)
                        {
                            for (int k = 0; k < NZ; ++k)
                            {
                                if (moments2[get_idx(i, j, k, 0)] > IRL::global_constants::VF_HIGH && moments1[get_idx(i, j, k, 0)] > IRL::global_constants::VF_LOW && moments1[get_idx(i, j, k, 0)] < IRL::global_constants::VF_HIGH)
                                {
                                    moments[get_idx(i, j, k, 0)] = 1 - moments1[get_idx(i, j, k, 0)];
                                    moments[get_idx(i, j, k, 1)] = moments1[get_idx(i, j, k, 4)];
                                    moments[get_idx(i, j, k, 2)] = moments1[get_idx(i, j, k, 5)];
                                    moments[get_idx(i, j, k, 3)] = moments1[get_idx(i, j, k, 6)];
                                    moments[get_idx(i, j, k, 4)] = moments1[get_idx(i, j, k, 1)];
                                    moments[get_idx(i, j, k, 5)] = moments1[get_idx(i, j, k, 2)];
                                    moments[get_idx(i, j, k, 6)] = moments1[get_idx(i, j, k, 3)];
                                }
                                else
                                {
                                    moments[get_idx(i, j, k, 0)] = moments2[get_idx(i, j, k, 0)] - moments1[get_idx(i, j, k, 0)];
                                    for (int m = 1; m < 4; ++m)
                                    {
                                        if (moments[get_idx(i, j, k, 0)] > IRL::global_constants::VF_LOW)
                                        {
                                            moments[get_idx(i, j, k, m)] = (moments2[get_idx(i, j, k, 0)]*moments2[get_idx(i, j, k, m)] - moments1[get_idx(i, j, k, 0)]*moments1[get_idx(i, j, k, m)])/moments[get_idx(i, j, k, 0)];
                                        }
                                        else
                                        {
                                            moments[get_idx(i, j, k, m)] = 0;
                                        }
                                    }
                                    for (int m = 4; m < 7; ++m)
                                    {
                                        if (1-moments[get_idx(i, j, k, 0)] > IRL::global_constants::VF_LOW)
                                        {
                                            moments[get_idx(i, j, k, m)] = ((1-moments2[get_idx(i, j, k, 0)])*moments2[get_idx(i, j, k, m)] - (1-moments1[get_idx(i, j, k, 0)])*moments1[get_idx(i, j, k, m)])/(1-moments[get_idx(i, j, k, 0)]);
                                        }
                                        else
                                        {
                                            moments[get_idx(i, j, k, m)] = 0;
                                        }
                                    }
                                }
                                if (moments[get_idx(i, j, k, 0)] > IRL::global_constants::VF_LOW) 
                                {
                                    points.push_back(IRL::Pt(moments[get_idx(i, j, k, 1)],moments[get_idx(i, j, k, 2)],moments[get_idx(i, j, k, 3)])+IRL::Pt(gen->getStencil()->get_xm(i),gen->getStencil()->get_ym(j),gen->getStencil()->get_zm(k)));
                                }
                            }
                        }
                    }
                }

                //points = smoothPoints(points,0.75);

                    // for (const auto& element : points) {
                    //     std::cout << element << std::endl;
                    // }

                IRL::Paraboloid fit = fitConstrainedParaboloidFromPoints(points);//fitParaboloidFromPoints(points);
                //std::cout << fit.getAlignedParaboloid().a() << " " << fit.getAlignedParaboloid().b() << " " << fit.getReferenceFrame()[2] << std::endl;

                auto cell = gen->getStencil()->getCell(gen->getStencil()->get_ic(),gen->getStencil()->get_jc(),gen->getStencil()->get_kc());
                auto surface_and_moments = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParaboloidParametrizedSurfaceOutput>>(cell, paraboloid1);
                auto surface = surface_and_moments.getSurface();
                auto normal1 = surface.getAverageNormalNonAligned();

                surface_and_moments = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParaboloidParametrizedSurfaceOutput>>(cell, paraboloid2);
                surface = surface_and_moments.getSurface();
                auto normal2 = surface.getAverageNormalNonAligned();

                surface_and_moments = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParaboloidParametrizedSurfaceOutput>>(cell, fit);
                surface = surface_and_moments.getSurface();
                auto normal_fit = fit.getReferenceFrame()[2];//surface.getAverageNormalNonAligned();

                //std::cout << normal1 << std::endl << std::endl;
                IRL::Pt center1 = get_global_centroid(moments);
                if (IRL::dotProduct(center1,normal_fit) < 0)
                {
                    //std::cout << fit.getAlignedParaboloid().a() << " " << fit.getAlignedParaboloid().b() << " " << fit.getReferenceFrame()[2] << std::endl;
                    //std::cout << normal1 << std::endl << std::endl;
                    //normal_fit = -normal_fit;
                }

                // IRL::Pt surface_centroid(0, 0, 0);
                // for (const auto& pt : points) {
                //     surface_centroid[0] += pt[0];
                //     surface_centroid[1] += pt[1];
                //     surface_centroid[2] += pt[2];
                // }
                // surface_centroid[0] /= points.size();
                // surface_centroid[1] /= points.size();
                // surface_centroid[2] /= points.size();

                // // Force the normal to always point "outward" relative to the cell center
                // std::cout << normal1 << " " << normal_fit << " " << surface_centroid << std::endl;
                // if (IRL::dotProduct(normal_fit, surface_centroid) < 0) {
                //     normal_fit = -normal_fit;
                // }
                // std::cout << normal1 << " " << normal_fit << std::endl << std::endl;


                const double eps = 1e-10;

                // if (normal_fit[0] < -eps) {
                //     normal_fit = -normal_fit;
                // } else if (std::abs(normal_fit[0]) <= eps) {
                //     if (normal_fit[1] < -eps) {
                //         normal_fit = -normal_fit;
                //     } else if (std::abs(normal_fit[1]) <= eps && normal_fit[2] < -eps) {
                //         normal_fit = -normal_fit;
                //     }
                // }

                auto moments_fit = gen->get_moments(fit, 1, true, flip);

                IRL::Pt center = IRL::Pt(normal_fit[0],normal_fit[1],normal_fit[2]);
                reflectMoments(moments, direction, direction2, center);
                //IRL::Pt center = get_global_centroid(moments_fit);
                //reflectMoments(moments, direction, direction2, center);
                //reflectMoments(moments_fit, direction, direction2, center);
                //moments = moments_fit;

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

                double tmp;
                switch (direction)
                {
                    case 1:
                        normal1[0] = -normal1[0];

                        normal2[0] = -normal2[0];

                        normal_fit[0] = -normal_fit[0];
                    break;
                    case 2:
                        normal1[1] = -normal1[1];

                        normal2[1] = -normal2[1];

                        normal_fit[1] = -normal_fit[1];
                    break;
                    case 3:
                        normal1[2] = -normal1[2];

                        normal2[2] = -normal2[2];

                        normal_fit[2] = -normal_fit[2];
                    break;
                    case 4:
                        normal1[0] = -normal1[0];
                        normal1[1] = -normal1[1];

                        normal2[0] = -normal2[0];
                        normal2[1] = -normal2[1];

                        normal_fit[0] = -normal_fit[0];
                        normal_fit[1] = -normal_fit[1];
                    break;
                    case 5:
                        normal1[0] = -normal1[0];
                        normal1[2] = -normal1[2];

                        normal2[0] = -normal2[0];
                        normal2[2] = -normal2[2];

                        normal_fit[0] = -normal_fit[0];
                        normal_fit[2] = -normal_fit[2];
                    break;
                    case 6:
                        normal1[1] = -normal1[1];
                        normal1[2] = -normal1[2];

                        normal2[1] = -normal2[1];
                        normal2[2] = -normal2[2];

                        normal_fit[1] = -normal_fit[1];
                        normal_fit[2] = -normal_fit[2];
                    break;
                    case 7:
                        normal1[0] = -normal1[0];
                        normal1[1] = -normal1[1];
                        normal1[2] = -normal1[2];

                        normal2[0] = -normal2[0];
                        normal2[1] = -normal2[1];
                        normal2[2] = -normal2[2];  

                        normal_fit[0] = -normal_fit[0];
                        normal_fit[1] = -normal_fit[1];
                        normal_fit[2] = -normal_fit[2];                     
                    break;
                }
                switch (direction2)
                {
                    case 1:
                        tmp=normal1[0]; 
                        normal1[0]=normal1[1]; 
                        normal1[1]=tmp;

                        tmp=normal2[0]; 
                        normal2[0]=normal2[1]; 
                        normal2[1]=tmp;    

                        tmp=normal_fit[0]; 
                        normal_fit[0]=normal_fit[1]; 
                        normal_fit[1]=tmp;                       
                    break;
                    case 2:
                        tmp=normal1[1]; 
                        normal1[1]=normal1[2]; 
                        normal1[2]=tmp;

                        tmp=normal2[1]; 
                        normal2[1]=normal2[2]; 
                        normal2[2]=tmp;    

                        tmp=normal_fit[1]; 
                        normal_fit[1]=normal_fit[2]; 
                        normal_fit[2]=tmp;                        
                    break;
                    case 3:
                        tmp=normal1[0]; 
                        normal1[0]=normal1[2]; 
                        normal1[2]=tmp;

                        tmp=normal2[0]; 
                        normal2[0]=normal2[2]; 
                        normal2[2]=tmp;     

                        tmp=normal_fit[0]; 
                        normal_fit[0]=normal_fit[2]; 
                        normal_fit[2]=tmp;                     
                    break;
                    case 4:
                        tmp=normal1[0]; 
                        normal1[0]=normal1[1]; 
                        normal1[1]=tmp;
                        tmp=normal1[1]; 
                        normal1[1]=normal1[2]; 
                        normal1[2]=tmp;

                        tmp=normal2[0]; 
                        normal2[0]=normal2[1]; 
                        normal2[1]=tmp;
                        tmp=normal2[1]; 
                        normal2[1]=normal2[2]; 
                        normal2[2]=tmp;     

                        tmp=normal_fit[0]; 
                        normal_fit[0]=normal_fit[1]; 
                        normal_fit[1]=tmp;
                        tmp=normal_fit[1]; 
                        normal_fit[1]=normal_fit[2]; 
                        normal_fit[2]=tmp;                                            
                    break;
                    case 5:
                        tmp=normal1[0]; 
                        normal1[0]=normal1[1]; 
                        normal1[1]=tmp;
                        tmp=normal1[0]; 
                        normal1[0]=normal1[2]; 
                        normal1[2]=tmp;

                        tmp=normal2[0]; 
                        normal2[0]=normal2[1]; 
                        normal2[1]=tmp;
                        tmp=normal2[0]; 
                        normal2[0]=normal2[2]; 
                        normal2[2]=tmp;     

                        tmp=normal_fit[0]; 
                        normal_fit[0]=normal_fit[1]; 
                        normal_fit[1]=tmp;
                        tmp=normal_fit[0]; 
                        normal_fit[0]=normal_fit[2]; 
                        normal_fit[2]=tmp;                    
                    break;
                }

                normal_fit.normalize();
                // if (!flip)
                // {
                //     normal_fit = -normal_fit;
                // }
                output << normal_fit[0] << ",";
                output << normal_fit[1] << ",";
                output << normal_fit[2] << ",";
                output << "\n";

                // std::cout << normal1 << std::endl;
                // std::cout << normal2 << std::endl;
                // std::cout << normal_fit << std::endl;

                normal1.normalize();
                normal2.normalize();
                // std::cout << normal1 << " " << normal2 << " " << normal_fit << std::endl;

                bool flip_normals = false;
                if (normal1[0] < -eps) {
                    flip_normals = true;
                } else if (std::abs(normal1[0]) <= eps) {
                    if (normal1[1] < -eps) {
                        flip_normals = true;
                    } else if (std::abs(normal1[1]) <= eps && normal1[2] < -eps) {
                        flip_normals = true;
                    }
                }

                if (flip_normals) {
                    //if (normal2.calculateMagnitude() < 0.1)
                    {
                        //normal1 = -normal1;
                    }
                    //else
                    {
                        auto tmp = normal1;
                        normal1 = -normal2;
                        normal2 = -tmp;
                    }
                }

                double theta1 = std::atan2(normal1[1], normal1[0])/(M_PI/4.0);
                double phi1   = std::asin(std::clamp(normal1[2], -1.0, 1.0))/(M_PI/4.0);
                //double n2x = -normal2[0], n2y = -normal2[1], n2z = -normal2[2];
                double theta2 = std::atan2(normal2[1], normal2[0])/(M_PI/4.0);
                double phi2   = std::asin(std::clamp(normal2[2], -1.0, 1.0))/(M_PI/4.0);
                //normals << theta1 << "," << phi1 << "," << theta2 << "," << phi2 << "\n";
                normals << normal1[0] << "," << normal1[1] << "," << normal1[2] << "," << -normal2[0] << "," << -normal2[1] << "," << -normal2[2] << "\n";                
           
           
                // const auto bottom_corner = IRL::Pt(-1.5, -1.5, -1.5);
                // const auto top_corner = IRL::Pt(1.5, 1.5, 1.5);
                // const auto cube = IRL::StoredRectangularCuboid<IRL::Pt>::fromBoundingPts(bottom_corner, top_corner);

                // const auto first_moments_and_surface = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParaboloidParametrizedSurfaceOutput>, IRL::HalfEdgeCutting>(cube, paraboloid1);
                // const auto first_moments_and_surface2 = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParaboloidParametrizedSurfaceOutput>, IRL::HalfEdgeCutting>(cube, paraboloid2);
                // const auto first_moments_and_surface3 = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParaboloidParametrizedSurfaceOutput>, IRL::HalfEdgeCutting>(cube, fit);
                // const double length_scale = 0.01;
                // IRL::TriangulatedSurfaceOutput triangulated_surface = first_moments_and_surface.getSurface().triangulate(length_scale);
                // IRL::TriangulatedSurfaceOutput triangulated_surface2 = first_moments_and_surface2.getSurface().triangulate(length_scale);
                // IRL::TriangulatedSurfaceOutput triangulated_surface3 = first_moments_and_surface3.getSurface().triangulate(length_scale);
                // std::string name3 = "p1_"+std::to_string(n);
                // std::string name4 = "p2_"+std::to_string(n);
                // std::string name5 = "pf_"+std::to_string(n);
                // triangulated_surface.write(name3);
                // triangulated_surface2.write(name4);  
                // triangulated_surface3.write(name5);  
           
           
            }  
            output.close();  
            normals.close(); 
            coefficients1.close(); 
            coefficients2.close(); 
        }; 

        struct Interval 
        {
            double min_val;
            double max_val;

            Interval()
            {
                min_val = 0;
                max_val = 0;
            }
            Interval(double val)
            {
                min_val = val;
                max_val = val;
            }
            Interval(double min_v, double max_v)
            {
                min_val = min_v;
                max_val = max_v;
            }

            Interval operator+(const Interval& other) const 
            {
                return Interval(min_val + other.min_val, max_val + other.max_val);
            }

            Interval operator-(const Interval& other) const 
            {
                return Interval(min_val - other.max_val, max_val - other.min_val);
            }

            Interval operator*(double scalar) const 
            {
                double a = min_val * scalar;
                double b = max_val * scalar;
                return Interval(std::min(a, b), std::max(a, b));
            }

            Interval scaledSquare(double coeff) const 
            {
                if (coeff == 0.0) return Interval(0.0, 0.0);

                double sq_lo, sq_hi;

                if (min_val <= 0.0 && max_val >= 0.0) 
                {
                    sq_lo = 0.0;
                    sq_hi = std::max(min_val * min_val, max_val * max_val);
                } 
                else 
                {
                    double a = min_val * min_val;
                    double b = max_val * max_val;
                    sq_lo = std::min(a, b);
                    sq_hi = std::max(a, b);
                }

                double p = sq_lo * coeff;
                double q = sq_hi * coeff;
                return Interval(std::min(p, q), std::max(p, q));
            }

            bool containsZero() const 
            {
                return (min_val <= 0.0 && max_val >= 0.0);
            }
        };

        struct Interval3 
        {
            Interval x, y, z;
            Interval3(Interval _x, Interval _y, Interval _z)
            {
                x = _x, y = _y, z = _z;
            }
        };

        Interval intervalDot(const Interval3& v_int, const IRL::Normal& n) 
        {
            return v_int.x * n[0] + v_int.y * n[1] + v_int.z * n[2];
        }

        Interval evaluateParaboloid(const IRL::Paraboloid& p, const Interval3& box) 
        {
            IRL::Pt datum = p.getDatum();
            IRL::ReferenceFrame frame = p.getReferenceFrame();
            double a = p.getAlignedParaboloid().a();
            double b = p.getAlignedParaboloid().b();

            Interval3 shift(
                box.x - Interval(datum[0]),
                box.y - Interval(datum[1]),
                box.z - Interval(datum[2])
            );

            Interval U = intervalDot(shift, frame[0]);
            Interval V = intervalDot(shift, frame[1]);
            Interval W = intervalDot(shift, frame[2]);

            return W + U.scaledSquare(a) + V.scaledSquare(b);
        }

        bool checkIntersectionRecursive(const IRL::Paraboloid& p1, const IRL::Paraboloid& p2, double min_x, double max_x, double min_y, double max_y, double min_z, double max_z, int depth) 
        {
            
            Interval3 box(Interval(min_x, max_x), Interval(min_y, max_y), Interval(min_z, max_z));

            Interval f1 = evaluateParaboloid(p1, box);
            Interval f2 = evaluateParaboloid(p2, box);

            if (!f1.containsZero() || !f2.containsZero()) 
            {
                return false;
            }

            if (depth >= 10) 
            {
                return true; 
            }

            double mid_x = (min_x + max_x) / 2.0;
            double mid_y = (min_y + max_y) / 2.0;
            double mid_z = (min_z + max_z) / 2.0;

            double x_bounds[3] = {min_x, mid_x, max_x};
            double y_bounds[3] = {min_y, mid_y, max_y};
            double z_bounds[3] = {min_z, mid_z, max_z};

            for (int i = 0; i < 2; ++i) 
            {
                for (int j = 0; j < 2; ++j) 
                {
                    for (int k = 0; k < 2; ++k) 
                    {
                        if (checkIntersectionRecursive(p1, p2, x_bounds[i], x_bounds[i+1], y_bounds[j], y_bounds[j+1], z_bounds[k], z_bounds[k+1], depth + 1)) 
                        {
                            return true;
                        }
                    }
                }
            }

            return false;
        }

        bool intersect_in_domain(const IRL::Paraboloid& p1, const IRL::Paraboloid& p2, double min_bound, double max_bound) 
        {
            return checkIntersectionRecursive(p1, p2, min_bound, max_bound, min_bound, max_bound, min_bound, max_bound, 0);
        }

        std::vector<IRL::Pt> smoothPoints(const std::vector<IRL::Pt>& points, double merge_radius) 
        {
            const size_t N = points.size();
            if (N < 14) return points;

            std::vector<IRL::Pt> smoothed_points;

            // Track which points have already been absorbed into a cluster
            std::vector<bool> merged(N, false);
            
            // Pre-calculate squared radius to avoid expensive sqrt() in the inner loop
            const double r2 = merge_radius * merge_radius;

            for (size_t i = 0; i < N; ++i) 
            {
                if (merged[i]) continue;

                // Start a new cluster
                double sum_x = points[i][0];
                double sum_y = points[i][1];
                double sum_z = points[i][2];
                double count = 1.0;
                
                merged[i] = true;

                // Find all other unmerged points within the radius
                for (size_t j = i + 1; j < N; ++j) 
                {
                    if (merged[j]) continue;

                    double dx = points[i][0] - points[j][0];
                    double dy = points[i][1] - points[j][1];
                    double dz = points[i][2] - points[j][2];
                    
                    if ((dx*dx + dy*dy + dz*dz) <= r2) 
                    {
                        sum_x += points[j][0];
                        sum_y += points[j][1];
                        sum_z += points[j][2];
                        count += 1.0;
                        merged[j] = true;
                    }
                }

                // Add the local average to the smoothed dataset
                smoothed_points.emplace_back(sum_x / count, sum_y / count, sum_z / count);
            }

            return smoothed_points;
        }

        IRL::Paraboloid fitParaboloidFromPoints(const std::vector<IRL::Pt>& points) 
        {
            using MatrixX = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
            using VectorX = Eigen::Matrix<double, Eigen::Dynamic, 1>;
            using Vector3 = Eigen::Vector<double, 3>;
            using Matrix33 = Eigen::Matrix<double, 3, 3>;

            const size_t N = points.size();
            assert(N >= 3 && "At least 3 points are required for fit.");

            double xc = 0;
            double yc = 0;
            double zc = 0;
            for (const auto& pt : points) 
            {
                xc += pt[0];
                yc += pt[1];
                zc += pt[2];
            }
            xc = xc / double(N);
            yc = yc / double(N);
            zc = zc / double(N);

            Vector3 L;
            Matrix33 hessF = Matrix33::Zero();
            double C_offset;

            if (N>=10)
            {
                // Set up the homogeneous least squares system: A * C = 0
                MatrixX A(N, 10);
                for (size_t i = 0; i < N; ++i) 
                {
                    double x = points[i][0] - xc;
                    double y = points[i][1] - yc;
                    double z = points[i][2] - zc;
                    
                    A(i, 0) = x * x;
                    A(i, 1) = y * y;
                    A(i, 2) = z * z;
                    A(i, 3) = x * y;
                    A(i, 4) = y * z;
                    A(i, 5) = x * z;
                    A(i, 6) = x;
                    A(i, 7) = y;
                    A(i, 8) = z;
                    A(i, 9) = 1;
                }

                // Solve for C using SVD (Right singular vector of smallest singular value)
                Eigen::JacobiSVD<MatrixX> svd(A, Eigen::ComputeFullV);
                VectorX C = svd.matrixV().col(9);

                L = Vector3(C(6), C(7), C(8));

                hessF(0, 0) = 2 * C(0);
                hessF(1, 1) = 2 * C(1);
                hessF(2, 2) = 2 * C(2);
                hessF(0, 1) = C(3); hessF(1, 0) = C(3);
                hessF(1, 2) = C(4); hessF(2, 1) = C(4);
                hessF(0, 2) = C(5); hessF(2, 0) = C(5);

                C_offset = C(9);
            }
            else if (N >= 4)
            {
                // --- SPHERE FALLBACK (4 <= N < 10) ---
                // Equation: C0*(x^2 + y^2 + z^2) + C1*x + C2*y + C3*z + C4 = 0
                MatrixX A(N, 5);
                for (size_t i = 0; i < N; ++i) 
                {
                    double x = points[i][0] - xc;
                    double y = points[i][1] - yc;
                    double z = points[i][2] - zc;
                    A.row(i) << (x*x + y*y + z*z), x, y, z, 1;
                }

                Eigen::JacobiSVD<MatrixX> svd(A, Eigen::ComputeFullV);
                VectorX C = svd.matrixV().col(4);

                L = Vector3(C(1), C(2), C(3));

                hessF(0, 0) = 2 * C(0);
                hessF(1, 1) = 2 * C(0);
                hessF(2, 2) = 2 * C(0);

                C_offset = C(4);
            }
            else
            {
                // --- PLANAR FALLBACK (N == 3) ---
                // Equation: C0*x + C1*y + C2*z + C3 = 0
                MatrixX A(N, 4);
                for (size_t i = 0; i < N; ++i) 
                {
                    double x = points[i][0] - xc;
                    double y = points[i][1] - yc;
                    double z = points[i][2] - zc;
                    A.row(i) << x, y, z, 1;
                }

                Eigen::JacobiSVD<MatrixX> svd(A, Eigen::ComputeFullV);
                VectorX C = svd.matrixV().col(3);

                L = Vector3(C(0), C(1), C(2));
                C_offset = C(3);
            }

            // Find a datum exactly on the surface F(x,y,z) = 0
            double a = 0.5 * (L.transpose() * hessF * L).value();
            double b = L.squaredNorm();
            double c = C_offset;
            
            double t = 0;
            
            // Check if the surface is practically linear in this direction
            if (std::abs(a) < Eigen::NumTraits<double>::epsilon() && std::abs(b) > Eigen::NumTraits<double>::epsilon()) 
            {
                t = -c / b;
            } 
            else if (std::abs(a) > Eigen::NumTraits<double>::epsilon())
            {
                double discriminant = b * b - 4 * a * c;
                if (discriminant >= 0)
                {
                    double sqrt_disc = std::sqrt(discriminant);
                    double t1 = (-b + sqrt_disc) / (2 * a);
                    double t2 = (-b - sqrt_disc) / (2 * a);
                    t = (std::abs(t1) < std::abs(t2)) ? t1 : t2;
                }
            }

            Vector3 d_local = t * L;
            
            PtBase<double> datum(xc + d_local(0), yc + d_local(1), zc + d_local(2));

            // Evaluate the Gradient at the precise datum location
            Vector3 gradF = hessF * d_local + L;

            // Build and return the IRL Paraboloid
            return IRL::Paraboloid::fromDerivatives(datum, gradF, hessF);
        }

        IRL::Paraboloid fitConstrainedParaboloidFromPoints(const std::vector<IRL::Pt>& points) 
        {
            using MatrixX = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
            using VectorX = Eigen::Matrix<double, Eigen::Dynamic, 1>;
            using Vector3 = Eigen::Vector<double, 3>;
            using Matrix33 = Eigen::Matrix<double, 3, 3>;

            const size_t N = points.size();
            if (N < 6)
            {
                return fitParaboloidFromPoints(points);
            }
            assert(N >= 6 && "At least 6 points required for constrained paraboloid fit.");

            // 1. Calculate Centroid
            Vector3 centroid = Vector3::Zero();
            for (const auto& pt : points) 
            {
                centroid += Vector3(pt[0], pt[1], pt[2]);
            }
            centroid = centroid / double(N);

            // 2. PCA to find the plane of best fit
            Matrix33 covariance = Matrix33::Zero();
            for (const auto& pt : points) 
            {
                Vector3 d = Vector3(pt[0], pt[1], pt[2]) - centroid;
                covariance += d * d.transpose();
            }
            
            Eigen::SelfAdjointEigenSolver<Matrix33> eigensolver(covariance);
            // The eigenvector with the smallest eigenvalue is the normal to the plane
            Vector3 local_z = eigensolver.eigenvectors().col(0).normalized();
            
            // We want the normal to generally point towards the positive global hemisphere to maintain consistency
            //if (local_z.sum() < 0.0) local_z *= -1.0;

            // 3. Create a rotation to align local_z with global Z (0,0,1)
            Vector3 global_z(0, 0, 1);
            Eigen::Quaterniond rot;
            rot.setFromTwoVectors(local_z, global_z);
            Matrix33 R = rot.toRotationMatrix(); 
            Matrix33 R_inv = R.transpose(); // To go back to global later

            // 4. Rotate points into the local tangent frame
            MatrixX X(N, 6);
            VectorX Z(N);
            
            auto populate_matrices = [&]() 
            {
                for (size_t i = 0; i < N; ++i) 
                {
                    // Shift to centroid, then rotate
                    Vector3 p = Vector3(points[i][0], points[i][1], points[i][2]) - centroid;
                    Vector3 p_local = R * p;
                    
                    double x = p_local(0);
                    double y = p_local(1);
                    
                    X(i, 0) = x * x;
                    X(i, 1) = y * y;
                    X(i, 2) = x * y;
                    X(i, 3) = x;
                    X(i, 4) = y;
                    X(i, 5) = 1.0;
                    
                    Z(i) = p_local(2); // The local height
                }
            };

            populate_matrices();

            // 5. Least Squares Fit: z = C0*x^2 + C1*y^2 + C2*xy + C3*x + C4*y + C5
            VectorX C = X.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeFullV).solve(Z);

            // // Check if the paraboloid is generally opening downwards (dome instead of bowl)
            // if (C(0) + C(1) > 0.0)
            // {
            //     // The arbitrary PCA normal is pointing away from the bowl opening. Flip it.
            //     local_z = -local_z;
            //     rot.setFromTwoVectors(local_z, global_z);
            //     R = rot.toRotationMatrix();
            //     R_inv = R.transpose();
                
            //     // Repopulate and re-solve with the corrected orientation
            //     populate_matrices();
            //     C = X.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeFullV).solve(Z);
            // }

            // // Enforce positive semi-definiteness on the quadratic form
            // Eigen::Matrix2d Q;
            // Q << C(0), C(2) / 2.0,
            //      C(2) / 2.0, C(1);

            // Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es2d(Q);
            // Eigen::Vector2d evals = es2d.eigenvalues();

            // // Define a small positive epsilon to ensure strictly positive curvatures
            // const double epsilon = 1e-6; 
            // bool needs_projection = false;

            // if (evals(0) > -epsilon) { evals(0) = -epsilon; needs_projection = true; }
            // if (evals(1) > -epsilon) { evals(1) = -epsilon; needs_projection = true; }

            // if (needs_projection)
            // {
            //     // Reconstruct the clamped quadratic matrix
            //     Eigen::Matrix2d Q_proj = es2d.eigenvectors() * evals.asDiagonal() * es2d.eigenvectors().transpose();
                
            //     C(0) = Q_proj(0, 0);
            //     C(1) = Q_proj(1, 1);
            //     C(2) = 2.0 * Q_proj(0, 1);

            //     // Re-solve for the linear terms (C3, C4, C5) to minimize error given the forced quadratic terms
            //     VectorX Z_adj = Z - (X.col(0) * C(0) + X.col(1) * C(1) + X.col(2) * C(2));
            //     MatrixX X_lin = X.block(0, 3, N, 3); // Cols 3, 4, 5
                
            //     Vector3 C_lin = X_lin.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeFullV).solve(Z_adj);
            //     C(3) = C_lin(0);
            //     C(4) = C_lin(1);
            //     C(5) = C_lin(2);
            // }

            // 6. Evaluate local derivatives at (x=0, y=0)
            double z_offset = C(5);
            
            // F(x,y,z) = z - (C0*x^2 + C1*y^2 + C2*xy + C3*x + C4*y + C5) = 0
            // Gradient: del_F / del_x = -C3, del_F / del_y = -C4, del_F / del_z = 1
            Vector3 grad_local(-C(3), -C(4), 1.0);
            
            // Hessian:
            Matrix33 hess_local = Matrix33::Zero();
            hess_local(0, 0) = -2.0 * C(0);
            hess_local(1, 1) = -2.0 * C(1);
            hess_local(0, 1) = -C(2);
            hess_local(1, 0) = -C(2);

            // 7. Transform back to global coordinates
            // Local datum is (0, 0, z_offset)
            Vector3 datum_local(0.0, 0.0, z_offset);
            Vector3 datum_global = centroid + R_inv * datum_local;
            IRL::Pt datum(datum_global(0), datum_global(1), datum_global(2));

            // Gradient transforms cleanly as a vector
            Vector3 grad_global = R_inv * grad_local;
            
            // Hessian transforms via H_global = R^T * H_local * R
            // Because R_inv IS R^T, this is:
            Matrix33 hess_global = R_inv * hess_local * R;

            // 8. Return the Paraboloid
            IRL::Paraboloid solution = IRL::Paraboloid::fromDerivatives(datum, grad_global, hess_global);
            // std::cout << solution.getAlignedParaboloid().a() << " " << solution.getAlignedParaboloid().b() << " " << solution.getReferenceFrame()[2] << std::endl;
            return solution;
        }























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

        void reflectMoments(std::vector<double>& moments, int& dir, int& dir2, IRL::Pt center) 
        {
            //IRL::Pt center = get_global_centroid(moments);
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