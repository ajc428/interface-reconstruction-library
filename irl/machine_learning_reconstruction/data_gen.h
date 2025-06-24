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

#include "irl/machine_learning_reconstruction/fractions.h"
#include "irl/machine_learning_reconstruction/spatial_moments.h"
#include <iostream>
#include <cstdlib>

// #include "tensor.h"
// #include "runningtime.h"
// #include "Tensor3D.h"
// #include "Tensor3D.cpp"
// #include "cpd_als.cpp"
// #include "tucker_hosvd.cpp"
// #include "tensor_hooi.cpp"
// #include "t_svd.cpp"
// #include "tensor_train.cpp"
// #include "cpd_gen.cpp"
// #include "mode_n_product.cpp"
// #include <mkl.h>

//using namespace TensorLet_decomposition;

namespace IRL 
{
    class data_gen
    {
    private:
        int number_of_cells;     
        int Ntests;  

        IRL::fractions *gen;
        std::array<double, 3> angles;
        IRL::spatial_moments *sm;

    public:
        data_gen(int x, int y)
        {
            number_of_cells = x;
            Ntests = y;
            gen = new IRL::fractions(number_of_cells);
            sm = new IRL::spatial_moments();
        };

        ~data_gen()
        {
            delete gen;
            delete sm;
        };

        void generate(double rota_l, double rota_h, double rotb_l, double rotb_h, double rotc_l, double rotc_h, double coa_l, double coa_h, double cob_l, double cob_h, double ox_l, double ox_h, double oy_l, double oy_h, double oz_l, double oz_h, bool all)
        {
            for (int n = 0; n < Ntests; ++n) 
            {
                torch::Tensor result;
                std::cout << n << endl;
                IRL::Paraboloid paraboloid = gen->new_random_parabaloid(rota_l, rota_h, rotb_l, rotb_h, rotc_l, rotc_h, coa_l, coa_h, cob_l, cob_h, ox_l, ox_h, oy_l, oy_h, oz_l, oz_h);
                // result = gen->get_fractions_all(paraboloid);
                // while (result[((result.sizes()[0]-7)/2)].item<double>() > 0.5)
                // {
                //     paraboloid = gen->new_random_parabaloid(rota_l, rota_h, rotb_l, rotb_h, rotc_l, rotc_h, coa_l, coa_h, cob_l, cob_h, ox_l, ox_h, oy_l, oy_h, oz_l, oz_h);
                //     result = gen->get_fractions_all(paraboloid);
                // }
                angles = gen->getAngles();

                std::ofstream coefficients;
                std::string name = "coefficients.txt";
                coefficients.open(name, std::ios_base::app);
                coefficients << paraboloid.getDatum().x() << "," << paraboloid.getDatum().y() << "," << paraboloid.getDatum().z()
                << "," << angles[0] << "," << angles[1] << "," << angles[2]
                << "," << paraboloid.getAlignedParaboloid().a() << "," << paraboloid.getAlignedParaboloid().b() << "\n";
                coefficients.close();
                
                bool flip = false;
                if (!all)
                {
                    result = gen->get_fractions(paraboloid, true);
                    if (result[((result.sizes()[0]-/*7*/4)/2)].item<double>() > 0.5)
                    {
                        flip = true;
                        result = gen->get_fractions_gas(paraboloid, true);
                    }
                }
                else
                {
                    result = gen->get_fractions_all(paraboloid);
                    if (result[((result.sizes()[0]-7)/2)].item<double>() > 0.5)
                    {
                        flip = true;
                        result = gen->get_fractions_gas_all(paraboloid);
                    }                    
                }

                std::vector<double> fractions;
                for (int i = 0; i < result.sizes()[0]; ++i)
                {
                    fractions.push_back(result[i].item<double>());
                }
                

                int direction = 0;
                std::vector<double> center;
                auto sm = IRL::spatial_moments();
                if (!all)
                {
                    center = sm.get_mass_centers(fractions);
                    direction = rotateFractions(&fractions,center);
                }
                else
                {
                    center = sm.get_mass_centers_all(&fractions);
                    direction = rotateFractions_all(&fractions,center);
                }
                
                std::ofstream output;
                std::string data_name = "fractions.txt";
                output.open(data_name, std::ios_base::app);

                for (int i = 0; i < result.sizes()[0]; ++i)
                {
                    output << fractions[i] << ",";
                }
                output << "\n";
                output.close();  

                std::ofstream normals;
                std::string normals_name = "normals.txt";
                normals.open(normals_name, std::ios_base::app);
                auto cube = IRL::RectangularCuboid::fromBoundingPts(IRL::Pt(-0.5, -0.5, -0.5), IRL::Pt(0.5, 0.5, 0.5));
                auto surface_and_moments = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParaboloidParametrizedSurfaceOutput>>(cube, paraboloid);
                auto surface = surface_and_moments.getSurface();
                auto normal = surface.getAverageNormalNonAligned();

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
                if (!flip)
                {
                    normal[0] = -normal[0];
                    normal[1] = -normal[1];
                    normal[2] = -normal[2];
                }

                normals << normal[0] << "," << normal[1] << "," << normal[2] << "\n";
                normals.close();                  
            }  
        }; 

        void generate_with_disturbance(double rota_l, double rota_h, double rotb_l, double rotb_h, double rotc_l, double rotc_h, double coa_l, double coa_h, double cob_l, double cob_h, double ox_l, double ox_h, double oy_l, double oy_h, double oz_l, double oz_h, bool all)
        {
            srand((unsigned) time(NULL));
            for (int n = 0; n < Ntests; ++n) 
            {
                std::cout << n << endl;
                IRL::Paraboloid paraboloid = gen->new_random_parabaloid(rota_l, rota_h, rotb_l, rotb_h, rotc_l, rotc_h, coa_l, coa_h, cob_l, cob_h, ox_l, ox_h, oy_l, oy_h, oz_l, oz_h);
                angles = gen->getAngles();

                std::ofstream coefficients;
                std::string name = "coefficients.txt";
                coefficients.open(name, std::ios_base::app);
                coefficients << paraboloid.getDatum().x() << "," << paraboloid.getDatum().y() << "," << paraboloid.getDatum().z()
                << "," << angles[0] << "," << angles[1] << "," << angles[2]
                << "," << paraboloid.getAlignedParaboloid().a() << "," << paraboloid.getAlignedParaboloid().b() << "\n";
                coefficients.close();

                torch::Tensor result;
                bool flip = false;
                if (!all)
                {
                    result = gen->get_fractions(paraboloid, true);
                    if (result[((result.sizes()[0]-/*7*/4)/2)].item<double>() > 0.5)
                    {
                        flip = true;
                        result = gen->get_fractions_gas(paraboloid, true);
                    }
                }
                else
                {
                    result = gen->get_fractions_all(paraboloid);
                    if (result[((result.sizes()[0]-7)/2)].item<double>() > 0.5)
                    {
                        flip = true;
                        result = gen->get_fractions_gas_all(paraboloid);
                    }                    
                }

                std::vector<double> fractions;
                for (int i = 0; i < result.sizes()[0]; ++i)
                {
                    fractions.push_back(result[i].item<double>());
                }

                int direction = 0;
                std::vector<double> center;
                auto sm = IRL::spatial_moments();
                if (!all)
                {
                    center = sm.get_mass_centers(fractions);
                    direction = rotateFractions(&fractions,center);
                }
                else
                {
                    center = sm.get_mass_centers_all(&fractions);
                    direction = rotateFractions_all(&fractions,center);
                }

                std::ofstream output;
                std::string data_name = "fractions.txt";
                output.open(data_name, std::ios_base::app);

                int p = rand() % 8;//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                int mod = 4;
                if (all)
                {
                    mod = 7;
                }
                for (int i = 0; i < result.sizes()[0]; ++i)
                {
                    if (p == 0)
                    {
                        output << fractions[i] << ",";
                    }
                    else
                    {
                        double c = (rand() % 201 - 100) / 1000.0;
                        if (i % mod != 0 && fractions[i] > -9)
                        {
                            if (fractions[i] + c > 0.5)
                            {
                                fractions[i] = 0.5;
                                output << 0.5 << ",";
                            }
                            else if (fractions[i] + c < -0.5)
                            {
                                fractions[i] = -0.5;
                                output << -0.5 << ",";
                            }
                            else
                            {
                                output << fractions[i] + c << ",";
                                fractions[i] = fractions[i] + c;
                            }
                        }
                        else
                        {
                            output << fractions[i] << ",";
                        }
                    }
                }
                output << "\n";
                output.close();  

                auto cube = IRL::RectangularCuboid::fromBoundingPts(IRL::Pt(-0.5, -0.5, -0.5), IRL::Pt(0.5, 0.5, 0.5));
                auto surface_and_moments = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParaboloidParametrizedSurfaceOutput>>(cube, paraboloid);
                auto surface = surface_and_moments.getSurface();
                auto normal = surface.getAverageNormalNonAligned();
                double curv = surface.getAverageGaussianCurvature();

                IRL::Normal axis;
                IRL::Normal a_dir;
                IRL::Normal b_dir;
                axis = paraboloid.getReferenceFrame()[2];
                a_dir = paraboloid.getReferenceFrame()[0];
                b_dir = paraboloid.getReferenceFrame()[1];
                double co_a = paraboloid.getAlignedParaboloid().a();
                double co_b = paraboloid.getAlignedParaboloid().b();
                IRL::Pt origin = paraboloid.getDatum();
                IRL::UnitQuaternion rot;

                switch (direction)
                {
                    case 1:
                    normal[0] = -normal[0];
                    axis[0] = -axis[0];
                    a_dir[0] = -a_dir[0];
                    origin[0] = -origin[0];
                    break;
                    case 2:
                    normal[1] = -normal[1];
                    axis[1] = -axis[1];
                    a_dir[1] = -a_dir[1];
                    origin[1] = -origin[1];
                    break;
                    case 3:
                    normal[2] = -normal[2];
                    axis[2] = -axis[2];
                    a_dir[2] = -a_dir[2];
                    origin[2] = -origin[2];
                    break;
                    case 4:
                    normal[0] = -normal[0];
                    normal[1] = -normal[1];
                    axis[0] = -axis[0];
                    axis[1] = -axis[1];
                    a_dir[0] = -a_dir[0];
                    a_dir[1] = -a_dir[1];
                    origin[0] = -origin[0];
                    origin[1] = -origin[1]; 
                    break;
                    case 5:
                    normal[0] = -normal[0];
                    normal[2] = -normal[2];
                    axis[0] = -axis[0];
                    axis[2] = -axis[2];
                    a_dir[0] = -a_dir[0];
                    a_dir[2] = -a_dir[2];
                    origin[0] = -origin[0];
                    origin[2] = -origin[2]; 
                    break;
                    case 6:
                    normal[1] = -normal[1];
                    normal[2] = -normal[2];
                    axis[1] = -axis[1];
                    axis[2] = -axis[2];
                    a_dir[1] = -a_dir[1];
                    a_dir[2] = -a_dir[2];
                    origin[1] = -origin[1];
                    origin[2] = -origin[2]; 
                    break;
                    case 7:
                    normal[0] = -normal[0];
                    normal[1] = -normal[1];
                    normal[2] = -normal[2];
                    axis[0] = -axis[0];
                    axis[1] = -axis[1];
                    axis[2] = -axis[2];
                    a_dir[0] = -a_dir[0];
                    a_dir[1] = -a_dir[1];
                    a_dir[2] = -a_dir[2];   
                    origin[0] = -origin[0]; 
                    origin[1] = -origin[1];
                    origin[2] = -origin[2];       
                    break;
                }
                if (!flip)
                {
                    normal[0] = -normal[0];
                    normal[1] = -normal[1];
                    normal[2] = -normal[2];
                    axis[0] = -axis[0];
                    axis[1] = -axis[1];
                    axis[2] = -axis[2];
                    a_dir[0] = -a_dir[0];
                    a_dir[1] = -a_dir[1];
                    a_dir[2] = -a_dir[2];
                    co_a = -co_a;
                    co_b = -co_b;
                }
                b_dir = IRL::crossProduct(axis,a_dir);

                std::ofstream normals;
                std::string normals_name = "normals.txt";
                normals.open(normals_name, std::ios_base::app);
                normals << normal[0] << "," << normal[1] << "," << normal[2] << "\n";
                normals.close();   

                IRL::Normal par = ((origin[0]*normal[0] + origin[1]*normal[1] + origin[2]*normal[2])) * normal;
                IRL::Normal offset = origin - par;

                if (co_a < co_b)
                {
                    double temp = co_b;
                    co_b = co_a;
                    co_a = temp;
                    rot = IRL::UnitQuaternion(M_PI/2,axis);
                    a_dir = rot*a_dir;
                    b_dir = IRL::crossProduct(axis,a_dir);
                }

                if ((a_dir[0] < 0 && a_dir[1] < 0) || (a_dir[0] < 0 && a_dir[2] < 0) || (a_dir[1] < 0 && a_dir[2] < 0))
                {
                    rot = IRL::UnitQuaternion(M_PI,axis);
                    a_dir = rot*a_dir;
                    b_dir = IRL::crossProduct(axis,a_dir);
                }

                double n2 = axis[0]/(sqrt(axis[1]*axis[1]+axis[0]*axis[0]));
                double n1 = (-n2*axis[1])/axis[0];
                double theta = acos(n1*a_dir[0]+n2*a_dir[1]);
                if (sqrt(axis[1]*axis[1]+axis[0]*axis[0]) == 0 || axis[0] == 0)
                {
                    theta = 0;
                }
                // double theta2 = acos(n1*offset[0]+n2*offset[1]);
                rot = IRL::UnitQuaternion(theta,axis);
                IRL::Normal v1;
                v1[0] = n1; v1[1] = n2; v1[2] = 0;
                IRL::Normal v = rot*v1;
                // rot = IRL::UnitQuaternion(theta2,axis);
                // IRL::Normal v2 = rot*v1;
                if (abs(v[0]-a_dir[0]) >= 1e-8 && abs(v[1]-a_dir[1]) >= 1e-8)
                {
                    theta = -theta;
                } 
                if (theta < 0)
                {
                    theta = theta + 2*M_PI;
                }
                if (theta >= 2*M_PI)
                {
                    theta = theta - 2*M_PI;
                }
                // if (abs(v2[0]-offset[0]) >= 1e-8 && abs(v2[1]-offset[1]) >= 1e-8)
                // {
                //     theta2 = -theta2;
                // } 
                // if (theta2 < 0)
                // {
                //     theta2 = theta2 + 2*M_PI;
                // }
                // if (theta2 >= 2*M_PI)
                // {
                //     theta2 = theta2 - 2*M_PI;
                // }
                

                axis.normalize();
                std::ofstream input;
                // name = "input.txt";
                // input.open(name, std::ios_base::app);
                // input << axis[0] << "," << axis[1] << "," << axis[2] << "\n";
                // input.close();   

                a_dir.normalize();
                // name = "input1.txt";
                // input.open(name, std::ios_base::app);
                // input << co_a << "," << co_b << "," << a_dir[0] << "," << a_dir[1] << "," << a_dir[2] << "\n";
                // input.close();

                double a = co_a*origin[1]*origin[1] - origin[2] + co_b*origin[0]*origin[0] + co_a*origin[0]*origin[0]*pow(cos(theta),2.0) - co_a*origin[1]*origin[1]*pow(cos(theta),2.0) - co_b*origin[0]*origin[0]*pow(cos(theta),2.0) + co_b*origin[1]*origin[1]*pow(cos(theta),2.0) + co_a*origin[0]*origin[1]*sin(2*theta) - co_b*origin[0]*origin[1]*sin(2*theta);
                
                double b = co_a*origin[0] + co_b*origin[0] + co_a*origin[0]*cos(2*theta) - co_b*origin[0]*cos(2*theta) + co_a*origin[1]*sin(2*theta) - co_b*origin[1]*sin(2*theta);
                double c = co_a*origin[1] + co_b*origin[1] - co_a*origin[1]*cos(2*theta) + co_b*origin[1]*cos(2*theta) + co_a*origin[0]*sin(2*theta) - co_b*origin[0]*sin(2*theta);

                double d = co_a - co_a * pow(sin(theta),2.0) + co_b * pow(sin(theta),2.0);
                double f = co_b - co_b * pow(sin(theta),2.0) + co_a * pow(sin(theta),2.0);
                double e = (co_a-co_b) * sin(2.0*theta);

                // name = "input1.txt";
                // input.open(name, std::ios_base::app);
                // input << co_a << "," << co_b << "," << (co_a-co_b)*theta << "\n";
                // input.close();

                // name = "input1.txt";
                // input.open(name, std::ios_base::app);
                // input << d << "," << f << "," << e << "\n";
                // input.close();

                // name = "input2.txt";
                // input.open(name, std::ios_base::app);
                // input << origin[0] << "," << origin[1] << "," << origin[2] << "\n";
                // input.close();

                // name = "input3.txt";
                // input.open(name, std::ios_base::app);
                // input << a << "," << b << "," << c << "\n";
                // input.close();

                // name = "input4.txt";
                // input.open(name, std::ios_base::app);
                // input << offset[0] << "," << offset[1] << "," << offset[2] << "\n";
                // input.close();

                // name = "input5.txt";
                // input.open(name, std::ios_base::app);
                // input << IRL::magnitude(offset) << "\n";
                // input.close();

                // name = "input6.txt";
                // input.open(name, std::ios_base::app);
                // input << theta2 << "\n";
                // input.close();

                // data_name = "fractions_w_coeffs.txt";
                // output.open(data_name, std::ios_base::app);

                // output << d << "," << e << "," << f << "," << normal[0] << "," << normal[1] << "," << normal[2] << ",";
                // for (int i = 0; i < result.sizes()[0]; ++i)
                // {
                //     output << fractions[i] << ",";
                // }
                // output << "\n";
                // output.close();  
            }           
        };

        void generate_sub_grid(double rota_l, double rota_h, double rotb_l, double rotb_h, double rotc_l, double rotc_h, double coa_l, double coa_h, double cob_l, double cob_h, double ox_l, double ox_h, double oy_l, double oy_h, double oz_l, double oz_h, bool all)
        {
            srand((unsigned) time(NULL));
            for (int n = 0; n < Ntests; ++n) 
            {
                std::cout << n << endl;
                IRL::Paraboloid paraboloid = gen->new_random_sub_grid2_parabaloid(rota_l, rota_h, rotb_l, rotb_h, rotc_l, rotc_h, coa_l, coa_h, ox_l, ox_h, oy_l, oy_h, oz_l, oz_h);
                angles = gen->getAngles();

                std::ofstream coefficients;
                std::string name = "coefficients.txt";
                coefficients.open(name, std::ios_base::app);
                coefficients << paraboloid.getDatum().x() << "," << paraboloid.getDatum().y() << "," << paraboloid.getDatum().z()
                << "," << angles[0] << "," << angles[1] << "," << angles[2]
                << "," << paraboloid.getAlignedParaboloid().a() << "," << paraboloid.getAlignedParaboloid().b() << "\n";
                coefficients.close();

                torch::Tensor result;
                bool flip = false;
                if (!all)
                {
                    result = gen->get_fractions(paraboloid, true);
                    if (result[((result.sizes()[0]-/*7*/4)/2)].item<double>() > 0.5)
                    {
                        flip = true;
                        result = gen->get_fractions_gas(paraboloid, true);
                    }
                }
                else
                {
                    result = gen->get_fractions_all(paraboloid);
                    if (result[((result.sizes()[0]-7)/2)].item<double>() > 0.5)
                    {
                        flip = true;
                        result = gen->get_fractions_gas_all(paraboloid);
                    }                    
                }

                std::vector<double> fractions;
                for (int i = 0; i < result.sizes()[0]; ++i)
                {
                    fractions.push_back(result[i].item<double>());
                }

                int direction = 0;
                std::vector<double> center;
                auto sm = IRL::spatial_moments();
                if (!all)
                {
                    center = sm.get_mass_centers(fractions);
                    direction = rotateFractions(&fractions,center);
                }
                else
                {
                    center = sm.get_mass_centers_all(&fractions);
                    direction = rotateFractions_all(&fractions,center);
                }

                std::ofstream output;
                std::string data_name = "fractions.txt";
                output.open(data_name, std::ios_base::app);

                int p = 0;//rand() % 8;//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                int mod = 4;
                if (all)
                {
                    mod = 7;
                }
                for (int i = 0; i < result.sizes()[0]; ++i)
                {
                    if (p == 0)
                    {
                        output << fractions[i] << ",";
                    }
                    else
                    {
                        double c = (rand() % 201 - 100) / 1000.0;
                        if (i % mod != 0 && fractions[i] > -9)
                        {
                            if (fractions[i] + c > 0.5)
                            {
                                fractions[i] = 0.5;
                                output << 0.5 << ",";
                            }
                            else if (fractions[i] + c < -0.5)
                            {
                                fractions[i] = -0.5;
                                output << -0.5 << ",";
                            }
                            else
                            {
                                output << fractions[i] + c << ",";
                                fractions[i] = fractions[i] + c;
                            }
                        }
                        else
                        {
                            output << fractions[i] << ",";
                        }
                    }
                }
                output << "\n";
                output.close();  

                auto cube = IRL::RectangularCuboid::fromBoundingPts(IRL::Pt(-0.5, -0.5, -0.5), IRL::Pt(0.5, 0.5, 0.5));
                auto surface_and_moments = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParaboloidParametrizedSurfaceOutput>>(cube, paraboloid);
                auto surface = surface_and_moments.getSurface();
                auto normal = surface.getAverageNormalNonAligned();
                double curv = surface.getAverageGaussianCurvature();

                IRL::Normal axis;
                IRL::Normal a_dir;
                IRL::Normal b_dir;
                axis = paraboloid.getReferenceFrame()[2];
                a_dir = paraboloid.getReferenceFrame()[0];
                b_dir = paraboloid.getReferenceFrame()[1];
                double co_a = paraboloid.getAlignedParaboloid().a();
                double co_b = paraboloid.getAlignedParaboloid().b();
                IRL::Pt origin = paraboloid.getDatum();
                IRL::UnitQuaternion rot;

                switch (direction)
                {
                    case 1:
                    normal[0] = -normal[0];
                    axis[0] = -axis[0];
                    a_dir[0] = -a_dir[0];
                    origin[0] = -origin[0];
                    break;
                    case 2:
                    normal[1] = -normal[1];
                    axis[1] = -axis[1];
                    a_dir[1] = -a_dir[1];
                    origin[1] = -origin[1];
                    break;
                    case 3:
                    normal[2] = -normal[2];
                    axis[2] = -axis[2];
                    a_dir[2] = -a_dir[2];
                    origin[2] = -origin[2];
                    break;
                    case 4:
                    normal[0] = -normal[0];
                    normal[1] = -normal[1];
                    axis[0] = -axis[0];
                    axis[1] = -axis[1];
                    a_dir[0] = -a_dir[0];
                    a_dir[1] = -a_dir[1];
                    origin[0] = -origin[0];
                    origin[1] = -origin[1]; 
                    break;
                    case 5:
                    normal[0] = -normal[0];
                    normal[2] = -normal[2];
                    axis[0] = -axis[0];
                    axis[2] = -axis[2];
                    a_dir[0] = -a_dir[0];
                    a_dir[2] = -a_dir[2];
                    origin[0] = -origin[0];
                    origin[2] = -origin[2]; 
                    break;
                    case 6:
                    normal[1] = -normal[1];
                    normal[2] = -normal[2];
                    axis[1] = -axis[1];
                    axis[2] = -axis[2];
                    a_dir[1] = -a_dir[1];
                    a_dir[2] = -a_dir[2];
                    origin[1] = -origin[1];
                    origin[2] = -origin[2]; 
                    break;
                    case 7:
                    normal[0] = -normal[0];
                    normal[1] = -normal[1];
                    normal[2] = -normal[2];
                    axis[0] = -axis[0];
                    axis[1] = -axis[1];
                    axis[2] = -axis[2];
                    a_dir[0] = -a_dir[0];
                    a_dir[1] = -a_dir[1];
                    a_dir[2] = -a_dir[2];   
                    origin[0] = -origin[0]; 
                    origin[1] = -origin[1];
                    origin[2] = -origin[2];       
                    break;
                }
                if (!flip)
                {
                    normal[0] = -normal[0];
                    normal[1] = -normal[1];
                    normal[2] = -normal[2];
                    axis[0] = -axis[0];
                    axis[1] = -axis[1];
                    axis[2] = -axis[2];
                    a_dir[0] = -a_dir[0];
                    a_dir[1] = -a_dir[1];
                    a_dir[2] = -a_dir[2];
                    co_a = -co_a;
                    co_b = -co_b;
                }
                b_dir = IRL::crossProduct(axis,a_dir);

                std::ofstream normals;
                std::string normals_name = "normals.txt";
                normals.open(normals_name, std::ios_base::app);
                normals << normal[0] << "," << normal[1] << "," << normal[2] << "\n";
                normals.close();   

                IRL::Normal par = ((origin[0]*normal[0] + origin[1]*normal[1] + origin[2]*normal[2])) * normal;
                IRL::Normal offset = origin - par;

                // if (co_a < co_b)
                // {
                //     double temp = co_b;
                //     co_b = co_a;
                //     co_a = temp;
                //     rot = IRL::UnitQuaternion(M_PI/2,axis);
                //     a_dir = rot*a_dir;
                //     b_dir = IRL::crossProduct(axis,a_dir);
                // }

                if ((a_dir[0] < 0 && a_dir[1] < 0) || (a_dir[0] < 0 && a_dir[2] < 0) || (a_dir[1] < 0 && a_dir[2] < 0))
                {
                    rot = IRL::UnitQuaternion(M_PI,axis);
                    a_dir = rot*a_dir;
                    b_dir = IRL::crossProduct(axis,a_dir);
                }

                double n2 = axis[0]/(sqrt(axis[1]*axis[1]+axis[0]*axis[0]));
                double n1 = (-n2*axis[1])/axis[0];
                double theta = acos(n1*a_dir[0]+n2*a_dir[1]);
                if (sqrt(axis[1]*axis[1]+axis[0]*axis[0]) == 0 || axis[0] == 0)
                {
                    theta = 0;
                }
                rot = IRL::UnitQuaternion(theta,axis);
                IRL::Normal v1;
                v1[0] = n1; v1[1] = n2; v1[2] = 0;
                IRL::Normal v = rot*v1;
                if (abs(v[0]-a_dir[0]) >= 1e-8 && abs(v[1]-a_dir[1]) >= 1e-8)
                {
                    theta = -theta;
                } 
                if (theta < 0)
                {
                    theta = theta + 2*M_PI;
                }
                if (theta >= 2*M_PI)
                {
                    theta = theta - 2*M_PI;
                }
                

                axis.normalize();
                std::ofstream input;
                name = "input.txt";
                input.open(name, std::ios_base::app);
                input << axis[0] << "," << axis[1] << "," << axis[2] << "\n";
                input.close();   

                a_dir.normalize();

                double a = co_a*origin[1]*origin[1] - origin[2] + co_b*origin[0]*origin[0] + co_a*origin[0]*origin[0]*pow(cos(theta),2.0) - co_a*origin[1]*origin[1]*pow(cos(theta),2.0) - co_b*origin[0]*origin[0]*pow(cos(theta),2.0) + co_b*origin[1]*origin[1]*pow(cos(theta),2.0) + co_a*origin[0]*origin[1]*sin(2*theta) - co_b*origin[0]*origin[1]*sin(2*theta);
                
                double b = co_a*origin[0] + co_b*origin[0] + co_a*origin[0]*cos(2*theta) - co_b*origin[0]*cos(2*theta) + co_a*origin[1]*sin(2*theta) - co_b*origin[1]*sin(2*theta);
                double c = co_a*origin[1] + co_b*origin[1] - co_a*origin[1]*cos(2*theta) + co_b*origin[1]*cos(2*theta) + co_a*origin[0]*sin(2*theta) - co_b*origin[0]*sin(2*theta);

                double d = co_a - co_a * pow(sin(theta),2.0) + co_b * pow(sin(theta),2.0);
                double f = co_b - co_b * pow(sin(theta),2.0) + co_a * pow(sin(theta),2.0);
                double e = (co_a-co_b) * sin(2.0*theta);

                // name = "input1.txt";
                // input.open(name, std::ios_base::app);
                // input << d << "," << f << "," << e << "\n";
                // input.close();

                name = "input1.txt";
                input.open(name, std::ios_base::app);
                if (co_a < 0)
                {
                    input << -2*sqrt(1/-co_a) << "," << theta << "\n";
                }
                else
                {
                    input << 2*sqrt(1/co_a) << "," << theta << "\n";
                }
                input.close();

                name = "input2.txt";
                input.open(name, std::ios_base::app);
                input << origin[0] << "," << origin[1] << "," << origin[2] << "\n";
                input.close();   

                // name = "fractions_origin.txt";
                // input.open(name, std::ios_base::app);
                // if (co_a < 0)
                // {
                //     input << fractions[(fractions.size()-7)/2] << "," << fractions[(fractions.size()-7)/2+1] << "," << fractions[(fractions.size()-7)/2+2] << "," << fractions[(fractions.size()-7)/2+3] << "," << fractions[(fractions.size()-7)/2+4] << "," << fractions[(fractions.size()-7)/2+5] << "," << fractions[(fractions.size()-7)/2+6] << "," << axis[0] << "," << axis[1] << "," << axis[2] << "," << -2*sqrt(1/-co_a) << "," << theta << "," << "\n";
                // }
                // else
                // {
                //     input << fractions[(fractions.size()-7)/2] << "," << fractions[(fractions.size()-7)/2+1] << "," << fractions[(fractions.size()-7)/2+2] << "," << fractions[(fractions.size()-7)/2+3] << "," << fractions[(fractions.size()-7)/2+4] << "," << fractions[(fractions.size()-7)/2+5] << "," << fractions[(fractions.size()-7)/2+6] << "," << axis[0] << "," << axis[1] << "," << axis[2] << "," << 2*sqrt(1/co_a) << "," << theta << "," << "\n";
                // }
                // input.close();  
            }           
        };

        void generate_two_paraboloids(double rota_l, double rota_h, double rotb_l, double rotb_h, double rotc_l, double rotc_h, double coa_l, double coa_h, double cob_l, double cob_h, double ox_l, double ox_h, double oy_l, double oy_h, double oz_l, double oz_h, bool all)
        {
            for (int n = 0; n < Ntests; ++n) 
            {
                std::cout << n << endl;
                IRL::Paraboloid paraboloid = gen->new_random_parabaloid(rota_l, rota_h, rotb_l, rotb_h, rotc_l, rotc_h, coa_l, coa_h, cob_l, cob_h, ox_l, ox_h, oy_l, oy_h, oz_l, oz_h);
                IRL::Paraboloid interface = gen->new_interface_parabaloid(rota_l, rota_h, rotb_l, rotb_h, rotc_l, rotc_h, coa_l, coa_h, cob_l, cob_h, ox_l, ox_h, oy_l, oy_h, oz_l, oz_h, paraboloid);
                angles = gen->getAngles();

                std::ofstream coefficients;
                std::string name = "coefficients.txt";
                coefficients.open(name, std::ios_base::app);
                coefficients << paraboloid.getDatum().x() << "," << paraboloid.getDatum().y() << "," << paraboloid.getDatum().z()
                << "," << angles[0] << "," << angles[1] << "," << angles[2]
                << "," << paraboloid.getAlignedParaboloid().a() << "," << paraboloid.getAlignedParaboloid().b() << "\n";
                coefficients.close();

                std::ofstream coefficients2;
                std::string name2 = "coefficients2.txt";
                coefficients2.open(name2, std::ios_base::app);
                coefficients2 << interface.getDatum().x() << "," << interface.getDatum().y() << "," << interface.getDatum().z()
                << "," << angles[0] << "," << angles[1] << "," << angles[2]
                << "," << interface.getAlignedParaboloid().a() << "," << interface.getAlignedParaboloid().b() << "\n";
                coefficients2.close();

                std::ofstream classification;
                std::string data_name = "type.txt";
                classification.open(data_name, std::ios_base::app);
                if ((abs(paraboloid.getAlignedParaboloid().a() - paraboloid.getAlignedParaboloid().b()) > 1) && (paraboloid.getAlignedParaboloid().a() < 0.2 || paraboloid.getAlignedParaboloid().b() < 0.2))
                {
                    classification << "0,1,0" << " \n";
                }
                else if ((abs(paraboloid.getAlignedParaboloid().a() - paraboloid.getAlignedParaboloid().b()) < 1) && (paraboloid.getAlignedParaboloid().a() > 2 || paraboloid.getAlignedParaboloid().b() > 2))
                {
                    classification << "1,0,0" << " \n";
                }
                else
                {
                    classification << "0,0,1" << " \n";
                }
                classification.close();

                std::ofstream inter;
                std::string interface_name = "interface.txt";
                inter.open(interface_name, std::ios_base::app);
                inter << "0,1" << " \n";
                inter.close();

                torch::Tensor result;
                torch::Tensor result1;
                bool flip = false;
                bool option = false;
                bool type = true;
                if (!all)
                {
                    result = gen->get_fractions(paraboloid, true);
                    result1 = gen->get_fractions(interface, true);
                    if (result[((result.sizes()[0]-/*7*/4)/2)].item<double>() > 0.5)
                    {
                        flip = true;
                        result = gen->get_fractions_gas(paraboloid, true);
                        result1 = gen->get_fractions_gas(interface, true);
                    }
                }
                else
                {
                    result = gen->get_fractions_all(paraboloid);
                    result1 = gen->get_fractions_all(interface);
                    if (result[((result.sizes()[0]-7)/2)].item<double>() > 0.5)
                    {
                        flip = true;
                        result = gen->get_fractions_gas_all(paraboloid);
                        result1 = gen->get_fractions_gas_all(interface);
                    }                    
                }

                std::vector<double> fractions;
                int mod = 4;
                if (all)
                {
                    mod = 7;
                }
                
                for (int i = 0; i < result.sizes()[0]; ++i)
                {
                    if (i%mod == 0)
                    {
                        if (result[i].item<double>() > IRL::global_constants::VF_LOW && result1[i].item<double>() > IRL::global_constants::VF_LOW)
                        {
                            type = false;
                            break;
                        }
                    }
                }
                for (int i = 0; i < result.sizes()[0]; ++i)
                {
                    if (type)
                    {
                        if (i%mod == 0)
                        {
                            if (result1[i].item<double>() > IRL::global_constants::VF_LOW)
                            {
                                option = true;
                            }
                            else
                            {
                                option = false;
                            }
                        }
                        if (option)
                        {
                            fractions.push_back(result1[i].item<double>());
                        }
                        else
                        {
                            fractions.push_back(result[i].item<double>());
                        }
                    }
                    else
                    {
                        if (i%mod == 0)
                        {
                            if (result1[i].item<double>() < IRL::global_constants::VF_HIGH)
                            {
                                option = true;
                            }
                        }
                        if (option)
                        {
                            fractions.push_back(result1[i].item<double>());
                        }
                        else
                        {
                            fractions.push_back(result[i].item<double>());
                        }
                    }
                }

                int direction = 0;
                std::vector<double> center;
                auto sm = IRL::spatial_moments();
                if (!all)
                {
                    center = sm.get_mass_centers(fractions);
                    direction = rotateFractions(&fractions,center);
                }
                else
                {
                    center = sm.get_mass_centers_all(&fractions);
                    direction = rotateFractions_all(&fractions,center);
                }

                std::ofstream output;
                data_name = "fractions.txt";
                output.open(data_name, std::ios_base::app);

                for (int i = 0; i < result.sizes()[0]; ++i)
                {
                    output << fractions[i] << ",";
                }
                output << "\n";
                output.close();  

                std::ofstream normals;
                std::string normals_name = "normals.txt";
                normals.open(normals_name, std::ios_base::app);
                auto cube = IRL::RectangularCuboid::fromBoundingPts(IRL::Pt(-0.5, -0.5, -0.5), IRL::Pt(0.5, 0.5, 0.5));
                auto surface_and_moments = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParaboloidParametrizedSurfaceOutput>>(cube, paraboloid);
                auto surface_and_moments1 = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParaboloidParametrizedSurfaceOutput>>(cube, interface);
                auto surface = surface_and_moments.getSurface();
                auto surface1 = surface_and_moments1.getSurface();
                auto normal = surface.getAverageNormalNonAligned();
                auto normal1 = surface1.getAverageNormalNonAligned();

                switch (direction)
                {
                    case 1:
                    normal[0] = -normal[0];
                    normal1[0] = -normal1[0];
                    break;
                    case 2:
                    normal[1] = -normal[1];
                    normal1[1] = -normal1[1];
                    break;
                    case 3:
                    normal[2] = -normal[2];
                    normal1[2] = -normal1[2];
                    break;
                    case 4:
                    normal[0] = -normal[0];
                    normal[1] = -normal[1];
                    normal1[0] = -normal1[0];
                    normal1[1] = -normal1[1];
                    break;
                    case 5:
                    normal[0] = -normal[0];
                    normal[2] = -normal[2];
                    normal1[0] = -normal1[0];
                    normal1[2] = -normal1[2];
                    break;
                    case 6:
                    normal[1] = -normal[1];
                    normal[2] = -normal[2];
                    normal1[1] = -normal1[1];
                    normal1[2] = -normal1[2];
                    break;
                    case 7:
                    normal[0] = -normal[0];
                    normal[1] = -normal[1];
                    normal[2] = -normal[2];
                    normal1[0] = -normal1[0];
                    normal1[1] = -normal1[1];
                    normal1[2] = -normal1[2];
                    break;
                }
                if (!flip)
                {
                    normal[0] = -normal[0];
                    normal[1] = -normal[1];
                    normal[2] = -normal[2];
                    normal1[0] = -normal1[0];
                    normal1[1] = -normal1[1];
                    normal1[2] = -normal1[2];
                }

                normals << normal[0] << "," << normal[1] << "," << normal[2] << "," << normal1[0] << "," << normal1[1] << "," << normal1[2] << "\n";
                normals.close();      

                // const auto bottom_corner = IRL::Pt(-1.5, -1.5, -1.5);
                // const auto top_corner = IRL::Pt(1.5, 1.5, 1.5);
                // const auto cell = IRL::StoredRectangularCuboid<IRL::Pt>::fromBoundingPts(bottom_corner, top_corner);

                // const auto first_moments_and_surface = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParaboloidParametrizedSurfaceOutput>, IRL::HalfEdgeCutting>(cell, paraboloid);
                // const auto first_moments_and_surface2 = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParaboloidParametrizedSurfaceOutput>, IRL::HalfEdgeCutting>(cell, interface);
                // const double length_scale = 0.05;
                // IRL::TriangulatedSurfaceOutput triangulated_surface = first_moments_and_surface.getSurface().triangulate(length_scale);
                // IRL::TriangulatedSurfaceOutput triangulated_surface2 = first_moments_and_surface2.getSurface().triangulate(length_scale);
                // string name3 = "p";
                // string name4 = "i";
                // triangulated_surface.write(name3);
                // triangulated_surface2.write(name4);       
            }       
        };

        void generate_two_paraboloids_with_disturbance(double rota_l, double rota_h, double rotb_l, double rotb_h, double rotc_l, double rotc_h, double coa_l, double coa_h, double cob_l, double cob_h, double ox_l, double ox_h, double oy_l, double oy_h, double oz_l, double oz_h, bool all)
        {
            srand((unsigned) time(NULL));
            for (int n = 0; n < Ntests; ++n) 
            {
                std::cout << n << endl;
                IRL::Paraboloid paraboloid = gen->new_random_parabaloid(rota_l, rota_h, rotb_l, rotb_h, rotc_l, rotc_h, coa_l, coa_h, cob_l, cob_h, ox_l, ox_h, oy_l, oy_h, oz_l, oz_h);
                IRL::Paraboloid interface = gen->new_interface_parabaloid(rota_l, rota_h, rotb_l, rotb_h, rotc_l, rotc_h, coa_l, coa_h, cob_l, cob_h, ox_l, ox_h, oy_l, oy_h, oz_l, oz_h, paraboloid);
                angles = gen->getAngles();

                std::ofstream coefficients;
                std::string name = "coefficients.txt";
                coefficients.open(name, std::ios_base::app);
                coefficients << paraboloid.getDatum().x() << "," << paraboloid.getDatum().y() << "," << paraboloid.getDatum().z()
                << "," << angles[0] << "," << angles[1] << "," << angles[2]
                << "," << paraboloid.getAlignedParaboloid().a() << "," << paraboloid.getAlignedParaboloid().b() << "\n";
                coefficients.close();

                std::ofstream coefficients2;
                std::string name2 = "coefficients2.txt";
                coefficients2.open(name2, std::ios_base::app);
                coefficients2 << interface.getDatum().x() << "," << interface.getDatum().y() << "," << interface.getDatum().z()
                << "," << angles[0] << "," << angles[1] << "," << angles[2]
                << "," << interface.getAlignedParaboloid().a() << "," << interface.getAlignedParaboloid().b() << "\n";
                coefficients2.close();

                std::ofstream classification;
                std::string data_name = "type.txt";
                classification.open(data_name, std::ios_base::app);
                if ((abs(paraboloid.getAlignedParaboloid().a() - paraboloid.getAlignedParaboloid().b()) > 1) && (paraboloid.getAlignedParaboloid().a() < 0.2 || paraboloid.getAlignedParaboloid().b() < 0.2))
                {
                    classification << "0,1,0" << " \n";
                }
                else if ((abs(paraboloid.getAlignedParaboloid().a() - paraboloid.getAlignedParaboloid().b()) < 1) && (paraboloid.getAlignedParaboloid().a() > 2 || paraboloid.getAlignedParaboloid().b() > 2))
                {
                    classification << "1,0,0" << " \n";
                }
                else
                {
                    classification << "0,0,1" << " \n";
                }
                classification.close();

                std::ofstream inter;
                std::string interface_name = "interface.txt";
                inter.open(interface_name, std::ios_base::app);
                inter << "0,1" << " \n";
                inter.close();

                torch::Tensor result;
                torch::Tensor result1;
                bool flip = false;
                bool option = false;
                bool type = true;
                if (!all)
                {
                    result = gen->get_fractions(paraboloid, true);
                    result1 = gen->get_fractions(interface, true);
                    if (result[((result.sizes()[0]-/*7*/4)/2)].item<double>() > 0.5)
                    {
                        flip = true;
                        result = gen->get_fractions_gas(paraboloid, true);
                        result1 = gen->get_fractions_gas(interface, true);
                    }
                }
                else
                {
                    result = gen->get_fractions_all(paraboloid);
                    result1 = gen->get_fractions_all(interface);
                    if (result[((result.sizes()[0]-7)/2)].item<double>() > 0.5)
                    {
                        flip = true;
                        result = gen->get_fractions_gas_all(paraboloid);
                        result1 = gen->get_fractions_gas_all(interface);
                    }                    
                }

                std::vector<double> fractions;
                int mod = 4;
                if (all)
                {
                    mod = 7;
                }
                
                for (int i = 0; i < result.sizes()[0]; ++i)
                {
                    if (i%mod == 0)
                    {
                        if (result[i].item<double>() > IRL::global_constants::VF_LOW && result1[i].item<double>() > IRL::global_constants::VF_LOW)
                        {
                            type = false;
                            break;
                        }
                    }
                }
                for (int i = 0; i < result.sizes()[0]; ++i)
                {
                    if (type)
                    {
                        if (i%mod == 0)
                        {
                            if (result1[i].item<double>() > IRL::global_constants::VF_LOW)
                            {
                                option = true;
                            }
                            else
                            {
                                option = false;
                            }
                        }
                        if (option)
                        {
                            fractions.push_back(result1[i].item<double>());
                        }
                        else
                        {
                            fractions.push_back(result[i].item<double>());
                        }
                    }
                    else
                    {
                        if (i%mod == 0)
                        {
                            if (result1[i].item<double>() < IRL::global_constants::VF_HIGH)
                            {
                                option = true;
                            }
                        }
                        if (option)
                        {
                            fractions.push_back(result1[i].item<double>());
                        }
                        else
                        {
                            fractions.push_back(result[i].item<double>());
                        }
                    }
                }

                int direction = 0;
                std::vector<double> center;
                auto sm = IRL::spatial_moments();
                if (!all)
                {
                    center = sm.get_mass_centers(fractions);
                    direction = rotateFractions(&fractions,center);
                }
                else
                {
                    center = sm.get_mass_centers_all(&fractions);
                    direction = rotateFractions_all(&fractions,center);
                }

                std::ofstream output;
                data_name = "fractions.txt";
                output.open(data_name, std::ios_base::app);

                int p = rand() % 8;
                for (int i = 0; i < result.sizes()[0]; ++i)
                {
                    if (p == 0)
                    {
                        output << fractions[i] << ",";
                    }
                    else
                    {
                        //int r = rand() % 3 - 1;
                        double c = (rand() % 201 - 100) / 1000.0;
                        if (i % mod != 0)
                        {
                            if (fractions[i] + c > 0.5)
                            {
                                output << 0.5 << ",";
                            }
                            else if (fractions[i] + c < -0.5)
                            {
                                output << -0.5 << ",";
                            }
                            else
                            {
                                output << fractions[i] + c << ",";
                            }
                        }
                        else
                        {
                            output << fractions[i] << ",";
                        }
                    }
                }
                output << "\n";
                output.close(); 

                std::ofstream normals;
                std::string normals_name = "normals.txt";
                normals.open(normals_name, std::ios_base::app);
                auto cube = IRL::RectangularCuboid::fromBoundingPts(IRL::Pt(-0.5, -0.5, -0.5), IRL::Pt(0.5, 0.5, 0.5));
                auto surface_and_moments = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParaboloidParametrizedSurfaceOutput>>(cube, paraboloid);
                auto surface_and_moments1 = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParaboloidParametrizedSurfaceOutput>>(cube, interface);
                auto surface = surface_and_moments.getSurface();
                auto surface1 = surface_and_moments1.getSurface();
                auto normal = surface.getAverageNormalNonAligned();
                auto normal1 = surface1.getAverageNormalNonAligned();

                switch (direction)
                {
                    case 1:
                    normal[0] = -normal[0];
                    normal1[0] = -normal1[0];
                    break;
                    case 2:
                    normal[1] = -normal[1];
                    normal1[1] = -normal1[1];
                    break;
                    case 3:
                    normal[2] = -normal[2];
                    normal1[2] = -normal1[2];
                    break;
                    case 4:
                    normal[0] = -normal[0];
                    normal[1] = -normal[1];
                    normal1[0] = -normal1[0];
                    normal1[1] = -normal1[1];
                    break;
                    case 5:
                    normal[0] = -normal[0];
                    normal[2] = -normal[2];
                    normal1[0] = -normal1[0];
                    normal1[2] = -normal1[2];
                    break;
                    case 6:
                    normal[1] = -normal[1];
                    normal[2] = -normal[2];
                    normal1[1] = -normal1[1];
                    normal1[2] = -normal1[2];
                    break;
                    case 7:
                    normal[0] = -normal[0];
                    normal[1] = -normal[1];
                    normal[2] = -normal[2];
                    normal1[0] = -normal1[0];
                    normal1[1] = -normal1[1];
                    normal1[2] = -normal1[2];
                    break;
                }
                if (!flip)
                {
                    normal[0] = -normal[0];
                    normal[1] = -normal[1];
                    normal[2] = -normal[2];
                    normal1[0] = -normal1[0];
                    normal1[1] = -normal1[1];
                    normal1[2] = -normal1[2];
                }

                normals << normal[0] << "," << normal[1] << "," << normal[2]/* << "," << normal1[0] << "," << normal1[1] << "," << normal1[2]*/ << "\n";
                normals.close();      

                /*const auto bottom_corner = IRL::Pt(-1.5, -1.5, -1.5);
                const auto top_corner = IRL::Pt(1.5, 1.5, 1.5);
                const auto cell = IRL::StoredRectangularCuboid<IRL::Pt>::fromBoundingPts(bottom_corner, top_corner);

                const auto first_moments_and_surface = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParaboloidParametrizedSurfaceOutput>, IRL::HalfEdgeCutting>(cell, paraboloid);
                const auto first_moments_and_surface2 = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParaboloidParametrizedSurfaceOutput>, IRL::HalfEdgeCutting>(cell, interface);
                const double length_scale = 0.05;
                IRL::TriangulatedSurfaceOutput triangulated_surface = first_moments_and_surface.getSurface().triangulate(length_scale);
                IRL::TriangulatedSurfaceOutput triangulated_surface2 = first_moments_and_surface2.getSurface().triangulate(length_scale);
                string name3 = "p";
                string name4 = "i";
                triangulated_surface.write(name3);
                triangulated_surface2.write(name4);    */   
            }
        };

        void generate_two_paraboloids_in_cell(double rota_l, double rota_h, double rotb_l, double rotb_h, double rotc_l, double rotc_h, double coa_l, double coa_h, double cob_l, double cob_h, double ox_l, double ox_h, double oy_l, double oy_h, double oz_l, double oz_h, bool all)
        {
            for (int n = 0; n < Ntests; ++n) 
            {
                std::cout << n << endl;
                IRL::Paraboloid paraboloid = gen->new_random_parabaloid(rota_l, rota_h, rotb_l, rotb_h, rotc_l, rotc_h, coa_l, coa_h, cob_l, cob_h, ox_l, ox_h, oy_l, oy_h, oz_l, oz_h);
                IRL::Paraboloid interface = gen->new_interface_parabaloid_in_cell(rota_l, rota_h, rotb_l, rotb_h, rotc_l, rotc_h, coa_l, coa_h, cob_l, cob_h, ox_l, ox_h, oy_l, oy_h, oz_l, oz_h, paraboloid);
                angles = gen->getAngles();

                std::ofstream coefficients;
                std::string name = "coefficients.txt";
                coefficients.open(name, std::ios_base::app);
                coefficients << paraboloid.getDatum().x() << "," << paraboloid.getDatum().y() << "," << paraboloid.getDatum().z()
                << "," << angles[0] << "," << angles[1] << "," << angles[2]
                << "," << paraboloid.getAlignedParaboloid().a() << "," << paraboloid.getAlignedParaboloid().b() << "\n";
                coefficients.close();

                std::ofstream coefficients2;
                std::string name2 = "coefficients2.txt";
                coefficients2.open(name2, std::ios_base::app);
                coefficients2 << interface.getDatum().x() << "," << interface.getDatum().y() << "," << interface.getDatum().z()
                << "," << angles[0] << "," << angles[1] << "," << angles[2]
                << "," << interface.getAlignedParaboloid().a() << "," << interface.getAlignedParaboloid().b() << "\n";
                coefficients2.close();

                // std::ofstream classification;
                // std::string data_name = "type.txt";
                // classification.open(data_name, std::ios_base::app);
                // if ((abs(paraboloid.getAlignedParaboloid().a() - paraboloid.getAlignedParaboloid().b()) > 1) && (paraboloid.getAlignedParaboloid().a() < 0.2 || paraboloid.getAlignedParaboloid().b() < 0.2))
                // {
                //     classification << "0,1,0" << " \n";
                // }
                // else if ((abs(paraboloid.getAlignedParaboloid().a() - paraboloid.getAlignedParaboloid().b()) < 1) && (paraboloid.getAlignedParaboloid().a() > 2 || paraboloid.getAlignedParaboloid().b() > 2))
                // {
                //     classification << "1,0,0" << " \n";
                // }
                // else
                // {
                //     classification << "0,0,1" << " \n";
                // }
                // classification.close();
                std::ofstream type1;
                name = "type.txt";
                type1.open(name, std::ios_base::app);
                type1 << std::to_string(0) << "\n";
                //type1 << std::to_string(1) << "\n";
                type1.close();

                torch::Tensor result;
                torch::Tensor result1;
                bool flip = false;
                bool option = false;
                bool type = true;
                bool same_cell = false;
                if (!all)
                {
                    result = gen->get_fractions(paraboloid, true);
                    result1 = gen->get_fractions(interface, true);
                    if (result[((result.sizes()[0]-/*7*/4)/2)].item<double>() > 0.5)
                    {
                        flip = true;
                        result = gen->get_fractions_gas(paraboloid, true);
                        result1 = gen->get_fractions_gas(interface, true);
                    }
                }
                else
                {
                    result = gen->get_fractions_all(paraboloid);
                    result1 = gen->get_fractions_all(interface);
                    // if (result[((result.sizes()[0]-7)/2)].item<double>() + result1[((result1.sizes()[0]-7)/2)].item<double>() > 0.5)
                    // {
                    //     flip = true;
                    //     result = gen->get_fractions_gas_all(paraboloid);
                    //     result1 = gen->get_fractions_gas_all(interface);
                    // }                    
                }

                std::vector<double> fractions;
                int mod = 4;
                if (all)
                {
                    mod = 7;
                }
                
                // for (int i = 0; i < result.sizes()[0]; ++i)
                // {
                //     if (i%mod == 0)
                //     {
                //         IRL::Pt pl;
                //         IRL::Pt pl1;
                //         IRL::Pt pg;
                //         IRL::Pt pg1;
                //         if (!flip)
                //         {
                //             pl = IRL::Pt(result[((result.sizes()[0]-7)/2) + 1].item<double>(), result[((result.sizes()[0]-7)/2) + 2].item<double>(), result[((result.sizes()[0]-7)/2) + 3].item<double>());
                //             pg = IRL::Pt(result[((result.sizes()[0]-7)/2) + 4].item<double>(), result[((result.sizes()[0]-7)/2) + 5].item<double>(), result[((result.sizes()[0]-7)/2) + 6].item<double>());
                //             pl1 = IRL::Pt(result1[((result1.sizes()[0]-7)/2) + 1].item<double>(), result1[((result1.sizes()[0]-7)/2) + 2].item<double>(), result1[((result1.sizes()[0]-7)/2) + 3].item<double>());
                //             pg1 = IRL::Pt(result1[((result1.sizes()[0]-7)/2) + 4].item<double>(), result1[((result1.sizes()[0]-7)/2) + 5].item<double>(), result1[((result1.sizes()[0]-7)/2) + 6].item<double>());
                //         }
                //         else
                //         {
                //             pg = IRL::Pt(result[((result.sizes()[0]-7)/2) + 1].item<double>(), result[((result.sizes()[0]-7)/2) + 2].item<double>(), result[((result.sizes()[0]-7)/2) + 3].item<double>());
                //             pl = IRL::Pt(result[((result.sizes()[0]-7)/2) + 4].item<double>(), result[((result.sizes()[0]-7)/2) + 5].item<double>(), result[((result.sizes()[0]-7)/2) + 6].item<double>());
                //             pg1 = IRL::Pt(result1[((result1.sizes()[0]-7)/2) + 1].item<double>(), result1[((result1.sizes()[0]-7)/2) + 2].item<double>(), result1[((result1.sizes()[0]-7)/2) + 3].item<double>());
                //             pl1 = IRL::Pt(result1[((result1.sizes()[0]-7)/2) + 4].item<double>(), result1[((result1.sizes()[0]-7)/2) + 5].item<double>(), result1[((result1.sizes()[0]-7)/2) + 6].item<double>());
                //         }
                //         if (IRL::distanceBetweenPts(pl,pl1) < IRL::distanceBetweenPts(pg,pg1))
                //         {
                //             type = false;
                //             break;
                //         }
                //     }
                // }
                int ind = 0;
                int count = 0;
                for (int i = 0; i < result.sizes()[0]; ++i)
                {
                    //if (type)
                    {
                        if (i%mod == 0)
                        {
                            ind = i;
                            count = 0;
                            if ((result[i].item<double>() < IRL::global_constants::VF_LOW || result[i].item<double>() > IRL::global_constants::VF_HIGH) && result1[i].item<double>() < IRL::global_constants::VF_LOW || result1[i].item<double>() > IRL::global_constants::VF_HIGH)
                            {
                                fractions.push_back(0);
                                option = true;
                            }
                            else
                            {
                                if (result[i].item<double>() < IRL::global_constants::VF_LOW || result[i].item<double>() > IRL::global_constants::VF_HIGH)
                                {
                                    fractions.push_back(1 - result1[i].item<double>());
                                }
                                else if (result1[i].item<double>() < IRL::global_constants::VF_LOW || result1[i].item<double>() > IRL::global_constants::VF_HIGH)
                                {
                                    fractions.push_back(1 - result[i].item<double>());
                                }
                                else
                                {
                                    fractions.push_back(1 - (result[i].item<double>() + result1[i].item<double>()));
                                }
                                option = false;
                            }
                            // if (result1[i].item<double>() > IRL::global_constants::VF_LOW && result[i].item<double>() <= IRL::global_constants::VF_LOW)
                            // {
                            //     option = true;
                            //     same_cell = false;
                            // }
                            // else if (result1[i].item<double>() > IRL::global_constants::VF_LOW && result[i].item<double>() > IRL::global_constants::VF_LOW && result1[i].item<double>() < IRL::global_constants::VF_HIGH && result[i].item<double>() < IRL::global_constants::VF_HIGH)
                            // {
                            //     option = false;
                            //     same_cell = true;
                            // }
                            // else
                            // {
                            //     option = false;
                            //     same_cell = false;
                            // }
                            // count = 0;
                        }
                        else
                        {
                            if (option)
                            {
                                fractions.push_back(0);
                            }
                            else
                            {
                                if (count < 3)
                                {
                                    if (result[i].item<double>() < IRL::global_constants::VF_LOW || result[i].item<double>() > IRL::global_constants::VF_HIGH)
                                    {
                                        fractions.push_back(result1[i+3].item<double>());
                                    }
                                    else if (result1[i].item<double>() < IRL::global_constants::VF_LOW || result1[i].item<double>() > IRL::global_constants::VF_HIGH)
                                    {
                                        fractions.push_back(result[i+3].item<double>());
                                    }
                                    else
                                    {
                                        fractions.push_back((result[i].item<double>() * (1-result[ind].item<double>()) + result1[i].item<double>() * (1-result1[ind].item<double>())) / (2 - result[i].item<double>() - result1[i].item<double>()));
                                    }
                                }
                                else
                                {
                                    if (result[i].item<double>() < IRL::global_constants::VF_LOW || result[i].item<double>() > IRL::global_constants::VF_HIGH)
                                    {
                                        fractions.push_back(result1[i-3].item<double>());
                                    }
                                    else if (result1[i].item<double>() < IRL::global_constants::VF_LOW || result1[i].item<double>() > IRL::global_constants::VF_HIGH)
                                    {
                                        fractions.push_back(result[i-3].item<double>());
                                    }
                                    else
                                    {
                                        fractions.push_back((result[i].item<double>() * (result[ind].item<double>()) + result1[i].item<double>() * (result1[ind].item<double>())) / (result[i].item<double>() + result1[i].item<double>()));
                                    }  
                                }
                            }
                            ++count;
                        }
                    }
                }

                int direction = 0;
                std::vector<double> center;
                auto sm = IRL::spatial_moments();
                if (!all)
                {
                    //center = sm.get_mass_centers(fractions);
                    //direction = rotateFractions(&fractions,center);
                }
                else
                {
                    //center = sm.get_mass_centers_all(&fractions);
                    //direction = rotateFractions_all(&fractions,center);
                }

                int track[number_of_cells][number_of_cells][number_of_cells];
                for (int i = 0; i < number_of_cells; ++i)
                {
                    for (int j = 0; j < number_of_cells; ++j)
                    {
                        for (int k = 0; k < number_of_cells; ++k)
                        {
                            track[i][j][k] = 1;
                        }
                    }
                } 

                std::ofstream output;
                string data_name = "fractions.txt";
                output.open(data_name, std::ios_base::app);

                for (int i = 0; i < result.sizes()[0]; ++i)
                {
                    output << fractions[i] << ",";
                }
                output << "\n";
                output.close();  

                std::ofstream normals;
                std::string normals_name = "normals.txt";
                normals.open(normals_name, std::ios_base::app);

                IRL::RectangularCuboid cell;
                IRL::Normal normal = IRL::Normal(0,0,0);
                IRL::Normal normal1 = IRL::Normal(0,0,0);
                double area = 0;
                track[1][1][1] = 1;
                for (int i = 0; i < number_of_cells; ++i)
                {
                    for (int j = 0; j < number_of_cells; ++j)
                    {
                        for (int k = 0; k < number_of_cells; ++k)
                        {
                            if (track[i][j][k] == 1)
                            {
                                cell = IRL::RectangularCuboid::fromBoundingPts(
                                            IRL::Pt(-number_of_cells/2.0+i,-number_of_cells/2.0+j,-number_of_cells/2.0+k),
                                            IRL::Pt(-number_of_cells/2.0+1+i,-number_of_cells/2.0+1+j,-number_of_cells/2.0+1+k));
                                auto surface_and_moments = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParaboloidParametrizedSurfaceOutput>, IRL::HalfEdgeCutting>(cell, paraboloid);
                                auto surface_and_moments1 = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParaboloidParametrizedSurfaceOutput>, IRL::HalfEdgeCutting>(cell, interface);
                                auto surface = surface_and_moments.getSurface();
                                auto surface1 = surface_and_moments1.getSurface();
                                normal = normal + surface.getAverageNormalNonAligned();
                                normal1 = normal1 + surface1.getAverageNormalNonAligned();
                                area = area + surface.getSurfaceArea();
                                area = area + surface1.getSurfaceArea();
                            }
                        }
                    }
                }
                normal.normalize();
                normal1.normalize();
                IRL::Normal dir = normal - normal1;
                dir.normalize();
                auto moments = sm.calculate_moments(fractions, dir, number_of_cells);

                std::ofstream moments_out;
                std::string moments_name = "moments.txt";
                moments_out.open(moments_name, std::ios_base::app);
                //moments[0] = moments[0] / pow(area,(1.0/5.0));
                // moments[1] = moments[1] / pow(area,(5.0));
                // moments[2] = moments[2] / pow(area,(15.0/2.0));
                //moments[moments.sizes()[0]-1] = moments[moments.sizes()[0]-1] / area;                        
                for (int i = 0; i < moments.sizes()[0]; ++i)
                {
                    moments_out << moments[i].item<double>() << ",";
                }
                //moments_out << paraboloid.getAlignedParaboloid().a() + paraboloid.getAlignedParaboloid().b() << ",";
                moments_out << "\n";
                moments_out.close();  

                switch (direction)
                {
                    case 1:
                    normal[0] = -normal[0];
                    normal1[0] = -normal1[0];
                    break;
                    case 2:
                    normal[1] = -normal[1];
                    normal1[1] = -normal1[1];
                    break;
                    case 3:
                    normal[2] = -normal[2];
                    normal1[2] = -normal1[2];
                    break;
                    case 4:
                    normal[0] = -normal[0];
                    normal[1] = -normal[1];
                    normal1[0] = -normal1[0];
                    normal1[1] = -normal1[1];
                    break;
                    case 5:
                    normal[0] = -normal[0];
                    normal[2] = -normal[2];
                    normal1[0] = -normal1[0];
                    normal1[2] = -normal1[2];
                    break;
                    case 6:
                    normal[1] = -normal[1];
                    normal[2] = -normal[2];
                    normal1[1] = -normal1[1];
                    normal1[2] = -normal1[2];
                    break;
                    case 7:
                    normal[0] = -normal[0];
                    normal[1] = -normal[1];
                    normal[2] = -normal[2];
                    normal1[0] = -normal1[0];
                    normal1[1] = -normal1[1];
                    normal1[2] = -normal1[2];
                    break;
                }
                if (!flip)
                {
                    normal[0] = -normal[0];
                    normal[1] = -normal[1];
                    normal[2] = -normal[2];
                    normal1[0] = -normal1[0];
                    normal1[1] = -normal1[1];
                    normal1[2] = -normal1[2];
                }

                normals << normal[0] << "," << normal[1] << "," << normal[2] << "," << normal1[0] << "," << normal1[1] << "," << normal1[2] << "\n";
                normals.close();      

                // const auto bottom_corner = IRL::Pt(-1.5, -1.5, -1.5);
                // const auto top_corner = IRL::Pt(1.5, 1.5, 1.5);
                // const auto cell = IRL::StoredRectangularCuboid<IRL::Pt>::fromBoundingPts(bottom_corner, top_corner);

                // const auto first_moments_and_surface = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParaboloidParametrizedSurfaceOutput>, IRL::HalfEdgeCutting>(cell, paraboloid);
                // const auto first_moments_and_surface2 = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParaboloidParametrizedSurfaceOutput>, IRL::HalfEdgeCutting>(cell, interface);
                // const double length_scale = 0.05;
                // IRL::TriangulatedSurfaceOutput triangulated_surface = first_moments_and_surface.getSurface().triangulate(length_scale);
                // IRL::TriangulatedSurfaceOutput triangulated_surface2 = first_moments_and_surface2.getSurface().triangulate(length_scale);
                // string name3 = "p";
                // string name4 = "i";
                // triangulated_surface.write(name3);
                // triangulated_surface2.write(name4);       
            }
        };

        void generate_two_paraboloids_in_cell_with_disturbance(double rota_l, double rota_h, double rotb_l, double rotb_h, double rotc_l, double rotc_h, double coa_l, double coa_h, double cob_l, double cob_h, double ox_l, double ox_h, double oy_l, double oy_h, double oz_l, double oz_h, bool all)
        {
                        for (int n = 0; n < Ntests; ++n) 
            {
                std::cout << n << endl;
                IRL::Paraboloid paraboloid = gen->new_random_parabaloid(rota_l, rota_h, rotb_l, rotb_h, rotc_l, rotc_h, coa_l, coa_h, cob_l, cob_h, ox_l, ox_h, oy_l, oy_h, oz_l, oz_h);
                IRL::Paraboloid interface = gen->new_interface_parabaloid_in_cell(rota_l, rota_h, rotb_l, rotb_h, rotc_l, rotc_h, coa_l, coa_h, cob_l, cob_h, ox_l, ox_h, oy_l, oy_h, oz_l, oz_h, paraboloid);
                angles = gen->getAngles();

                std::ofstream coefficients;
                std::string name = "coefficients.txt";
                coefficients.open(name, std::ios_base::app);
                coefficients << paraboloid.getDatum().x() << "," << paraboloid.getDatum().y() << "," << paraboloid.getDatum().z()
                << "," << angles[0] << "," << angles[1] << "," << angles[2]
                << "," << paraboloid.getAlignedParaboloid().a() << "," << paraboloid.getAlignedParaboloid().b() << "\n";
                coefficients.close();

                std::ofstream coefficients2;
                std::string name2 = "coefficients2.txt";
                coefficients2.open(name2, std::ios_base::app);
                coefficients2 << interface.getDatum().x() << "," << interface.getDatum().y() << "," << interface.getDatum().z()
                << "," << angles[0] << "," << angles[1] << "," << angles[2]
                << "," << interface.getAlignedParaboloid().a() << "," << interface.getAlignedParaboloid().b() << "\n";
                coefficients2.close();

                std::ofstream classification;
                std::string data_name = "type.txt";
                classification.open(data_name, std::ios_base::app);
                if ((abs(paraboloid.getAlignedParaboloid().a() - paraboloid.getAlignedParaboloid().b()) > 1) && (paraboloid.getAlignedParaboloid().a() < 0.2 || paraboloid.getAlignedParaboloid().b() < 0.2))
                {
                    classification << "0,1,0" << " \n";
                }
                else if ((abs(paraboloid.getAlignedParaboloid().a() - paraboloid.getAlignedParaboloid().b()) < 1) && (paraboloid.getAlignedParaboloid().a() > 2 || paraboloid.getAlignedParaboloid().b() > 2))
                {
                    classification << "1,0,0" << " \n";
                }
                else
                {
                    classification << "0,0,1" << " \n";
                }
                classification.close();

                std::ofstream inter;
                std::string interface_name = "interface.txt";
                inter.open(interface_name, std::ios_base::app);
                inter << "0,1" << " \n";
                inter.close();

                torch::Tensor result;
                torch::Tensor result1;
                bool flip = false;
                bool option = false;
                bool type = true;
                bool same_cell = false;
                if (!all)
                {
                    result = gen->get_fractions(paraboloid, true);
                    result1 = gen->get_fractions(interface, true);
                    if (result[((result.sizes()[0]-/*7*/4)/2)].item<double>() > 0.5)
                    {
                        flip = true;
                        result = gen->get_fractions_gas(paraboloid, true);
                        result1 = gen->get_fractions_gas(interface, true);
                    }
                }
                else
                {
                    result = gen->get_fractions_all(paraboloid);
                    result1 = gen->get_fractions_all(interface);
                    if (result[((result.sizes()[0]-7)/2)].item<double>() > 0.5)
                    {
                        flip = true;
                        result = gen->get_fractions_gas_all(paraboloid);
                        result1 = gen->get_fractions_gas_all(interface);
                    }                    
                }

                std::vector<double> fractions;
                int mod = 4;
                if (all)
                {
                    mod = 7;
                }
                
                                for (int i = 0; i < result.sizes()[0]; ++i)
                {
                    if (i%mod == 0)
                    {
                        IRL::Pt pl;
                        IRL::Pt pl1;
                        IRL::Pt pg;
                        IRL::Pt pg1;
                        if (!flip)
                        {
                            pl = IRL::Pt(result[((result.sizes()[0]-7)/2) + 1].item<double>(), result[((result.sizes()[0]-7)/2) + 2].item<double>(), result[((result.sizes()[0]-7)/2) + 3].item<double>());
                            pg = IRL::Pt(result[((result.sizes()[0]-7)/2) + 4].item<double>(), result[((result.sizes()[0]-7)/2) + 5].item<double>(), result[((result.sizes()[0]-7)/2) + 6].item<double>());
                            pl1 = IRL::Pt(result1[((result1.sizes()[0]-7)/2) + 1].item<double>(), result1[((result1.sizes()[0]-7)/2) + 2].item<double>(), result1[((result1.sizes()[0]-7)/2) + 3].item<double>());
                            pg1 = IRL::Pt(result1[((result1.sizes()[0]-7)/2) + 4].item<double>(), result1[((result1.sizes()[0]-7)/2) + 5].item<double>(), result1[((result1.sizes()[0]-7)/2) + 6].item<double>());
                        }
                        else
                        {
                            pg = IRL::Pt(result[((result.sizes()[0]-7)/2) + 1].item<double>(), result[((result.sizes()[0]-7)/2) + 2].item<double>(), result[((result.sizes()[0]-7)/2) + 3].item<double>());
                            pl = IRL::Pt(result[((result.sizes()[0]-7)/2) + 4].item<double>(), result[((result.sizes()[0]-7)/2) + 5].item<double>(), result[((result.sizes()[0]-7)/2) + 6].item<double>());
                            pg1 = IRL::Pt(result1[((result1.sizes()[0]-7)/2) + 1].item<double>(), result1[((result1.sizes()[0]-7)/2) + 2].item<double>(), result1[((result1.sizes()[0]-7)/2) + 3].item<double>());
                            pl1 = IRL::Pt(result1[((result1.sizes()[0]-7)/2) + 4].item<double>(), result1[((result1.sizes()[0]-7)/2) + 5].item<double>(), result1[((result1.sizes()[0]-7)/2) + 6].item<double>());
                        }
                        if (IRL::distanceBetweenPts(pl,pl1) < IRL::distanceBetweenPts(pg,pg1))
                        {
                            type = false;
                            break;
                        }
                    }
                }
                int ind = 0;
                int count = 0;
                for (int i = 0; i < result.sizes()[0]; ++i)
                {
                    if (type)
                    {
                        if (i%mod == 0)
                        {
                            if (result1[i].item<double>() > IRL::global_constants::VF_LOW && result[i].item<double>() <= IRL::global_constants::VF_LOW)
                            {
                                option = true;
                                same_cell = false;
                            }
                            else if (result1[i].item<double>() > IRL::global_constants::VF_LOW && result[i].item<double>() > IRL::global_constants::VF_LOW && result1[i].item<double>() < IRL::global_constants::VF_HIGH && result[i].item<double>() < IRL::global_constants::VF_HIGH)
                            {
                                option = false;
                                same_cell = true;
                            }
                            else
                            {
                                option = false;
                                same_cell = false;
                            }
                            count = 0;
                        }
                        if (option)
                        {
                            fractions.push_back(result1[i].item<double>());
                            count = 0;
                        }
                        else if (!same_cell)
                        {
                            fractions.push_back(result[i].item<double>());
                            count = 0;
                        }
                        else
                        {
                            if (i%mod == 0)
                            {
                                fractions.push_back(result[i].item<double>() + result1[i].item<double>());
                                ind = i;
                            }
                            else
                            {
                                ++count;
                                double x = result[ind].item<double>();
                                double y = result1[ind].item<double>();
                                if (count > 3)
                                {
                                    x = 1 - x;
                                    y = 1 - y;
                                }
                                double c = x/(x + y) * result[i].item<double>() + y/(x + y) * result1[i].item<double>();
                                fractions.push_back(c);
                            }
                        }
                    }
                    else
                    {
                        if (i%mod == 0)
                        {
                            count = 0;
                            if (result1[i].item<double>() < IRL::global_constants::VF_HIGH && result[i].item<double>() >= IRL::global_constants::VF_HIGH)
                            {
                                option = true;
                                same_cell = false;
                            }
                            else if (result1[i].item<double>() > IRL::global_constants::VF_LOW && result[i].item<double>() > IRL::global_constants::VF_LOW && result1[i].item<double>() < IRL::global_constants::VF_HIGH && result[i].item<double>() < IRL::global_constants::VF_HIGH)
                            {
                                option = false;
                                same_cell = true;
                            }
                            else
                            {
                                option = false;
                                same_cell = false;
                            }
                        }
                        if (option)
                        {
                            fractions.push_back(result1[i].item<double>());
                            count = 0;
                        }
                        else if (!same_cell)
                        {
                            fractions.push_back(result[i].item<double>());
                            count = 0;
                        }
                        else
                        {
                            if (i%mod == 0)
                            {
                                fractions.push_back(result[i].item<double>() + result1[i].item<double>());
                                ind = i;
                            }
                            else
                            {
                                ++count;
                                double x = result[ind].item<double>();
                                double y = result1[ind].item<double>();
                                if (count > 3)
                                {
                                    x = 1 - x;
                                    y = 1 - y;
                                }
                                double c = x/(x + y) * result[i].item<double>() + y/(x + y) * result1[i].item<double>();
                                fractions.push_back(c);
                            }
                        }
                    }
                }

                int direction = 0;
                std::vector<double> center;
                auto sm = IRL::spatial_moments();
                if (!all)
                {
                    center = sm.get_mass_centers(fractions);
                    direction = rotateFractions(&fractions,center);
                }
                else
                {
                    center = sm.get_mass_centers_all(&fractions);
                    direction = rotateFractions_all(&fractions,center);
                }

                std::ofstream output;
                data_name = "fractions.txt";
                output.open(data_name, std::ios_base::app);

                int p = rand() % 8;
                for (int i = 0; i < result.sizes()[0]; ++i)
                {
                    if (p == 0)
                    {
                        output << fractions[i] << ",";
                    }
                    else
                    {
                        //int r = rand() % 3 - 1;
                        double c = (rand() % 201 - 100) / 1000.0;
                        if (i % mod != 0)
                        {
                            if (fractions[i] + c > 0.5)
                            {
                                output << 0.5 << ",";
                            }
                            else if (fractions[i] + c < -0.5)
                            {
                                output << -0.5 << ",";
                            }
                            else
                            {
                                output << fractions[i] + c << ",";
                            }
                        }
                        else
                        {
                            output << fractions[i] << ",";
                        }
                    }
                }
                output << "\n";
                output.close(); 

                std::ofstream normals;
                std::string normals_name = "normals.txt";
                normals.open(normals_name, std::ios_base::app);
                auto cube = IRL::RectangularCuboid::fromBoundingPts(IRL::Pt(-0.5, -0.5, -0.5), IRL::Pt(0.5, 0.5, 0.5));
                auto surface_and_moments = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParaboloidParametrizedSurfaceOutput>>(cube, paraboloid);
                auto surface_and_moments1 = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParaboloidParametrizedSurfaceOutput>>(cube, interface);
                auto surface = surface_and_moments.getSurface();
                auto surface1 = surface_and_moments1.getSurface();
                auto normal = surface.getAverageNormalNonAligned();
                auto normal1 = surface1.getAverageNormalNonAligned();

                switch (direction)
                {
                    case 1:
                    normal[0] = -normal[0];
                    normal1[0] = -normal1[0];
                    break;
                    case 2:
                    normal[1] = -normal[1];
                    normal1[1] = -normal1[1];
                    break;
                    case 3:
                    normal[2] = -normal[2];
                    normal1[2] = -normal1[2];
                    break;
                    case 4:
                    normal[0] = -normal[0];
                    normal[1] = -normal[1];
                    normal1[0] = -normal1[0];
                    normal1[1] = -normal1[1];
                    break;
                    case 5:
                    normal[0] = -normal[0];
                    normal[2] = -normal[2];
                    normal1[0] = -normal1[0];
                    normal1[2] = -normal1[2];
                    break;
                    case 6:
                    normal[1] = -normal[1];
                    normal[2] = -normal[2];
                    normal1[1] = -normal1[1];
                    normal1[2] = -normal1[2];
                    break;
                    case 7:
                    normal[0] = -normal[0];
                    normal[1] = -normal[1];
                    normal[2] = -normal[2];
                    normal1[0] = -normal1[0];
                    normal1[1] = -normal1[1];
                    normal1[2] = -normal1[2];
                    break;
                }
                if (!flip)
                {
                    normal[0] = -normal[0];
                    normal[1] = -normal[1];
                    normal[2] = -normal[2];
                    normal1[0] = -normal1[0];
                    normal1[1] = -normal1[1];
                    normal1[2] = -normal1[2];
                }

                normals << normal[0] << "," << normal[1] << "," << normal[2] << "," << normal1[0] << "," << normal1[1] << "," << normal1[2] << "\n";
                normals.close();      

                // const auto bottom_corner = IRL::Pt(-1.5, -1.5, -1.5);
                // const auto top_corner = IRL::Pt(1.5, 1.5, 1.5);
                // const auto cell = IRL::StoredRectangularCuboid<IRL::Pt>::fromBoundingPts(bottom_corner, top_corner);

                // const auto first_moments_and_surface = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParaboloidParametrizedSurfaceOutput>, IRL::HalfEdgeCutting>(cell, paraboloid);
                // const auto first_moments_and_surface2 = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParaboloidParametrizedSurfaceOutput>, IRL::HalfEdgeCutting>(cell, interface);
                // const double length_scale = 0.05;
                // IRL::TriangulatedSurfaceOutput triangulated_surface = first_moments_and_surface.getSurface().triangulate(length_scale);
                // IRL::TriangulatedSurfaceOutput triangulated_surface2 = first_moments_and_surface2.getSurface().triangulate(length_scale);
                // string name3 = "p";
                // string name4 = "i";
                // triangulated_surface.write(name3);
                // triangulated_surface2.write(name4);       
            }
        };

        void generate_noise(double rota_l, double rota_h, double rotb_l, double rotb_h, double rotc_l, double rotc_h)
        {
            srand((unsigned) time(NULL));
            for (int n = 0; n < Ntests; ++n) 
            {
                std::cout << n << endl;
                IRL::Paraboloid paraboloid[27] = { };
                //int r = rand() % 27;
                for (int i = 0; i < 3; ++i)
                {
                    for (int j = 0; j < 3; ++j)
                    {
                        for (int k = 0; k < 3; ++k)
                        {
                            //int r = rand() % 10;
                            //if (k+j*3+i*9 == r)
                            //if (r == 0)
                            {
                                paraboloid[k+3*j+9*i] = gen->new_random_parabaloid(rota_l, rota_h, rotb_l, rotb_h, rotc_l, rotc_h, 0.01, 0.01, 0.01, 0.01, -0.3+(i-1), 0.3+(i-1), -0.3+(j-1), 0.3+(j-1), -0.3+(k-1), 0.3+(k-1));
                            }
                        }
                    }
                }

                std::ofstream normals;
                std::string normals_name = "normals.txt";
                normals.open(normals_name, std::ios_base::app);
                auto cube = IRL::RectangularCuboid::fromBoundingPts(IRL::Pt(-0.5, -0.5, -0.5), IRL::Pt(0.5, 0.5, 0.5));
                auto surface_and_moments = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParaboloidParametrizedSurfaceOutput>>(cube, paraboloid[13]);
                auto surface = surface_and_moments.getSurface();
                auto normal = surface.getAverageNormalNonAligned();
                normals << normal[0] << "," << normal[1] << "," << normal[2] << "\n";
                normals.close();

                std::ofstream output;
                std::string data_name = "fractions.txt";
                output.open(data_name, std::ios_base::app);
                for (int i = 0; i < 3; ++i)
                {
                    for (int j = 0; j < 3; ++j)
                    {
                        for (int k = 0; k < 3; ++k)
                        {
                            auto cube = IRL::RectangularCuboid::fromBoundingPts(IRL::Pt(-0.5+(i-1), -0.5+(j-1), -0.5+(k-1)), IRL::Pt(0.5+(i-1), 0.5+(j-1), 0.5+(k-1)));
                            auto moments = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParaboloidParametrizedSurfaceOutput>>(cube, paraboloid[k+3*j+9*i]);
                            if (moments.getMoments().volume() > IRL::global_constants::VF_LOW)
                            {
                                moments.getMoments().centroid()[0] = moments.getMoments().centroid()[0] / moments.getMoments().volume() - (i-1);
                                moments.getMoments().centroid()[1] = moments.getMoments().centroid()[1] / moments.getMoments().volume() - (j-1);
                                moments.getMoments().centroid()[2] = moments.getMoments().centroid()[2] / moments.getMoments().volume() - (k-1);
                            }
                            output << moments.getMoments().volume() << ",";
                            output << moments.getMoments().centroid()[0] << ",";
                            output << moments.getMoments().centroid()[1] << ",";
                            output << moments.getMoments().centroid()[2] << ",";
                            // auto surface = moments.getSurface();
                            // const double length_scale = 0.05;
                            // IRL::TriangulatedSurfaceOutput triangulated_surface = surface.triangulate(length_scale);
                            // string name = "p" + std::to_string(n)+std::to_string(i)+std::to_string(j)+std::to_string(k);
                            // triangulated_surface.write(name);
                        }
                    }
                }
                output << "\n";
                output.close();                    
            }           
        };







        //************************************************************
        //**************CYLINDERS*************************************
        //************************************************************


        void generate_cylinders(double rota_l, double rota_h, double rotb_l, double rotb_h, double rotc_l, double rotc_h, double r_l, double r_h, double cob_l, double cob_h, double ox_l, double ox_h, double oy_l, double oy_h, double oz_l, double oz_h, bool all)
        {
            for (int n = 0; n < Ntests; ++n) 
            {
                torch::Tensor result;
                std::cout << n << endl;
                IRL::Cylinder cylinder = gen->new_random_cylinder(rota_l, rota_h, rotb_l, rotb_h, rotc_l, rotc_h, r_l, r_h, cob_l, cob_h, ox_l, ox_h, oy_l, oy_h, oz_l, oz_h);
                angles = gen->getAngles();

                std::ofstream coefficients;
                std::string name = "coefficients.txt";
                coefficients.open(name, std::ios_base::app);
                coefficients << cylinder.getDatum().x() << "," << cylinder.getDatum().y() << "," << cylinder.getDatum().z()
                << "," << angles[0] << "," << angles[1] << "," << angles[2]
                << "," << cylinder.getAlignedCylinder().r() << "," << cylinder.getAlignedCylinder().b() << "\n";
                coefficients.close();
                
                bool flip = false;
                result = gen->get_fractions_all(cylinder);
                if (result[((result.sizes()[0]-7)/2)].item<double>() > 0.5)
                {
                    flip = true;
                    result = gen->get_fractions_gas_all(cylinder);
                }                    

                std::vector<double> fractions;
                for (int i = 0; i < result.sizes()[0]; ++i)
                {
                    fractions.push_back(result[i].item<double>());
                }
                

                int direction = 0;
                std::vector<double> center;
                auto sm = IRL::spatial_moments();
                if (!all)
                {
                    center = sm.get_mass_centers(fractions);
                    direction = rotateFractions(&fractions,center);
                }
                else
                {
                    center = sm.get_mass_centers_all(&fractions);
                    direction = rotateFractions_all(&fractions,center);
                }
                
                std::ofstream output;
                std::string data_name = "fractions.txt";
                output.open(data_name, std::ios_base::app);

                for (int i = 0; i < result.sizes()[0]; ++i)
                {
                    output << fractions[i] << ",";
                }
                output << "\n";
                output.close();  

                std::ofstream normals;
                std::string normals_name = "normals.txt";
                normals.open(normals_name, std::ios_base::app);
                auto cube = IRL::RectangularCuboid::fromBoundingPts(IRL::Pt(-0.5, -0.5, -0.5), IRL::Pt(0.5, 0.5, 0.5));
                auto surface_and_moments = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::CylinderParametrizedSurfaceOutput>>(cube, cylinder);
                auto surface = surface_and_moments.getSurface();
                auto normal = surface.getAverageNormalNonAligned();

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
                if (!flip)
                {
                    normal[0] = -normal[0];
                    normal[1] = -normal[1];
                    normal[2] = -normal[2];
                }

                normals << normal[0] << "," << normal[1] << "," << normal[2] << "\n";
                normals.close();  
                
                // const auto bottom_corner = IRL::Pt(-0.5, -0.5, -0.5);
                // const auto top_corner = IRL::Pt(0.5, 0.5, 0.5);
                // const auto cell = IRL::StoredRectangularCuboid<IRL::Pt>::fromBoundingPts(bottom_corner, top_corner);

                // const auto first_moments_and_surface = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::CylinderParametrizedSurfaceOutput>>(cell, cylinder);
                // const double length_scale = 0.05;
                // IRL::TriangulatedSurfaceOutput triangulated_surface = first_moments_and_surface.getSurface().triangulate(length_scale);
                // string name3 = "p" + to_string(n);
                // triangulated_surface.write(name3);

                // IRL::HalfEdgePolyhedronQuadratic<IRL::Pt> half_edge;
                // cell.setHalfEdgeVersion(&half_edge);
                // auto seg_half_edge = half_edge.generateSegmentedPolyhedron();
                // std::ofstream myfile;
                // myfile.open(name3 + ".vtu");
                // myfile << seg_half_edge;
                // myfile.close();
            }  
        }; 

        void generate_cylinders_with_disturbance(double rota_l, double rota_h, double rotb_l, double rotb_h, double rotc_l, double rotc_h, double r_l, double r_h, double cob_l, double cob_h, double ox_l, double ox_h, double oy_l, double oy_h, double oz_l, double oz_h, bool all)
        {
            for (int n = 0; n < Ntests; ++n) 
            {
                torch::Tensor result;
                std::cout << n << endl;
                IRL::Cylinder cylinder = gen->new_random_cylinder(rota_l, rota_h, rotb_l, rotb_h, rotc_l, rotc_h, r_l, r_h, cob_l, cob_h, ox_l, ox_h, oy_l, oy_h, oz_l, oz_h);
                angles = gen->getAngles();

                std::ofstream coefficients;
                std::string name = "coefficients.txt";
                coefficients.open(name, std::ios_base::app);
                coefficients << cylinder.getDatum().x() << "," << cylinder.getDatum().y() << "," << cylinder.getDatum().z()
                << "," << angles[0] << "," << angles[1] << "," << angles[2]
                << "," << cylinder.getAlignedCylinder().r() << "," << cylinder.getAlignedCylinder().b() << "\n";
                coefficients.close();
                
                bool flip = false;
                result = gen->get_fractions_all(cylinder);
                if (result[((result.sizes()[0]-7)/2)].item<double>() > 0.5)
                {
                    flip = true;
                    result = gen->get_fractions_gas_all(cylinder);
                }                    

                std::vector<double> fractions;
                for (int i = 0; i < result.sizes()[0]; ++i)
                {
                    fractions.push_back(result[i].item<double>());
                }
                

                int direction = 0;
                std::vector<double> center;
                auto sm = IRL::spatial_moments();
                if (!all)
                {
                    center = sm.get_mass_centers(fractions);
                    direction = rotateFractions(&fractions,center);
                }
                else
                {
                    center = sm.get_mass_centers_all(&fractions);
                    direction = rotateFractions_all(&fractions,center);
                }
                
                std::ofstream output;
                std::string data_name = "fractions.txt";
                output.open(data_name, std::ios_base::app);

                int p = rand() % 8;
                int mod = 4;
                if (all)
                {
                    mod = 7;
                }
                for (int i = 0; i < result.sizes()[0]; ++i)
                {
                    if (p == 0)
                    {
                        output << fractions[i] << ",";
                    }
                    else
                    {
                        double c = (rand() % 201 - 100) / 1000.0;
                        if (i % mod != 0)
                        {
                            if (fractions[i] + c > 0.5)
                            {
                                fractions[i] = 0.5;
                                output << 0.5 << ",";
                            }
                            else if (fractions[i] + c < -0.5)
                            {
                                fractions[i] = -0.5;
                                output << -0.5 << ",";
                            }
                            else
                            {
                                output << fractions[i] + c << ",";
                                fractions[i] = fractions[i] + c;
                            }
                        }
                        else
                        {
                            output << fractions[i] << ",";
                        }
                    }
                }
                output << "\n";
                output.close();  

                std::ofstream normals;
                std::string normals_name = "normals.txt";
                normals.open(normals_name, std::ios_base::app);
                auto cube = IRL::RectangularCuboid::fromBoundingPts(IRL::Pt(-0.5, -0.5, -0.5), IRL::Pt(0.5, 0.5, 0.5));
                auto surface_and_moments = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::CylinderParametrizedSurfaceOutput>>(cube, cylinder);
                auto surface = surface_and_moments.getSurface();
                auto normal = surface.getAverageNormalNonAligned();
                normal.normalize();

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
                if (!flip)
                {
                    normal[0] = -normal[0];
                    normal[1] = -normal[1];
                    normal[2] = -normal[2];
                }

                normals << normal[0] << "," << normal[1] << "," << normal[2] << "\n";
                normals.close();  
                
                // const auto bottom_corner = IRL::Pt(-0.5, -0.5, -0.5);
                // const auto top_corner = IRL::Pt(0.5, 0.5, 0.5);
                // const auto cell = IRL::StoredRectangularCuboid<IRL::Pt>::fromBoundingPts(bottom_corner, top_corner);

                // const auto first_moments_and_surface = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::CylinderParametrizedSurfaceOutput>>(cell, cylinder);
                // const double length_scale = 0.05;
                // IRL::TriangulatedSurfaceOutput triangulated_surface = first_moments_and_surface.getSurface().triangulate(length_scale);
                // string name3 = "p" + to_string(n);
                // triangulated_surface.write(name3);

                // IRL::HalfEdgePolyhedronQuadratic<IRL::Pt> half_edge;
                // cell.setHalfEdgeVersion(&half_edge);
                // auto seg_half_edge = half_edge.generateSegmentedPolyhedron();
                // std::ofstream myfile;
                // myfile.open(name3 + ".vtu");
                // myfile << seg_half_edge;
                // myfile.close();
            }  
        }; 







        //************************************************************
        //**************PLANES****************************************
        //************************************************************

        void generate_plane(double rota1_l, double rota1_h, double rotb1_l, double rotb1_h, double d1_l, double d1_h, bool R2P)
        {
            for (int n = 0; n < Ntests; ++n) 
            {
                std::cout << n << endl;
                IRL::PlanarSeparator plane = gen->new_random_plane(rota1_l, rota1_h, rotb1_l, rotb1_h, d1_l, d1_h);

                std::ofstream coefficients;
                std::string name = "coefficients.txt";
                coefficients.open(name, std::ios_base::app);
                coefficients << plane[0].normal()[0] << "," << plane[0].normal()[1] << "," << plane[0].normal()[2] << "," << plane[0].distance() << "\n";
                coefficients.close();

                std::ofstream type;
                name = "type.txt";
                type.open(name, std::ios_base::app);
                type << std::to_string(0) << "\n";
                type.close();

                torch::Tensor result;
                bool flip = false;
                result = gen->get_fractions_all(plane);
                if (result[((result.sizes()[0]-7)/2)].item<double>() > 0.5)
                {
                    flip = true;
                    result = gen->get_fractions_gas_all(plane);
                }                    

                std::vector<double> fractions;
                for (int i = 0; i < result.sizes()[0]; ++i)
                {
                    fractions.push_back(result[i].item<double>());
                }
                

                int direction = 0;
                std::vector<double> center;
                auto sm = IRL::spatial_moments();
                center = sm.get_mass_centers_all(&fractions);
                //direction = rotateFractions_all(&fractions,center);
                
                std::ofstream output;
                std::string data_name = "fractions.txt";
                output.open(data_name, std::ios_base::app);

                for (int i = 0; i < result.sizes()[0]; ++i)
                {
                    output << fractions[i] << ",";
                }
                output << "\n";
                output.close();  

                std::ofstream normals;
                std::string normals_name = "normals.txt";
                normals.open(normals_name, std::ios_base::app);
                //std::ofstream normals2;
                //std::string normals2_name = "normals2.txt";
                //normals2.open(normals2_name, std::ios_base::app);
                auto normal = plane[0].normal();

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
                if (!flip)
                {
                    normal[0] = -normal[0];
                    normal[1] = -normal[1];
                    normal[2] = -normal[2];
                }

                vector<double> angles = gen->getPLICAngles();
                double theta1 = angles[0];
                double phi1 = angles[1];
                //normals2 << theta1 << "," << phi1 << "\n";

                if (!R2P)
                {
                    normals << normal[0] << "," << normal[1] << "," << normal[2] << "\n";
                }
                else
                {
                    normals << normal[0] << "," << normal[1] << "," << normal[2] << ",0,0,0" << "\n";
                }

                normals.close(); 
            }  
        }; 

        void generate_plane_with_disturbance(double rota1_l, double rota1_h, double rotb1_l, double rotb1_h, double d1_l, double d1_h, bool R2P)
        {
            srand((unsigned) time(NULL));
            for (int n = 0; n < Ntests; ++n) 
            {
                std::cout << n << endl;
                IRL::PlanarSeparator plane = gen->new_random_plane(rota1_l, rota1_h, rotb1_l, rotb1_h, d1_l, d1_h);

                std::ofstream coefficients;
                std::string name = "coefficients.txt";
                coefficients.open(name, std::ios_base::app);
                coefficients << plane[0].normal()[0] << "," << plane[0].normal()[1] << "," << plane[0].normal()[2] << "," << plane[0].distance() << "\n";
                coefficients.close();

                torch::Tensor result;
                bool flip = false;
                result = gen->get_fractions_all(plane);
                if (result[((result.sizes()[0]-7)/2)].item<double>() > 0.5)
                {
                    flip = true;
                    result = gen->get_fractions_gas_all(plane);
                }                    

                std::vector<double> fractions;
                for (int i = 0; i < result.sizes()[0]; ++i)
                {
                    fractions.push_back(result[i].item<double>());
                }
                

                int direction = 0;
                std::vector<double> center;
                auto sm = IRL::spatial_moments();
                center = sm.get_mass_centers_all(&fractions);
                direction = rotateFractions_all(&fractions,center);
                
                std::ofstream output;
                std::string data_name = "fractions.txt";
                output.open(data_name, std::ios_base::app);

                int p = rand() % 8;
                int mod = 7;
                for (int i = 0; i < result.sizes()[0]; ++i)
                {
                    if (p == 0)
                    {
                        output << fractions[i] << ",";
                    }
                    else
                    {
                        double c = (rand() % 201 - 100) / 1000.0;
                        if (i % mod != 0)
                        {
                            if (fractions[i] + c > 0.5)
                            {
                                output << 0.5 << ",";
                            }
                            else if (fractions[i] + c < -0.5)
                            {
                                output << -0.5 << ",";
                            }
                            else
                            {
                                output << fractions[i] + c << ",";
                            }
                        }
                        else
                        {
                            output << fractions[i] << ",";
                        }
                    }
                }
                output << "\n";
                output.close(); 

                std::ofstream normals;
                std::string normals_name = "normals.txt";
                normals.open(normals_name, std::ios_base::app);
                auto normal = plane[0].normal();

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
                if (!flip)
                {
                    normal[0] = -normal[0];
                    normal[1] = -normal[1];
                    normal[2] = -normal[2];
                }

                if (!R2P)
                {
                    normals << normal[0] << "," << normal[1] << "," << normal[2] << "\n";
                }
                else
                {
                    normals << normal[0] << "," << normal[1] << "," << normal[2] << ",0,0,0" << "\n";
                }
                normals.close();                  
            }  
        };

        void generate_R2P(double rota1_l, double rota1_h, double rotb1_l, double rotb1_h, double rota2_l, double rota2_h, double rotb2_l, double rotb2_h, double d1_l, double d1_h, double d2_l, double d2_h, bool inter, bool same)
        {
            srand((unsigned) time(NULL));
            torch::Tensor result_all = torch::zeros({162, Ntests});
            for (int n = 0; n < Ntests; ++n) 
            {
                std::cout << n << endl;
                IRL::PlanarSeparator plane = gen->new_random_R2P(rota1_l, rota1_h, rotb1_l, rotb1_h, rota2_l, rota2_h, rotb2_l, rotb2_h, d1_l, d1_h, d2_l, d2_h, inter, same);
                //IRL::PlanarSeparator plane = gen->new_step_R2P(inter, n, Ntests);
                torch::Tensor result;
                result = gen->get_fractions_only(plane);
                while (result[((result.sizes()[0]/*-7*/)/2)].item<double>() > 0.5/* && result[((result.sizes()[0]-7)/2)].item<double>() < 0.85*/)
                {
                    //IRL::PlanarSeparator plane = gen->new_step_R2P(inter, n, Ntests);
                    plane = gen->new_random_R2P(rota1_l, rota1_h, rotb1_l, rotb1_h, rota2_l, rota2_h, rotb2_l, rotb2_h, d1_l, d1_h, d2_l, d2_h, inter, same);
                    //IRL::PlanarSeparator plane = gen->new_step_R2P(inter, n, Ntests);
                    result = gen->get_fractions_only(plane);
                }
                //std::cout << result[((result.sizes()[0]-7)/2)].item<double>() << std::endl;
                //result_all.index_put_({torch::indexing::Slice(), n}, result);
                std::ofstream coefficients;
                std::string name = "coefficients.txt";
                coefficients.open(name, std::ios_base::app);
                coefficients << plane[0].normal()[0] << "," << plane[0].normal()[1] << "," << plane[0].normal()[2] << "," << plane[0].distance() 
                << "," << plane[1].normal()[0] << "," << plane[1].normal()[1] << "," << plane[1].normal()[2] << "," << plane[1].distance() << "," << plane.flip() << "\n";
                coefficients.close();

                std::ofstream type;
                name = "type.txt";
                type.open(name, std::ios_base::app);
                if (!inter)
                {
                    type << std::to_string(0) << "\n";                   
                }
                else
                {
                    type << std::to_string(1) << "\n";
                }
                //type << std::to_string(1) << "\n";
                //type << std::to_string(0) << "\n";
                //type << std::to_string(1) << "," << std::to_string(0) << "," << std::to_string(0) << "\n";
                // if (!same)
                // {
                //     type << std::to_string(0) << "\n";
                // }
                // else
                // {
                //     type << std::to_string(1) << "\n";
                // }
                type.close();

                //torch::Tensor result;
                bool flip = false;
                //result = gen->get_fractions_all(plane);
                //std::cout << result[((result.sizes()[0]-7)/2)].item<double>() << std::endl;
                //result = gen->get_barycenters(plane);
                //result_all.index_put_({torch::indexing::Slice(), n}, result);
                torch::Tensor result1;
                IRL::PlanarSeparator p = IRL::PlanarSeparator::fromOnePlane(plane[0]);
                //result1 = gen->get_fractions_all(p);
                if (result[((result.sizes()[0]-7)/2)].item<double>() > 0.5)
                {
                    result = gen->get_fractions_gas_all(plane);
                    //result1 = gen->get_fractions_gas_all(p);
                }  

                int track[number_of_cells][number_of_cells][number_of_cells];
                for (int i = 0; i < number_of_cells; ++i)
                {
                    for (int j = 0; j < number_of_cells; ++j)
                    {
                        for (int k = 0; k < number_of_cells; ++k)
                        {
                            track[i][j][k] = 1;
                        }
                    }
                }                   

                std::vector<double> fractions;
                //std::vector<double> fractions1;
                for (int i = 0; i < result.sizes()[0]; ++i)
                {
                    fractions.push_back(result[i].item<double>());
                    //fractions1.push_back(result1[i].item<double>());
                }

                int direction = 0;
                std::vector<double> center;
                double c = (rand() % 601 - 300) / 1000.0;
                center.push_back(plane[0].normal()[0] + c);
                center.push_back(plane[0].normal()[1] + c);
                center.push_back(plane[0].normal()[2] + c);
                //direction = rotateFractions_all(&fractions,center);

                std::ofstream output;
                std::string data_name = "fractions.txt";
                output.open(data_name, std::ios_base::app);

                for (int i = 0; i < result.sizes()[0]; ++i)
                {
                    //if (i%7!=0)
                    {
                        output << fractions[i] << ",";
                    }
                }
                output << "\n";
                output.close();  

                std::ofstream normals;
                std::string normals_name = "normals.txt";
                normals.open(normals_name, std::ios_base::app);
                auto normal = IRL::Normal();
                auto normal1 = IRL::Normal();
                if (gen->arePlanesInSameCenterCell(plane))
                {
                    normal = plane[0].normal();
                    normal1 = plane[1].normal();
                }
                else
                {
                    normal = plane[0].normal();
                    normal1 = IRL::Normal(0,0,0);
                }

                vector<double> angles = gen->getR2PAngles();
                double theta1 = angles[0];
                double phi1 = angles[1];
                double theta2 = angles[2];
                double phi2 = angles[3];

                switch (direction)
                {
                    case 1:
                    normal[0] = -normal[0];
                    normal1[0] = -normal1[0];
                    theta1 = theta1 + 2*(M_PI/2-theta1);
                    if (theta1 > 2*M_PI)
                    {
                        theta1 = theta1 - 2*M_PI;
                    }
                    else if (theta1 < 0)
                    {
                        theta1 = theta1 + 2*M_PI;
                    }
                    theta2 = theta2 + 2*(M_PI/2-theta2);
                    if (theta2 > 2*M_PI)
                    {
                        theta2 = theta2 - 2*M_PI;
                    }
                    else if (theta2 < 0)
                    {
                        theta2 = theta2 + 2*M_PI;
                    }
                    break;
                    case 2:
                    normal[1] = -normal[1];
                    normal1[1] = -normal1[1];
                    theta1 = theta1 + 2*(M_PI-theta1);
                    if (theta1 > 2*M_PI)
                    {
                        theta1 = theta1 - 2*M_PI;
                    }
                    else if (theta1 < 0)
                    {
                        theta1 = theta1 + 2*M_PI;
                    }                    
                    theta2 = theta2 + 2*(M_PI-theta2);
                    if (theta2 > 2*M_PI)
                    {
                        theta2 = theta2 - 2*M_PI;
                    }
                    else if (theta2 < 0)
                    {
                        theta2 = theta2 + 2*M_PI;
                    }
                    break;
                    case 3:
                    normal[2] = -normal[2];
                    normal1[2] = -normal1[2];
                    phi1 = -phi1;
                    phi2 = -phi2;
                    break;
                    case 4:
                    normal[0] = -normal[0];
                    normal[1] = -normal[1];
                    normal1[0] = -normal1[0];
                    normal1[1] = -normal1[1];
                    theta1 = theta1 + 2*(M_PI/2-theta1);
                    if (theta1 > 2*M_PI)
                    {
                        theta1 = theta1 - 2*M_PI;
                    }
                    else if (theta1 < 0)
                    {
                        theta1 = theta1 + 2*M_PI;
                    }
                    theta2 = theta2 + 2*(M_PI/2-theta2);
                    if (theta2 > 2*M_PI)
                    {
                        theta2 = theta2 - 2*M_PI;
                    }
                    else if (theta2 < 0)
                    {
                        theta2 = theta2 + 2*M_PI;
                    }
                    theta1 = theta1 + 2*(M_PI-theta1);
                    if (theta1 > 2*M_PI)
                    {
                        theta1 = theta1 - 2*M_PI;
                    }
                    else if (theta1 < 0)
                    {
                        theta1 = theta1 + 2*M_PI;
                    }
                    theta2 = theta2 + 2*(M_PI-theta2);
                    if (theta2 > 2*M_PI)
                    {
                        theta2 = theta2 - 2*M_PI;
                    }
                    else if (theta2 < 0)
                    {
                        theta2 = theta2 + 2*M_PI;
                    }
                    break;
                    case 5:
                    normal[0] = -normal[0];
                    normal[2] = -normal[2];
                    normal1[0] = -normal1[0];
                    normal1[2] = -normal1[2]; 
                    theta1 = theta1 + 2*(M_PI/2-theta1);
                    if (theta1 > 2*M_PI)
                    {
                        theta1 = theta1 - 2*M_PI;
                    }
                    else if (theta1 < 0)
                    {
                        theta1 = theta1 + 2*M_PI;
                    }
                    theta2 = theta2 + 2*(M_PI/2-theta2);
                    if (theta2 > 2*M_PI)
                    {
                        theta2 = theta2 - 2*M_PI;
                    }
                    else if (theta2 < 0)
                    {
                        theta2 = theta2 + 2*M_PI;
                    }
                    phi1 = -phi1;
                    phi2 = -phi2;
                    break;
                    case 6:
                    normal[1] = -normal[1];
                    normal[2] = -normal[2];
                    normal1[1] = -normal1[1];
                    normal1[2] = -normal1[2];
                    theta1 = theta1 + 2*(M_PI-theta1);
                    if (theta1 > 2*M_PI)
                    {
                        theta1 = theta1 - 2*M_PI;
                    }
                    else if (theta1 < 0)
                    {
                        theta1 = theta1 + 2*M_PI;
                    }
                    theta2 = theta2 + 2*(M_PI-theta2);
                    if (theta2 > 2*M_PI)
                    {
                        theta2 = theta2 - 2*M_PI;
                    }
                    else if (theta2 < 0)
                    {
                        theta2 = theta2 + 2*M_PI;
                    }
                    phi1 = -phi1;
                    phi2 = -phi2;
                    break;
                    case 7:
                    normal[0] = -normal[0];
                    normal[1] = -normal[1];
                    normal[2] = -normal[2];
                    normal1[0] = -normal1[0];
                    normal1[1] = -normal1[1];
                    normal1[2] = -normal1[2];
                    theta1 = theta1 + 2*(M_PI/2-theta1);
                    if (theta1 > 2*M_PI)
                    {
                        theta1 = theta1 - 2*M_PI;
                    }
                    else if (theta1 < 0)
                    {
                        theta1 = theta1 + 2*M_PI;
                    }
                    theta2 = theta2 + 2*(M_PI/2-theta2);
                    if (theta2 > 2*M_PI)
                    {
                        theta2 = theta2 - 2*M_PI;
                    }
                    else if (theta2 < 0)
                    {
                        theta2 = theta2 + 2*M_PI;
                    }
                    theta1 = theta1 + 2*(M_PI-theta1);
                    if (theta1 > 2*M_PI)
                    {
                        theta1 = theta1 - 2*M_PI;
                    }
                    else if (theta1 < 0)
                    {
                        theta1 = theta1 + 2*M_PI;
                    }
                    theta2 = theta2 + 2*(M_PI-theta2);
                    if (theta2 > 2*M_PI)
                    {
                        theta2 = theta2 - 2*M_PI;
                    }
                    else if (theta2 < 0)
                    {
                        theta2 = theta2 + 2*M_PI;
                    }
                    phi1 = -phi1;
                    phi2 = -phi2;
                    break;
                }
                if (!flip)
                {
                    normal[0] = -normal[0];
                    normal[1] = -normal[1];
                    normal[2] = -normal[2];
                    normal1[0] = -normal1[0];
                    normal1[1] = -normal1[1];
                    normal1[2] = -normal1[2];
                }

                //double theta1 = atan2(normal[1],normal[0]);
                //double phi1 = acos(normal[2]/normal.calculateMagnitude());
                //double theta2 = atan2(normal1[1],normal1[0]);
                //double phi2 = acos(normal1[2]/normal1.calculateMagnitude());
                //std::cout << theta1 << "," << phi1 << "," << theta2 << "," << phi2 << std::endl << std::endl;
                //theta1 = sin(theta1);
                //phi1 = sin(phi1);
                //theta2 = sin(theta2);
                //phi2 = sin(phi2);
                //normals << normal[0] << "," << normal[1] << "," << normal[2] << "," << normal1[0] << "," << normal1[1] << "," << normal1[2] << "\n"; //<< "," << plane[0].distance() << "," << plane[1].distance() << "\n";
                normals << /*theta1 << "," << phi1 << "," <<*/ theta2 << "," << phi2 /*<< "," << plane[0].distance() << "," << plane[1].distance()*/ << "\n";
                normals.close();    
            } 
        }; 

        void generate_R2P_with_disturbance(double rota1_l, double rota1_h, double rotb1_l, double rotb1_h, double rota2_l, double rota2_h, double rotb2_l, double rotb2_h, double d1_l, double d1_h, double d2_l, double d2_h, bool inter, bool same)
        {
            srand((unsigned) time(NULL));
            for (int n = 0; n < Ntests; ++n) 
            {
                std::cout << n << endl;
                IRL::PlanarSeparator plane = gen->new_random_R2P(rota1_l, rota1_h, rotb1_l, rotb1_h, rota2_l, rota2_h, rotb2_l, rotb2_h, d1_l, d1_h, d2_l, d2_h, inter, same);
                //torch::Tensor result;
                //result = gen->get_fractions_all(plane);
                //while (result[((result.sizes()[0]-7)/2)].item<double>() > 0.5)
                {
                    //plane = gen->new_random_R2P(rota1_l, rota1_h, rotb1_l, rotb1_h, rota2_l, rota2_h, rotb2_l, rotb2_h, d1_l, d1_h, d2_l, d2_h, inter, same);
                    //result = gen->get_fractions_all(plane);
                }
                //result_all.index_put_({torch::indexing::Slice(), n}, result);
                std::ofstream coefficients;
                std::string name = "coefficients.txt";
                coefficients.open(name, std::ios_base::app);
                coefficients << plane[0].normal()[0] << "," << plane[0].normal()[1] << "," << plane[0].normal()[2] << "," << plane[0].distance() 
                << "," << plane[1].normal()[0] << "," << plane[1].normal()[1] << "," << plane[1].normal()[2] << "," << plane[1].distance() << "," << plane.flip() << "\n";
                coefficients.close();

                // std::ofstream type;
                // name = "type.txt";
                // type.open(name, std::ios_base::app);
                // type << std::to_string(1) << "\n";
                // type.close();

                torch::Tensor result;
                bool flip = true;//false;
                result = gen->get_fractions_all(plane);
                //result = gen->get_barycenters(plane);
                //result_all.index_put_({torch::indexing::Slice(), n}, result);
                torch::Tensor result1;
                IRL::PlanarSeparator p = IRL::PlanarSeparator::fromOnePlane(plane[0]);
                //result1 = gen->get_fractions_all(p);
                if (result[((result.sizes()[0]-7)/2)].item<double>() > 0.5)
                {
                    //flip = true;
                    //result = gen->get_fractions_gas_all(plane);
                    //result1 = gen->get_fractions_gas_all(p);
                }                    

                std::vector<double> fractions;
                //std::vector<double> fractions1;
                for (int i = 0; i < result.sizes()[0]; ++i)
                {
                    fractions.push_back(result[i].item<double>());
                    //fractions1.push_back(result1[i].item<double>());
                }

                int direction = 0;
                std::vector<double> center;
                double c = (rand() % 601 - 300) / 1000.0;
                center.push_back(plane[0].normal()[0] + c);
                center.push_back(plane[0].normal()[1] + c);
                center.push_back(plane[0].normal()[2] + c);
                direction = rotateFractions_all(&fractions,center);

                std::ofstream output;
                std::string data_name = "fractions.txt";
                output.open(data_name, std::ios_base::app);

                int pr = rand() % 8;
                int mod = 7;
                for (int i = 0; i < result.sizes()[0]; ++i)
                {
                    if (pr == 0)
                    {
                        output << fractions[i] << ",";
                    }
                    else
                    {
                        double c = (rand() % 101 - 50) / 1000.0;
                        if (i % mod != 0)
                        {
                            if (fractions[i] + c > 0.5)
                            {
                                output << 0.5 << ",";
                            }
                            else if (fractions[i] + c < -0.5)
                            {
                                output << -0.5 << ",";
                            }
                            else
                            {
                                output << fractions[i] + c << ",";
                            }
                        }
                        else
                        {
                            output << fractions[i] << ",";
                        }
                    }
                }
                output << "\n";
                output.close();

                std::ofstream normals;
                std::string normals_name = "normals.txt";
                normals.open(normals_name, std::ios_base::app);
                auto normal = IRL::Normal();
                auto normal1 = IRL::Normal();
                if (gen->arePlanesInSameCenterCell(plane))
                {
                    normal = plane[0].normal();
                    normal1 = plane[1].normal();
                }
                else
                {
                    normal = plane[0].normal();
                    normal1 = IRL::Normal(0,0,0);
                }

                vector<double> angles = gen->getR2PAngles();
                double theta1 = angles[0];
                double phi1 = angles[1];
                double theta2 = angles[2];
                double phi2 = angles[3];

                switch (direction)
                {
                    case 1:
                    normal[0] = -normal[0];
                    normal1[0] = -normal1[0];
                    theta1 = theta1 + 2*(M_PI/2-theta1);
                    if (theta1 > 2*M_PI)
                    {
                        theta1 = theta1 - 2*M_PI;
                    }
                    else if (theta1 < 0)
                    {
                        theta1 = theta1 + 2*M_PI;
                    }
                    theta2 = theta2 + 2*(M_PI/2-theta2);
                    if (theta2 > 2*M_PI)
                    {
                        theta2 = theta2 - 2*M_PI;
                    }
                    else if (theta2 < 0)
                    {
                        theta2 = theta2 + 2*M_PI;
                    }
                    break;
                    case 2:
                    normal[1] = -normal[1];
                    normal1[1] = -normal1[1];
                    theta1 = theta1 + 2*(M_PI-theta1);
                    if (theta1 > 2*M_PI)
                    {
                        theta1 = theta1 - 2*M_PI;
                    }
                    else if (theta1 < 0)
                    {
                        theta1 = theta1 + 2*M_PI;
                    }                    
                    theta2 = theta2 + 2*(M_PI-theta2);
                    if (theta2 > 2*M_PI)
                    {
                        theta2 = theta2 - 2*M_PI;
                    }
                    else if (theta2 < 0)
                    {
                        theta2 = theta2 + 2*M_PI;
                    }
                    break;
                    case 3:
                    normal[2] = -normal[2];
                    normal1[2] = -normal1[2];
                    phi1 = -phi1;
                    phi2 = -phi2;
                    break;
                    case 4:
                    normal[0] = -normal[0];
                    normal[1] = -normal[1];
                    normal1[0] = -normal1[0];
                    normal1[1] = -normal1[1];
                    theta1 = theta1 + 2*(M_PI/2-theta1);
                    if (theta1 > 2*M_PI)
                    {
                        theta1 = theta1 - 2*M_PI;
                    }
                    else if (theta1 < 0)
                    {
                        theta1 = theta1 + 2*M_PI;
                    }
                    theta2 = theta2 + 2*(M_PI/2-theta2);
                    if (theta2 > 2*M_PI)
                    {
                        theta2 = theta2 - 2*M_PI;
                    }
                    else if (theta2 < 0)
                    {
                        theta2 = theta2 + 2*M_PI;
                    }
                    theta1 = theta1 + 2*(M_PI-theta1);
                    if (theta1 > 2*M_PI)
                    {
                        theta1 = theta1 - 2*M_PI;
                    }
                    else if (theta1 < 0)
                    {
                        theta1 = theta1 + 2*M_PI;
                    }
                    theta2 = theta2 + 2*(M_PI-theta2);
                    if (theta2 > 2*M_PI)
                    {
                        theta2 = theta2 - 2*M_PI;
                    }
                    else if (theta2 < 0)
                    {
                        theta2 = theta2 + 2*M_PI;
                    }
                    break;
                    case 5:
                    normal[0] = -normal[0];
                    normal[2] = -normal[2];
                    normal1[0] = -normal1[0];
                    normal1[2] = -normal1[2]; 
                    theta1 = theta1 + 2*(M_PI/2-theta1);
                    if (theta1 > 2*M_PI)
                    {
                        theta1 = theta1 - 2*M_PI;
                    }
                    else if (theta1 < 0)
                    {
                        theta1 = theta1 + 2*M_PI;
                    }
                    theta2 = theta2 + 2*(M_PI/2-theta2);
                    if (theta2 > 2*M_PI)
                    {
                        theta2 = theta2 - 2*M_PI;
                    }
                    else if (theta2 < 0)
                    {
                        theta2 = theta2 + 2*M_PI;
                    }
                    phi1 = -phi1;
                    phi2 = -phi2;
                    break;
                    case 6:
                    normal[1] = -normal[1];
                    normal[2] = -normal[2];
                    normal1[1] = -normal1[1];
                    normal1[2] = -normal1[2];
                    theta1 = theta1 + 2*(M_PI-theta1);
                    if (theta1 > 2*M_PI)
                    {
                        theta1 = theta1 - 2*M_PI;
                    }
                    else if (theta1 < 0)
                    {
                        theta1 = theta1 + 2*M_PI;
                    }
                    theta2 = theta2 + 2*(M_PI-theta2);
                    if (theta2 > 2*M_PI)
                    {
                        theta2 = theta2 - 2*M_PI;
                    }
                    else if (theta2 < 0)
                    {
                        theta2 = theta2 + 2*M_PI;
                    }
                    phi1 = -phi1;
                    phi2 = -phi2;
                    break;
                    case 7:
                    normal[0] = -normal[0];
                    normal[1] = -normal[1];
                    normal[2] = -normal[2];
                    normal1[0] = -normal1[0];
                    normal1[1] = -normal1[1];
                    normal1[2] = -normal1[2];
                    theta1 = theta1 + 2*(M_PI/2-theta1);
                    if (theta1 > 2*M_PI)
                    {
                        theta1 = theta1 - 2*M_PI;
                    }
                    else if (theta1 < 0)
                    {
                        theta1 = theta1 + 2*M_PI;
                    }
                    theta2 = theta2 + 2*(M_PI/2-theta2);
                    if (theta2 > 2*M_PI)
                    {
                        theta2 = theta2 - 2*M_PI;
                    }
                    else if (theta2 < 0)
                    {
                        theta2 = theta2 + 2*M_PI;
                    }
                    theta1 = theta1 + 2*(M_PI-theta1);
                    if (theta1 > 2*M_PI)
                    {
                        theta1 = theta1 - 2*M_PI;
                    }
                    else if (theta1 < 0)
                    {
                        theta1 = theta1 + 2*M_PI;
                    }
                    theta2 = theta2 + 2*(M_PI-theta2);
                    if (theta2 > 2*M_PI)
                    {
                        theta2 = theta2 - 2*M_PI;
                    }
                    else if (theta2 < 0)
                    {
                        theta2 = theta2 + 2*M_PI;
                    }
                    phi1 = -phi1;
                    phi2 = -phi2;
                    break;
                }
                if (!flip)
                {
                    normal[0] = -normal[0];
                    normal[1] = -normal[1];
                    normal[2] = -normal[2];
                    normal1[0] = -normal1[0];
                    normal1[1] = -normal1[1];
                    normal1[2] = -normal1[2];
                }

                //double theta1 = atan2(normal[1],normal[0]);
                //double phi1 = acos(normal[2]/normal.calculateMagnitude());
                //double theta2 = atan2(normal1[1],normal1[0]);
                //double phi2 = acos(normal1[2]/normal1.calculateMagnitude());
                //std::cout << theta1 << "," << phi1 << "," << theta2 << "," << phi2 << std::endl << std::endl;
                //theta1 = sin(theta1);
                //phi1 = sin(phi1);
                //theta2 = sin(theta2);
                //phi2 = sin(phi2);
                //normals << normal[0] << "," << normal[1] << "," << normal[2] << "," << normal1[0] << "," << normal1[1] << "," << normal1[2] << "\n"; //<< "," << plane[0].distance() << "," << plane[1].distance() << "\n";
                normals << /*theta1 << "," << phi1 << "," <<*/ theta2 << "," << phi2 /*<< "," << plane[0].distance() << "," << plane[1].distance()*/ << "\n";
                normals.close();   
            }  
        };





        //************************************************************
        //**************ROTATE****************************************
        //************************************************************

        int rotateFractions(std::vector<double>* fractions1, std::vector<double> center)
        {
            std::vector<double> fractions = *fractions1;
            int direction = 0;
            if (center[0] < 0 && center[1] >= 0 && center[2] >= 0)
            {
                direction = 1;
                for (int i = 0; i < 3; ++i)
                {
                for (int j = 0; j < 3; ++j)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                    if (i == 0)
                    {
                        double temp = fractions[/*7*/4*(i*9+j*3+k)+0];
                        fractions[/*7*/4*(i*9+j*3+k)+0] = fractions[/*7*/4*(2*9+j*3+k)+0];
                        fractions[/*7*/4*(2*9+j*3+k)+0] = temp;
                        temp = fractions[/*7*/4*(i*9+j*3+k)+1];
                        fractions[/*7*/4*(i*9+j*3+k)+1] = -fractions[/*7*/4*(2*9+j*3+k)+1];
                        fractions[/*7*/4*(2*9+j*3+k)+1] = -temp;
                        temp = fractions[/*7*/4*(i*9+j*3+k)+2];
                        fractions[/*7*/4*(i*9+j*3+k)+2] = fractions[/*7*/4*(2*9+j*3+k)+2];
                        fractions[/*7*/4*(2*9+j*3+k)+2] = temp;
                        temp = fractions[/*7*/4*(i*9+j*3+k)+3];
                        fractions[/*7*/4*(i*9+j*3+k)+3] = fractions[/*7*/4*(2*9+j*3+k)+3];
                        fractions[/*7*/4*(2*9+j*3+k)+3] = temp;
                        /*temp = fractions[7*(i*9+j*3+k)+4];
                        fractions[7*(i*9+j*3+k)+4] = -fractions[7*(2*9+j*3+k)+4];
                        fractions[7*(2*9+j*3+k)+4] = -temp;
                        temp = fractions[7*(i*9+j*3+k)+5];
                        fractions[7*(i*9+j*3+k)+5] = fractions[7*(2*9+j*3+k)+5];
                        fractions[7*(2*9+j*3+k)+5] = temp;
                        temp = fractions[7*(i*9+j*3+k)+6];
                        fractions[7*(i*9+j*3+k)+6] = fractions[7*(2*9+j*3+k)+6];
                        fractions[7*(2*9+j*3+k)+6] = temp;*/
                    }
                    else if (i == 1)
                    {
                        fractions[/*7*/4*(i*9+j*3+k)+1] = -fractions[/*7*/4*(i*9+j*3+k)+1];
                        //fractions[7*(i*9+j*3+k)+4] = -fractions[7*(i*9+j*3+k)+4];
                    }
                    }
                }
                }
            }
            else if (center[0] >= 0 && center[1] < 0 && center[2] >= 0)
            {
                direction = 2;
                for (int i = 0; i < 3; ++i)
                {
                for (int j = 0; j < 3; ++j)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                    if (j == 0)
                    {
                        double temp = fractions[/*7*/4*(i*9+j*3+k)+0];
                        fractions[/*7*/4*(i*9+j*3+k)+0] = fractions[/*7*/4*(i*9+2*3+k)+0];
                        fractions[/*7*/4*(i*9+2*3+k)+0] = temp;
                        temp = fractions[/*7*/4*(i*9+j*3+k)+1];
                        fractions[/*7*/4*(i*9+j*3+k)+1] = fractions[/*7*/4*(i*9+2*3+k)+1];
                        fractions[/*7*/4*(i*9+2*3+k)+1] = temp;
                        temp = fractions[/*7*/4*(i*9+j*3+k)+2];
                        fractions[/*7*/4*(i*9+j*3+k)+2] = -fractions[/*7*/4*(i*9+2*3+k)+2];
                        fractions[/*7*/4*(i*9+2*3+k)+2] = -temp;
                        temp = fractions[/*7*/4*(i*9+j*3+k)+3];
                        fractions[/*7*/4*(i*9+j*3+k)+3] = fractions[/*7*/4*(i*9+2*3+k)+3];
                        fractions[/*7*/4*(i*9+2*3+k)+3] = temp;
                        /*temp = fractions[7*(i*9+j*3+k)+4];
                        fractions[7*(i*9+j*3+k)+4] = fractions[7*(i*9+2*3+k)+4];
                        fractions[7*(i*9+2*3+k)+4] = temp;
                        temp = fractions[7*(i*9+j*3+k)+5];
                        fractions[7*(i*9+j*3+k)+5] = -fractions[7*(i*9+2*3+k)+5];
                        fractions[7*(i*9+2*3+k)+5] = -temp;
                        temp = fractions[7*(i*9+j*3+k)+6];
                        fractions[7*(i*9+j*3+k)+6] = fractions[7*(i*9+2*3+k)+6];
                        fractions[7*(i*9+2*3+k)+6] = temp;*/
                    }
                    else if (j == 1)
                    {
                        fractions[/*7*/4*(i*9+j*3+k)+2] = -fractions[/*7*/4*(i*9+j*3+k)+2];
                        //fractions[7*(i*9+j*3+k)+5] = -fractions[7*(i*9+j*3+k)+5];
                    }
                    }
                }
                }
            }
            else if (center[0] >= 0 && center[1] >= 0 && center[2] < 0)
            {
                direction = 3;
                for (int i = 0; i < 3; ++i)
                {
                for (int j = 0; j < 3; ++j)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                    if (k == 0)
                    {
                        double temp = fractions[/*7*/4*(i*9+j*3+k)+0];
                        fractions[/*7*/4*(i*9+j*3+k)+0] = fractions[/*7*/4*(i*9+j*3+2)+0];
                        fractions[/*7*/4*(i*9+j*3+2)+0] = temp;
                        temp = fractions[/*7*/4*(i*9+j*3+k)+1];
                        fractions[/*7*/4*(i*9+j*3+k)+1] = fractions[/*7*/4*(i*9+j*3+2)+1];
                        fractions[/*7*/4*(i*9+j*3+2)+1] = temp;
                        temp = fractions[/*7*/4*(i*9+j*3+k)+2];
                        fractions[/*7*/4*(i*9+j*3+k)+2] = fractions[/*7*/4*(i*9+j*3+2)+2];
                        fractions[/*7*/4*(i*9+j*3+2)+2] = temp;
                        temp = fractions[/*7*/4*(i*9+j*3+k)+3];
                        fractions[/*7*/4*(i*9+j*3+k)+3] = -fractions[/*7*/4*(i*9+j*3+2)+3];
                        fractions[/*7*/4*(i*9+j*3+2)+3] = -temp;
                        /*temp = fractions[7*(i*9+j*3+k)+4];
                        fractions[7*(i*9+j*3+k)+4] = fractions[7*(i*9+j*3+2)+4];
                        fractions[7*(i*9+j*3+2)+4] = temp;
                        temp = fractions[7*(i*9+j*3+k)+5];
                        fractions[7*(i*9+j*3+k)+5] = fractions[7*(i*9+j*3+2)+5];
                        fractions[7*(i*9+j*3+2)+5] = temp;
                        temp = fractions[7*(i*9+j*3+k)+6];
                        fractions[7*(i*9+j*3+k)+6] = -fractions[7*(i*9+j*3+2)+6];
                        fractions[7*(i*9+j*3+2)+6] = -temp;*/
                    }
                    else if (k == 1)
                    {
                        fractions[/*7*/4*(i*9+j*3+k)+3] = -fractions[/*7*/4*(i*9+j*3+k)+3];
                        //fractions[7*(i*9+j*3+k)+6] = -fractions[7*(i*9+j*3+k)+6];
                    }
                    }
                }
                }
            }
            else if (center[0] < 0 && center[1] < 0 && center[2] >= 0)
            {
                direction = 4;
                for (int i = 0; i < 3; ++i)
                {
                for (int j = 0; j < 3; ++j)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                    if (i == 0)
                    {
                        double temp = fractions[/*7*/4*(i*9+j*3+k)+0];
                        fractions[/*7*/4*(i*9+j*3+k)+0] = fractions[/*7*/4*(2*9+j*3+k)+0];
                        fractions[/*7*/4*(2*9+j*3+k)+0] = temp;
                        temp = fractions[/*7*/4*(i*9+j*3+k)+1];
                        fractions[/*7*/4*(i*9+j*3+k)+1] = -fractions[/*7*/4*(2*9+j*3+k)+1];
                        fractions[/*7*/4*(2*9+j*3+k)+1] = -temp;
                        temp = fractions[/*7*/4*(i*9+j*3+k)+2];
                        fractions[/*7*/4*(i*9+j*3+k)+2] = fractions[/*7*/4*(2*9+j*3+k)+2];
                        fractions[/*7*/4*(2*9+j*3+k)+2] = temp;
                        temp = fractions[/*7*/4*(i*9+j*3+k)+3];
                        fractions[/*7*/4*(i*9+j*3+k)+3] = fractions[/*7*/4*(2*9+j*3+k)+3];
                        fractions[/*7*/4*(2*9+j*3+k)+3] = temp;
                        /*temp = fractions[7*(i*9+j*3+k)+4];
                        fractions[7*(i*9+j*3+k)+4] = -fractions[7*(2*9+j*3+k)+4];
                        fractions[7*(2*9+j*3+k)+4] = -temp;
                        temp = fractions[7*(i*9+j*3+k)+5];
                        fractions[7*(i*9+j*3+k)+5] = fractions[7*(2*9+j*3+k)+5];
                        fractions[7*(2*9+j*3+k)+5] = temp;
                        temp = fractions[7*(i*9+j*3+k)+6];
                        fractions[7*(i*9+j*3+k)+6] = fractions[7*(2*9+j*3+k)+6];
                        fractions[7*(2*9+j*3+k)+6] = temp;*/
                    }
                    else if (i == 1)
                    {
                        fractions[/*7*/4*(i*9+j*3+k)+1] = -fractions[/*7*/4*(i*9+j*3+k)+1];
                        //fractions[7*(i*9+j*3+k)+4] = -fractions[7*(i*9+j*3+k)+4];
                    }
                    }
                }
                }
                for (int i = 0; i < 3; ++i)
                {
                for (int j = 0; j < 3; ++j)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                    if (j == 0)
                    {
                        double temp = fractions[/*7*/4*(i*9+j*3+k)+0];
                        fractions[/*7*/4*(i*9+j*3+k)+0] = fractions[/*7*/4*(i*9+2*3+k)+0];
                        fractions[/*7*/4*(i*9+2*3+k)+0] = temp;
                        temp = fractions[/*7*/4*(i*9+j*3+k)+1];
                        fractions[/*7*/4*(i*9+j*3+k)+1] = fractions[/*7*/4*(i*9+2*3+k)+1];
                        fractions[/*7*/4*(i*9+2*3+k)+1] = temp;
                        temp = fractions[/*7*/4*(i*9+j*3+k)+2];
                        fractions[/*7*/4*(i*9+j*3+k)+2] = -fractions[/*7*/4*(i*9+2*3+k)+2];
                        fractions[/*7*/4*(i*9+2*3+k)+2] = -temp;
                        temp = fractions[/*7*/4*(i*9+j*3+k)+3];
                        fractions[/*7*/4*(i*9+j*3+k)+3] = fractions[/*7*/4*(i*9+2*3+k)+3];
                        fractions[/*7*/4*(i*9+2*3+k)+3] = temp;
                        /*temp = fractions[7*(i*9+j*3+k)+4];
                        fractions[7*(i*9+j*3+k)+4] = fractions[7*(i*9+2*3+k)+4];
                        fractions[7*(i*9+2*3+k)+4] = temp;
                        temp = fractions[7*(i*9+j*3+k)+5];
                        fractions[7*(i*9+j*3+k)+5] = -fractions[7*(i*9+2*3+k)+5];
                        fractions[7*(i*9+2*3+k)+5] = -temp;
                        temp = fractions[7*(i*9+j*3+k)+6];
                        fractions[7*(i*9+j*3+k)+6] = fractions[7*(i*9+2*3+k)+6];
                        fractions[7*(i*9+2*3+k)+6] = temp;*/
                    }
                    else if (j == 1)
                    {
                        fractions[/*7*/4*(i*9+j*3+k)+2] = -fractions[/*7*/4*(i*9+j*3+k)+2];
                        //fractions[7*(i*9+j*3+k)+5] = -fractions[7*(i*9+j*3+k)+5];
                    }
                    }
                }
                }
            }
            else if (center[0] < 0 && center[1] >= 0 && center[2] < 0)
            {
                direction = 5;
                for (int i = 0; i < 3; ++i)
                {
                for (int j = 0; j < 3; ++j)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                    if (i == 0)
                    {
                        double temp = fractions[/*7*/4*(i*9+j*3+k)+0];
                        fractions[/*7*/4*(i*9+j*3+k)+0] = fractions[/*7*/4*(2*9+j*3+k)+0];
                        fractions[/*7*/4*(2*9+j*3+k)+0] = temp;
                        temp = fractions[/*7*/4*(i*9+j*3+k)+1];
                        fractions[/*7*/4*(i*9+j*3+k)+1] = -fractions[/*7*/4*(2*9+j*3+k)+1];
                        fractions[/*7*/4*(2*9+j*3+k)+1] = -temp;
                        temp = fractions[/*7*/4*(i*9+j*3+k)+2];
                        fractions[/*7*/4*(i*9+j*3+k)+2] = fractions[/*7*/4*(2*9+j*3+k)+2];
                        fractions[/*7*/4*(2*9+j*3+k)+2] = temp;
                        temp = fractions[/*7*/4*(i*9+j*3+k)+3];
                        fractions[/*7*/4*(i*9+j*3+k)+3] = fractions[/*7*/4*(2*9+j*3+k)+3];
                        fractions[/*7*/4*(2*9+j*3+k)+3] = temp;
                        /*temp = fractions[7*(i*9+j*3+k)+4];
                        fractions[7*(i*9+j*3+k)+4] = -fractions[7*(2*9+j*3+k)+4];
                        fractions[7*(2*9+j*3+k)+4] = -temp;
                        temp = fractions[7*(i*9+j*3+k)+5];
                        fractions[7*(i*9+j*3+k)+5] = fractions[7*(2*9+j*3+k)+5];
                        fractions[7*(2*9+j*3+k)+5] = temp;
                        temp = fractions[7*(i*9+j*3+k)+6];
                        fractions[7*(i*9+j*3+k)+6] = fractions[7*(2*9+j*3+k)+6];
                        fractions[7*(2*9+j*3+k)+6] = temp;*/
                    }
                    else if (i == 1)
                    {
                        fractions[/*7*/4*(i*9+j*3+k)+1] = -fractions[/*7*/4*(i*9+j*3+k)+1];
                        //fractions[7*(i*9+j*3+k)+4] = -fractions[7*(i*9+j*3+k)+4];
                    }
                    }
                }
                }
                for (int i = 0; i < 3; ++i)
                {
                for (int j = 0; j < 3; ++j)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                    if (k == 0)
                    {
                        double temp = fractions[/*7*/4*(i*9+j*3+k)+0];
                        fractions[/*7*/4*(i*9+j*3+k)+0] = fractions[/*7*/4*(i*9+j*3+2)+0];
                        fractions[/*7*/4*(i*9+j*3+2)+0] = temp;
                        temp = fractions[/*7*/4*(i*9+j*3+k)+1];
                        fractions[/*7*/4*(i*9+j*3+k)+1] = fractions[/*7*/4*(i*9+j*3+2)+1];
                        fractions[/*7*/4*(i*9+j*3+2)+1] = temp;
                        temp = fractions[/*7*/4*(i*9+j*3+k)+2];
                        fractions[/*7*/4*(i*9+j*3+k)+2] = fractions[/*7*/4*(i*9+j*3+2)+2];
                        fractions[/*7*/4*(i*9+j*3+2)+2] = temp;
                        temp = fractions[/*7*/4*(i*9+j*3+k)+3];
                        fractions[/*7*/4*(i*9+j*3+k)+3] = -fractions[/*7*/4*(i*9+j*3+2)+3];
                        fractions[/*7*/4*(i*9+j*3+2)+3] = -temp;
                        /*temp = fractions[7*(i*9+j*3+k)+4];
                        fractions[7*(i*9+j*3+k)+4] = fractions[7*(i*9+j*3+2)+4];
                        fractions[7*(i*9+j*3+2)+4] = temp;
                        temp = fractions[7*(i*9+j*3+k)+5];
                        fractions[7*(i*9+j*3+k)+5] = fractions[7*(i*9+j*3+2)+5];
                        fractions[7*(i*9+j*3+2)+5] = temp;
                        temp = fractions[7*(i*9+j*3+k)+6];
                        fractions[7*(i*9+j*3+k)+6] = -fractions[7*(i*9+j*3+2)+6];
                        fractions[7*(i*9+j*3+2)+6] = -temp;*/
                    }
                    else if (k == 1)
                    {
                        fractions[/*7*/4*(i*9+j*3+k)+3] = -fractions[/*7*/4*(i*9+j*3+k)+3];
                        //fractions[7*(i*9+j*3+k)+6] = -fractions[7*(i*9+j*3+k)+6];
                    }
                    }
                }
                }
            }
            else if (center[0] >= 0 && center[1] < 0 && center[2] < 0)
            {
                direction = 6;
                for (int i = 0; i < 3; ++i)
                {
                for (int j = 0; j < 3; ++j)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                    if (j == 0)
                    {
                        double temp = fractions[/*7*/4*(i*9+j*3+k)+0];
                        fractions[/*7*/4*(i*9+j*3+k)+0] = fractions[/*7*/4*(i*9+2*3+k)+0];
                        fractions[/*7*/4*(i*9+2*3+k)+0] = temp;
                        temp = fractions[/*7*/4*(i*9+j*3+k)+1];
                        fractions[/*7*/4*(i*9+j*3+k)+1] = fractions[/*7*/4*(i*9+2*3+k)+1];
                        fractions[/*7*/4*(i*9+2*3+k)+1] = temp;
                        temp = fractions[/*7*/4*(i*9+j*3+k)+2];
                        fractions[/*7*/4*(i*9+j*3+k)+2] = -fractions[/*7*/4*(i*9+2*3+k)+2];
                        fractions[/*7*/4*(i*9+2*3+k)+2] = -temp;
                        temp = fractions[/*7*/4*(i*9+j*3+k)+3];
                        fractions[/*7*/4*(i*9+j*3+k)+3] = fractions[/*7*/4*(i*9+2*3+k)+3];
                        fractions[/*7*/4*(i*9+2*3+k)+3] = temp;
                        /*temp = fractions[7*(i*9+j*3+k)+4];
                        fractions[7*(i*9+j*3+k)+4] = fractions[7*(i*9+2*3+k)+4];
                        fractions[7*(i*9+2*3+k)+4] = temp;
                        temp = fractions[7*(i*9+j*3+k)+5];
                        fractions[7*(i*9+j*3+k)+5] = -fractions[7*(i*9+2*3+k)+5];
                        fractions[7*(i*9+2*3+k)+5] = -temp;
                        temp = fractions[7*(i*9+j*3+k)+6];
                        fractions[7*(i*9+j*3+k)+6] = fractions[7*(i*9+2*3+k)+6];
                        fractions[7*(i*9+2*3+k)+6] = temp;*/
                    }
                    else if (j == 1)
                    {
                        fractions[/*7*/4*(i*9+j*3+k)+2] = -fractions[/*7*/4*(i*9+j*3+k)+2];
                        //fractions[7*(i*9+j*3+k)+5] = -fractions[7*(i*9+j*3+k)+5];
                    }
                    }
                }
                }
                for (int i = 0; i < 3; ++i)
                {
                for (int j = 0; j < 3; ++j)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                    if (k == 0)
                    {
                        double temp = fractions[/*7*/4*(i*9+j*3+k)+0];
                        fractions[/*7*/4*(i*9+j*3+k)+0] = fractions[/*7*/4*(i*9+j*3+2)+0];
                        fractions[/*7*/4*(i*9+j*3+2)+0] = temp;
                        temp = fractions[/*7*/4*(i*9+j*3+k)+1];
                        fractions[/*7*/4*(i*9+j*3+k)+1] = fractions[/*7*/4*(i*9+j*3+2)+1];
                        fractions[/*7*/4*(i*9+j*3+2)+1] = temp;
                        temp = fractions[/*7*/4*(i*9+j*3+k)+2];
                        fractions[/*7*/4*(i*9+j*3+k)+2] = fractions[/*7*/4*(i*9+j*3+2)+2];
                        fractions[/*7*/4*(i*9+j*3+2)+2] = temp;
                        temp = fractions[/*7*/4*(i*9+j*3+k)+3];
                        fractions[/*7*/4*(i*9+j*3+k)+3] = -fractions[/*7*/4*(i*9+j*3+2)+3];
                        fractions[/*7*/4*(i*9+j*3+2)+3] = -temp;
                        /*temp = fractions[7*(i*9+j*3+k)+4];
                        fractions[7*(i*9+j*3+k)+4] = fractions[7*(i*9+j*3+2)+4];
                        fractions[7*(i*9+j*3+2)+4] = temp;
                        temp = fractions[7*(i*9+j*3+k)+5];
                        fractions[7*(i*9+j*3+k)+5] = fractions[7*(i*9+j*3+2)+5];
                        fractions[7*(i*9+j*3+2)+5] = temp;
                        temp = fractions[7*(i*9+j*3+k)+6];
                        fractions[7*(i*9+j*3+k)+6] = -fractions[7*(i*9+j*3+2)+6];
                        fractions[7*(i*9+j*3+2)+6] = -temp;*/
                    }
                    else if (k == 1)
                    {
                        fractions[/*7*/4*(i*9+j*3+k)+3] = -fractions[/*7*/4*(i*9+j*3+k)+3];
                        //fractions[7*(i*9+j*3+k)+6] = -fractions[7*(i*9+j*3+k)+6];
                    }
                    }
                }
                }
            }
            else if (center[0] < 0 && center[1] < 0 && center[2] < 0)
            {
                direction = 7;
                for (int i = 0; i < 3; ++i)
                {
                for (int j = 0; j < 3; ++j)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                    if (i == 0)
                    {
                        double temp = fractions[/*7*/4*(i*9+j*3+k)+0];
                        fractions[/*7*/4*(i*9+j*3+k)+0] = fractions[/*7*/4*(2*9+j*3+k)+0];
                        fractions[/*7*/4*(2*9+j*3+k)+0] = temp;
                        temp = fractions[/*7*/4*(i*9+j*3+k)+1];
                        fractions[/*7*/4*(i*9+j*3+k)+1] = -fractions[/*7*/4*(2*9+j*3+k)+1];
                        fractions[/*7*/4*(2*9+j*3+k)+1] = -temp;
                        temp = fractions[/*7*/4*(i*9+j*3+k)+2];
                        fractions[/*7*/4*(i*9+j*3+k)+2] = fractions[/*7*/4*(2*9+j*3+k)+2];
                        fractions[/*7*/4*(2*9+j*3+k)+2] = temp;
                        temp = fractions[/*7*/4*(i*9+j*3+k)+3];
                        fractions[/*7*/4*(i*9+j*3+k)+3] = fractions[/*7*/4*(2*9+j*3+k)+3];
                        fractions[/*7*/4*(2*9+j*3+k)+3] = temp;
                        /*temp = fractions[7*(i*9+j*3+k)+4];
                        fractions[7*(i*9+j*3+k)+4] = -fractions[7*(2*9+j*3+k)+4];
                        fractions[7*(2*9+j*3+k)+4] = -temp;
                        temp = fractions[7*(i*9+j*3+k)+5];
                        fractions[7*(i*9+j*3+k)+5] = fractions[7*(2*9+j*3+k)+5];
                        fractions[7*(2*9+j*3+k)+5] = temp;
                        temp = fractions[7*(i*9+j*3+k)+6];
                        fractions[7*(i*9+j*3+k)+6] = fractions[7*(2*9+j*3+k)+6];
                        fractions[7*(2*9+j*3+k)+6] = temp;*/
                    }
                    else if (i == 1)
                    {
                        fractions[/*7*/4*(i*9+j*3+k)+1] = -fractions[/*7*/4*(i*9+j*3+k)+1];
                        //fractions[7*(i*9+j*3+k)+4] = -fractions[7*(i*9+j*3+k)+4];
                    }
                    }
                }
                }
                for (int i = 0; i < 3; ++i)
                {
                for (int j = 0; j < 3; ++j)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                    if (j == 0)
                    {
                        double temp = fractions[/*7*/4*(i*9+j*3+k)+0];
                        fractions[/*7*/4*(i*9+j*3+k)+0] = fractions[/*7*/4*(i*9+2*3+k)+0];
                        fractions[/*7*/4*(i*9+2*3+k)+0] = temp;
                        temp = fractions[/*7*/4*(i*9+j*3+k)+1];
                        fractions[/*7*/4*(i*9+j*3+k)+1] = fractions[/*7*/4*(i*9+2*3+k)+1];
                        fractions[/*7*/4*(i*9+2*3+k)+1] = temp;
                        temp = fractions[/*7*/4*(i*9+j*3+k)+2];
                        fractions[/*7*/4*(i*9+j*3+k)+2] = -fractions[/*7*/4*(i*9+2*3+k)+2];
                        fractions[/*7*/4*(i*9+2*3+k)+2] = -temp;
                        temp = fractions[/*7*/4*(i*9+j*3+k)+3];
                        fractions[/*7*/4*(i*9+j*3+k)+3] = fractions[/*7*/4*(i*9+2*3+k)+3];
                        fractions[/*7*/4*(i*9+2*3+k)+3] = temp;
                        /*temp = fractions[7*(i*9+j*3+k)+4];
                        fractions[7*(i*9+j*3+k)+4] = fractions[7*(i*9+2*3+k)+4];
                        fractions[7*(i*9+2*3+k)+4] = temp;
                        temp = fractions[7*(i*9+j*3+k)+5];
                        fractions[7*(i*9+j*3+k)+5] = -fractions[7*(i*9+2*3+k)+5];
                        fractions[7*(i*9+2*3+k)+5] = -temp;
                        temp = fractions[7*(i*9+j*3+k)+6];
                        fractions[7*(i*9+j*3+k)+6] = fractions[7*(i*9+2*3+k)+6];
                        fractions[7*(i*9+2*3+k)+6] = temp;*/
                    }
                    else if (j == 1)
                    {
                        fractions[/*7*/4*(i*9+j*3+k)+2] = -fractions[/*7*/4*(i*9+j*3+k)+2];
                        //fractions[7*(i*9+j*3+k)+5] = -fractions[7*(i*9+j*3+k)+5];
                    }
                    }
                }
                }
                for (int i = 0; i < 3; ++i)
                {
                for (int j = 0; j < 3; ++j)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                    if (k == 0)
                    {
                        double temp = fractions[/*7*/4*(i*9+j*3+k)+0];
                        fractions[/*7*/4*(i*9+j*3+k)+0] = fractions[/*7*/4*(i*9+j*3+2)+0];
                        fractions[/*7*/4*(i*9+j*3+2)+0] = temp;
                        temp = fractions[/*7*/4*(i*9+j*3+k)+1];
                        fractions[/*7*/4*(i*9+j*3+k)+1] = fractions[/*7*/4*(i*9+j*3+2)+1];
                        fractions[/*7*/4*(i*9+j*3+2)+1] = temp;
                        temp = fractions[/*7*/4*(i*9+j*3+k)+2];
                        fractions[/*7*/4*(i*9+j*3+k)+2] = fractions[/*7*/4*(i*9+j*3+2)+2];
                        fractions[/*7*/4*(i*9+j*3+2)+2] = temp;
                        temp = fractions[/*7*/4*(i*9+j*3+k)+3];
                        fractions[/*7*/4*(i*9+j*3+k)+3] = -fractions[/*7*/4*(i*9+j*3+2)+3];
                        fractions[/*7*/4*(i*9+j*3+2)+3] = -temp;
                        /*temp = fractions[7*(i*9+j*3+k)+4];
                        fractions[7*(i*9+j*3+k)+4] = fractions[7*(i*9+j*3+2)+4];
                        fractions[7*(i*9+j*3+2)+4] = temp;
                        temp = fractions[7*(i*9+j*3+k)+5];
                        fractions[7*(i*9+j*3+k)+5] = fractions[7*(i*9+j*3+2)+5];
                        fractions[7*(i*9+j*3+2)+5] = temp;
                        temp = fractions[7*(i*9+j*3+k)+6];
                        fractions[7*(i*9+j*3+k)+6] = -fractions[7*(i*9+j*3+2)+6];
                        fractions[7*(i*9+j*3+2)+6] = -temp;*/
                    }
                    else if (k == 1)
                    {
                        fractions[/*7*/4*(i*9+j*3+k)+3] = -fractions[/*7*/4*(i*9+j*3+k)+3];
                        //fractions[7*(i*9+j*3+k)+6] = -fractions[7*(i*9+j*3+k)+6];
                    }
                    }
                }
                }
            }
            *fractions1 = fractions;
            return direction;
        };

        int rotateFractions_all(std::vector<double>* fractions1, std::vector<double> center)
        {
            std::vector<double> fractions = *fractions1;
            int direction = 0;
            if (center[0] < 0 && center[1] >= 0 && center[2] >= 0)
            {
                direction = 1;
                for (int i = 0; i < 3; ++i)
                {
                for (int j = 0; j < 3; ++j)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                    if (i == 0)
                    {
                        double temp = fractions[7*(i*9+j*3+k)+0];
                        fractions[7*(i*9+j*3+k)+0] = fractions[7*(2*9+j*3+k)+0];
                        fractions[7*(2*9+j*3+k)+0] = temp;
                        temp = fractions[7*(i*9+j*3+k)+1];
                        fractions[7*(i*9+j*3+k)+1] = -fractions[7*(2*9+j*3+k)+1];
                        fractions[7*(2*9+j*3+k)+1] = -temp;
                        temp = fractions[7*(i*9+j*3+k)+2];
                        fractions[7*(i*9+j*3+k)+2] = fractions[7*(2*9+j*3+k)+2];
                        fractions[7*(2*9+j*3+k)+2] = temp;
                        temp = fractions[7*(i*9+j*3+k)+3];
                        fractions[7*(i*9+j*3+k)+3] = fractions[7*(2*9+j*3+k)+3];
                        fractions[7*(2*9+j*3+k)+3] = temp;
                        temp = fractions[7*(i*9+j*3+k)+4];
                        fractions[7*(i*9+j*3+k)+4] = -fractions[7*(2*9+j*3+k)+4];
                        fractions[7*(2*9+j*3+k)+4] = -temp;
                        temp = fractions[7*(i*9+j*3+k)+5];
                        fractions[7*(i*9+j*3+k)+5] = fractions[7*(2*9+j*3+k)+5];
                        fractions[7*(2*9+j*3+k)+5] = temp;
                        temp = fractions[7*(i*9+j*3+k)+6];
                        fractions[7*(i*9+j*3+k)+6] = fractions[7*(2*9+j*3+k)+6];
                        fractions[7*(2*9+j*3+k)+6] = temp;
                    }
                    else if (i == 1)
                    {
                        fractions[7*(i*9+j*3+k)+1] = -fractions[7*(i*9+j*3+k)+1];
                        fractions[7*(i*9+j*3+k)+4] = -fractions[7*(i*9+j*3+k)+4];
                    }
                    }
                }
                }
            }
            else if (center[0] >= 0 && center[1] < 0 && center[2] >= 0)
            {
                direction = 2;
                for (int i = 0; i < 3; ++i)
                {
                for (int j = 0; j < 3; ++j)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                    if (j == 0)
                    {
                        double temp = fractions[7*(i*9+j*3+k)+0];
                        fractions[7*(i*9+j*3+k)+0] = fractions[7*(i*9+2*3+k)+0];
                        fractions[7*(i*9+2*3+k)+0] = temp;
                        temp = fractions[7*(i*9+j*3+k)+1];
                        fractions[7*(i*9+j*3+k)+1] = fractions[7*(i*9+2*3+k)+1];
                        fractions[7*(i*9+2*3+k)+1] = temp;
                        temp = fractions[7*(i*9+j*3+k)+2];
                        fractions[7*(i*9+j*3+k)+2] = -fractions[7*(i*9+2*3+k)+2];
                        fractions[7*(i*9+2*3+k)+2] = -temp;
                        temp = fractions[7*(i*9+j*3+k)+3];
                        fractions[7*(i*9+j*3+k)+3] = fractions[7*(i*9+2*3+k)+3];
                        fractions[7*(i*9+2*3+k)+3] = temp;
                        temp = fractions[7*(i*9+j*3+k)+4];
                        fractions[7*(i*9+j*3+k)+4] = fractions[7*(i*9+2*3+k)+4];
                        fractions[7*(i*9+2*3+k)+4] = temp;
                        temp = fractions[7*(i*9+j*3+k)+5];
                        fractions[7*(i*9+j*3+k)+5] = -fractions[7*(i*9+2*3+k)+5];
                        fractions[7*(i*9+2*3+k)+5] = -temp;
                        temp = fractions[7*(i*9+j*3+k)+6];
                        fractions[7*(i*9+j*3+k)+6] = fractions[7*(i*9+2*3+k)+6];
                        fractions[7*(i*9+2*3+k)+6] = temp;
                    }
                    else if (j == 1)
                    {
                        fractions[7*(i*9+j*3+k)+2] = -fractions[7*(i*9+j*3+k)+2];
                        fractions[7*(i*9+j*3+k)+5] = -fractions[7*(i*9+j*3+k)+5];
                    }
                    }
                }
                }
            }
            else if (center[0] >= 0 && center[1] >= 0 && center[2] < 0)
            {
                direction = 3;
                for (int i = 0; i < 3; ++i)
                {
                for (int j = 0; j < 3; ++j)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                    if (k == 0)
                    {
                        double temp = fractions[7*(i*9+j*3+k)+0];
                        fractions[7*(i*9+j*3+k)+0] = fractions[7*(i*9+j*3+2)+0];
                        fractions[7*(i*9+j*3+2)+0] = temp;
                        temp = fractions[7*(i*9+j*3+k)+1];
                        fractions[7*(i*9+j*3+k)+1] = fractions[7*(i*9+j*3+2)+1];
                        fractions[7*(i*9+j*3+2)+1] = temp;
                        temp = fractions[7*(i*9+j*3+k)+2];
                        fractions[7*(i*9+j*3+k)+2] = fractions[7*(i*9+j*3+2)+2];
                        fractions[7*(i*9+j*3+2)+2] = temp;
                        temp = fractions[7*(i*9+j*3+k)+3];
                        fractions[7*(i*9+j*3+k)+3] = -fractions[7*(i*9+j*3+2)+3];
                        fractions[7*(i*9+j*3+2)+3] = -temp;
                        temp = fractions[7*(i*9+j*3+k)+4];
                        fractions[7*(i*9+j*3+k)+4] = fractions[7*(i*9+j*3+2)+4];
                        fractions[7*(i*9+j*3+2)+4] = temp;
                        temp = fractions[7*(i*9+j*3+k)+5];
                        fractions[7*(i*9+j*3+k)+5] = fractions[7*(i*9+j*3+2)+5];
                        fractions[7*(i*9+j*3+2)+5] = temp;
                        temp = fractions[7*(i*9+j*3+k)+6];
                        fractions[7*(i*9+j*3+k)+6] = -fractions[7*(i*9+j*3+2)+6];
                        fractions[7*(i*9+j*3+2)+6] = -temp;
                    }
                    else if (k == 1)
                    {
                        fractions[7*(i*9+j*3+k)+3] = -fractions[7*(i*9+j*3+k)+3];
                        fractions[7*(i*9+j*3+k)+6] = -fractions[7*(i*9+j*3+k)+6];
                    }
                    }
                }
                }
            }
            else if (center[0] < 0 && center[1] < 0 && center[2] >= 0)
            {
                direction = 4;
                for (int i = 0; i < 3; ++i)
                {
                for (int j = 0; j < 3; ++j)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                    if (i == 0)
                    {
                        double temp = fractions[7*(i*9+j*3+k)+0];
                        fractions[7*(i*9+j*3+k)+0] = fractions[7*(2*9+j*3+k)+0];
                        fractions[7*(2*9+j*3+k)+0] = temp;
                        temp = fractions[7*(i*9+j*3+k)+1];
                        fractions[7*(i*9+j*3+k)+1] = -fractions[7*(2*9+j*3+k)+1];
                        fractions[7*(2*9+j*3+k)+1] = -temp;
                        temp = fractions[7*(i*9+j*3+k)+2];
                        fractions[7*(i*9+j*3+k)+2] = fractions[7*(2*9+j*3+k)+2];
                        fractions[7*(2*9+j*3+k)+2] = temp;
                        temp = fractions[7*(i*9+j*3+k)+3];
                        fractions[7*(i*9+j*3+k)+3] = fractions[7*(2*9+j*3+k)+3];
                        fractions[7*(2*9+j*3+k)+3] = temp;
                        temp = fractions[7*(i*9+j*3+k)+4];
                        fractions[7*(i*9+j*3+k)+4] = -fractions[7*(2*9+j*3+k)+4];
                        fractions[7*(2*9+j*3+k)+4] = -temp;
                        temp = fractions[7*(i*9+j*3+k)+5];
                        fractions[7*(i*9+j*3+k)+5] = fractions[7*(2*9+j*3+k)+5];
                        fractions[7*(2*9+j*3+k)+5] = temp;
                        temp = fractions[7*(i*9+j*3+k)+6];
                        fractions[7*(i*9+j*3+k)+6] = fractions[7*(2*9+j*3+k)+6];
                        fractions[7*(2*9+j*3+k)+6] = temp;
                    }
                    else if (i == 1)
                    {
                        fractions[7*(i*9+j*3+k)+1] = -fractions[7*(i*9+j*3+k)+1];
                        fractions[7*(i*9+j*3+k)+4] = -fractions[7*(i*9+j*3+k)+4];
                    }
                    }
                }
                }
                for (int i = 0; i < 3; ++i)
                {
                for (int j = 0; j < 3; ++j)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                    if (j == 0)
                    {
                        double temp = fractions[7*(i*9+j*3+k)+0];
                        fractions[7*(i*9+j*3+k)+0] = fractions[7*(i*9+2*3+k)+0];
                        fractions[7*(i*9+2*3+k)+0] = temp;
                        temp = fractions[7*(i*9+j*3+k)+1];
                        fractions[7*(i*9+j*3+k)+1] = fractions[7*(i*9+2*3+k)+1];
                        fractions[7*(i*9+2*3+k)+1] = temp;
                        temp = fractions[7*(i*9+j*3+k)+2];
                        fractions[7*(i*9+j*3+k)+2] = -fractions[7*(i*9+2*3+k)+2];
                        fractions[7*(i*9+2*3+k)+2] = -temp;
                        temp = fractions[7*(i*9+j*3+k)+3];
                        fractions[7*(i*9+j*3+k)+3] = fractions[7*(i*9+2*3+k)+3];
                        fractions[7*(i*9+2*3+k)+3] = temp;
                        temp = fractions[7*(i*9+j*3+k)+4];
                        fractions[7*(i*9+j*3+k)+4] = fractions[7*(i*9+2*3+k)+4];
                        fractions[7*(i*9+2*3+k)+4] = temp;
                        temp = fractions[7*(i*9+j*3+k)+5];
                        fractions[7*(i*9+j*3+k)+5] = -fractions[7*(i*9+2*3+k)+5];
                        fractions[7*(i*9+2*3+k)+5] = -temp;
                        temp = fractions[7*(i*9+j*3+k)+6];
                        fractions[7*(i*9+j*3+k)+6] = fractions[7*(i*9+2*3+k)+6];
                        fractions[7*(i*9+2*3+k)+6] = temp;
                    }
                    else if (j == 1)
                    {
                        fractions[7*(i*9+j*3+k)+2] = -fractions[7*(i*9+j*3+k)+2];
                        fractions[7*(i*9+j*3+k)+5] = -fractions[7*(i*9+j*3+k)+5];
                    }
                    }
                }
                }
            }
            else if (center[0] < 0 && center[1] >= 0 && center[2] < 0)
            {
                direction = 5;
                for (int i = 0; i < 3; ++i)
                {
                for (int j = 0; j < 3; ++j)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                    if (i == 0)
                    {
                        double temp = fractions[7*(i*9+j*3+k)+0];
                        fractions[7*(i*9+j*3+k)+0] = fractions[7*(2*9+j*3+k)+0];
                        fractions[7*(2*9+j*3+k)+0] = temp;
                        temp = fractions[7*(i*9+j*3+k)+1];
                        fractions[7*(i*9+j*3+k)+1] = -fractions[7*(2*9+j*3+k)+1];
                        fractions[7*(2*9+j*3+k)+1] = -temp;
                        temp = fractions[7*(i*9+j*3+k)+2];
                        fractions[7*(i*9+j*3+k)+2] = fractions[7*(2*9+j*3+k)+2];
                        fractions[7*(2*9+j*3+k)+2] = temp;
                        temp = fractions[7*(i*9+j*3+k)+3];
                        fractions[7*(i*9+j*3+k)+3] = fractions[7*(2*9+j*3+k)+3];
                        fractions[7*(2*9+j*3+k)+3] = temp;
                        temp = fractions[7*(i*9+j*3+k)+4];
                        fractions[7*(i*9+j*3+k)+4] = -fractions[7*(2*9+j*3+k)+4];
                        fractions[7*(2*9+j*3+k)+4] = -temp;
                        temp = fractions[7*(i*9+j*3+k)+5];
                        fractions[7*(i*9+j*3+k)+5] = fractions[7*(2*9+j*3+k)+5];
                        fractions[7*(2*9+j*3+k)+5] = temp;
                        temp = fractions[7*(i*9+j*3+k)+6];
                        fractions[7*(i*9+j*3+k)+6] = fractions[7*(2*9+j*3+k)+6];
                        fractions[7*(2*9+j*3+k)+6] = temp;
                    }
                    else if (i == 1)
                    {
                        fractions[7*(i*9+j*3+k)+1] = -fractions[7*(i*9+j*3+k)+1];
                        fractions[7*(i*9+j*3+k)+4] = -fractions[7*(i*9+j*3+k)+4];
                    }
                    }
                }
                }
                for (int i = 0; i < 3; ++i)
                {
                for (int j = 0; j < 3; ++j)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                    if (k == 0)
                    {
                        double temp = fractions[7*(i*9+j*3+k)+0];
                        fractions[7*(i*9+j*3+k)+0] = fractions[7*(i*9+j*3+2)+0];
                        fractions[7*(i*9+j*3+2)+0] = temp;
                        temp = fractions[7*(i*9+j*3+k)+1];
                        fractions[7*(i*9+j*3+k)+1] = fractions[7*(i*9+j*3+2)+1];
                        fractions[7*(i*9+j*3+2)+1] = temp;
                        temp = fractions[7*(i*9+j*3+k)+2];
                        fractions[7*(i*9+j*3+k)+2] = fractions[7*(i*9+j*3+2)+2];
                        fractions[7*(i*9+j*3+2)+2] = temp;
                        temp = fractions[7*(i*9+j*3+k)+3];
                        fractions[7*(i*9+j*3+k)+3] = -fractions[7*(i*9+j*3+2)+3];
                        fractions[7*(i*9+j*3+2)+3] = -temp;
                        temp = fractions[7*(i*9+j*3+k)+4];
                        fractions[7*(i*9+j*3+k)+4] = fractions[7*(i*9+j*3+2)+4];
                        fractions[7*(i*9+j*3+2)+4] = temp;
                        temp = fractions[7*(i*9+j*3+k)+5];
                        fractions[7*(i*9+j*3+k)+5] = fractions[7*(i*9+j*3+2)+5];
                        fractions[7*(i*9+j*3+2)+5] = temp;
                        temp = fractions[7*(i*9+j*3+k)+6];
                        fractions[7*(i*9+j*3+k)+6] = -fractions[7*(i*9+j*3+2)+6];
                        fractions[7*(i*9+j*3+2)+6] = -temp;
                    }
                    else if (k == 1)
                    {
                        fractions[7*(i*9+j*3+k)+3] = -fractions[7*(i*9+j*3+k)+3];
                        fractions[7*(i*9+j*3+k)+6] = -fractions[7*(i*9+j*3+k)+6];
                    }
                    }
                }
                }
            }
            else if (center[0] >= 0 && center[1] < 0 && center[2] < 0)
            {
                direction = 6;
                for (int i = 0; i < 3; ++i)
                {
                for (int j = 0; j < 3; ++j)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                    if (j == 0)
                    {
                        double temp = fractions[7*(i*9+j*3+k)+0];
                        fractions[7*(i*9+j*3+k)+0] = fractions[7*(i*9+2*3+k)+0];
                        fractions[7*(i*9+2*3+k)+0] = temp;
                        temp = fractions[7*(i*9+j*3+k)+1];
                        fractions[7*(i*9+j*3+k)+1] = fractions[7*(i*9+2*3+k)+1];
                        fractions[7*(i*9+2*3+k)+1] = temp;
                        temp = fractions[7*(i*9+j*3+k)+2];
                        fractions[7*(i*9+j*3+k)+2] = -fractions[7*(i*9+2*3+k)+2];
                        fractions[7*(i*9+2*3+k)+2] = -temp;
                        temp = fractions[7*(i*9+j*3+k)+3];
                        fractions[7*(i*9+j*3+k)+3] = fractions[7*(i*9+2*3+k)+3];
                        fractions[7*(i*9+2*3+k)+3] = temp;
                        temp = fractions[7*(i*9+j*3+k)+4];
                        fractions[7*(i*9+j*3+k)+4] = fractions[7*(i*9+2*3+k)+4];
                        fractions[7*(i*9+2*3+k)+4] = temp;
                        temp = fractions[7*(i*9+j*3+k)+5];
                        fractions[7*(i*9+j*3+k)+5] = -fractions[7*(i*9+2*3+k)+5];
                        fractions[7*(i*9+2*3+k)+5] = -temp;
                        temp = fractions[7*(i*9+j*3+k)+6];
                        fractions[7*(i*9+j*3+k)+6] = fractions[7*(i*9+2*3+k)+6];
                        fractions[7*(i*9+2*3+k)+6] = temp;
                    }
                    else if (j == 1)
                    {
                        fractions[7*(i*9+j*3+k)+2] = -fractions[7*(i*9+j*3+k)+2];
                        fractions[7*(i*9+j*3+k)+5] = -fractions[7*(i*9+j*3+k)+5];
                    }
                    }
                }
                }
                for (int i = 0; i < 3; ++i)
                {
                for (int j = 0; j < 3; ++j)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                    if (k == 0)
                    {
                        double temp = fractions[7*(i*9+j*3+k)+0];
                        fractions[7*(i*9+j*3+k)+0] = fractions[7*(i*9+j*3+2)+0];
                        fractions[7*(i*9+j*3+2)+0] = temp;
                        temp = fractions[7*(i*9+j*3+k)+1];
                        fractions[7*(i*9+j*3+k)+1] = fractions[7*(i*9+j*3+2)+1];
                        fractions[7*(i*9+j*3+2)+1] = temp;
                        temp = fractions[7*(i*9+j*3+k)+2];
                        fractions[7*(i*9+j*3+k)+2] = fractions[7*(i*9+j*3+2)+2];
                        fractions[7*(i*9+j*3+2)+2] = temp;
                        temp = fractions[7*(i*9+j*3+k)+3];
                        fractions[7*(i*9+j*3+k)+3] = -fractions[7*(i*9+j*3+2)+3];
                        fractions[7*(i*9+j*3+2)+3] = -temp;
                        temp = fractions[7*(i*9+j*3+k)+4];
                        fractions[7*(i*9+j*3+k)+4] = fractions[7*(i*9+j*3+2)+4];
                        fractions[7*(i*9+j*3+2)+4] = temp;
                        temp = fractions[7*(i*9+j*3+k)+5];
                        fractions[7*(i*9+j*3+k)+5] = fractions[7*(i*9+j*3+2)+5];
                        fractions[7*(i*9+j*3+2)+5] = temp;
                        temp = fractions[7*(i*9+j*3+k)+6];
                        fractions[7*(i*9+j*3+k)+6] = -fractions[7*(i*9+j*3+2)+6];
                        fractions[7*(i*9+j*3+2)+6] = -temp;
                    }
                    else if (k == 1)
                    {
                        fractions[7*(i*9+j*3+k)+3] = -fractions[7*(i*9+j*3+k)+3];
                        fractions[7*(i*9+j*3+k)+6] = -fractions[7*(i*9+j*3+k)+6];
                    }
                    }
                }
                }
            }
            else if (center[0] < 0 && center[1] < 0 && center[2] < 0)
            {
                direction = 7;
                for (int i = 0; i < 3; ++i)
                {
                for (int j = 0; j < 3; ++j)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                    if (i == 0)
                    {
                        double temp = fractions[7*(i*9+j*3+k)+0];
                        fractions[7*(i*9+j*3+k)+0] = fractions[7*(2*9+j*3+k)+0];
                        fractions[7*(2*9+j*3+k)+0] = temp;
                        temp = fractions[7*(i*9+j*3+k)+1];
                        fractions[7*(i*9+j*3+k)+1] = -fractions[7*(2*9+j*3+k)+1];
                        fractions[7*(2*9+j*3+k)+1] = -temp;
                        temp = fractions[7*(i*9+j*3+k)+2];
                        fractions[7*(i*9+j*3+k)+2] = fractions[7*(2*9+j*3+k)+2];
                        fractions[7*(2*9+j*3+k)+2] = temp;
                        temp = fractions[7*(i*9+j*3+k)+3];
                        fractions[7*(i*9+j*3+k)+3] = fractions[7*(2*9+j*3+k)+3];
                        fractions[7*(2*9+j*3+k)+3] = temp;
                        temp = fractions[7*(i*9+j*3+k)+4];
                        fractions[7*(i*9+j*3+k)+4] = -fractions[7*(2*9+j*3+k)+4];
                        fractions[7*(2*9+j*3+k)+4] = -temp;
                        temp = fractions[7*(i*9+j*3+k)+5];
                        fractions[7*(i*9+j*3+k)+5] = fractions[7*(2*9+j*3+k)+5];
                        fractions[7*(2*9+j*3+k)+5] = temp;
                        temp = fractions[7*(i*9+j*3+k)+6];
                        fractions[7*(i*9+j*3+k)+6] = fractions[7*(2*9+j*3+k)+6];
                        fractions[7*(2*9+j*3+k)+6] = temp;
                    }
                    else if (i == 1)
                    {
                        fractions[7*(i*9+j*3+k)+1] = -fractions[7*(i*9+j*3+k)+1];
                        fractions[7*(i*9+j*3+k)+4] = -fractions[7*(i*9+j*3+k)+4];
                    }
                    }
                }
                }
                for (int i = 0; i < 3; ++i)
                {
                for (int j = 0; j < 3; ++j)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                    if (j == 0)
                    {
                        double temp = fractions[7*(i*9+j*3+k)+0];
                        fractions[7*(i*9+j*3+k)+0] = fractions[7*(i*9+2*3+k)+0];
                        fractions[7*(i*9+2*3+k)+0] = temp;
                        temp = fractions[7*(i*9+j*3+k)+1];
                        fractions[7*(i*9+j*3+k)+1] = fractions[7*(i*9+2*3+k)+1];
                        fractions[7*(i*9+2*3+k)+1] = temp;
                        temp = fractions[7*(i*9+j*3+k)+2];
                        fractions[7*(i*9+j*3+k)+2] = -fractions[7*(i*9+2*3+k)+2];
                        fractions[7*(i*9+2*3+k)+2] = -temp;
                        temp = fractions[7*(i*9+j*3+k)+3];
                        fractions[7*(i*9+j*3+k)+3] = fractions[7*(i*9+2*3+k)+3];
                        fractions[7*(i*9+2*3+k)+3] = temp;
                        temp = fractions[7*(i*9+j*3+k)+4];
                        fractions[7*(i*9+j*3+k)+4] = fractions[7*(i*9+2*3+k)+4];
                        fractions[7*(i*9+2*3+k)+4] = temp;
                        temp = fractions[7*(i*9+j*3+k)+5];
                        fractions[7*(i*9+j*3+k)+5] = -fractions[7*(i*9+2*3+k)+5];
                        fractions[7*(i*9+2*3+k)+5] = -temp;
                        temp = fractions[7*(i*9+j*3+k)+6];
                        fractions[7*(i*9+j*3+k)+6] = fractions[7*(i*9+2*3+k)+6];
                        fractions[7*(i*9+2*3+k)+6] = temp;
                    }
                    else if (j == 1)
                    {
                        fractions[7*(i*9+j*3+k)+2] = -fractions[7*(i*9+j*3+k)+2];
                        fractions[7*(i*9+j*3+k)+5] = -fractions[7*(i*9+j*3+k)+5];
                    }
                    }
                }
                }
                for (int i = 0; i < 3; ++i)
                {
                for (int j = 0; j < 3; ++j)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                    if (k == 0)
                    {
                        double temp = fractions[7*(i*9+j*3+k)+0];
                        fractions[7*(i*9+j*3+k)+0] = fractions[7*(i*9+j*3+2)+0];
                        fractions[7*(i*9+j*3+2)+0] = temp;
                        temp = fractions[7*(i*9+j*3+k)+1];
                        fractions[7*(i*9+j*3+k)+1] = fractions[7*(i*9+j*3+2)+1];
                        fractions[7*(i*9+j*3+2)+1] = temp;
                        temp = fractions[7*(i*9+j*3+k)+2];
                        fractions[7*(i*9+j*3+k)+2] = fractions[7*(i*9+j*3+2)+2];
                        fractions[7*(i*9+j*3+2)+2] = temp;
                        temp = fractions[7*(i*9+j*3+k)+3];
                        fractions[7*(i*9+j*3+k)+3] = -fractions[7*(i*9+j*3+2)+3];
                        fractions[7*(i*9+j*3+2)+3] = -temp;
                        temp = fractions[7*(i*9+j*3+k)+4];
                        fractions[7*(i*9+j*3+k)+4] = fractions[7*(i*9+j*3+2)+4];
                        fractions[7*(i*9+j*3+2)+4] = temp;
                        temp = fractions[7*(i*9+j*3+k)+5];
                        fractions[7*(i*9+j*3+k)+5] = fractions[7*(i*9+j*3+2)+5];
                        fractions[7*(i*9+j*3+2)+5] = temp;
                        temp = fractions[7*(i*9+j*3+k)+6];
                        fractions[7*(i*9+j*3+k)+6] = -fractions[7*(i*9+j*3+2)+6];
                        fractions[7*(i*9+j*3+2)+6] = -temp;
                    }
                    else if (k == 1)
                    {
                        fractions[7*(i*9+j*3+k)+3] = -fractions[7*(i*9+j*3+k)+3];
                        fractions[7*(i*9+j*3+k)+6] = -fractions[7*(i*9+j*3+k)+6];
                    }
                    }
                }
                }
            }
            *fractions1 = fractions;
            return direction;
        };

        void rotateMoments(std::vector<double>* moments1, int direction)
        {
            std::vector<double> moments = *moments1;
            switch(direction)
            {
                case 1:
                    moments[1] = -moments[1];
                    moments[5] = -moments[5];
                    moments[6] = -moments[6];
                    moments[10] = -moments[10];
                    moments[14] = -moments[14];
                    moments[15] = -moments[15];                    
                break;
                case 2:
                    moments[2] = -moments[2];
                    moments[5] = -moments[5];
                    moments[8] = -moments[8];
                    moments[11] = -moments[11];
                    moments[14] = -moments[14];
                    moments[17] = -moments[17];
                break;
                case 3:
                    moments[3] = -moments[3];
                    moments[6] = -moments[6];
                    moments[8] = -moments[8];
                    moments[12] = -moments[12];
                    moments[15] = -moments[15];
                    moments[17] = -moments[17];
                break;
                case 4:
                    moments[1] = -moments[1];
                    moments[5] = -moments[5];
                    moments[6] = -moments[6];
                    moments[10] = -moments[10];
                    moments[14] = -moments[14];
                    moments[15] = -moments[15];     
                    moments[2] = -moments[2];
                    moments[5] = -moments[5];
                    moments[8] = -moments[8];
                    moments[11] = -moments[11];
                    moments[14] = -moments[14];
                    moments[17] = -moments[17];               
                break;
                case 5:
                    moments[1] = -moments[1];
                    moments[5] = -moments[5];
                    moments[6] = -moments[6];
                    moments[10] = -moments[10];
                    moments[14] = -moments[14];
                    moments[15] = -moments[15]; 
                    moments[2] = -moments[2];
                    moments[5] = -moments[5];
                    moments[8] = -moments[8];
                    moments[11] = -moments[11];
                    moments[14] = -moments[14];
                    moments[17] = -moments[17];                
                break;
                case 6:
                    moments[2] = -moments[2];
                    moments[5] = -moments[5];
                    moments[8] = -moments[8];
                    moments[11] = -moments[11];
                    moments[14] = -moments[14];
                    moments[17] = -moments[17];
                    moments[3] = -moments[3];
                    moments[6] = -moments[6];
                    moments[8] = -moments[8];
                    moments[12] = -moments[12];
                    moments[15] = -moments[15];
                    moments[17] = -moments[17];
                break;
                case 7:
                    moments[1] = -moments[1];
                    moments[5] = -moments[5];
                    moments[6] = -moments[6];
                    moments[10] = -moments[10];
                    moments[14] = -moments[14];
                    moments[15] = -moments[15];   
                    moments[2] = -moments[2];
                    moments[5] = -moments[5];
                    moments[8] = -moments[8];
                    moments[11] = -moments[11];
                    moments[14] = -moments[14];
                    moments[17] = -moments[17];
                    moments[3] = -moments[3];
                    moments[6] = -moments[6];
                    moments[8] = -moments[8];
                    moments[12] = -moments[12];
                    moments[15] = -moments[15];
                    moments[17] = -moments[17];
                break;
            }
            *moments1 = moments;
        };




















        void generate_with_disturbance2(double rota_l, double rota_h, double rotb_l, double rotb_h, double rotc_l, double rotc_h, double coa_l, double coa_h, double cob_l, double cob_h, double ox_l, double ox_h, double oy_l, double oy_h, double oz_l, double oz_h)
        {
            for (int n = 0; n < Ntests; ++n) 
            {
                std::cout << n << endl;
                IRL::Paraboloid paraboloid = gen->new_random_parabaloid(rota_l, rota_h, rotb_l, rotb_h, rotc_l, rotc_h, coa_l, coa_h, cob_l, cob_h, ox_l, ox_h, oy_l, oy_h, oz_l, oz_h);
                angles = gen->getAngles();

                std::ofstream coefficients;
                std::string name = "coefficients.txt";
                coefficients.open(name, std::ios_base::app);
                coefficients << paraboloid.getDatum().x() << "," << paraboloid.getDatum().y() << "," << paraboloid.getDatum().z()
                << "," << angles[0] << "," << angles[1] << "," << angles[2]
                << "," << paraboloid.getAlignedParaboloid().a() << "," << paraboloid.getAlignedParaboloid().b() << "\n";
                coefficients.close();

                std::ofstream curvatures;
                std::string curv_name = "curvatures.txt";
                curvatures.open(curv_name, std::ios_base::app);
                curvatures << paraboloid.getAlignedParaboloid().a() << "," << paraboloid.getAlignedParaboloid().b() << "\n";
                coefficients.close();

                std::ofstream classification;
                std::string data_name = "type.txt";
                classification.open(data_name, std::ios_base::app);
                if ((abs(paraboloid.getAlignedParaboloid().a() - paraboloid.getAlignedParaboloid().b()) > 1) && (paraboloid.getAlignedParaboloid().a() < 0.2 || paraboloid.getAlignedParaboloid().b() < 0.2))
                {
                    classification << "0,1,0" << " \n";
                }
                else if ((abs(paraboloid.getAlignedParaboloid().a() - paraboloid.getAlignedParaboloid().b()) < 1) && (paraboloid.getAlignedParaboloid().a() > 2 || paraboloid.getAlignedParaboloid().b() > 2))
                {
                    classification << "1,0,0" << " \n";
                }
                else
                {
                    classification << "0,0,1" << " \n";
                }
                classification.close();

                auto result = gen->get_fractions(paraboloid, true);
                bool flip = false;
                if (result[((result.sizes()[0]-/*7*/4)/2)].item<double>() > 0.5)
                {
                    flip = true;
                    result = gen->get_fractions_gas(paraboloid, true);
                }
                std::vector<double> fractions;
                for (int i = 0; i < result.sizes()[0]; ++i)
                {
                    fractions.push_back(result[i].item<double>());
                }
                

                auto sm = IRL::spatial_moments();
                std::vector<double> center = sm.get_mass_centers(fractions);
                int direction = 0;
                srand((unsigned) time(NULL));
                if (center[0] < 0 && center[1] >= 0 && center[2] >= 0)
                {
                    direction = 1;
                    for (int i = 0; i < 3; ++i)
                    {
                    for (int j = 0; j < 3; ++j)
                    {
                        for (int k = 0; k < 3; ++k)
                        {
                        if (i == 0)
                        {
                            double temp = fractions[/*7*/4*(i*9+j*3+k)+0];
                            fractions[/*7*/4*(i*9+j*3+k)+0] = fractions[/*7*/4*(2*9+j*3+k)+0];
                            fractions[/*7*/4*(2*9+j*3+k)+0] = temp;
                            temp = fractions[/*7*/4*(i*9+j*3+k)+1];
                            int r = rand() % 3 - 1;
                            double c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+k)+1] = -fractions[/*7*/4*(2*9+j*3+k)+1] + r*fractions[/*7*/4*(2*9+j*3+k)+1]*c;
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(2*9+j*3+k)+1] = -temp + r*temp*c;
                            temp = fractions[/*7*/4*(i*9+j*3+k)+2];
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+k)+2] = fractions[/*7*/4*(2*9+j*3+k)+2] + r*fractions[/*7*/4*(2*9+j*3+k)+2]*c;
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(2*9+j*3+k)+2] = temp + r*temp*c;
                            temp = fractions[/*7*/4*(i*9+j*3+k)+3];
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+k)+3] = fractions[/*7*/4*(2*9+j*3+k)+3] + r*fractions[/*7*/4*(2*9+j*3+k)+3]*c;
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(2*9+j*3+k)+3] = temp + r*temp*c;
                            /*temp = fractions[7*(i*9+j*3+k)+4];
                            fractions[7*(i*9+j*3+k)+4] = -fractions[7*(2*9+j*3+k)+4];
                            fractions[7*(2*9+j*3+k)+4] = -temp;
                            temp = fractions[7*(i*9+j*3+k)+5];
                            fractions[7*(i*9+j*3+k)+5] = fractions[7*(2*9+j*3+k)+5];
                            fractions[7*(2*9+j*3+k)+5] = temp;
                            temp = fractions[7*(i*9+j*3+k)+6];
                            fractions[7*(i*9+j*3+k)+6] = fractions[7*(2*9+j*3+k)+6];
                            fractions[7*(2*9+j*3+k)+6] = temp;*/
                        }
                        else if (i == 1)
                        {
                            int r = rand() % 3 - 1;
                            double c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+k)+1] = -fractions[/*7*/4*(i*9+j*3+k)+1] + r*fractions[/*7*/4*(i*9+j*3+k)+1]*c;
                            //fractions[7*(i*9+j*3+k)+4] = -fractions[7*(i*9+j*3+k)+4];
                        }
                        }
                    }
                    }
                }
                else if (center[0] >= 0 && center[1] < 0 && center[2] >= 0)
                {
                    direction = 2;
                    for (int i = 0; i < 3; ++i)
                    {
                    for (int j = 0; j < 3; ++j)
                    {
                        for (int k = 0; k < 3; ++k)
                        {
                        if (j == 0)
                        {
                            double temp = fractions[/*7*/4*(i*9+j*3+k)+0];
                            fractions[/*7*/4*(i*9+j*3+k)+0] = fractions[/*7*/4*(i*9+2*3+k)+0];
                            fractions[/*7*/4*(i*9+2*3+k)+0] = temp;
                            temp = fractions[/*7*/4*(i*9+j*3+k)+1];
                            int r = rand() % 3 - 1;
                            double c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+k)+1] = fractions[/*7*/4*(i*9+2*3+k)+1] + r*fractions[/*7*/4*(i*9+2*3+k)+1]*c;
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+2*3+k)+1] = temp + r*temp*c;
                            temp = fractions[/*7*/4*(i*9+j*3+k)+2];
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+k)+2] = -fractions[/*7*/4*(i*9+2*3+k)+2] + r*fractions[/*7*/4*(i*9+2*3+k)+2]*c;
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+2*3+k)+2] = -temp + r*temp*c;
                            temp = fractions[/*7*/4*(i*9+j*3+k)+3];
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+k)+3] = fractions[/*7*/4*(i*9+2*3+k)+3] + r*fractions[/*7*/4*(i*9+2*3+k)+3]*c;
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+2*3+k)+3] = temp + r*temp*c;
                            /*temp = fractions[7*(i*9+j*3+k)+4];
                            fractions[7*(i*9+j*3+k)+4] = fractions[7*(i*9+2*3+k)+4];
                            fractions[7*(i*9+2*3+k)+4] = temp;
                            temp = fractions[7*(i*9+j*3+k)+5];
                            fractions[7*(i*9+j*3+k)+5] = -fractions[7*(i*9+2*3+k)+5];
                            fractions[7*(i*9+2*3+k)+5] = -temp;
                            temp = fractions[7*(i*9+j*3+k)+6];
                            fractions[7*(i*9+j*3+k)+6] = fractions[7*(i*9+2*3+k)+6];
                            fractions[7*(i*9+2*3+k)+6] = temp;*/
                        }
                        else if (j == 1)
                        {
                            int r = rand() % 3 - 1;
                            double c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+k)+2] = -fractions[/*7*/4*(i*9+j*3+k)+2] + r*fractions[/*7*/4*(i*9+j*3+k)+2]*c;
                            //fractions[7*(i*9+j*3+k)+5] = -fractions[7*(i*9+j*3+k)+5];
                        }
                        }
                    }
                    }
                }
                else if (center[0] >= 0 && center[1] >= 0 && center[2] < 0)
                {
                    direction = 3;
                    for (int i = 0; i < 3; ++i)
                    {
                    for (int j = 0; j < 3; ++j)
                    {
                        for (int k = 0; k < 3; ++k)
                        {
                        if (k == 0)
                        {
                            double temp = fractions[/*7*/4*(i*9+j*3+k)+0];
                            fractions[/*7*/4*(i*9+j*3+k)+0] = fractions[/*7*/4*(i*9+j*3+2)+0];
                            fractions[/*7*/4*(i*9+j*3+2)+0] = temp;
                            temp = fractions[/*7*/4*(i*9+j*3+k)+1];
                            int r = rand() % 3 - 1;
                            double c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+k)+1] = fractions[/*7*/4*(i*9+j*3+2)+1] + r*fractions[/*7*/4*(i*9+j*3+2)+1]*c;
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+2)+1] = temp + r*temp*c;
                            temp = fractions[/*7*/4*(i*9+j*3+k)+2];
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+k)+2] = fractions[/*7*/4*(i*9+j*3+2)+2] + r*fractions[/*7*/4*(i*9+j*3+2)+2]*c;
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;                           
                            fractions[/*7*/4*(i*9+j*3+2)+2] = temp + r*temp*c;
                            temp = fractions[/*7*/4*(i*9+j*3+k)+3];
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+k)+3] = -fractions[/*7*/4*(i*9+j*3+2)+3] + r*fractions[/*7*/4*(i*9+j*3+2)+3]*c;
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;                           
                            fractions[/*7*/4*(i*9+j*3+2)+3] = -temp + r*temp*c;
                            /*temp = fractions[7*(i*9+j*3+k)+4];
                            fractions[7*(i*9+j*3+k)+4] = fractions[7*(i*9+j*3+2)+4];
                            fractions[7*(i*9+j*3+2)+4] = temp;
                            temp = fractions[7*(i*9+j*3+k)+5];
                            fractions[7*(i*9+j*3+k)+5] = fractions[7*(i*9+j*3+2)+5];
                            fractions[7*(i*9+j*3+2)+5] = temp;
                            temp = fractions[7*(i*9+j*3+k)+6];
                            fractions[7*(i*9+j*3+k)+6] = -fractions[7*(i*9+j*3+2)+6];
                            fractions[7*(i*9+j*3+2)+6] = -temp;*/
                        }
                        else if (k == 1)
                        {
                            int r = rand() % 3 - 1;
                            double c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+k)+3] = -fractions[/*7*/4*(i*9+j*3+k)+3] + r*fractions[/*7*/4*(i*9+j*3+k)+3]*c;
                            //fractions[7*(i*9+j*3+k)+6] = -fractions[7*(i*9+j*3+k)+6];
                        }
                        }
                    }
                    }
                }
                else if (center[0] < 0 && center[1] < 0 && center[2] >= 0)
                {
                    direction = 4;
                    for (int i = 0; i < 3; ++i)
                    {
                    for (int j = 0; j < 3; ++j)
                    {
                        for (int k = 0; k < 3; ++k)
                        {
                        if (i == 0)
                        {
                            double temp = fractions[/*7*/4*(i*9+j*3+k)+0];
                            fractions[/*7*/4*(i*9+j*3+k)+0] = fractions[/*7*/4*(2*9+j*3+k)+0];
                            fractions[/*7*/4*(2*9+j*3+k)+0] = temp;
                            temp = fractions[/*7*/4*(i*9+j*3+k)+1];
                            int r = rand() % 3 - 1;
                            double c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+k)+1] = -fractions[/*7*/4*(2*9+j*3+k)+1] + r*fractions[/*7*/4*(2*9+j*3+k)+1]*c;
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(2*9+j*3+k)+1] = -temp + r*temp*c;
                            temp = fractions[/*7*/4*(i*9+j*3+k)+2];
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+k)+2] = fractions[/*7*/4*(2*9+j*3+k)+2] + r*fractions[/*7*/4*(2*9+j*3+k)+2]*c;
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(2*9+j*3+k)+2] = temp + r*temp*c;
                            temp = fractions[/*7*/4*(i*9+j*3+k)+3];
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+k)+3] = fractions[/*7*/4*(2*9+j*3+k)+3] + r*fractions[/*7*/4*(2*9+j*3+k)+3]*c;
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(2*9+j*3+k)+3] = temp + r*temp*c;
                            /*temp = fractions[7*(i*9+j*3+k)+4];
                            fractions[7*(i*9+j*3+k)+4] = -fractions[7*(2*9+j*3+k)+4];
                            fractions[7*(2*9+j*3+k)+4] = -temp;
                            temp = fractions[7*(i*9+j*3+k)+5];
                            fractions[7*(i*9+j*3+k)+5] = fractions[7*(2*9+j*3+k)+5];
                            fractions[7*(2*9+j*3+k)+5] = temp;
                            temp = fractions[7*(i*9+j*3+k)+6];
                            fractions[7*(i*9+j*3+k)+6] = fractions[7*(2*9+j*3+k)+6];
                            fractions[7*(2*9+j*3+k)+6] = temp;*/
                        }
                        else if (i == 1)
                        {
                            int r = rand() % 3 - 1;
                            double c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+k)+1] = -fractions[/*7*/4*(i*9+j*3+k)+1] + r*fractions[/*7*/4*(i*9+j*3+k)+1]*c;
                            //fractions[7*(i*9+j*3+k)+4] = -fractions[7*(i*9+j*3+k)+4];
                        }
                        }
                    }
                    }
                    for (int i = 0; i < 3; ++i)
                    {
                    for (int j = 0; j < 3; ++j)
                    {
                        for (int k = 0; k < 3; ++k)
                        {
                        if (j == 0)
                        {
                            double temp = fractions[/*7*/4*(i*9+j*3+k)+0];
                            fractions[/*7*/4*(i*9+j*3+k)+0] = fractions[/*7*/4*(i*9+2*3+k)+0];
                            fractions[/*7*/4*(i*9+2*3+k)+0] = temp;
                            temp = fractions[/*7*/4*(i*9+j*3+k)+1];
                            int r = rand() % 3 - 1;
                            double c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+k)+1] = fractions[/*7*/4*(i*9+2*3+k)+1] + r*fractions[/*7*/4*(i*9+2*3+k)+1]*c;
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+2*3+k)+1] = temp + r*temp*c;
                            temp = fractions[/*7*/4*(i*9+j*3+k)+2];
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+k)+2] = -fractions[/*7*/4*(i*9+2*3+k)+2] + r*fractions[/*7*/4*(i*9+2*3+k)+2]*c;
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+2*3+k)+2] = -temp + r*temp*c;
                            temp = fractions[/*7*/4*(i*9+j*3+k)+3];
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+k)+3] = fractions[/*7*/4*(i*9+2*3+k)+3] + r*fractions[/*7*/4*(i*9+2*3+k)+3]*c;
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+2*3+k)+3] = temp + r*temp*c;
                            /*temp = fractions[7*(i*9+j*3+k)+4];
                            fractions[7*(i*9+j*3+k)+4] = fractions[7*(i*9+2*3+k)+4];
                            fractions[7*(i*9+2*3+k)+4] = temp;
                            temp = fractions[7*(i*9+j*3+k)+5];
                            fractions[7*(i*9+j*3+k)+5] = -fractions[7*(i*9+2*3+k)+5];
                            fractions[7*(i*9+2*3+k)+5] = -temp;
                            temp = fractions[7*(i*9+j*3+k)+6];
                            fractions[7*(i*9+j*3+k)+6] = fractions[7*(i*9+2*3+k)+6];
                            fractions[7*(i*9+2*3+k)+6] = temp;*/
                        }
                        else if (j == 1)
                        {
                            int r = rand() % 3 - 1;
                            double c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+k)+2] = -fractions[/*7*/4*(i*9+j*3+k)+2] + r*fractions[/*7*/4*(i*9+j*3+k)+2]*c;
                            //fractions[7*(i*9+j*3+k)+5] = -fractions[7*(i*9+j*3+k)+5];
                        }
                        }
                    }
                    }
                }
                else if (center[0] < 0 && center[1] >= 0 && center[2] < 0)
                {
                    direction = 5;
                    for (int i = 0; i < 3; ++i)
                    {
                    for (int j = 0; j < 3; ++j)
                    {
                        for (int k = 0; k < 3; ++k)
                        {
                        if (i == 0)
                        {
                            double temp = fractions[/*7*/4*(i*9+j*3+k)+0];
                            fractions[/*7*/4*(i*9+j*3+k)+0] = fractions[/*7*/4*(2*9+j*3+k)+0];
                            fractions[/*7*/4*(2*9+j*3+k)+0] = temp;
                            temp = fractions[/*7*/4*(i*9+j*3+k)+1];
                            int r = rand() % 3 - 1;
                            double c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+k)+1] = -fractions[/*7*/4*(2*9+j*3+k)+1] + r*fractions[/*7*/4*(2*9+j*3+k)+1]*c;
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(2*9+j*3+k)+1] = -temp + r*temp*c;
                            temp = fractions[/*7*/4*(i*9+j*3+k)+2];
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+k)+2] = fractions[/*7*/4*(2*9+j*3+k)+2] + r*fractions[/*7*/4*(2*9+j*3+k)+2]*c;
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(2*9+j*3+k)+2] = temp + r*temp*c;
                            temp = fractions[/*7*/4*(i*9+j*3+k)+3];
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+k)+3] = fractions[/*7*/4*(2*9+j*3+k)+3] + r*fractions[/*7*/4*(2*9+j*3+k)+3]*c;
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(2*9+j*3+k)+3] = temp + r*temp*c;
                            /*temp = fractions[7*(i*9+j*3+k)+4];
                            fractions[7*(i*9+j*3+k)+4] = -fractions[7*(2*9+j*3+k)+4];
                            fractions[7*(2*9+j*3+k)+4] = -temp;
                            temp = fractions[7*(i*9+j*3+k)+5];
                            fractions[7*(i*9+j*3+k)+5] = fractions[7*(2*9+j*3+k)+5];
                            fractions[7*(2*9+j*3+k)+5] = temp;
                            temp = fractions[7*(i*9+j*3+k)+6];
                            fractions[7*(i*9+j*3+k)+6] = fractions[7*(2*9+j*3+k)+6];
                            fractions[7*(2*9+j*3+k)+6] = temp;*/
                        }
                        else if (i == 1)
                        {
                            int r = rand() % 3 - 1;
                            double c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+k)+1] = -fractions[/*7*/4*(i*9+j*3+k)+1] + r*fractions[/*7*/4*(i*9+j*3+k)+1]*c;
                            //fractions[7*(i*9+j*3+k)+4] = -fractions[7*(i*9+j*3+k)+4];
                        }
                        }
                    }
                    }
                    for (int i = 0; i < 3; ++i)
                    {
                    for (int j = 0; j < 3; ++j)
                    {
                        for (int k = 0; k < 3; ++k)
                        {
                        if (k == 0)
                        {
                            double temp = fractions[/*7*/4*(i*9+j*3+k)+0];
                            fractions[/*7*/4*(i*9+j*3+k)+0] = fractions[/*7*/4*(i*9+j*3+2)+0];
                            fractions[/*7*/4*(i*9+j*3+2)+0] = temp;
                            temp = fractions[/*7*/4*(i*9+j*3+k)+1];
                            int r = rand() % 3 - 1;
                            double c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+k)+1] = fractions[/*7*/4*(i*9+j*3+2)+1] + r*fractions[/*7*/4*(i*9+j*3+2)+1]*c;
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+2)+1] = temp + r*temp*c;
                            temp = fractions[/*7*/4*(i*9+j*3+k)+2];
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+k)+2] = fractions[/*7*/4*(i*9+j*3+2)+2] + r*fractions[/*7*/4*(i*9+j*3+2)+2]*c;
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;                           
                            fractions[/*7*/4*(i*9+j*3+2)+2] = temp + r*temp*c;
                            temp = fractions[/*7*/4*(i*9+j*3+k)+3];
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+k)+3] = -fractions[/*7*/4*(i*9+j*3+2)+3] + r*fractions[/*7*/4*(i*9+j*3+2)+3]*c;
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;                           
                            fractions[/*7*/4*(i*9+j*3+2)+3] = -temp + r*temp*c;
                            /*temp = fractions[7*(i*9+j*3+k)+4];
                            fractions[7*(i*9+j*3+k)+4] = fractions[7*(i*9+j*3+2)+4];
                            fractions[7*(i*9+j*3+2)+4] = temp;
                            temp = fractions[7*(i*9+j*3+k)+5];
                            fractions[7*(i*9+j*3+k)+5] = fractions[7*(i*9+j*3+2)+5];
                            fractions[7*(i*9+j*3+2)+5] = temp;
                            temp = fractions[7*(i*9+j*3+k)+6];
                            fractions[7*(i*9+j*3+k)+6] = -fractions[7*(i*9+j*3+2)+6];
                            fractions[7*(i*9+j*3+2)+6] = -temp;*/
                        }
                        else if (k == 1)
                        {
                            int r = rand() % 3 - 1;
                            double c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+k)+3] = -fractions[/*7*/4*(i*9+j*3+k)+3] + r*fractions[/*7*/4*(i*9+j*3+k)+3]*c;
                            //fractions[7*(i*9+j*3+k)+6] = -fractions[7*(i*9+j*3+k)+6];
                        }
                        }
                    }
                    }
                }
                else if (center[0] >= 0 && center[1] < 0 && center[2] < 0)
                {
                    direction = 6;
                    for (int i = 0; i < 3; ++i)
                    {
                    for (int j = 0; j < 3; ++j)
                    {
                        for (int k = 0; k < 3; ++k)
                        {
                        if (j == 0)
                        {
                            double temp = fractions[/*7*/4*(i*9+j*3+k)+0];
                            fractions[/*7*/4*(i*9+j*3+k)+0] = fractions[/*7*/4*(i*9+2*3+k)+0];
                            fractions[/*7*/4*(i*9+2*3+k)+0] = temp;
                            temp = fractions[/*7*/4*(i*9+j*3+k)+1];
                            int r = rand() % 3 - 1;
                            double c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+k)+1] = fractions[/*7*/4*(i*9+2*3+k)+1] + r*fractions[/*7*/4*(i*9+2*3+k)+1]*c;
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+2*3+k)+1] = temp + r*temp*c;
                            temp = fractions[/*7*/4*(i*9+j*3+k)+2];
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+k)+2] = -fractions[/*7*/4*(i*9+2*3+k)+2] + r*fractions[/*7*/4*(i*9+2*3+k)+2]*c;
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+2*3+k)+2] = -temp + r*temp*c;
                            temp = fractions[/*7*/4*(i*9+j*3+k)+3];
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+k)+3] = fractions[/*7*/4*(i*9+2*3+k)+3] + r*fractions[/*7*/4*(i*9+2*3+k)+3]*c;
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+2*3+k)+3] = temp + r*temp*c;
                            /*temp = fractions[7*(i*9+j*3+k)+4];
                            fractions[7*(i*9+j*3+k)+4] = fractions[7*(i*9+2*3+k)+4];
                            fractions[7*(i*9+2*3+k)+4] = temp;
                            temp = fractions[7*(i*9+j*3+k)+5];
                            fractions[7*(i*9+j*3+k)+5] = -fractions[7*(i*9+2*3+k)+5];
                            fractions[7*(i*9+2*3+k)+5] = -temp;
                            temp = fractions[7*(i*9+j*3+k)+6];
                            fractions[7*(i*9+j*3+k)+6] = fractions[7*(i*9+2*3+k)+6];
                            fractions[7*(i*9+2*3+k)+6] = temp;*/
                        }
                        else if (j == 1)
                        {
                            int r = rand() % 3 - 1;
                            double c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+k)+2] = -fractions[/*7*/4*(i*9+j*3+k)+2] + r*fractions[/*7*/4*(i*9+j*3+k)+2]*c;
                            //fractions[7*(i*9+j*3+k)+5] = -fractions[7*(i*9+j*3+k)+5];
                        }
                        }
                    }
                    }
                    for (int i = 0; i < 3; ++i)
                    {
                    for (int j = 0; j < 3; ++j)
                    {
                        for (int k = 0; k < 3; ++k)
                        {
                        if (k == 0)
                        {
                            double temp = fractions[/*7*/4*(i*9+j*3+k)+0];
                            fractions[/*7*/4*(i*9+j*3+k)+0] = fractions[/*7*/4*(i*9+j*3+2)+0];
                            fractions[/*7*/4*(i*9+j*3+2)+0] = temp;
                            temp = fractions[/*7*/4*(i*9+j*3+k)+1];
                            int r = rand() % 3 - 1;
                            double c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+k)+1] = fractions[/*7*/4*(i*9+j*3+2)+1] + r*fractions[/*7*/4*(i*9+j*3+2)+1]*c;
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+2)+1] = temp + r*temp*c;
                            temp = fractions[/*7*/4*(i*9+j*3+k)+2];
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+k)+2] = fractions[/*7*/4*(i*9+j*3+2)+2] + r*fractions[/*7*/4*(i*9+j*3+2)+2]*c;
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;                           
                            fractions[/*7*/4*(i*9+j*3+2)+2] = temp + r*temp*c;
                            temp = fractions[/*7*/4*(i*9+j*3+k)+3];
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+k)+3] = -fractions[/*7*/4*(i*9+j*3+2)+3] + r*fractions[/*7*/4*(i*9+j*3+2)+3]*c;
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;                           
                            fractions[/*7*/4*(i*9+j*3+2)+3] = -temp + r*temp*c;
                            /*temp = fractions[7*(i*9+j*3+k)+4];
                            fractions[7*(i*9+j*3+k)+4] = fractions[7*(i*9+j*3+2)+4];
                            fractions[7*(i*9+j*3+2)+4] = temp;
                            temp = fractions[7*(i*9+j*3+k)+5];
                            fractions[7*(i*9+j*3+k)+5] = fractions[7*(i*9+j*3+2)+5];
                            fractions[7*(i*9+j*3+2)+5] = temp;
                            temp = fractions[7*(i*9+j*3+k)+6];
                            fractions[7*(i*9+j*3+k)+6] = -fractions[7*(i*9+j*3+2)+6];
                            fractions[7*(i*9+j*3+2)+6] = -temp;*/
                        }
                        else if (k == 1)
                        {
                            int r = rand() % 3 - 1;
                            double c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+k)+3] = -fractions[/*7*/4*(i*9+j*3+k)+3] + r*fractions[/*7*/4*(i*9+j*3+k)+3]*c;
                            //fractions[7*(i*9+j*3+k)+6] = -fractions[7*(i*9+j*3+k)+6];
                        }
                        }
                    }
                    }
                }
                else if (center[0] < 0 && center[1] < 0 && center[2] < 0)
                {
                    direction = 7;
                    for (int i = 0; i < 3; ++i)
                    {
                    for (int j = 0; j < 3; ++j)
                    {
                        for (int k = 0; k < 3; ++k)
                        {
                        if (i == 0)
                        {
                            double temp = fractions[/*7*/4*(i*9+j*3+k)+0];
                            fractions[/*7*/4*(i*9+j*3+k)+0] = fractions[/*7*/4*(2*9+j*3+k)+0];
                            fractions[/*7*/4*(2*9+j*3+k)+0] = temp;
                            temp = fractions[/*7*/4*(i*9+j*3+k)+1];
                            int r = rand() % 3 - 1;
                            double c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+k)+1] = -fractions[/*7*/4*(2*9+j*3+k)+1] + r*fractions[/*7*/4*(2*9+j*3+k)+1]*c;
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(2*9+j*3+k)+1] = -temp + r*temp*c;
                            temp = fractions[/*7*/4*(i*9+j*3+k)+2];
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+k)+2] = fractions[/*7*/4*(2*9+j*3+k)+2] + r*fractions[/*7*/4*(2*9+j*3+k)+2]*c;
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(2*9+j*3+k)+2] = temp + r*temp*c;
                            temp = fractions[/*7*/4*(i*9+j*3+k)+3];
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+k)+3] = fractions[/*7*/4*(2*9+j*3+k)+3] + r*fractions[/*7*/4*(2*9+j*3+k)+3]*c;
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(2*9+j*3+k)+3] = temp + r*temp*c;
                            /*temp = fractions[7*(i*9+j*3+k)+4];
                            fractions[7*(i*9+j*3+k)+4] = -fractions[7*(2*9+j*3+k)+4];
                            fractions[7*(2*9+j*3+k)+4] = -temp;
                            temp = fractions[7*(i*9+j*3+k)+5];
                            fractions[7*(i*9+j*3+k)+5] = fractions[7*(2*9+j*3+k)+5];
                            fractions[7*(2*9+j*3+k)+5] = temp;
                            temp = fractions[7*(i*9+j*3+k)+6];
                            fractions[7*(i*9+j*3+k)+6] = fractions[7*(2*9+j*3+k)+6];
                            fractions[7*(2*9+j*3+k)+6] = temp;*/
                        }
                        else if (i == 1)
                        {
                            int r = rand() % 3 - 1;
                            double c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+k)+1] = -fractions[/*7*/4*(i*9+j*3+k)+1] + r*fractions[/*7*/4*(i*9+j*3+k)+1]*c;
                            //fractions[7*(i*9+j*3+k)+4] = -fractions[7*(i*9+j*3+k)+4];
                        }
                        }
                    }
                    }
                    for (int i = 0; i < 3; ++i)
                    {
                    for (int j = 0; j < 3; ++j)
                    {
                        for (int k = 0; k < 3; ++k)
                        {
                        if (j == 0)
                        {
                            double temp = fractions[/*7*/4*(i*9+j*3+k)+0];
                            fractions[/*7*/4*(i*9+j*3+k)+0] = fractions[/*7*/4*(i*9+2*3+k)+0];
                            fractions[/*7*/4*(i*9+2*3+k)+0] = temp;
                            temp = fractions[/*7*/4*(i*9+j*3+k)+1];
                            int r = rand() % 3 - 1;
                            double c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+k)+1] = fractions[/*7*/4*(i*9+2*3+k)+1] + r*fractions[/*7*/4*(i*9+2*3+k)+1]*c;
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+2*3+k)+1] = temp + r*temp*c;
                            temp = fractions[/*7*/4*(i*9+j*3+k)+2];
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+k)+2] = -fractions[/*7*/4*(i*9+2*3+k)+2] + r*fractions[/*7*/4*(i*9+2*3+k)+2]*c;
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+2*3+k)+2] = -temp + r*temp*c;
                            temp = fractions[/*7*/4*(i*9+j*3+k)+3];
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+k)+3] = fractions[/*7*/4*(i*9+2*3+k)+3] + r*fractions[/*7*/4*(i*9+2*3+k)+3]*c;
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+2*3+k)+3] = temp + r*temp*c;
                            /*temp = fractions[7*(i*9+j*3+k)+4];
                            fractions[7*(i*9+j*3+k)+4] = fractions[7*(i*9+2*3+k)+4];
                            fractions[7*(i*9+2*3+k)+4] = temp;
                            temp = fractions[7*(i*9+j*3+k)+5];
                            fractions[7*(i*9+j*3+k)+5] = -fractions[7*(i*9+2*3+k)+5];
                            fractions[7*(i*9+2*3+k)+5] = -temp;
                            temp = fractions[7*(i*9+j*3+k)+6];
                            fractions[7*(i*9+j*3+k)+6] = fractions[7*(i*9+2*3+k)+6];
                            fractions[7*(i*9+2*3+k)+6] = temp;*/
                        }
                        else if (j == 1)
                        {
                            int r = rand() % 3 - 1;
                            double c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+k)+2] = -fractions[/*7*/4*(i*9+j*3+k)+2] + r*fractions[/*7*/4*(i*9+j*3+k)+2]*c;
                            //fractions[7*(i*9+j*3+k)+5] = -fractions[7*(i*9+j*3+k)+5];
                        }
                        }
                    }
                    }
                    for (int i = 0; i < 3; ++i)
                    {
                    for (int j = 0; j < 3; ++j)
                    {
                        for (int k = 0; k < 3; ++k)
                        {
                        if (k == 0)
                        {
                            double temp = fractions[/*7*/4*(i*9+j*3+k)+0];
                            fractions[/*7*/4*(i*9+j*3+k)+0] = fractions[/*7*/4*(i*9+j*3+2)+0];
                            fractions[/*7*/4*(i*9+j*3+2)+0] = temp;
                            temp = fractions[/*7*/4*(i*9+j*3+k)+1];
                            int r = rand() % 3 - 1;
                            double c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+k)+1] = fractions[/*7*/4*(i*9+j*3+2)+1] + r*fractions[/*7*/4*(i*9+j*3+2)+1]*c;
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+2)+1] = temp + r*temp*c;
                            temp = fractions[/*7*/4*(i*9+j*3+k)+2];
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+k)+2] = fractions[/*7*/4*(i*9+j*3+2)+2] + r*fractions[/*7*/4*(i*9+j*3+2)+2]*c;
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;                           
                            fractions[/*7*/4*(i*9+j*3+2)+2] = temp + r*temp*c;
                            temp = fractions[/*7*/4*(i*9+j*3+k)+3];
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+k)+3] = -fractions[/*7*/4*(i*9+j*3+2)+3] + r*fractions[/*7*/4*(i*9+j*3+2)+3]*c;
                            r = rand() % 3 - 1;
                            c = (rand() % 200 + 1) / 1000.0;                           
                            fractions[/*7*/4*(i*9+j*3+2)+3] = -temp + r*temp*c;
                            /*temp = fractions[7*(i*9+j*3+k)+4];
                            fractions[7*(i*9+j*3+k)+4] = fractions[7*(i*9+j*3+2)+4];
                            fractions[7*(i*9+j*3+2)+4] = temp;
                            temp = fractions[7*(i*9+j*3+k)+5];
                            fractions[7*(i*9+j*3+k)+5] = fractions[7*(i*9+j*3+2)+5];
                            fractions[7*(i*9+j*3+2)+5] = temp;
                            temp = fractions[7*(i*9+j*3+k)+6];
                            fractions[7*(i*9+j*3+k)+6] = -fractions[7*(i*9+j*3+2)+6];
                            fractions[7*(i*9+j*3+2)+6] = -temp;*/
                        }
                        else if (k == 1)
                        {
                            int r = rand() % 3 - 1;
                            double c = (rand() % 200 + 1) / 1000.0;
                            fractions[/*7*/4*(i*9+j*3+k)+3] = -fractions[/*7*/4*(i*9+j*3+k)+3] + r*fractions[/*7*/4*(i*9+j*3+k)+3]*c;
                            //fractions[7*(i*9+j*3+k)+6] = -fractions[7*(i*9+j*3+k)+6];
                        }
                        }
                    }
                    }
                }
                std::ofstream output;
                data_name = "fractions.txt";
                output.open(data_name, std::ios_base::app);

                for (int i = 0; i < result.sizes()[0]; ++i)
                {
                    output << fractions[i] << ",";
                }
                output << "\n";
                output.close();  

                std::ofstream normals;
                std::string normals_name = "normals.txt";
                normals.open(normals_name, std::ios_base::app);
                auto cube = IRL::RectangularCuboid::fromBoundingPts(IRL::Pt(-0.5, -0.5, -0.5), IRL::Pt(0.5, 0.5, 0.5));
                auto surface_and_moments = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParaboloidParametrizedSurfaceOutput>>(cube, paraboloid);
                auto surface = surface_and_moments.getSurface();
                auto normal = surface.getAverageNormalNonAligned();

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
                if (!flip)
                {
                    normal[0] = -normal[0];
                    normal[1] = -normal[1];
                    normal[2] = -normal[2];
                }

                normals << normal[0] << "," << normal[1] << "," << normal[2] << "\n";
                normals.close();                  
            }  
        };
    };
}

#endif