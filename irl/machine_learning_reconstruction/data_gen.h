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
#include <iostream>
#include <cstdlib>

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
                    center = sm.get_mass_centers_all(fractions);
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
                auto surface_and_moments = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParametrizedSurfaceOutput>>(cube, paraboloid);
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

                std::ofstream curvatures;
                std::string curv_name = "curvatures.txt";
                curvatures.open(curv_name, std::ios_base::app);
                curvatures << paraboloid.getAlignedParaboloid().a() << "," << paraboloid.getAlignedParaboloid().b() << "\n";
                curvatures.close();

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
                inter << "1,0" << " \n";
                inter.close();

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
                    center = sm.get_mass_centers_all(fractions);
                    direction = rotateFractions_all(&fractions,center);
                }

                std::ofstream output;
                data_name = "fractions.txt";
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
                        //int r = rand() % 3 - 1;
                        double c = (rand() % 401 - 200) / 1000.0;
                        if (i % mod != 0)
                        {
                            if (fractions[i] + fractions[i]*c > 0.5)
                            {
                                output << 0.5 << ",";
                            }
                            else if (fractions[i] + fractions[i]*c < -0.5)
                            {
                                output << -0.5 << ",";
                            }
                            else
                            {
                                output << fractions[i] + fractions[i]*c << ",";
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
                auto surface_and_moments = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParametrizedSurfaceOutput>>(cube, paraboloid);
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

                std::ofstream inter;
                std::string interface_name = "interface.txt";
                inter.open(interface_name, std::ios_base::app);
                inter << "0,1" << " \n";
                inter.close();

                torch::Tensor result;
                torch::Tensor result1;
                bool flip = false;
                bool option;
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
                    if (!flip)
                    {
                        if (i%mod == 0 && result1[i].item<double>() > IRL::global_constants::VF_LOW)
                        {
                            option = true;
                        }
                        else if (i%mod == 0)
                        {
                            option = false;
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
                        if (i%mod == 0 && result1[i].item<double>() < IRL::global_constants::VF_HIGH)
                        {
                            option = true;
                        }
                        else if (i%mod == 0)
                        {
                            option = false;
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
                    center = sm.get_mass_centers_all(fractions);
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
                auto surface_and_moments = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParametrizedSurfaceOutput>>(cube, paraboloid);
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

                /*for (int i = 0; i < 3; ++i)
                {
                    for (int j = 0; j < 3; ++j)
                    {
                        for (int k = 0; k < 3; ++k)
                        {
                            const auto bottom_corner = IRL::Pt(-1.5+i, -1.5+j, -1.5+k);
                            const auto top_corner = IRL::Pt(-0.5+i, -0.5+j, -0.5+k);
                            const auto cell = IRL::StoredRectangularCuboid<IRL::Pt>::fromBoundingPts(bottom_corner, top_corner);

                            const auto first_moments_and_surface = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParametrizedSurfaceOutput>, IRL::HalfEdgeCutting>(cell, paraboloid);
                            const auto first_moments_and_surface2 = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParametrizedSurfaceOutput>, IRL::HalfEdgeCutting>(cell, interface);
                            auto surface = first_moments_and_surface.getSurface();
                            auto surface2 = first_moments_and_surface2.getSurface();
                            const double length_scale = 0.05;
                            IRL::TriangulatedSurfaceOutput triangulated_surface = first_moments_and_surface.getSurface().triangulate(length_scale);
                            IRL::TriangulatedSurfaceOutput triangulated_surface2 = first_moments_and_surface2.getSurface().triangulate(length_scale);
                            string name = "p" + std::to_string(n)+std::to_string(i)+std::to_string(j)+std::to_string(k);
                            string name2 = "i" + std::to_string(n)+std::to_string(i)+std::to_string(j)+std::to_string(k);
                            triangulated_surface.write(name);
                            triangulated_surface2.write(name2);
                        }
                    }
                }*/              
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

                std::ofstream inter;
                std::string interface_name = "interface.txt";
                inter.open(interface_name, std::ios_base::app);
                inter << "0,1" << " \n";
                inter.close();

                torch::Tensor result;
                torch::Tensor result1;
                bool flip = false;
                bool option;
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
                    if (!flip)
                    {
                        if (i%mod == 0 && result1[i].item<double>() > IRL::global_constants::VF_LOW)
                        {
                            option = true;
                        }
                        else if (i%mod == 0)
                        {
                            option = false;
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
                        if (i%mod == 0 && result1[i].item<double>() < IRL::global_constants::VF_HIGH)
                        {
                            option = true;
                        }
                        else if (i%mod == 0)
                        {
                            option = false;
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
                    center = sm.get_mass_centers_all(fractions);
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
                        double c = (rand() % 401 - 200) / 1000.0;
                        if (i % 4 != 0)
                        {
                            if (fractions[i] + fractions[i]*c > 0.5)
                            {
                                output << 0.5 << ",";
                            }
                            else if (fractions[i] + fractions[i]*c < -0.5)
                            {
                                output << -0.5 << ",";
                            }
                            else
                            {
                                output << fractions[i] + fractions[i]*c << ",";
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
                auto surface_and_moments = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParametrizedSurfaceOutput>>(cube, paraboloid);
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

                /*for (int i = 0; i < 3; ++i)
                {
                    for (int j = 0; j < 3; ++j)
                    {
                        for (int k = 0; k < 3; ++k)
                        {
                            const auto bottom_corner = IRL::Pt(-1.5+i, -1.5+j, -1.5+k);
                            const auto top_corner = IRL::Pt(-0.5+i, -0.5+j, -0.5+k);
                            const auto cell = IRL::StoredRectangularCuboid<IRL::Pt>::fromBoundingPts(bottom_corner, top_corner);

                            const auto first_moments_and_surface = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParametrizedSurfaceOutput>, IRL::HalfEdgeCutting>(cell, paraboloid);
                            const auto first_moments_and_surface2 = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParametrizedSurfaceOutput>, IRL::HalfEdgeCutting>(cell, interface);
                            auto surface = first_moments_and_surface.getSurface();
                            auto surface2 = first_moments_and_surface2.getSurface();
                            const double length_scale = 0.05;
                            IRL::TriangulatedSurfaceOutput triangulated_surface = first_moments_and_surface.getSurface().triangulate(length_scale);
                            IRL::TriangulatedSurfaceOutput triangulated_surface2 = first_moments_and_surface2.getSurface().triangulate(length_scale);
                            string name = "p" + std::to_string(n)+std::to_string(i)+std::to_string(j)+std::to_string(k);
                            string name2 = "i" + std::to_string(n)+std::to_string(i)+std::to_string(j)+std::to_string(k);
                            triangulated_surface.write(name);
                            triangulated_surface2.write(name2);
                        }
                    }
                }*/           
            }       
        };

        void generate_two_paraboloids_in_cell(double rota_l, double rota_h, double rotb_l, double rotb_h, double rotc_l, double rotc_h, double coa_l, double coa_h, double cob_l, double cob_h, double ox_l, double ox_h, double oy_l, double oy_h, double oz_l, double oz_h)
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

                std::ofstream inter;
                std::string interface_name = "interface.txt";
                inter.open(interface_name, std::ios_base::app);
                inter << "0,1" << " \n";
                inter.close();

                auto result = gen->get_fractions(paraboloid, true);
                
                auto result1 = gen->get_fractions(interface, true);

                bool flip = false;
                if (result[((result.sizes()[0]-/*7*/4)/2)].item<double>() + result[((result1.sizes()[0]-/*7*/4)/2)].item<double>() > 0.5)
                {
                    flip = true;
                    result = gen->get_fractions_gas(paraboloid, true);
                    result1 = gen->get_fractions_gas(interface, true);
                }

                std::vector<double> fractions;
                int option;
                int count = 0;
                for (int i = 0; i < result.sizes()[0]; ++i)
                {
                    if (!flip)
                    {
                        if (i%4 == 0 && result[i].item<double>() > IRL::global_constants::VF_LOW && result1[i].item<double>() > IRL::global_constants::VF_LOW)
                        {
                            double v = result[i].item<double>() + result1[i].item<double>();
                            fractions.push_back(v);
                            option = 2;
                            count = 0;
                        }
                        else if (i%4 == 0 && result1[i].item<double>() > IRL::global_constants::VF_LOW)
                        {
                            option = 0;
                        }
                        else if (i%4 == 0)
                        {
                            option = 1;
                        }
                        if (option == 0)
                        {
                            fractions.push_back(result1[i].item<double>());
                        }
                        else if (option == 1)
                        {
                            fractions.push_back(result[i].item<double>());
                        }
                        else if (option == 2 && i%4 != 0)
                        {
                            ++count;
                            double x = result[i-count].item<double>();
                            double y = result1[i-count].item<double>();
                            double c = x/(x + y) * result[i].item<double>() + y/(x + y) * result1[i].item<double>();
                            //std::cout << x << " " << result[i].item<double>() << " " << y << " " << result1[i].item<double>() << " " << c << std::endl;
                            fractions.push_back(c);
                        }
                    }
                    else
                    {
                        if (i%4 == 0 && result[i].item<double>() > IRL::global_constants::VF_LOW && result1[i].item<double>() > IRL::global_constants::VF_LOW)
                        {
                            double v = result[i].item<double>() + result1[i].item<double>();
                            fractions.push_back(v);
                            option = 2;
                            count = 0;
                        }
                        else if (i%4 == 0 && result1[i].item<double>() < IRL::global_constants::VF_HIGH)
                        {
                            option = 0;
                        }
                        else if (i%4 == 0)
                        {
                            option = 1;
                        }
                        if (option == 0)
                        {
                            fractions.push_back(result1[i].item<double>());
                        }
                        else if (option == 1)
                        {
                            fractions.push_back(result[i].item<double>());
                        }
                        else if (option == 2 && i%4 != 0)
                        {
                            ++count;
                            double x = result[i-count].item<double>();
                            double y = result1[i-count].item<double>();
                            double c = x/(x + y) * result[i].item<double>() + y/(x + y) * result1[i].item<double>();
                            //std::cout << x << " " << result[i].item<double>() << " " << y << " " << result1[i].item<double>() << " " << c << std::endl;
                            fractions.push_back(c);
                        }
                    }
                }

                auto sm = IRL::spatial_moments();
                std::vector<double> center = sm.get_mass_centers(fractions);

                int direction = 0;
                direction = rotateFractions(&fractions,center);

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
                auto surface_and_moments = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParametrizedSurfaceOutput>>(cube, paraboloid);
                auto surface = surface_and_moments.getSurface();
                auto normal = surface.getAverageNormalNonAligned();
                auto surface_and_moments2 = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParametrizedSurfaceOutput>>(cube, interface);
                auto surface2 = surface_and_moments2.getSurface();
                auto normal2 = surface2.getAverageNormalNonAligned();

                switch (direction)
                {
                    case 1:
                    normal[0] = -normal[0];
                    normal2[0] = -normal2[0];
                    break;
                    case 2:
                    normal[1] = -normal[1];
                    normal2[1] = -normal2[1];
                    break;
                    case 3:
                    normal[2] = -normal[2];
                    normal2[2] = -normal2[2];
                    break;
                    case 4:
                    normal[0] = -normal[0];
                    normal[1] = -normal[1];
                    normal2[0] = -normal2[0];
                    normal2[1] = -normal2[1];
                    break;
                    case 5:
                    normal[0] = -normal[0];
                    normal[2] = -normal[2];
                    normal2[0] = -normal2[0];
                    normal2[2] = -normal2[2];
                    break;
                    case 6:
                    normal[1] = -normal[1];
                    normal[2] = -normal[2];
                    normal2[1] = -normal2[1];
                    normal2[2] = -normal2[2];
                    break;
                    case 7:
                    normal[0] = -normal[0];
                    normal[1] = -normal[1];
                    normal[2] = -normal[2];
                    normal2[0] = -normal2[0];
                    normal2[1] = -normal2[1];
                    normal2[2] = -normal2[2];
                    break;
                }
                if (!flip)
                {
                    normal[0] = -normal[0];
                    normal[1] = -normal[1];
                    normal[2] = -normal[2];
                    normal2[0] = -normal2[0];
                    normal2[1] = -normal2[1];
                    normal2[2] = -normal2[2];
                }

                normals << normal[0] << "," << normal[1] << "," << normal[2] << "," << normal2[0] << "," << normal2[1] << "," << normal2[2] << "\n";
                normals.close();    

                /*for (int i = 0; i < 3; ++i)
                {
                    for (int j = 0; j < 3; ++j)
                    {
                        for (int k = 0; k < 3; ++k)
                        {
                            const auto bottom_corner = IRL::Pt(-1.5+i, -1.5+j, -1.5+k);
                            const auto top_corner = IRL::Pt(-0.5+i, -0.5+j, -0.5+k);
                            const auto cell = IRL::StoredRectangularCuboid<IRL::Pt>::fromBoundingPts(bottom_corner, top_corner);

                            const auto first_moments_and_surface = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParametrizedSurfaceOutput>, IRL::HalfEdgeCutting>(cell, paraboloid);
                            const auto first_moments_and_surface2 = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParametrizedSurfaceOutput>, IRL::HalfEdgeCutting>(cell, interface);
                            auto surface = first_moments_and_surface.getSurface();
                            auto surface2 = first_moments_and_surface2.getSurface();
                            const double length_scale = 0.05;
                            IRL::TriangulatedSurfaceOutput triangulated_surface = first_moments_and_surface.getSurface().triangulate(length_scale);
                            IRL::TriangulatedSurfaceOutput triangulated_surface2 = first_moments_and_surface2.getSurface().triangulate(length_scale);
                            string name = "p" + std::to_string(n)+std::to_string(i)+std::to_string(j)+std::to_string(k);
                            string name2 = "i" + std::to_string(n)+std::to_string(i)+std::to_string(j)+std::to_string(k);
                            triangulated_surface.write(name);
                            triangulated_surface2.write(name2);
                        }
                    }
                }*/      
            }       
        };

        void generate_two_paraboloids_in_cell_with_disturbance(double rota_l, double rota_h, double rotb_l, double rotb_h, double rotc_l, double rotc_h, double coa_l, double coa_h, double cob_l, double cob_h, double ox_l, double ox_h, double oy_l, double oy_h, double oz_l, double oz_h)
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
                coefficients << paraboloid.getDatum().x() << "," << paraboloid.getDatum().y() << "," << paraboloid.getDatum().z()
                << "," << angles[0] << "," << angles[1] << "," << angles[2]
                << "," << paraboloid.getAlignedParaboloid().a() << "," << paraboloid.getAlignedParaboloid().b() << "\n";
                coefficients << paraboloid.getDatum().x() << "," << paraboloid.getDatum().y() << "," << paraboloid.getDatum().z()
                << "," << angles[0] << "," << angles[1] << "," << angles[2]
                << "," << paraboloid.getAlignedParaboloid().a() << "," << paraboloid.getAlignedParaboloid().b() << "\n";
                coefficients.close();

                std::ofstream curvatures;
                std::string curv_name = "curvatures.txt";
                curvatures.open(curv_name, std::ios_base::app);
                curvatures << paraboloid.getAlignedParaboloid().a() << "," << paraboloid.getAlignedParaboloid().b() << "\n";
                curvatures << paraboloid.getAlignedParaboloid().a() << "," << paraboloid.getAlignedParaboloid().b() << "\n";
                curvatures << paraboloid.getAlignedParaboloid().a() << "," << paraboloid.getAlignedParaboloid().b() << "\n";
                coefficients.close();

                std::ofstream normals;
                std::string normals_name = "normals.txt";
                normals.open(normals_name, std::ios_base::app);
                auto cube = IRL::RectangularCuboid::fromBoundingPts(IRL::Pt(-0.5, -0.5, -0.5), IRL::Pt(0.5, 0.5, 0.5));
                auto surface_and_moments = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParametrizedSurfaceOutput>>(cube, paraboloid);
                auto surface = surface_and_moments.getSurface();
                auto normal = surface.getAverageNormalNonAligned();
                normals << normal[0] << "," << normal[1] << "," << normal[2] << "\n";
                normals << normal[0] << "," << normal[1] << "," << normal[2] << "\n";
                normals << normal[0] << "," << normal[1] << "," << normal[2] << "\n";
                normals.close();

                std::ofstream classification;
                std::string data_name = "type.txt";
                classification.open(data_name, std::ios_base::app);
                if ((abs(paraboloid.getAlignedParaboloid().a() - paraboloid.getAlignedParaboloid().b()) > 1) && (paraboloid.getAlignedParaboloid().a() < 0.2 || paraboloid.getAlignedParaboloid().b() < 0.2))
                {
                    classification << "0,1,0" << " \n";
                    classification << "0,1,0" << " \n";
                    classification << "0,1,0" << " \n";
                }
                else if ((abs(paraboloid.getAlignedParaboloid().a() - paraboloid.getAlignedParaboloid().b()) < 1) && (paraboloid.getAlignedParaboloid().a() > 2 || paraboloid.getAlignedParaboloid().b() > 2))
                {
                    classification << "1,0,0" << " \n";
                    classification << "1,0,0" << " \n";
                    classification << "1,0,0" << " \n";
                }
                else
                {
                    classification << "0,0,1" << " \n";
                    classification << "0,0,1" << " \n";
                    classification << "0,0,1" << " \n";
                }
                classification.close();

                std::ofstream inter;
                std::string interface_name = "interface.txt";
                inter.open(interface_name, std::ios_base::app);
                inter << "0,1" << " \n";
                inter << "0,1" << " \n";
                inter << "0,1" << " \n";
                inter.close();

                auto result = gen->get_fractions(paraboloid, true);
                auto result1 = gen->get_fractions_gas(interface, true);
                std::ofstream output;
                data_name = "fractions.txt";
                output.open(data_name, std::ios_base::app);
                int option;
                int count = 0;

                for (int i = 0; i < result.sizes()[0]; ++i)
                {
                    if (i%4 == 0 && result[i].item<double>() > IRL::global_constants::VF_LOW && result1[i].item<double>() > IRL::global_constants::VF_LOW)
                    {
                        double v = result[i].item<double>() + result1[i].item<double>();
                        output << v << ",";
                        result[i] = v;
                        option = 2;
                        count = 0;
                    }
                    else if (i%4 == 0 && result1[i].item<double>() > IRL::global_constants::VF_LOW)
                    {
                        option = 0;
                    }
                    else if (i%4 == 0)
                    {
                        option = 1;
                    }
                    if (option == 0)
                    {
                        output << result1[i].item<double>() << ",";
                        result[i] = result1[i];
                    }
                    else if (option == 1)
                    {
                        output << result[i].item<double>() << ",";
                    }
                    else if (option == 2 && i%4 != 0)
                    {
                        ++count;
                        double x = result[i-count].item<double>() - result1[i-count].item<double>();
                        double y = result1[i-count].item<double>();
                        double c = x/(x + y) * result[i].item<double>() + y/(x + y) * result1[i].item<double>();
                        //std::cout << x << " " << result[i].item<double>() << " " << y << " " << result1[i].item<double>() << " " << c << std::endl;
                        output << c << ",";
                        result[i] = c;
                    }
                }
                output << "\n";
                srand((unsigned) time(NULL));
                for (int i = 0; i < result.sizes()[0]; ++i)
                {   
                    int r = rand() % 3 - 1;
                    double x = result[i].item<double>() + r*result[i].item<double>()*0.01;
                    if (i % 4 == 0 && x < 0)
                    {
                        x = 0;
                    }
                    else if (i % 4 == 0 && x > 1)
                    {
                        x = 1;
                    }
                    else if (i % 4 != 0 && x > 0.5)
                    {
                        x = 0.5;
                    }
                    else if (i % 4 != 0 && x < -0.5)
                    {
                        x = -0.5;
                    }
                    output << x << ",";
                }
                output << "\n";
                for (int i = 0; i < result.sizes()[0]; ++i)
                {   
                    int r = rand() % 3 - 1;
                    double x = result[i].item<double>() + r*result[i].item<double>()*0.01;
                    if (i % 4 == 0 && x < 0)
                    {
                        x = 0;
                    }
                    else if (i % 4 == 0 && x > 1)
                    {
                        x = 1;
                    }
                    else if (i % 4 != 0 && x > 0.5)
                    {
                        x = 0.5;
                    }
                    else if (i % 4 != 0 && x < -0.5)
                    {
                        x = -0.5;
                    }
                    output << x << ",";
                }
                output << "\n";
                output.close();      

                /*for (int i = 0; i < 3; ++i)
                {
                    for (int j = 0; j < 3; ++j)
                    {
                        for (int k = 0; k < 3; ++k)
                        {
                            const auto bottom_corner = IRL::Pt(-1.5+i, -1.5+j, -1.5+k);
                            const auto top_corner = IRL::Pt(-0.5+i, -0.5+j, -0.5+k);
                            const auto cell = IRL::StoredRectangularCuboid<IRL::Pt>::fromBoundingPts(bottom_corner, top_corner);

                            const auto first_moments_and_surface = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParametrizedSurfaceOutput>, IRL::HalfEdgeCutting>(cell, paraboloid);
                            const auto first_moments_and_surface2 = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParametrizedSurfaceOutput>, IRL::HalfEdgeCutting>(cell, interface);
                            auto surface = first_moments_and_surface.getSurface();
                            auto surface2 = first_moments_and_surface2.getSurface();
                            const double length_scale = 0.05;
                            IRL::TriangulatedSurfaceOutput triangulated_surface = first_moments_and_surface.getSurface().triangulate(length_scale);
                            IRL::TriangulatedSurfaceOutput triangulated_surface2 = first_moments_and_surface2.getSurface().triangulate(length_scale);
                            string name = "p" + std::to_string(n)+std::to_string(i)+std::to_string(j)+std::to_string(k);
                            string name2 = "i" + std::to_string(n)+std::to_string(i)+std::to_string(j)+std::to_string(k);
                            triangulated_surface.write(name);
                            triangulated_surface2.write(name2);
                        }
                    }
                }   */    
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
                auto surface_and_moments = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParametrizedSurfaceOutput>>(cube, paraboloid[13]);
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
                            auto moments = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParametrizedSurfaceOutput>>(cube, paraboloid[k+3*j+9*i]);
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
                auto surface_and_moments = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParametrizedSurfaceOutput>>(cube, paraboloid);
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