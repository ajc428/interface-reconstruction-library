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
                    center = sm.get_mass_centers_all(&fractions);
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
                        /*double c = (rand() % 401 - 200) / 1000.0;
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
                        }*/
                        double c = (rand() % 101 - 200) / 1000.0;
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
                auto surface_and_moments = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParametrizedSurfaceOutput>>(cube, paraboloid);
                auto surface_and_moments1 = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParametrizedSurfaceOutput>>(cube, interface);
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

                const auto bottom_corner = IRL::Pt(-1.5, -1.5, -1.5);
                const auto top_corner = IRL::Pt(1.5, 1.5, 1.5);
                const auto cell = IRL::StoredRectangularCuboid<IRL::Pt>::fromBoundingPts(bottom_corner, top_corner);

                const auto first_moments_and_surface = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParametrizedSurfaceOutput>, IRL::HalfEdgeCutting>(cell, paraboloid);
                const auto first_moments_and_surface2 = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParametrizedSurfaceOutput>, IRL::HalfEdgeCutting>(cell, interface);
                const double length_scale = 0.05;
                IRL::TriangulatedSurfaceOutput triangulated_surface = first_moments_and_surface.getSurface().triangulate(length_scale);
                IRL::TriangulatedSurfaceOutput triangulated_surface2 = first_moments_and_surface2.getSurface().triangulate(length_scale);
                string name3 = "p";
                string name4 = "i";
                triangulated_surface.write(name3);
                triangulated_surface2.write(name4);       
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
                        double c = (rand() % 101 - 200) / 1000.0;
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
                auto surface_and_moments = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParametrizedSurfaceOutput>>(cube, paraboloid);
                auto surface_and_moments1 = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParametrizedSurfaceOutput>>(cube, interface);
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

                const auto first_moments_and_surface = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParametrizedSurfaceOutput>, IRL::HalfEdgeCutting>(cell, paraboloid);
                const auto first_moments_and_surface2 = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParametrizedSurfaceOutput>, IRL::HalfEdgeCutting>(cell, interface);
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
                    if (result[((result.sizes()[0]-7)/2)].item<double>() + result1[((result1.sizes()[0]-7)/2)].item<double>() > 0.5)
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
                auto surface_and_moments1 = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParametrizedSurfaceOutput>>(cube, interface);
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

                const auto bottom_corner = IRL::Pt(-1.5, -1.5, -1.5);
                const auto top_corner = IRL::Pt(1.5, 1.5, 1.5);
                const auto cell = IRL::StoredRectangularCuboid<IRL::Pt>::fromBoundingPts(bottom_corner, top_corner);

                const auto first_moments_and_surface = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParametrizedSurfaceOutput>, IRL::HalfEdgeCutting>(cell, paraboloid);
                const auto first_moments_and_surface2 = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParametrizedSurfaceOutput>, IRL::HalfEdgeCutting>(cell, interface);
                const double length_scale = 0.05;
                IRL::TriangulatedSurfaceOutput triangulated_surface = first_moments_and_surface.getSurface().triangulate(length_scale);
                IRL::TriangulatedSurfaceOutput triangulated_surface2 = first_moments_and_surface2.getSurface().triangulate(length_scale);
                string name3 = "p";
                string name4 = "i";
                triangulated_surface.write(name3);
                triangulated_surface2.write(name4);       
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
                        double c = (rand() % 101 - 200) / 1000.0;
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
                auto surface_and_moments = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParametrizedSurfaceOutput>>(cube, paraboloid);
                auto surface_and_moments1 = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParametrizedSurfaceOutput>>(cube, interface);
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

                const auto bottom_corner = IRL::Pt(-1.5, -1.5, -1.5);
                const auto top_corner = IRL::Pt(1.5, 1.5, 1.5);
                const auto cell = IRL::StoredRectangularCuboid<IRL::Pt>::fromBoundingPts(bottom_corner, top_corner);

                const auto first_moments_and_surface = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParametrizedSurfaceOutput>, IRL::HalfEdgeCutting>(cell, paraboloid);
                const auto first_moments_and_surface2 = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParametrizedSurfaceOutput>, IRL::HalfEdgeCutting>(cell, interface);
                const double length_scale = 0.05;
                IRL::TriangulatedSurfaceOutput triangulated_surface = first_moments_and_surface.getSurface().triangulate(length_scale);
                IRL::TriangulatedSurfaceOutput triangulated_surface2 = first_moments_and_surface2.getSurface().triangulate(length_scale);
                string name3 = "p";
                string name4 = "i";
                triangulated_surface.write(name3);
                triangulated_surface2.write(name4);       
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

                for (int i = 0; i < result.sizes()[0]; ++i)
                {
                    output << fractions[i] << ",";
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

                /*const Mesh& mesh = gen->getMesh();
                FILE* viz_file;
                std::string file_name = "interface_0.vtu";
                viz_file = fopen(file_name.c_str(), "w");

                // Build vectors of vertex locations and connectivities
                std::size_t n_vert = 0;
                std::size_t n_faces = 0;
                std::size_t current_face_size = 0;
                std::string vert_loc;
                std::string connectivity;
                std::string offsets;

                auto add_polyhedron = [&](const auto& a_poly) {
                    using T = std::decay_t<decltype(a_poly)>;
                    std::unordered_map<const typename T::vertex_type*, IRL::UnsignedIndex_t>
                        unique_vertices;
                    for (IRL::UnsignedIndex_t n = 0; n < a_poly.getNumberOfVertices(); ++n) {
                    unique_vertices[a_poly.getVertex(n)] = n + n_vert;
                    const auto& vert_pt = a_poly.getVertex(n)->getLocation();
                    vert_loc += std::to_string(vert_pt[0]) + " " +
                                std::to_string(vert_pt[1]) + " " +
                                std::to_string(vert_pt[2]) + "\n";
                    }
                    assert(unique_vertices.size() == a_poly.getNumberOfVertices());

                    for (IRL::UnsignedIndex_t n = 0; n < a_poly.getNumberOfFaces(); ++n) {
                    const auto& face = a_poly[n];
                    auto current_half_edge = face->getStartingHalfEdge();
                    do {
                        ++current_face_size;
                        connectivity +=
                            std::to_string(unique_vertices[current_half_edge->getVertex()]) +
                            " ";
                        current_half_edge = current_half_edge->getNextHalfEdge();
                    } while (current_half_edge != face->getStartingHalfEdge());
                    offsets += std::to_string(current_face_size) + " ";
                    connectivity += "\n";
                    }

                    n_vert += a_poly.getNumberOfVertices();
                    n_faces += a_poly.getNumberOfFaces();
                };

                for (int i = mesh.imin(); i <= mesh.imax(); ++i) {
                    for (int j = mesh.jmin(); j <= mesh.jmax(); ++j) {
                        for (int k = mesh.kmin(); k <= mesh.kmax(); ++k) {
                            const auto& recon = plane;
                            auto cell = IRL::RectangularCuboid::fromBoundingPts(
                                IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)),
                                IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));

                            if (recon.isFlipped()) {
                            auto he_poly = cell.generateHalfEdgeVersion();
                            auto seg = he_poly.generateSegmentedPolyhedron();
                            for (const auto& plane : recon) {
                                decltype(seg) clipped;
                                auto new_plane = plane.generateFlippedPlane();
                                IRL::splitHalfEdgePolytope(&seg, &clipped, &he_poly, new_plane);
                                add_polyhedron(clipped);
                            }
                            } else {
                            auto he_poly = cell.generateHalfEdgeVersion();
                            auto seg = he_poly.generateSegmentedPolyhedron();
                            for (const auto& plane : recon) {
                                decltype(seg) clipped;
                                IRL::splitHalfEdgePolytope(&seg, &clipped, &he_poly, plane);
                            }
                            add_polyhedron(seg);
                            }
                        }
                    }
                }

                // Write header
                {
                    fprintf(viz_file, "<?xml version=\"1.0\"?>\n");
                    fprintf(viz_file,
                            "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" "
                            "byte_order=\"LittleEndian\">\n");
                    fprintf(viz_file, "<UnstructuredGrid>\n");
                    fprintf(viz_file, "<Piece NumberOfPoints=\"%zu\" NumberOfCells=\"%zu\">\n",
                            n_vert, n_faces);

                    fprintf(viz_file, "<Points>\n");
                    fprintf(viz_file,
                            "<DataArray type=\"Float32\" NumberOfComponents=\"3\">\n");
                    fprintf(viz_file, "%s", vert_loc.c_str());
                    fprintf(viz_file, "</DataArray>\n");
                    fprintf(viz_file, "</Points>\n");

                    fprintf(viz_file, "<Cells>\n");
                    fprintf(viz_file,
                            "<DataArray type=\"Int32\" Name=\"connectivity\" "
                            "format=\"ascii\">\n");
                    fprintf(viz_file, "%s", connectivity.c_str());
                    fprintf(viz_file, "</DataArray>\n");

                    fprintf(viz_file,
                            "<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n");
                    fprintf(viz_file, "%s", offsets.c_str());
                    fprintf(viz_file, "\n</DataArray>\n");

                    // Cell type - General Polygon type
                    fprintf(viz_file,
                            "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n");
                    for (std::size_t n = 0; n < n_faces; ++n) {
                    fprintf(viz_file, "7 ");  // General polygon type
                    }
                    fprintf(viz_file,
                            "\n</DataArray>\n</Cells>\n</Piece>\n</UnstructuredGrid>\n</"
                            "VTKFile>\n");
                }
                fclose(viz_file);*/       
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
                        double c = (rand() % 101 - 200) / 1000.0;
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
            for (int n = 0; n < Ntests; ++n) 
            {
                std::cout << n << endl;
                IRL::PlanarSeparator plane = gen->new_random_R2P(rota1_l, rota1_h, rotb1_l, rotb1_h, rota2_l, rota2_h, rotb2_l, rotb2_h, d1_l, d1_h, d2_l, d2_h, inter, same);

                std::ofstream coefficients;
                std::string name = "coefficients.txt";
                coefficients.open(name, std::ios_base::app);
                coefficients << plane[0].normal()[0] << "," << plane[0].normal()[1] << "," << plane[0].normal()[2] << "," << plane[0].distance() 
                << "," << plane[1].normal()[0] << "," << plane[1].normal()[1] << "," << plane[1].normal()[2] << "," << plane[1].distance() << "," << plane.flip() << "\n";
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

                for (int i = 0; i < result.sizes()[0]; ++i)
                {
                    output << fractions[i] << ",";
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

                // const Mesh& mesh = gen->getMesh();
                // FILE* viz_file;
                // std::string file_name = "interface_0.vtu";
                // viz_file = fopen(file_name.c_str(), "w");

                // // Build vectors of vertex locations and connectivities
                // std::size_t n_vert = 0;
                // std::size_t n_faces = 0;
                // std::size_t current_face_size = 0;
                // std::string vert_loc;
                // std::string connectivity;
                // std::string offsets;

                // auto add_polyhedron = [&](const auto& a_poly) {
                //     using T = std::decay_t<decltype(a_poly)>;
                //     std::unordered_map<const typename T::vertex_type*, IRL::UnsignedIndex_t>
                //         unique_vertices;
                //     for (IRL::UnsignedIndex_t n = 0; n < a_poly.getNumberOfVertices(); ++n) {
                //     unique_vertices[a_poly.getVertex(n)] = n + n_vert;
                //     const auto& vert_pt = a_poly.getVertex(n)->getLocation();
                //     vert_loc += std::to_string(vert_pt[0]) + " " +
                //                 std::to_string(vert_pt[1]) + " " +
                //                 std::to_string(vert_pt[2]) + "\n";
                //     }
                //     assert(unique_vertices.size() == a_poly.getNumberOfVertices());

                //     for (IRL::UnsignedIndex_t n = 0; n < a_poly.getNumberOfFaces(); ++n) {
                //     const auto& face = a_poly[n];
                //     auto current_half_edge = face->getStartingHalfEdge();
                //     do {
                //         ++current_face_size;
                //         connectivity +=
                //             std::to_string(unique_vertices[current_half_edge->getVertex()]) +
                //             " ";
                //         current_half_edge = current_half_edge->getNextHalfEdge();
                //     } while (current_half_edge != face->getStartingHalfEdge());
                //     offsets += std::to_string(current_face_size) + " ";
                //     connectivity += "\n";
                //     }

                //     n_vert += a_poly.getNumberOfVertices();
                //     n_faces += a_poly.getNumberOfFaces();
                // };

                // for (int i = mesh.imin(); i <= mesh.imax(); ++i) {
                //     for (int j = mesh.jmin(); j <= mesh.jmax(); ++j) {
                //         for (int k = mesh.kmin(); k <= mesh.kmax(); ++k) {
                //             const auto& recon = plane;
                //             auto cell = IRL::RectangularCuboid::fromBoundingPts(
                //                 IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)),
                //                 IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));

                //             if (recon.isFlipped()) {
                //             auto he_poly = cell.generateHalfEdgeVersion();
                //             auto seg = he_poly.generateSegmentedPolyhedron();
                //             for (const auto& plane : recon) {
                //                 decltype(seg) clipped;
                //                 auto new_plane = plane.generateFlippedPlane();
                //                 IRL::splitHalfEdgePolytope(&seg, &clipped, &he_poly, new_plane);
                //                 add_polyhedron(clipped);
                //             }
                //             } else {
                //             auto he_poly = cell.generateHalfEdgeVersion();
                //             auto seg = he_poly.generateSegmentedPolyhedron();
                //             for (const auto& plane : recon) {
                //                 decltype(seg) clipped;
                //                 IRL::splitHalfEdgePolytope(&seg, &clipped, &he_poly, plane);
                //             }
                //             add_polyhedron(seg);
                //             }
                //         }
                //     }
                // }

                // // Write header
                // {
                //     fprintf(viz_file, "<?xml version=\"1.0\"?>\n");
                //     fprintf(viz_file,
                //             "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" "
                //             "byte_order=\"LittleEndian\">\n");
                //     fprintf(viz_file, "<UnstructuredGrid>\n");
                //     fprintf(viz_file, "<Piece NumberOfPoints=\"%zu\" NumberOfCells=\"%zu\">\n",
                //             n_vert, n_faces);

                //     fprintf(viz_file, "<Points>\n");
                //     fprintf(viz_file,
                //             "<DataArray type=\"Float32\" NumberOfComponents=\"3\">\n");
                //     fprintf(viz_file, "%s", vert_loc.c_str());
                //     fprintf(viz_file, "</DataArray>\n");
                //     fprintf(viz_file, "</Points>\n");

                //     fprintf(viz_file, "<Cells>\n");
                //     fprintf(viz_file,
                //             "<DataArray type=\"Int32\" Name=\"connectivity\" "
                //             "format=\"ascii\">\n");
                //     fprintf(viz_file, "%s", connectivity.c_str());
                //     fprintf(viz_file, "</DataArray>\n");

                //     fprintf(viz_file,
                //             "<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n");
                //     fprintf(viz_file, "%s", offsets.c_str());
                //     fprintf(viz_file, "\n</DataArray>\n");

                //     // Cell type - General Polygon type
                //     fprintf(viz_file,
                //             "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n");
                //     for (std::size_t n = 0; n < n_faces; ++n) {
                //     fprintf(viz_file, "7 ");  // General polygon type
                //     }
                //     fprintf(viz_file,
                //             "\n</DataArray>\n</Cells>\n</Piece>\n</UnstructuredGrid>\n</"
                //             "VTKFile>\n");
                // }
                // fclose(viz_file);        
            }  
        }; 

        void generate_R2P_with_disturbance(double rota1_l, double rota1_h, double rotb1_l, double rotb1_h, double rota2_l, double rota2_h, double rotb2_l, double rotb2_h, double d1_l, double d1_h, double d2_l, double d2_h, bool inter, bool same)
        {
            srand((unsigned) time(NULL));
            for (int n = 0; n < Ntests; ++n) 
            {
                std::cout << n << endl;
                IRL::PlanarSeparator plane = gen->new_random_R2P(rota1_l, rota1_h, rotb1_l, rotb1_h, rota2_l, rota2_h, rotb2_l, rotb2_h, d1_l, d1_h, d2_l, d2_h, inter, same);

                std::ofstream coefficients;
                std::string name = "coefficients.txt";
                coefficients.open(name, std::ios_base::app);
                coefficients << plane[0].normal()[0] << "," << plane[0].normal()[1] << "," << plane[0].normal()[2] << "," << plane[0].distance() 
                << plane[1].normal()[0] << "," << plane[1].normal()[1] << "," << plane[1].normal()[2] << "," << plane[1].distance() << "\n";
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
                        double c = (rand() % 101 - 200) / 1000.0;
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