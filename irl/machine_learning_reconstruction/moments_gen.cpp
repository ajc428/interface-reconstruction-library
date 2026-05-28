// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2023 Andrew Cahaly <andrew.cahaly@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "irl/machine_learning_reconstruction/moments_gen.h"

using namespace std;

namespace IRL
{
    moments_gen::moments_gen(int nx, int ny, int nz, int sx, int sy, int sz, double lx, double ly, double lz)
    {
        srand((unsigned) time(NULL));
        mesh = new IRL::stencil(nx, ny, nz, sx, sy, sz, lx, ly, lz);
        IRL::setVolumeFractionBounds(1.0e-15);
        std::cout.precision(15);
        distribution1 = std::normal_distribution<double>(0.0,0.3);
        distribution2 = std::normal_distribution<double>(0.0,0.3);
    }

    moments_gen::~moments_gen()
    {
        delete mesh;
    }

    IRL::Paraboloid moments_gen::new_paraboloid(double x, double y, double z, double a, double b, double c, double alpha, double beta)
    {
        IRL::Pt datum;
        IRL::ReferenceFrame frame;
        std::array<double, 3> angle;

        frame = IRL::ReferenceFrame(IRL::Normal(1.0, 0.0, 0.0), IRL::Normal(0.0, 1.0, 0.0), IRL::Normal(0.0, 0.0, 1.0));
        datum = IRL::Pt(x,y,z);
        angle = {a, b, c};

        IRL::UnitQuaternion x_rotation(angle[0], frame[0]);
        IRL::UnitQuaternion y_rotation(angle[1], frame[1]);
        IRL::UnitQuaternion z_rotation(angle[2], frame[2]);
        frame = x_rotation * y_rotation * z_rotation * frame;

        return IRL::Paraboloid(datum, frame, alpha, beta);
    }

    IRL::Paraboloid moments_gen::new_paraboloid(double x, double y, double z, IRL::ReferenceFrame frame, double alpha, double beta)
    {
        IRL::Pt datum = IRL::Pt(x,y,z);
        return IRL::Paraboloid(datum, frame, alpha, beta);
    }

    IRL::Paraboloid moments_gen::new_random_paraboloid(double rota_l, double rota_h, double rotb_l, double rotb_h, double rotc_l, double rotc_h, double coa_l, double coa_h, double cob_l, double cob_h, double ox_l, double ox_h, double oy_l, double oy_h, double oz_l, double oz_h)
    {
        std::random_device rd;  
        std::mt19937_64 a_eng(rd());

        IRL::Pt datum;
        IRL::ReferenceFrame frame;
        double alpha;
        double beta;

        std::uniform_real_distribution<double> random_rotationa(rota_l, rota_h);
        std::uniform_real_distribution<double> random_rotationb(rotb_l, rotb_h);
        std::uniform_real_distribution<double> random_rotationc(rotc_l, rotc_h);
        std::uniform_real_distribution<double> random_coeffsa(coa_l, coa_h);
        std::uniform_real_distribution<double> random_coeffsb(cob_l, cob_h);
        std::uniform_real_distribution<double> random_translationx(ox_l, ox_h);
        std::uniform_real_distribution<double> random_translationy(oy_l, oy_h);
        std::uniform_real_distribution<double> random_translationz(oz_l, oz_h);
        IRL::Paraboloid p;

        // alpha = random_coeffsa(a_eng);
        // beta = random_coeffsb(a_eng);
        alpha = distribution1(generator);
        beta = distribution2(generator);
        do
        {
            frame = IRL::ReferenceFrame(IRL::Normal(1.0, 0.0, 0.0), IRL::Normal(0.0, 1.0, 0.0), IRL::Normal(0.0, 0.0, 1.0));
            datum = IRL::Pt(random_translationx(a_eng), random_translationy(a_eng), random_translationz(a_eng));
            angles = {random_rotationa(a_eng), random_rotationb(a_eng), random_rotationc(a_eng)};

            IRL::UnitQuaternion x_rotation(angles[0], frame[0]);
            IRL::UnitQuaternion y_rotation(angles[1], frame[1]);
            IRL::UnitQuaternion z_rotation(angles[2], frame[2]);
            frame = x_rotation * y_rotation * z_rotation * frame;
            p = IRL::Paraboloid(datum, frame, alpha, beta);
        } while (!(isParaboloidInCenterCell(p)));

        return p;
    }

    std::vector<double> moments_gen::get_moments(IRL::Paraboloid p, int order, bool sym, bool& flip)
    {
        vector<double> f;

        const auto moments = getCellMoments(p, mesh->get_ic(), mesh->get_jc(), mesh->get_kc());  
        double volume = moments[0].volume();  
        flip = false; 
        if (volume > 0.5) 
        {
            flip = true;
        } 

        for (int i = 0; i < mesh->getNX(); ++i)
        {
            for (int j = 0; j < mesh->getNY(); ++j)
            {
                for (int k = 0; k < mesh->getNZ(); ++k)
                {
                    const auto moments = getCellMoments(p, i, j, k);  
                    double volume = moments[0].volume();   
                    IRL::Pt centroid = moments[0].centroid();  
                    IRL::Pt centroid_gas = moments[1].centroid(); 
                    flip = false;
                    if (sym && flip) 
                    {
                        volume = moments[1].volume(); 
                        centroid = moments[1].centroid(); 
                        centroid_gas = moments[0].centroid();  
                    }  
                   
                    f.push_back(volume);
                    if (order > 0)
                    {
                        if (volume < IRL::global_constants::VF_LOW || volume > IRL::global_constants::VF_HIGH)
                        {
                            f.push_back(0);
                            f.push_back(0);
                            f.push_back(0);    
                            f.push_back(0);
                            f.push_back(0);
                            f.push_back(0);    
                        }
                        else
                        {
                            f.push_back(centroid[0] - mesh->get_xm(i));
                            f.push_back(centroid[1] - mesh->get_ym(j));
                            f.push_back(centroid[2] - mesh->get_zm(k));   
                            f.push_back(centroid_gas[0] - mesh->get_xm(i));
                            f.push_back(centroid_gas[1] - mesh->get_ym(j));
                            f.push_back(centroid_gas[2] - mesh->get_zm(k));    
                        }
                    }
                }
            }
        }
        return f;  
    }

    IRL::SeparatedMoments<IRL::VolumeMoments> moments_gen::getCellMoments(IRL::Paraboloid p, int i, int j, int k)
    {
        auto cell = mesh->getCell(i,j,k);
        auto moments = IRL::getNormalizedVolumeMoments<IRL::SeparatedMoments<IRL::VolumeMoments>>(cell, p);
        return moments;
    }

    bool moments_gen::isParaboloidInCenterCell(IRL::Paraboloid p) 
    {
        auto cell = mesh->getCell(mesh->get_ic(),mesh->get_jc(),mesh->get_kc());
        const double volume_fraction = IRL::getVolumeMoments<IRL::Volume, IRL::HalfEdgeCutting>(cell, p);
        return volume_fraction < IRL::global_constants::VF_HIGH && volume_fraction > IRL::global_constants::VF_LOW;
    }
}