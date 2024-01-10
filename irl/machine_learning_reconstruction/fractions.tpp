// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2023 Andrew Cahaly <andrew.cahaly@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef IRL_MACHINE_LEARNING_RECONSTRUCTION_FRACTIONS_TPP_
#define IRL_MACHINE_LEARNING_RECONSTRUCTION_FRACTIONS_TPP_

using namespace std;

namespace IRL
{
    fractions::fractions(const int a)
    {
        a_number_of_cells = a;
        mesh = initializeMesh(a_number_of_cells);
        IRL::setVolumeFractionBounds(1.0e-14);
        std::cout.precision(15);
    }

    IRL::Paraboloid fractions::new_parabaloid(double x, double y, double z, double a, double b, double c, double alpha, double beta)
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

    IRL::Paraboloid fractions::new_parabaloid(double x, double y, double z, IRL::ReferenceFrame frame, double alpha, double beta)
    {
        IRL::Pt datum = IRL::Pt(x,y,z);
        return IRL::Paraboloid(datum, frame, alpha, beta);
    }

    IRL::Paraboloid fractions::new_random_parabaloid(double rota_l, double rota_h, double rotb_l, double rotb_h, double rotc_l, double rotc_h, double coa_l, double coa_h, double cob_l, double cob_h, double ox_l, double ox_h, double oy_l, double oy_h, double oz_l, double oz_h)
    {
        std::random_device rd;  
        std::mt19937_64 a_eng(rd());
        DataMesh<double> liquid_volume_fraction(mesh);
        IRL::Pt datum;
        IRL::ReferenceFrame frame;
        double z_offset;
        double alpha;
        double beta;
        int attempt = 0;
        std::uniform_real_distribution<double> random_rotationa(rota_l, rota_h);
        std::uniform_real_distribution<double> random_rotationb(rotb_l, rotb_h);
        std::uniform_real_distribution<double> random_rotationc(rotc_l, rotc_h);
        std::uniform_real_distribution<double> random_coeffsa(coa_l, coa_h);
        std::uniform_real_distribution<double> random_coeffsb(cob_l, cob_h);
        std::uniform_real_distribution<double> random_translationx(ox_l, ox_h);
        std::uniform_real_distribution<double> random_translationy(oy_l, oy_h);
        std::uniform_real_distribution<double> random_translationz(oz_l, oz_h);
        IRL::Paraboloid p;

        do
        {
            alpha = random_coeffsa(a_eng);
            beta = random_coeffsb(a_eng);
            frame = IRL::ReferenceFrame(IRL::Normal(1.0, 0.0, 0.0), IRL::Normal(0.0, 1.0, 0.0), IRL::Normal(0.0, 0.0, 1.0));
            datum = IRL::Pt(random_translationx(a_eng), random_translationy(a_eng), random_translationz(a_eng));
            angles = {random_rotationa(a_eng), random_rotationb(a_eng), random_rotationc(a_eng)};

            IRL::UnitQuaternion x_rotation(angles[0], frame[0]);
            IRL::UnitQuaternion y_rotation(angles[1], frame[1]);
            IRL::UnitQuaternion z_rotation(angles[2], frame[2]);
            frame = x_rotation * y_rotation * z_rotation * frame;
            p = IRL::Paraboloid(datum, frame, alpha, beta);
        } while (!(isParaboloidInCenterCell(p, liquid_volume_fraction)));

        return p;
    }

    IRL::Paraboloid fractions::new_random_parabaloid_not_center(double rota_l, double rota_h, double rotb_l, double rotb_h, double rotc_l, double rotc_h, double coa_l, double coa_h, double cob_l, double cob_h, double ox_l, double ox_h, double oy_l, double oy_h, double oz_l, double oz_h)
    {
        std::random_device rd;  
        std::mt19937_64 a_eng(rd());
        DataMesh<double> liquid_volume_fraction(mesh);
        IRL::Pt datum;
        IRL::ReferenceFrame frame;
        double z_offset;
        double alpha;
        double beta;
        int attempt = 0;
        std::uniform_real_distribution<double> random_rotationa(rota_l, rota_h);
        std::uniform_real_distribution<double> random_rotationb(rotb_l, rotb_h);
        std::uniform_real_distribution<double> random_rotationc(rotc_l, rotc_h);
        std::uniform_real_distribution<double> random_coeffsa(coa_l, coa_h);
        std::uniform_real_distribution<double> random_coeffsb(cob_l, cob_h);
        std::uniform_real_distribution<double> random_translationx(ox_l, ox_h);
        std::uniform_real_distribution<double> random_translationy(oy_l, oy_h);
        std::uniform_real_distribution<double> random_translationz(oz_l, oz_h);
        IRL::Paraboloid p;

        alpha = random_coeffsa(a_eng);
        beta = random_coeffsb(a_eng);
        frame = IRL::ReferenceFrame(IRL::Normal(1.0, 0.0, 0.0), IRL::Normal(0.0, 1.0, 0.0), IRL::Normal(0.0, 0.0, 1.0));
        datum = IRL::Pt(random_translationx(a_eng), random_translationy(a_eng), random_translationz(a_eng));
        angles = {random_rotationa(a_eng), random_rotationb(a_eng), random_rotationc(a_eng)};

        IRL::UnitQuaternion x_rotation(angles[0], frame[0]);
        IRL::UnitQuaternion y_rotation(angles[1], frame[1]);
        IRL::UnitQuaternion z_rotation(angles[2], frame[2]);
        frame = x_rotation * y_rotation * z_rotation * frame;
        p = IRL::Paraboloid(datum, frame, alpha, beta);

        return p;
    }

    IRL::Paraboloid fractions::new_interface_parabaloid(double rota_l, double rota_h, double rotb_l, double rotb_h, double rotc_l, double rotc_h, double coa_l, double coa_h, double cob_l, double cob_h, double ox_l, double ox_h, double oy_l, double oy_h, double oz_l, double oz_h, IRL::Paraboloid para)
    {
        std::random_device rd;  
        std::mt19937_64 a_eng(rd());
        DataMesh<double> liquid_volume_fraction(mesh);
        IRL::Pt datum;
        IRL::ReferenceFrame frame;
        double z_offset;
        double alpha;
        double beta;
        int attempt = 0;
        std::uniform_real_distribution<double> random_rotationa(rota_l, rota_h);
        std::uniform_real_distribution<double> random_rotationb(rotb_l, rotb_h);
        std::uniform_real_distribution<double> random_rotationc(rotc_l, rotc_h);
        std::uniform_real_distribution<double> random_coeffsa(coa_l, coa_h);
        std::uniform_real_distribution<double> random_coeffsb(cob_l, cob_h);
        std::uniform_real_distribution<double> random_translationx(-1.5, 1.5);
        std::uniform_real_distribution<double> random_translationy(-1.5, 1.5);
        std::uniform_real_distribution<double> random_translationz(-1.5, 1.5);
        IRL::Paraboloid p;

        do
        {
            alpha = random_coeffsa(a_eng);
            beta = random_coeffsb(a_eng);
            frame = IRL::ReferenceFrame(IRL::Normal(1.0, 0.0, 0.0), IRL::Normal(0.0, 1.0, 0.0), IRL::Normal(0.0, 0.0, 1.0));
            datum = IRL::Pt(random_translationx(a_eng), random_translationy(a_eng), random_translationz(a_eng));
            std::array<double, 3> angles2 = {random_rotationa(a_eng), random_rotationb(a_eng), random_rotationc(a_eng)};

            IRL::UnitQuaternion x_rotation(angles2[0], frame[0]);
            IRL::UnitQuaternion y_rotation(angles2[1], frame[1]);
            IRL::UnitQuaternion z_rotation(angles2[2], frame[2]);
            frame = x_rotation * y_rotation * z_rotation * frame;
            p = IRL::Paraboloid(datum, frame, alpha, beta);
        } while ((isParaboloidInCenterCell(p, liquid_volume_fraction)) || areParaboloidsInSameCell(p, para, liquid_volume_fraction));

        return p;
    }

    IRL::Paraboloid fractions::new_interface_parabaloid_in_cell(double rota_l, double rota_h, double rotb_l, double rotb_h, double rotc_l, double rotc_h, double coa_l, double coa_h, double cob_l, double cob_h, double ox_l, double ox_h, double oy_l, double oy_h, double oz_l, double oz_h, IRL::Paraboloid para)
    {
        std::random_device rd;  
        std::mt19937_64 a_eng(rd());
        DataMesh<double> liquid_volume_fraction(mesh);
        IRL::Pt datum;
        IRL::ReferenceFrame frame;
        double z_offset;
        double alpha;
        double beta;
        int attempt = 0;
        std::uniform_real_distribution<double> random_rotationa(rota_l, rota_h);
        std::uniform_real_distribution<double> random_rotationb(rotb_l, rotb_h);
        std::uniform_real_distribution<double> random_rotationc(rotc_l, rotc_h);
        std::uniform_real_distribution<double> random_coeffsa(coa_l, coa_h);
        std::uniform_real_distribution<double> random_coeffsb(cob_l, cob_h);
        if (para.getAlignedParaboloid().a() < 0 && para.getAlignedParaboloid().b() < 0)
        {
            std::uniform_real_distribution<double> random_coeffsa(coa_l, 0);
            std::uniform_real_distribution<double> random_coeffsb(cob_l, 0);
        }
        else
        {
            std::uniform_real_distribution<double> random_coeffsa(0, coa_h);
            std::uniform_real_distribution<double> random_coeffsb(0, cob_h);  
        }
        std::uniform_real_distribution<double> random_translationx(ox_l, ox_h);
        std::uniform_real_distribution<double> random_translationy(oy_l, oy_h);
        std::uniform_real_distribution<double> random_translationz(oz_l, oz_h);
        IRL::Paraboloid p;

        do
        {
            alpha = random_coeffsa(a_eng);
            beta = random_coeffsb(a_eng);
            frame = IRL::ReferenceFrame(IRL::Normal(1.0, 0.0, 0.0), IRL::Normal(0.0, 1.0, 0.0), IRL::Normal(0.0, 0.0, 1.0));
            datum = IRL::Pt(random_translationx(a_eng), random_translationy(a_eng), random_translationz(a_eng));
            std::array<double, 3> angles2 = {random_rotationa(a_eng), random_rotationb(a_eng), random_rotationc(a_eng)};

            IRL::UnitQuaternion x_rotation(angles2[0], frame[0]);
            IRL::UnitQuaternion y_rotation(angles2[1], frame[1]);
            IRL::UnitQuaternion z_rotation(angles2[2], frame[2]);
            frame = x_rotation * y_rotation * z_rotation * frame;
            p = IRL::Paraboloid(datum, frame, alpha, beta);
        } while (!(isParaboloidInCenterCell(p, liquid_volume_fraction)) || doParaboloidsIntersect(p, para, liquid_volume_fraction));

        return p;
    }

    torch::Tensor fractions::get_fractions(IRL::Paraboloid p, bool centroids)
    {
        DataMesh<double> liquid_volume_fraction(mesh);
        vector<double> f;

        for (int i = 0; i < a_number_of_cells; ++i)
        {
            for (int j = 0; j < a_number_of_cells; ++j)
            {
                for (int k = 0; k < a_number_of_cells; ++k)
                {
                    const auto volumes = getCellMoments<IRL::VolumeMoments>(p, liquid_volume_fraction, i, j, k);  
                    //const auto volumes_gas = getCellMomentsGas<IRL::VolumeMoments>(p, liquid_volume_fraction, i, j, k);  
                    auto& volume = volumes.volume();      
                    auto& centroid = volumes.centroid();  
                    //auto& volume_gas = volumes_gas.volume();      
                    //auto& centroid_gas = volumes_gas.centroid();   
                   
                    f.push_back(volume);
                    if (centroids)
                    {
                        if (volume < 10e-15/* || volume > 1-10e-15*/)
                        {
                            f.push_back(0);
                            f.push_back(0);
                            f.push_back(0);    
                            //f.push_back(0);
                            //f.push_back(0);
                            //f.push_back(0);    
                        }
                        else
                        {
                            f.push_back(centroid[0] - mesh.xm(i));
                            f.push_back(centroid[1] - mesh.ym(j));
                            f.push_back(centroid[2] - mesh.zm(k));   
                            //f.push_back(centroid_gas[0] - mesh.xm(i));
                            //f.push_back(centroid_gas[1] - mesh.ym(j));
                            //f.push_back(centroid_gas[2] - mesh.zm(k));    
                        }
                    }
                }
            }
        }
        return torch::tensor(f);  
    }

    torch::Tensor fractions::get_fractions_all(IRL::Paraboloid p)
    {
        DataMesh<double> liquid_volume_fraction(mesh);
        vector<double> f;

        for (int i = 0; i < a_number_of_cells; ++i)
        {
            for (int j = 0; j < a_number_of_cells; ++j)
            {
                for (int k = 0; k < a_number_of_cells; ++k)
                {
                    const auto volumes = getCellMoments<IRL::VolumeMoments>(p, liquid_volume_fraction, i, j, k);  
                    const auto volumes_gas = getCellMomentsGas<IRL::VolumeMoments>(p, liquid_volume_fraction, i, j, k);  
                    auto& volume = volumes.volume();      
                    auto& centroid = volumes.centroid();  
                    auto& volume_gas = volumes_gas.volume();      
                    auto& centroid_gas = volumes_gas.centroid();   
                   
                    f.push_back(volume);
                    if (volume < 10e-15 || volume > 1-10e-15)
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
                        f.push_back(centroid[0] - mesh.xm(i));
                        f.push_back(centroid[1] - mesh.ym(j));
                        f.push_back(centroid[2] - mesh.zm(k));   
                        f.push_back(centroid_gas[0] - mesh.xm(i));
                        f.push_back(centroid_gas[1] - mesh.ym(j));
                        f.push_back(centroid_gas[2] - mesh.zm(k));    
                    }
                }
            }
        }
        return torch::tensor(f);  
    }

    torch::Tensor fractions::get_fractions_gas(IRL::Paraboloid p, bool centroids)
    {
        DataMesh<double> liquid_volume_fraction(mesh);
        vector<double> f;

        for (int i = 0; i < a_number_of_cells; ++i)
        {
            for (int j = 0; j < a_number_of_cells; ++j)
            {
                for (int k = 0; k < a_number_of_cells; ++k)
                {
                    //const auto volumes = getCellMoments<IRL::VolumeMoments>(p, liquid_volume_fraction, i, j, k);  
                    const auto volumes_gas = getCellMomentsGas<IRL::VolumeMoments>(p, liquid_volume_fraction, i, j, k);  
                    //auto& volume = volumes.volume();      
                    //auto& centroid = volumes.centroid();  
                    auto& volume_gas = volumes_gas.volume();      
                    auto& centroid_gas = volumes_gas.centroid();   
                   
                    f.push_back(volume_gas);
                    if (centroids)
                    {
                        if (volume_gas < 10e-15/* || volume_gas > 1-10e-15*/)
                        {
                            f.push_back(0);
                            f.push_back(0);
                            f.push_back(0);    
                            //f.push_back(0);
                            //f.push_back(0);
                            //f.push_back(0);    
                        }
                        else
                        {
                            f.push_back(centroid_gas[0] - mesh.xm(i));
                            f.push_back(centroid_gas[1] - mesh.ym(j));
                            f.push_back(centroid_gas[2] - mesh.zm(k));    
                            //f.push_back(centroid[0] - mesh.xm(i));
                            //f.push_back(centroid[1] - mesh.ym(j));
                            //f.push_back(centroid[2] - mesh.zm(k));   
                        }
                    }
                }
            }
        }
        return torch::tensor(f);  
    }

    torch::Tensor fractions::get_fractions_gas_all(IRL::Paraboloid p)
    {
        DataMesh<double> liquid_volume_fraction(mesh);
        vector<double> f;

        for (int i = 0; i < a_number_of_cells; ++i)
        {
            for (int j = 0; j < a_number_of_cells; ++j)
            {
                for (int k = 0; k < a_number_of_cells; ++k)
                {
                    const auto volumes = getCellMoments<IRL::VolumeMoments>(p, liquid_volume_fraction, i, j, k);  
                    const auto volumes_gas = getCellMomentsGas<IRL::VolumeMoments>(p, liquid_volume_fraction, i, j, k);  
                    auto& volume = volumes.volume();      
                    auto& centroid = volumes.centroid();  
                    auto& volume_gas = volumes_gas.volume();      
                    auto& centroid_gas = volumes_gas.centroid();   
                   
                    f.push_back(volume_gas);
                    if (volume_gas < 10e-15 || volume_gas > 1-10e-15)
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
                        f.push_back(centroid_gas[0] - mesh.xm(i));
                        f.push_back(centroid_gas[1] - mesh.ym(j));
                        f.push_back(centroid_gas[2] - mesh.zm(k));    
                        f.push_back(centroid[0] - mesh.xm(i));
                        f.push_back(centroid[1] - mesh.ym(j));
                        f.push_back(centroid[2] - mesh.zm(k));   
                    }
                }
            }
        }
        return torch::tensor(f);  
    }

    torch::Tensor fractions::get_fractions(IRL::Plane p, bool centroids)
    {
        DataMesh<double> liquid_volume_fraction(mesh);
        vector<double> f;

        for (int i = 0; i < a_number_of_cells; ++i)
        {
            for (int j = 0; j < a_number_of_cells; ++j)
            {
                for (int k = 0; k < a_number_of_cells; ++k)
                {
                    const auto volumes = getCellMoments<IRL::VolumeMoments>(p, liquid_volume_fraction, i, j, k);  
                    auto& volume = volumes.volume();      
                    auto& centroid = volumes.centroid();   
                   
                    f.push_back(volume);
                    if (centroids)
                    {
                        f.push_back(centroid[0] - mesh.xm(i));
                        f.push_back(centroid[1] - mesh.ym(j));
                        f.push_back(centroid[2] - mesh.zm(k));    
                    }
                }
            }
        }
        return torch::tensor(f);  
    }

    torch::Tensor fractions::get_fractions_with_gradients(IRL::Paraboloid p, bool centroids)
    {
        DataMesh<double> liquid_volume_fraction(mesh);
        vector<double> f;
        for (int i = 0; i < 8; ++i)
        {
            gradients[i].clear();
        }

        for (int i = 0; i < a_number_of_cells; ++i)
        {
            for (int j = 0; j < a_number_of_cells; ++j)
            {
                for (int k = 0; k < a_number_of_cells; ++k)
                {
                    const auto volumesWithGradients = getCellMomentsWithGradients<IRL::VolumeMomentsWithGradient<IRL::ParaboloidGradientLocal>>(p, liquid_volume_fraction, i, j, k);    
                    auto& volumeWithGradients = volumesWithGradients.volume();      
                    auto& centroidWithGradients = volumesWithGradients.centroid();  

                    gradients[0].push_back(volumesWithGradients.volume_gradient().getGradTx());
                    if (centroids)
                    {
                        gradients[0].push_back(volumesWithGradients.centroid().getData()[0].getGradTx());
                        gradients[0].push_back(volumesWithGradients.centroid().getData()[1].getGradTx());
                        gradients[0].push_back(volumesWithGradients.centroid().getData()[2].getGradTx());
                    }

                    gradients[1].push_back(volumesWithGradients.volume_gradient().getGradTy()); 
                    if (centroids)
                    {
                        gradients[1].push_back(volumesWithGradients.centroid().getData()[0].getGradTy()); 
                        gradients[1].push_back(volumesWithGradients.centroid().getData()[1].getGradTy()); 
                        gradients[1].push_back(volumesWithGradients.centroid().getData()[2].getGradTy()); 
                    }

                    gradients[2].push_back(volumesWithGradients.volume_gradient().getGradTz()); 
                    if (centroids)
                    {
                        gradients[2].push_back(volumesWithGradients.centroid().getData()[0].getGradTz());
                        gradients[2].push_back(volumesWithGradients.centroid().getData()[1].getGradTz());
                        gradients[2].push_back(volumesWithGradients.centroid().getData()[2].getGradTz());
                    }

                    gradients[3].push_back(volumesWithGradients.volume_gradient().getGradRx()); 
                    if (centroids)
                    {
                        gradients[3].push_back(volumesWithGradients.centroid().getData()[0].getGradRx()); 
                        gradients[3].push_back(volumesWithGradients.centroid().getData()[1].getGradRx()); 
                        gradients[3].push_back(volumesWithGradients.centroid().getData()[2].getGradRx()); 
                    }

                    gradients[4].push_back(volumesWithGradients.volume_gradient().getGradRy()); 
                    if (centroids)
                    {
                        gradients[4].push_back(volumesWithGradients.centroid().getData()[0].getGradRy()); 
                        gradients[4].push_back(volumesWithGradients.centroid().getData()[1].getGradRy()); 
                        gradients[4].push_back(volumesWithGradients.centroid().getData()[2].getGradRy()); 
                    }

                    gradients[5].push_back(volumesWithGradients.volume_gradient().getGradRz()); 
                    if (centroids)
                    {
                        gradients[5].push_back(volumesWithGradients.centroid().getData()[0].getGradRz()); 
                        gradients[5].push_back(volumesWithGradients.centroid().getData()[1].getGradRz()); 
                        gradients[5].push_back(volumesWithGradients.centroid().getData()[2].getGradRz()); 
                    }

                    gradients[6].push_back(volumesWithGradients.volume_gradient().getGradA()); 
                    if (centroids)
                    {
                        gradients[6].push_back(volumesWithGradients.centroid().getData()[0].getGradA()); 
                        gradients[6].push_back(volumesWithGradients.centroid().getData()[1].getGradA()); 
                        gradients[6].push_back(volumesWithGradients.centroid().getData()[2].getGradA()); 
                    }

                    gradients[7].push_back(volumesWithGradients.volume_gradient().getGradB()); 
                    if (centroids)
                    {
                        gradients[7].push_back(volumesWithGradients.centroid().getData()[0].getGradB()); 
                        gradients[7].push_back(volumesWithGradients.centroid().getData()[1].getGradB()); 
                        gradients[7].push_back(volumesWithGradients.centroid().getData()[2].getGradB()); 
                    }

                    f.push_back(volumeWithGradients);
                    if (centroids)
                    {
                        f.push_back(centroidWithGradients[0]);
                        f.push_back(centroidWithGradients[1]);
                        f.push_back(centroidWithGradients[2]);    
                    }
                }
            }
        }
        return torch::tensor(f);  
    }

    torch::Tensor fractions::get_gradients(int index)
    {
        return torch::tensor(gradients[index]);
    }

    template <class MomentType, class SurfaceType>
    IRL::AddSurfaceOutput<MomentType, SurfaceType> fractions::getCellMomentsAndSurface(const IRL::Paraboloid& a_interface,
    const DataMesh<double>& a_liquid_volume_fraction, int x_loc, int y_loc, int z_loc) 
    {
        const Mesh& mesh = a_liquid_volume_fraction.getMesh();
        const int i(x_loc), j(y_loc), k(z_loc);
        auto cell = IRL::RectangularCuboid::fromBoundingPts(
            IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)),
            IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));
        return IRL::getVolumeMoments<IRL::AddSurfaceOutput<MomentType, SurfaceType>, IRL::HalfEdgeCutting>(cell, a_interface);
    }

    template <class MomentType>
    MomentType fractions::getCellMoments(const IRL::Paraboloid& a_interface,
    const DataMesh<double>& a_liquid_volume_fraction, int x_loc, int y_loc, int z_loc)
    {
        const Mesh& mesh = a_liquid_volume_fraction.getMesh();
        const int i(x_loc), j(y_loc), k(z_loc);
        auto cell = IRL::RectangularCuboid::fromBoundingPts(
            IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)),
            IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));
        auto moments = IRL::getVolumeMoments<MomentType, IRL::HalfEdgeCutting>(cell, a_interface);
        if (moments.volume() > 10e-15)
        {
            moments.centroid()[0] = moments.centroid()[0] / moments.volume();
            moments.centroid()[1] = moments.centroid()[1] / moments.volume();
            moments.centroid()[2] = moments.centroid()[2] / moments.volume();
        }
        return moments;
    }

    template <class MomentType>
    MomentType fractions::getCellMomentsGas(const IRL::Paraboloid& a_interface,
    const DataMesh<double>& a_liquid_volume_fraction, int x_loc, int y_loc, int z_loc)
    {
        const Mesh& mesh = a_liquid_volume_fraction.getMesh();
        const int i(x_loc), j(y_loc), k(z_loc);
        auto cell = IRL::RectangularCuboid::fromBoundingPts(
            IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)),
            IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));
        auto moments = IRL::getVolumeMoments<IRL::SeparatedMoments<MomentType>>(cell, a_interface);
        if (moments[1].volume() > 10e-15)
        {
            moments[1].centroid()[0] = moments[1].centroid()[0] / moments[1].volume();
            moments[1].centroid()[1] = moments[1].centroid()[1] / moments[1].volume();
            moments[1].centroid()[2] = moments[1].centroid()[2] / moments[1].volume();
        }
        return moments[1];
    }

    template <class MomentType>
    MomentType fractions::getCellMoments(const IRL::Plane& a_interface,
    const DataMesh<double>& a_liquid_volume_fraction, int x_loc, int y_loc, int z_loc)
    {
        const Mesh& mesh = a_liquid_volume_fraction.getMesh();
        const int i(x_loc), j(y_loc), k(z_loc);
        auto cell = IRL::RectangularCuboid::fromBoundingPts(
            IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)),
            IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));
        auto a = IRL::PlanarSeparator::fromOnePlane(a_interface);
        const auto moments = IRL::getVolumeMoments<MomentType>(cell, a);
        return moments;
    }

    template <class MomentType, class SurfaceType>
    IRL::AddSurfaceOutput<MomentType, SurfaceType> fractions::getCellMomentsAndSurfaceWithGradients(const IRL::Paraboloid& a_interface,
    const DataMesh<double>& a_liquid_volume_fraction, int x_loc, int y_loc, int z_loc) 
    {
        const Mesh& mesh = a_liquid_volume_fraction.getMesh();
        const int i(x_loc), j(y_loc), k(z_loc);
        auto cell = IRL::RectangularCuboid::fromBoundingPts(
            IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)),
            IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));
        using MyPtType = IRL::PtWithGradient<IRL::ParaboloidGradientLocal>;
        auto cube = IRL::StoredRectangularCuboid<MyPtType>::fromOtherPolytope(cell);
        return IRL::getVolumeMoments<IRL::AddSurfaceOutput<MomentType, SurfaceType>, IRL::HalfEdgeCutting>(cube, a_interface);
    }

    template <class MomentType>
    MomentType fractions::getCellMomentsWithGradients(const IRL::Paraboloid& a_interface,
    const DataMesh<double>& a_liquid_volume_fraction, int x_loc, int y_loc, int z_loc)
    {
        const Mesh& mesh = a_liquid_volume_fraction.getMesh();
        const int i(x_loc), j(y_loc), k(z_loc);
        auto cell = IRL::RectangularCuboid::fromBoundingPts(
            IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)),
            IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));
        using MyPtType = IRL::PtWithGradient<IRL::ParaboloidGradientLocal>;
        auto cube = IRL::StoredRectangularCuboid<MyPtType>::fromOtherPolytope(cell);
        const auto moments = IRL::getVolumeMoments<MomentType, IRL::HalfEdgeCutting>(cube, a_interface);

        return moments;
    }

    Mesh fractions::initializeMesh(const int a_number_of_cells) 
    {
        constexpr const int a_number_of_ghost_cells = 1;
        Mesh mesh(a_number_of_cells, a_number_of_cells, a_number_of_cells, a_number_of_ghost_cells);
        auto nx = mesh.getNx();
        auto ny = mesh.getNy();
        auto nz = mesh.getNz();
        IRL::Pt lower_domain(-0.5 * nx, -0.5 * ny, -0.5 * nz);
        IRL::Pt upper_domain(0.5 * nx, 0.5 * ny, 0.5 * nz);
        mesh.setCellBoundaries(lower_domain, upper_domain);
        return mesh;
    }

    bool fractions::isParaboloidInCenterCell(const IRL::Paraboloid& a_interface, const DataMesh<double>& a_liquid_volume_fraction) 
    {
        const Mesh& mesh = a_liquid_volume_fraction.getMesh();
        const int i(mesh.ic()), j(mesh.jc()), k(mesh.kc());
        auto cell = IRL::RectangularCuboid::fromBoundingPts(
            IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)),
            IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));
        const double volume_fraction = IRL::getVolumeMoments<IRL::Volume, IRL::HalfEdgeCutting>(cell, a_interface);
        return volume_fraction < IRL::global_constants::VF_HIGH && volume_fraction > IRL::global_constants::VF_LOW;
    }

    bool fractions::areParaboloidsInSameCell(IRL::Paraboloid& p, IRL::Paraboloid& p1, const DataMesh<double>& a_liquid_volume_fraction)
    {
        const Mesh& mesh = a_liquid_volume_fraction.getMesh();
        bool intersect = false;
        for (int i = 0; i < a_number_of_cells; ++i)
        {
            for (int j = 0; j < a_number_of_cells; ++j)
            {
                for (int k = 0; k < a_number_of_cells; ++k)
                {
                    auto cell = IRL::RectangularCuboid::fromBoundingPts(
                        IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)),
                        IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));
                    const double volume_fraction = IRL::getVolumeMoments<IRL::Volume, IRL::HalfEdgeCutting>(cell, p);
                    const double volume_fraction1 = IRL::getVolumeMoments<IRL::Volume, IRL::HalfEdgeCutting>(cell, p1);
                    if ((volume_fraction < IRL::global_constants::VF_HIGH && volume_fraction > IRL::global_constants::VF_LOW) && (volume_fraction1 < IRL::global_constants::VF_HIGH && volume_fraction1 > IRL::global_constants::VF_LOW))
                    {
                        intersect = true;
                    }
                }
            }
        }
        return intersect;
    }

    bool fractions::doParaboloidsIntersect(IRL::Paraboloid& p1, IRL::Paraboloid& p2, const DataMesh<double>& a_liquid_volume_fraction)
    {
        const Mesh& mesh = a_liquid_volume_fraction.getMesh();
        bool result = false;

        /*double a1 = p1.getAlignedParaboloid().a();
        double b1 = p1.getAlignedParaboloid().b();
        double a2 = p2.getAlignedParaboloid().a();
        double b2 = p2.getAlignedParaboloid().b();

        double r11 = p1.getReferenceFrame()[2][0];
        double r21 = p1.getReferenceFrame()[2][1];
        double r31 = p1.getReferenceFrame()[2][2];
        double x1 = p1.getDatum()[0];
        double y1 = p1.getDatum()[1];
        double z1 = p1.getDatum()[2];

        double r12 = p2.getReferenceFrame()[2][0];
        double r22 = p2.getReferenceFrame()[2][1];
        double r32 = p2.getReferenceFrame()[2][2];
        double x2 = p2.getDatum()[0];
        double y2 = p2.getDatum()[1];
        double z2 = p2.getDatum()[2];*/
        for (int i = 0; i < a_number_of_cells; ++i)
        {
            for (int j = 0; j < a_number_of_cells; ++j)
            {
                for (int k = 0; k < a_number_of_cells; ++k)
                {
                    auto cell = IRL::RectangularCuboid::fromBoundingPts(
                        IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)),
                        IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));
                    const auto first_moments_and_surface1 = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParametrizedSurfaceOutput>, IRL::HalfEdgeCutting>(cell, p1);
                    const auto first_moments_and_surface2 = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParametrizedSurfaceOutput>, IRL::HalfEdgeCutting>(cell, p2);
                    auto surface1 = first_moments_and_surface1.getSurface();
                    auto surface2 = first_moments_and_surface2.getSurface();
                    const double length_scale = 0.05;
                    IRL::TriangulatedSurfaceOutput triangulated_surface1 = first_moments_and_surface1.getSurface().triangulate(length_scale);
                    IRL::TriangulatedSurfaceOutput triangulated_surface2 = first_moments_and_surface2.getSurface().triangulate(length_scale);
                    auto l1 = triangulated_surface1.getVertexList();
                    auto l2 = triangulated_surface2.getVertexList();
                    for (int n = 0; n < l1.size(); ++n)
                    {
                        for (int m = 0; m < l2.size(); ++m)
                        {
                            if (abs(l1[n][0] - l2[m][0]) < 0.01 && abs(l1[n][1] - l2[m][1]) < 0.01 && abs(l1[n][2] - l2[m][2]) < 0.01)
                            {
                                result = true;
                                break;
                            }
                        }
                    }
                    if (result)
                    {
                        break;
                    }
                }
            }
        }
        /*Eigen::MatrixXd A1(l1.size(),8);
        Eigen::VectorXd b1(l1.size());
        Eigen::MatrixXd A2(l2.size(),8);
        Eigen::VectorXd b2(l2.size());
        for (int n = 0; n < 10; ++n)
        {
            if (n < l1.size())
            {
                A1(n,0) = l1[n][0]*l1[n][0];
                A1(n,1) = l1[n][1]*l1[n][1];
                A1(n,2) = l1[n][0]*l1[n][1];
                A1(n,3) = l1[n][1]*l1[n][2];
                A1(n,4) = l1[n][0]*l1[n][2];
                A1(n,5) = l1[n][0];
                A1(n,6) = l1[n][1];
                A1(n,7) = 1;
                b1(n) = l1[n][2];
            }
            if (n < l2.size())
            {
                A2(n,0) = l2[n][0]*l2[n][0];
                A2(n,1) = l2[n][1]*l2[n][1];
                A2(n,2) = l2[n][0]*l2[n][1];
                A2(n,3) = l2[n][1]*l2[n][2];
                A2(n,4) = l2[n][0]*l2[n][2];
                A2(n,5) = l2[n][0];
                A2(n,6) = l2[n][1];
                A2(n,7) = 1;
                b2(n) = l2[n][2]; 
            }
        }
        auto sol1 = A1.householderQr().solve(b1);
        auto sol2 = A2.householderQr().solve(b2);
        std::cout << sol1[0] << std::endl;
        std::cout << sol1[1] << std::endl;
        std::cout << sol1[2] << std::endl;
        std::cout << sol1[3] << std::endl;
        std::cout << sol1[4] << std::endl;
        std::cout << sol1[5] << std::endl;
        std::cout << sol1[6] << std::endl;
        std::cout << sol1[7] << std::endl;
        std::cout << sol2[0] << std::endl;
        std::cout << sol2[1] << std::endl;
        std::cout << sol2[2] << std::endl;
        std::cout << sol2[3] << std::endl;
        std::cout << sol2[4] << std::endl;
        std::cout << sol2[5] << std::endl;
        std::cout << sol2[6] << std::endl;
        std::cout << sol2[7] << std::endl;
        double a = sol1[0] - sol2[0];
        double b = sol1[1] - sol2[1];
        double c = sol1[2] - sol2[2];
        double d = sol1[3] - sol2[3];
        double e = sol1[4] - sol2[4];
        double f = sol1[5] - sol2[5];
        double g = sol1[6] - sol2[6];
        double h = sol1[7] - sol2[7];
        for (double y = -1.5; y <= 1.5; y=y+0.001)
        {
            for (double z = -1.5; z <= 1.5; z=z+0.001)
            {
                if (pow((c*y+e*z+f),2.0)-4*a*(b*y*y+d*y*z+g*y+h) < 0)
                {
                    result = true;
                    break;
                }
            }
        }
        if (!result)
        {
            for (double x = -1.5; x <= 1.5; x=x+0.001)
            {
                for (double z = -1.5; z <= 1.5; z=z+0.001)
                {
                    if (pow((c*x+d*z+g),2.0)-4*b*(a*z*z+e*x*z+f*x+h) < 0)
                    {
                        result = true;
                        break;
                    }
                }
            }    
        }*/

        /*if (a1 > 0 && x2 > x1)
        {
            result = true;
        }
        else if (a1 < 0 && x2 < x1)
        {
            result = true;
        }
        else if (b1 > 0 && y2 > y1)
        {
            result = true;
        }
        else if (b1 < 0 && y2 < y1)
        {
            result = true;
        }
        else if (c1 > 0 && z2 > z1)
        {
            result = true;
        }
        else if (c1 < 0 && z2 < z1)
        {
            result = true;
        }*/

        return result;
    }
}

#endif