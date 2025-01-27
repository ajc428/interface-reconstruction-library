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
        srand((unsigned) time(NULL));
        a_number_of_cells = a;
        mesh = initializeMesh(a_number_of_cells);
        IRL::setVolumeFractionBounds(1.0e-14);
        std::cout.precision(15);
        distribution1 = std::normal_distribution<double>(0.0,0.3);
        distribution2 = std::normal_distribution<double>(0.0,0.3);
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
            alpha = random_coeffsa(a_eng);//distribution1(generator);
            beta = random_coeffsb(a_eng);//distribution2(generator);
            frame = IRL::ReferenceFrame(IRL::Normal(1.0, 0.0, 0.0), IRL::Normal(0.0, 1.0, 0.0), IRL::Normal(0.0, 0.0, 1.0));
            datum = IRL::Pt(random_translationx(a_eng), random_translationy(a_eng), random_translationz(a_eng));
            angles = {random_rotationa(a_eng), random_rotationb(a_eng), random_rotationc(a_eng)};

            IRL::UnitQuaternion x_rotation(angles[0], frame[0]);
            IRL::UnitQuaternion y_rotation(angles[1], frame[1]);
            IRL::UnitQuaternion z_rotation(angles[2], frame[2]);
            frame = x_rotation * y_rotation * z_rotation * frame;
            p = IRL::Paraboloid(datum, frame, alpha, beta);
        } while (!(isParaboloidInCenterCell(p, liquid_volume_fraction))/* || (datum[0] > -0.5 && datum[0] < 0.5 && datum[1] > -0.5 && datum[1] < 0.5 && datum[2] > -0.5 && datum[2] < 0.5)*/);

        return p;
    }

    IRL::PlanarSeparator fractions::new_random_plane(double rota1_l, double rota1_h, double rotb1_l, double rotb1_h, double d1_l, double d1_h)
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
        std::uniform_real_distribution<double> randoma1(rota1_l, rota1_h);
        std::uniform_real_distribution<double> randomb1(rotb1_l, rotb1_h);
        std::uniform_real_distribution<double> randomd1(d1_l, d1_h);
        double s = rand() % 2;
        IRL::Plane p1;
        IRL::PlanarSeparator p;

        do
        {
            a1 = randoma1(a_eng);
            b1 = randomb1(a_eng);
            double d1 = randomd1(a_eng);

            IRL::Normal n1 = IRL::Normal(0,0,0);
            n1[0] = cos(b1) * cos(a1);
            n1[1] = cos(b1) * sin(a1);
            n1[2] = sin(b1);

            p1 = IRL::Plane(n1,d1);
            p = IRL::PlanarSeparator::fromOnePlane(p1);
        } while (!(isPlaneInCenterCell(p[0], liquid_volume_fraction)));

        return p;
    }

    IRL::PlanarSeparator fractions::new_random_R2P(double rota1_l, double rota1_h, double rotb1_l, double rotb1_h, double rota2_l, double rota2_h, double rotb2_l, double rotb2_h, double d1_l, double d1_h, double d2_l, double d2_h, bool inter, bool in)
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
        std::uniform_real_distribution<double> randoma1(rota1_l, rota1_h);
        std::uniform_real_distribution<double> randomb1(rotb1_l, rotb1_h);
        std::uniform_real_distribution<double> randoma2(rota2_l, rota2_h);
        std::uniform_real_distribution<double> randomb2(rotb2_l, rotb2_h);
        std::uniform_real_distribution<double> randomd1(d1_l, d1_h);
        //std::uniform_real_distribution<double> randomd2(d2_l, d2_h);

        std::uniform_real_distribution<double> randomNx1(-1, 1);
        std::uniform_real_distribution<double> randomNy1(-1, 1);
        std::uniform_real_distribution<double> randomNz1(-1, 1);
        std::uniform_real_distribution<double> randomNx2(-1, 1);
        std::uniform_real_distribution<double> randomNy2(-1, 1);
        std::uniform_real_distribution<double> randomNz2(-1, 1);
        double s = 1;//rand() % 2;
        if (s < 0.5)
        {
            //s = -1.0;
        }
        IRL::Plane p1;
        IRL::Plane p2;
        IRL::PlanarSeparator p;
        bool intersect = false;
        bool same = false;
        double d1;
        double d2;

        a1 = randoma1(a_eng);
        b1 = randomb1(a_eng);

        double nx1 = randomNx1(a_eng);
        double ny1 = randomNy1(a_eng);
        double nz1 = randomNz1(a_eng);

        IRL::Normal n1 = IRL::Normal(0,0,0);
        n1[0] = 0.0000000001;//cos(b1) * cos(a1);
        n1[1] = 1;//cos(b1) * sin(a1);
        n1[2] = 0.0000000001;//sin(b1);
        //if (inter)
        {
            // n1[0] = cos(b1) * cos(a1);
            // n1[1] = cos(b1) * sin(a1);
            // n1[2] = sin(b1);
            n1[0] = nx1;
            n1[1] = ny1;
            n1[2] = nz1;
        }
        do
        {
            double nx2 = randomNx2(a_eng);
            double ny2 = randomNy2(a_eng);
            double nz2 = randomNz2(a_eng);
            a2 = randoma2(a_eng);
            b2 = randomb2(a_eng);
            d1 = randomd1(a_eng);
            std::uniform_real_distribution<double> randomd2(d2_l, d2_h);
            d2 = randomd2(a_eng);
            IRL::Normal n2 = -n1;
            if(inter)
            {
                // n2[0] = cos(b2) * cos(a2);
                // n2[1] = cos(b2) * sin(a2);
                // n2[2] = sin(b2);
                n2[0] = nx2;
                n2[1] = ny2;
                n2[2] = nz2;
            }
            n1.normalize();
            n2.normalize();
            p1 = IRL::Plane(n1,d1);
            p2 = IRL::Plane(n2,d2);
            p = IRL::PlanarSeparator::fromTwoPlanes(p1,p2,s);
            if (!inter)
            {
                intersect = doPlanesIntersect(p,liquid_volume_fraction, 1.5);////////////////////////////////////////////////////IMPORTANT
            }
            else
            {
                intersect = !doPlanesIntersect(p,liquid_volume_fraction, 0.5);
            }
            if (in)
            {
                same = !arePlanesInSameCenterCell(p);
            }
            else
            {
                same = arePlanesInSameCenterCell(p);
            }
            double dot = n1[0]*n2[0] + n1[1]*n2[1] + n1[2]*n2[2];
            if (!inter)
            {
                if (dot >= -0.98)
                //if (dot <= -0.5)
                {
                    intersect = true;
                }
            }
            else
            {
                if (dot >= -0.98)
                //if (dot <= -0.5)
                {
                    intersect = true;
                }
            }
        } while (!(isPlaneInCenterCell(p[0], liquid_volume_fraction) && arePlanesInCenterCell(p, liquid_volume_fraction)) || intersect || same);
        //std::cout << a1 << " " << b1 << " " << a2 << " " << b2 << std::endl;

        return p;
    }

    IRL::PlanarSeparator fractions::new_step_R2P(bool inter, int current, int total)
    {
        DataMesh<double> liquid_volume_fraction(mesh);
        std::random_device rd;  
        std::mt19937_64 a_eng(rd());
        std::uniform_real_distribution<double> randomd1(0, 2);

        IRL::Plane p1;
        IRL::Plane p2;
        IRL::PlanarSeparator p;
        bool intersect = false;
        bool same = false;
        double d1;
        double d2;

        double nx1 = 0;
        double ny1 = 0;
        double nz1 = 1;

        IRL::Normal n1 = IRL::Normal(0,0,0);
        n1[0] = nx1;
        n1[1] = ny1;
        n1[2] = nz1;
        ++current;
        double nx2 = -nx1 + (double(current)/double(total))*0.45;
        double ny2 = -ny1;
        double nz2 = -nz1;
        IRL::Normal n2 = IRL::Normal(nx2,ny2,nz2);
        n1.normalize();
        n2.normalize();
        do
        {
            d1 = randomd1(a_eng);
            std::uniform_real_distribution<double> randomd2(0, 10);
            d2 = randomd2(a_eng);
            p1 = IRL::Plane(n1,d1);
            p2 = IRL::Plane(n2,d2);
            p = IRL::PlanarSeparator::fromTwoPlanes(p1,p2,1);
            if (!inter)
            {
                intersect = doPlanesIntersect(p,liquid_volume_fraction, 1.5);
            }
            else
            {
                intersect = !doPlanesIntersect(p,liquid_volume_fraction, 0.5);
            }
            same = !arePlanesInSameCenterCell(p);
            double dot = n1[0]*n2[0] + n1[1]*n2[1] + n1[2]*n2[2];
        } while (!(isPlaneInCenterCell(p[0], liquid_volume_fraction) && arePlanesInCenterCell(p, liquid_volume_fraction)) || intersect || same);

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
        //std::uniform_real_distribution<double> random_rotationa(rota_l, rota_h);
        //std::uniform_real_distribution<double> random_rotationb(rotb_l, rotb_h);
        //std::uniform_real_distribution<double> random_rotationc(rotc_l, rotc_h);
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
            frame = IRL::ReferenceFrame(-para.getReferenceFrame()[0], para.getReferenceFrame()[1], -para.getReferenceFrame()[2]);
            datum = IRL::Pt(random_translationx(a_eng), random_translationy(a_eng), random_translationz(a_eng));
            //std::array<double, 3> angles2 = {random_rotationa(a_eng), random_rotationb(a_eng), random_rotationc(a_eng)};

            //IRL::UnitQuaternion x_rotation(angles2[0], frame[0]);
            //IRL::UnitQuaternion y_rotation(angles2[1], frame[1]);
            //IRL::UnitQuaternion z_rotation(angles2[2], frame[2]);
            //frame = x_rotation * y_rotation * z_rotation * frame;
            p = IRL::Paraboloid(datum, frame, alpha, beta);
        } while (areParaboloidsInSameCell(p, para, liquid_volume_fraction));

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
        //std::uniform_real_distribution<double> random_rotationa(rota_l, rota_h);
        //std::uniform_real_distribution<double> random_rotationb(rotb_l, rotb_h);
        //std::uniform_real_distribution<double> random_rotationc(rotc_l, rotc_h);
        std::uniform_real_distribution<double> random_coeffsa(coa_l, coa_h);
        std::uniform_real_distribution<double> random_coeffsb(cob_l, cob_h);
        std::uniform_real_distribution<double> random_translationx(0.001, 0.5);
        std::uniform_real_distribution<double> random_translationy(-1.5, 1.5);
        std::uniform_real_distribution<double> random_translationz(-1.5, 1.5);
        IRL::Paraboloid p;

        do
        {
            double x = random_translationx(a_eng);
            alpha = -para.getAlignedParaboloid().a();//random_coeffsa(a_eng);
            beta = -para.getAlignedParaboloid().b();//random_coeffsb(a_eng);
            frame = IRL::ReferenceFrame(-para.getReferenceFrame()[0], para.getReferenceFrame()[1], -para.getReferenceFrame()[2]);
            double dx = para.getReferenceFrame()[2][0] * x;
            double dy = para.getReferenceFrame()[2][1] * x;
            double dz = para.getReferenceFrame()[2][2] * x;
            datum = IRL::Pt(para.getDatum()[0] + dx, para.getDatum()[1] + dy, para.getDatum()[2] + dz);//IRL::Pt(random_translationx(a_eng), random_translationy(a_eng), random_translationz(a_eng));
            //std::array<double, 3> angles2 = {random_rotationa(a_eng), random_rotationb(a_eng), random_rotationc(a_eng)};

            //IRL::UnitQuaternion x_rotation(angles2[0], frame[0]);
            //IRL::UnitQuaternion y_rotation(angles2[1], frame[1]);
            //IRL::UnitQuaternion z_rotation(angles2[2], frame[2]);
            //frame = x_rotation * y_rotation * z_rotation * frame;
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

    torch::Tensor fractions::get_fractions(IRL::Plane p1, bool centroids)
    {
        IRL::PlanarSeparator p = IRL::PlanarSeparator::fromOnePlane(p1);
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
                    }
                    else
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

    torch::Tensor fractions::get_fractions_all(IRL::PlanarSeparator p)
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

    torch::Tensor fractions::get_fractions_only(IRL::PlanarSeparator p)
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
                }
            }
        }
        return torch::tensor(f);  
    }

    torch::Tensor fractions::get_barycenters(IRL::PlanarSeparator p)
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
                   
                    //f.push_back(volume);
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

    torch::Tensor fractions::get_fractions_gas_all(IRL::PlanarSeparator p)
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

    torch::Tensor fractions::get_barycenters_gas(IRL::PlanarSeparator p)
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
                   
                    //f.push_back(volume_gas);
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

    torch::Tensor fractions::get_fractions_with_gradients(IRL::Paraboloid p, bool centroids)
    {
        // DataMesh<double> liquid_volume_fraction(mesh);
        vector<double> f;
        // for (int i = 0; i < 8; ++i)
        // {
        //     gradients[i].clear();
        // }

        // for (int i = 0; i < a_number_of_cells; ++i)
        // {
        //     for (int j = 0; j < a_number_of_cells; ++j)
        //     {
        //         for (int k = 0; k < a_number_of_cells; ++k)
        //         {
        //             const auto volumesWithGradients = getCellMomentsWithGradients<IRL::VolumeMomentsWithGradient<IRL::ParaboloidGradientLocal>>(p, liquid_volume_fraction, i, j, k);    
        //             auto& volumeWithGradients = volumesWithGradients.volume();      
        //             auto& centroidWithGradients = volumesWithGradients.centroid();  

        //             gradients[0].push_back(volumesWithGradients.volume_gradient().getGradTx());
        //             if (centroids)
        //             {
        //                 gradients[0].push_back(volumesWithGradients.centroid().getData()[0].getGradTx());
        //                 gradients[0].push_back(volumesWithGradients.centroid().getData()[1].getGradTx());
        //                 gradients[0].push_back(volumesWithGradients.centroid().getData()[2].getGradTx());
        //             }

        //             gradients[1].push_back(volumesWithGradients.volume_gradient().getGradTy()); 
        //             if (centroids)
        //             {
        //                 gradients[1].push_back(volumesWithGradients.centroid().getData()[0].getGradTy()); 
        //                 gradients[1].push_back(volumesWithGradients.centroid().getData()[1].getGradTy()); 
        //                 gradients[1].push_back(volumesWithGradients.centroid().getData()[2].getGradTy()); 
        //             }

        //             gradients[2].push_back(volumesWithGradients.volume_gradient().getGradTz()); 
        //             if (centroids)
        //             {
        //                 gradients[2].push_back(volumesWithGradients.centroid().getData()[0].getGradTz());
        //                 gradients[2].push_back(volumesWithGradients.centroid().getData()[1].getGradTz());
        //                 gradients[2].push_back(volumesWithGradients.centroid().getData()[2].getGradTz());
        //             }

        //             gradients[3].push_back(volumesWithGradients.volume_gradient().getGradRx()); 
        //             if (centroids)
        //             {
        //                 gradients[3].push_back(volumesWithGradients.centroid().getData()[0].getGradRx()); 
        //                 gradients[3].push_back(volumesWithGradients.centroid().getData()[1].getGradRx()); 
        //                 gradients[3].push_back(volumesWithGradients.centroid().getData()[2].getGradRx()); 
        //             }

        //             gradients[4].push_back(volumesWithGradients.volume_gradient().getGradRy()); 
        //             if (centroids)
        //             {
        //                 gradients[4].push_back(volumesWithGradients.centroid().getData()[0].getGradRy()); 
        //                 gradients[4].push_back(volumesWithGradients.centroid().getData()[1].getGradRy()); 
        //                 gradients[4].push_back(volumesWithGradients.centroid().getData()[2].getGradRy()); 
        //             }

        //             gradients[5].push_back(volumesWithGradients.volume_gradient().getGradRz()); 
        //             if (centroids)
        //             {
        //                 gradients[5].push_back(volumesWithGradients.centroid().getData()[0].getGradRz()); 
        //                 gradients[5].push_back(volumesWithGradients.centroid().getData()[1].getGradRz()); 
        //                 gradients[5].push_back(volumesWithGradients.centroid().getData()[2].getGradRz()); 
        //             }

        //             gradients[6].push_back(volumesWithGradients.volume_gradient().getGradA()); 
        //             if (centroids)
        //             {
        //                 gradients[6].push_back(volumesWithGradients.centroid().getData()[0].getGradA()); 
        //                 gradients[6].push_back(volumesWithGradients.centroid().getData()[1].getGradA()); 
        //                 gradients[6].push_back(volumesWithGradients.centroid().getData()[2].getGradA()); 
        //             }

        //             gradients[7].push_back(volumesWithGradients.volume_gradient().getGradB()); 
        //             if (centroids)
        //             {
        //                 gradients[7].push_back(volumesWithGradients.centroid().getData()[0].getGradB()); 
        //                 gradients[7].push_back(volumesWithGradients.centroid().getData()[1].getGradB()); 
        //                 gradients[7].push_back(volumesWithGradients.centroid().getData()[2].getGradB()); 
        //             }

        //             f.push_back(volumeWithGradients);
        //             if (centroids)
        //             {
        //                 f.push_back(centroidWithGradients[0]);
        //                 f.push_back(centroidWithGradients[1]);
        //                 f.push_back(centroidWithGradients[2]);    
        //             }
        //         }
        //     }
        // }
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
    MomentType fractions::getCellMoments(const IRL::PlanarSeparator& a_interface,
    const DataMesh<double>& a_liquid_volume_fraction, int x_loc, int y_loc, int z_loc)
    {
        const Mesh& mesh = a_liquid_volume_fraction.getMesh();
        const int i(x_loc), j(y_loc), k(z_loc);
        auto cell = IRL::RectangularCuboid::fromBoundingPts(
            IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)),
            IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));
        auto moments = IRL::getVolumeMoments<MomentType>(cell, a_interface);
        if (moments.volume() > 10e-15)
        {
            moments.centroid()[0] = moments.centroid()[0] / moments.volume();
            moments.centroid()[1] = moments.centroid()[1] / moments.volume();
            moments.centroid()[2] = moments.centroid()[2] / moments.volume();
        }
        return moments;
    }

    template <class MomentType>
    MomentType fractions::getCellMomentsGas(const IRL::PlanarSeparator& a_interface,
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

    template <class MomentType, class SurfaceType>
    IRL::AddSurfaceOutput<MomentType, SurfaceType> fractions::getCellMomentsAndSurfaceWithGradients(const IRL::Paraboloid& a_interface,
    const DataMesh<double>& a_liquid_volume_fraction, int x_loc, int y_loc, int z_loc) 
    {
        const Mesh& mesh = a_liquid_volume_fraction.getMesh();
        const int i(x_loc), j(y_loc), k(z_loc);
        auto cell = IRL::RectangularCuboid::fromBoundingPts(
            IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)),
            IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));
        //using MyPtType = IRL::PtWithGradient<IRL::ParaboloidGradientLocal>;
        //auto cube = IRL::StoredRectangularCuboid<MyPtType>::fromOtherPolytope(cell);
        return IRL::getVolumeMoments<IRL::AddSurfaceOutput<MomentType, SurfaceType>, IRL::HalfEdgeCutting>(/*cube*/cell, a_interface);
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
        //using MyPtType = IRL::PtWithGradient<IRL::ParaboloidGradientLocal>;
        //auto cube = IRL::StoredRectangularCuboid<MyPtType>::fromOtherPolytope(cell);
        const auto moments = IRL::getVolumeMoments<MomentType, IRL::HalfEdgeCutting>(/*cube*/cell, a_interface);

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

    bool fractions::isPlaneInCenterCell(const IRL::Plane& a_interface, const DataMesh<double>& a_liquid_volume_fraction) 
    {
        const Mesh& mesh = a_liquid_volume_fraction.getMesh();
        const int i(mesh.ic()), j(mesh.jc()), k(mesh.kc());
        auto cell = IRL::RectangularCuboid::fromBoundingPts(
            IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)),
            IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));
        IRL::PlanarSeparator p = IRL::PlanarSeparator::fromOnePlane(a_interface);
        const double volume_fraction = IRL::getVolumeMoments<IRL::Volume, IRL::HalfEdgeCutting>(cell, p);
        return volume_fraction < IRL::global_constants::VF_HIGH && volume_fraction > IRL::global_constants::VF_LOW;
    }

    bool fractions::arePlanesInCenterCell(const IRL::PlanarSeparator& a_interface, const DataMesh<double>& a_liquid_volume_fraction) 
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

    bool fractions::arePlanesInSameCenterCell(IRL::PlanarSeparator& p)
    {
        bool intersect = false;
        auto cell = IRL::RectangularCuboid::fromBoundingPts(
            IRL::Pt(-0.5, -0.5, -0.5),
            IRL::Pt(0.5, 0.5, 0.5));
        IRL::PlanarSeparator p1 = IRL::PlanarSeparator::fromOnePlane(p[0]);
        IRL::PlanarSeparator p2 = IRL::PlanarSeparator::fromOnePlane(p[1]);
        const double volume_fraction = IRL::getVolumeMoments<IRL::Volume, IRL::HalfEdgeCutting>(cell, p1);
        const double volume_fraction1 = IRL::getVolumeMoments<IRL::Volume, IRL::HalfEdgeCutting>(cell, p2);
        if ((volume_fraction < IRL::global_constants::VF_HIGH && volume_fraction > IRL::global_constants::VF_LOW) && (volume_fraction1 < IRL::global_constants::VF_HIGH && volume_fraction1 > IRL::global_constants::VF_LOW))
        {
            intersect = true;
        }
        /*IRL::Normal n1 = p[0].normal();
        IRL::Normal n2 = p[1].normal();
        double d1 = p[0].distance();
        double d2 = p[1].distance();
        double m1 = max(max(abs(n1[0]), abs(n1[1])), abs(n1[2]));
        double s1 = 0.5 / m1;
        double m2 = max(max(abs(n2[0]), abs(n2[1])), abs(n2[2]));
        double s2 = 0.5 / m2;
        if (abs(d1) <= s1 && abs(d2) <= s2)
        {
            intersect = true;
        }*/

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
        

        double A1 = p1.getAlignedParaboloid().a();
        double B1 = p1.getAlignedParaboloid().b();
        double u1 = p1.getDatum()[0];
        double v1 = p1.getDatum()[1];
        double w1 = p1.getDatum()[2];
        double theta1 = 0;

        double A2 = p2.getAlignedParaboloid().a();
        double B2 = p2.getAlignedParaboloid().b();
        double u2 = p2.getDatum()[0];
        double v2 = p2.getDatum()[1];
        double w2 = p2.getDatum()[2];
        double theta2 = 0;

        double a1 = A1 * v1 * v1 - w1 + B1 * u1 * u1 + A1 * u1 * u1 * cos(theta1) * cos(theta1) - A1 * v1 * v1 * cos(theta1) * cos(theta1) - B1 * u1 * u1 * cos(theta1) * cos(theta1) + B1 * v1 * v1 * cos(theta1) * cos(theta1) + A1 * u1 * v1 * sin(2*theta1) - B1 * u1 * v1 * sin(2*theta1);
        double b1 = A1 * u1 + B1 * u1 + A1 * u1 * cos(2*theta1) - B1 * u1 * cos(2*theta1) + A1 * v1 * sin(2 * theta1) - B1 * v1 * sin(2*theta1);
        double c1 = A1 * v1 + B1 * v1 - A1 * v1 * cos(2*theta1) + B1 * v1 * cos(2*theta1) + A1 * u1 * sin(2 * theta1) - B1 * u1 * sin(2*theta1);
        double d1 = A1 - A1 * sin(theta1) * sin(theta1) + B1 * sin(theta1) * sin(theta1);
        double e1 = (A1 - B1) * sin(2 * theta1);
        double f1 = B1 + A1 * sin(theta1) * sin(theta1) - B1 * sin(theta1) * sin(theta1);

        double a2 = A2 * v2 * v2 - w2 + B2 * u2 * u2 + A2 * u2 * u2 * cos(theta2) * cos(theta2) - A2 * v2 * v2 * cos(theta2) * cos(theta2) - B2 * u2 * u2 * cos(theta2) * cos(theta2) + B2 * v2 * v2 * cos(theta2) * cos(theta2) + A2 * u2 * v2 * sin(2*theta2) - B2 * u2 * v2 * sin(2*theta2);
        double b2 = A2 * u2 + B2 * u2 + A2 * u2 * cos(2*theta2) - B2 * u2 * cos(2*theta2) + A2 * v2 * sin(2 * theta2) - B2 * v2 * sin(2*theta2);
        double c2 = A2 * v2 + B2 * v2 - A2 * v2 * cos(2*theta2) + B2 * v2 * cos(2*theta2) + A2 * u2 * sin(2 * theta2) - B2 * u2 * sin(2*theta2);
        double d2 = A2 - A2 * sin(theta2) * sin(theta2) + B2 * sin(theta2) * sin(theta2);
        double e2 = (A2 - B2) * sin(2 * theta2);
        double f2 = B2 + A2 * sin(theta2) * sin(theta2) - B2 * sin(theta2) * sin(theta2);

        double a = a1 - a2;
        double b = b1 - b2;
        double c = c1 - c2;
        double d = d1 - d2;
        double e = e1 - e2;
        double f = f1 - f2;

        for (int i = 0; i <= 100; ++i)
        {
            double x = -10 + i * 20.0 / 10000.0;
            double det = (c + e * x) * (c + e * x) - 4 * f * (a + b * x + d * x * x);
            if (det >= 0)
            {result = true;
                double y1 = (-(c + e * x) + sqrt(det)) / (2 * f);
                double y2 = (-(c + e * x) - sqrt(det)) / (2 * f);
                if ((y1 >= -10 && y1 <= 10) || (y2 >= -10 && y2 <= 10))
                {
                    double z1 = a1 + b1 * x + c1 * y1 + d1 * x * x + e1 * x * y1 + f1 * y1 * y1;
                    double z2 = a1 + b1 * x + c1 * y2 + d1 * x * x + e1 * x * y2 + f1 * y2 * y2;
                    if ((z1 >= -10 && z1 <= 10) || (z2 >= -10 && z2 <= 10))
                    {
                        result = true;
                        break;
                    }
                }
            }
        }

        /*for (int i = 0; i < a_number_of_cells; ++i)
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
        }*/
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

    bool fractions::doPlanesIntersect(IRL::PlanarSeparator& p, const DataMesh<double>& a_liquid_volume_fraction, double limit)
    {
        bool result = false;
        double a1 = p[0].normal()[0];
        double b1 = p[0].normal()[1];
        double c1 = p[0].normal()[2];
        double d1 = p[0].distance();
        double a2 = p[1].normal()[0];
        double b2 = p[1].normal()[1];
        double c2 = p[1].normal()[2];
        double d2 = p[1].distance();
        double a = a1/c1 - a2/c2;
        double b = b1/c1 - b2/c2;
        double d = d1/c1 - d2/c2;
        // std::cout << a << " " << b << " " << d << std::endl;
        // if (b <= 10^(-8) && a <= 10^(-8))
        // {
        //     result = false;
        // }
        // else if (b <= 10^(-8))
        // {
        //     double x = d/a;
        //     if ((x >= -10 && x <= 10))
        //     {
        //         result = true;
        //     }
        // }
        // else
        {
            /*for (int i = 0; i <= 10000; ++i)
            {
                double x = -3 + i * 6.0 / 10000.0;
                double y = (1/b)*(d-a*x);
                double z = (d1-a1*x-b1*y)/c1;
                //std::cout << x << " " << y << " " << z << std::endl;
                if (y >= -3 && y <= 3 && z >= -3 && z <= 3)
                {
                    result = true;
                }
            }*/
            double bx1 = -limit;
            double bx2 = limit;
            double by1 = -limit;
            double by2 = limit;
            double bz1 = -limit;
            double bz2 = limit;


            double y1 = (1/b)*(d-a*bx1);
            double z1 = (d1-a1*bx1-b1*y1)/c1;
            if ((y1 >= -limit && y1 <= limit) && (z1 >= -limit && z1 <= limit))
            {
                return true;
            }
            double y2 = (1/b)*(d-a*bx2);
            double z2 = (d1-a1*bx2-b1*y2)/c1;
            if ((y2 >= -limit && y2 <= limit) && (z2 >= -limit && z2 <= limit))
            {
                return true;
            }
            double x1 = (-b/a)*by1 + d/a;
            z1 = (d1-a1*x1-b1*by1)/c1;
            if ((x1 >= -limit && x1 <= limit) && (z1 >= -limit && z1 <= limit))
            {
                return true;
            }
            double x2 = (-b/a)*by2 + d/a;
            z2 = (d1-a1*x2-b1*by2)/c1;
            if ((x2 >= -limit && x2 <= limit) && (z2 >= -limit && z2 <= limit))
            {
                return true;
            }
            x1 = (d1-b1*(d/b)-c1*bz1) / (a1+b1*(-a/b));
            y1 = (1/b)*(d-a*x1);
            if ((x1 >= -limit && x1 <= limit) && (y1 >= -limit && y1 <= limit))
            {
                return true;
            }
            x2 = (d1-b1*(d/b)-c1*bz2) / (a1+b1*(-a/b));
            y2 = (1/b)*(d-a*x2);
            if ((x2 >= -limit && x2 <= limit) && (y2 >= -limit && y2 <= limit))
            {
                return true;
            }
            //std::cout << x << " " << y << " " << z << std::endl;
        }
        return result;
    }
}

#endif