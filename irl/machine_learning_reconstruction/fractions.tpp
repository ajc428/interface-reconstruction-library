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
                    auto& volume = volumes.volume();      
                    auto& centroid = volumes.centroid();   
                   
                    f.push_back(volume);
                    if (centroids)
                    {
                        f.push_back(centroid[0]);
                        f.push_back(centroid[1]);
                        f.push_back(centroid[2]);    
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
                        f.push_back(centroid[0]);
                        f.push_back(centroid[1]);
                        f.push_back(centroid[2]);    
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
        IRL::Pt datum = IRL::Pt(a_interface.getDatum()[0] - (i-1), a_interface.getDatum()[1] - (j-1), a_interface.getDatum()[2] - (k-1));
        IRL::Paraboloid p = IRL::Paraboloid(datum, a_interface.getReferenceFrame(), a_interface.getAlignedParaboloid().a(), a_interface.getAlignedParaboloid().b());
        auto cell = IRL::RectangularCuboid::fromBoundingPts(
            IRL::Pt(mesh.x(1), mesh.y(1), mesh.z(1)),
            IRL::Pt(mesh.x(2), mesh.y(2), mesh.z(2)));
        return IRL::getVolumeMoments<IRL::AddSurfaceOutput<MomentType, SurfaceType>, IRL::HalfEdgeCutting>(cell, p);
    }

    template <class MomentType>
    MomentType fractions::getCellMoments(const IRL::Paraboloid& a_interface,
    const DataMesh<double>& a_liquid_volume_fraction, int x_loc, int y_loc, int z_loc)
    {
        const Mesh& mesh = a_liquid_volume_fraction.getMesh();
        const int i(x_loc), j(y_loc), k(z_loc);
        IRL::Pt datum = IRL::Pt(a_interface.getDatum()[0] - (i-1), a_interface.getDatum()[1] - (j-1), a_interface.getDatum()[2] - (k-1));
        IRL::Paraboloid p = IRL::Paraboloid(datum, a_interface.getReferenceFrame(), a_interface.getAlignedParaboloid().a(), a_interface.getAlignedParaboloid().b());
        auto cell = IRL::RectangularCuboid::fromBoundingPts(
            IRL::Pt(mesh.x(1), mesh.y(1), mesh.z(1)),
            IRL::Pt(mesh.x(2), mesh.y(2), mesh.z(2)));
        const auto moments = IRL::getVolumeMoments<MomentType, IRL::HalfEdgeCutting>(cell, p);
        /*auto cell = IRL::RectangularCuboid::fromBoundingPts(
            IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)),
            IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));
        const auto moments = IRL::getVolumeMoments<MomentType, IRL::HalfEdgeCutting>(cell, a_interface);*/
        return moments;
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
        IRL::Pt datum = IRL::Pt(a_interface.getDatum()[0] - (i-1), a_interface.getDatum()[1] - (j-1), a_interface.getDatum()[2] - (k-1));
        IRL::Paraboloid p = IRL::Paraboloid(datum, a_interface.getReferenceFrame(), a_interface.getAlignedParaboloid().a(), a_interface.getAlignedParaboloid().b());
        auto cell = IRL::RectangularCuboid::fromBoundingPts(
            IRL::Pt(mesh.x(1), mesh.y(1), mesh.z(1)),
            IRL::Pt(mesh.x(2), mesh.y(2), mesh.z(2)));
        using MyPtType = IRL::PtWithGradient<IRL::ParaboloidGradientLocal>;
        auto cube = IRL::StoredRectangularCuboid<MyPtType>::fromOtherPolytope(cell);
        return IRL::getVolumeMoments<IRL::AddSurfaceOutput<MomentType, SurfaceType>, IRL::HalfEdgeCutting>(cube, p);
    }

    template <class MomentType>
    MomentType fractions::getCellMomentsWithGradients(const IRL::Paraboloid& a_interface,
    const DataMesh<double>& a_liquid_volume_fraction, int x_loc, int y_loc, int z_loc)
    {
        const Mesh& mesh = a_liquid_volume_fraction.getMesh();
        const int i(x_loc), j(y_loc), k(z_loc);
        IRL::Pt datum = IRL::Pt(a_interface.getDatum()[0] - (i-1), a_interface.getDatum()[1] - (j-1), a_interface.getDatum()[2] - (k-1));
        IRL::Paraboloid p = IRL::Paraboloid(datum, a_interface.getReferenceFrame(), a_interface.getAlignedParaboloid().a(), a_interface.getAlignedParaboloid().b());
        auto cell = IRL::RectangularCuboid::fromBoundingPts(
            IRL::Pt(mesh.x(1), mesh.y(1), mesh.z(1)),
            IRL::Pt(mesh.x(2), mesh.y(2), mesh.z(2)));
        using MyPtType = IRL::PtWithGradient<IRL::ParaboloidGradientLocal>;
        auto cube = IRL::StoredRectangularCuboid<MyPtType>::fromOtherPolytope(cell);
        const auto moments = IRL::getVolumeMoments<MomentType, IRL::HalfEdgeCutting>(cube, p);

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
}

#endif