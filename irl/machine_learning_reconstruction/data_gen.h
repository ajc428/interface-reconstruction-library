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
#include "mpi.h"
#include <iostream>
#include <cstdlib>
#include <cstdio>
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

        // ------------------------------------------------------------------
        // Perturb the moment array in place, BEFORE the paraboloid fit / PCA
        // and before reflectMoments.
        //
        // Ordering matters: at inference the fit and the canonicalizing frame
        // are computed from real advected moments, which are noisy. If noise
        // is instead applied after the fit (as it previously was), the
        // canonical frame is chosen using clean information the deployed code
        // never has, and any noise-induced octant flip lands the network in a
        // frame it was never trained on.
        //
        // Empty and full cells are left untouched: their barycentres are
        // undefined, and their VF is a hard geometric fact that advection does
        // not blur. Interfacial cells get noise on VF and on both barycentres.
        // ------------------------------------------------------------------
        template <typename Engine>
        void perturbMoments(std::vector<double>& moments,
                            std::normal_distribution<double>& bary_noise,
                            Engine& eng) const
        {
            for (int i = 0; i < NX; ++i)
            {
                for (int j = 0; j < NY; ++j)
                {
                    for (int k = 0; k < NZ; ++k)
                    {
                        const int base = get_idx(i, j, k, 0);
                        const double vf = moments[base];

                        // Untouched if the cell carries no interface: the
                        // barycentres of empty/full cells are undefined.
                        if (vf <= IRL::global_constants::VF_LOW ||
                            vf >= IRL::global_constants::VF_HIGH)
                        {
                            continue;
                        }

                        // VF is deliberately NOT perturbed. It is matched
                        // exactly downstream by the distance solver to conserve
                        // volume, so it is not a noisy input in the way the
                        // barycentres are; and perturbing the barycentres
                        // already produces the moment inconsistency that
                        // advection introduces.

                        // Barycentre noise on both phases. Components are
                        // cell-relative and bounded by +/-0.5 of a cell width.
                        for (int m = 1; m < 7; ++m)
                        {
                            double c = moments[base + m] + bary_noise(eng);
                            moments[base + m] = std::min(0.5, std::max(-0.5, c));
                        }
                    }
                }
            }
        }

        // Collect liquid barycentres in global coordinates for the paraboloid
        // fit and the PCA.
        //
        // Full cells ARE included (vf > VF_LOW, matching the original
        // condition): for a thicker sheet there can be fully-liquid cells
        // sitting between the two interfaces, and those barycentres are part
        // of the film's shape that the fit and PCA need to see.
        // ------------------------------------------------------------------
        // PCA of the barycentre cloud.
        //
        // Returns the eigenvector of the SMALLEST eigenvalue -- the barycentres
        // lie near a surface, so the thin direction of the cloud is the surface
        // normal -- together with three scale-invariant shape descriptors built
        // from the eigenvalues (l0 >= l1 >= l2):
        //
        //   linearity  = (l0 - l1) / l0   cloud is a line  (ligament / edge-on)
        //   planarity  = (l1 - l2) / l0   cloud is a sheet (well-defined film)
        //   sphericity =  l2 / l0         cloud is isotropic (normal is noise)
        //
        // This is the same eigensolve fitConstrainedParaboloidFromPoints does
        // internally to set its frame, but the axis it returns is NOT this
        // vector: that function hands the result to Paraboloid::fromDerivatives,
        // which rebuilds the frame from the surface normal at the datum, and
        // the datum sits above the cloud centroid rather than at the vertex.
        // For a curved patch those differ by a degree or more. The PCA axis is
        // the patch-averaged orientation and is the more stable input, so the
        // sheet generator uses this directly and does not fit at all.
        //
        // The three descriptors are scalar functions of the eigenvalues, so
        // they are invariant under the reflections and axis permutations
        // reflectMoments applies and need no accompanying transform.
        // ------------------------------------------------------------------
        void computePCA(const std::vector<IRL::Pt>& points,
                        IRL::Normal& normal_pca,
                        double& linearity,
                        double& planarity,
                        double& sphericity) const
        {
            normal_pca = IRL::Normal(0.0, 0.0, 1.0);
            linearity = 0.0;
            planarity = 0.0;
            sphericity = 1.0;

            const int np = static_cast<int>(points.size());
            if (np < 3) return;   // covariance is rank-deficient; leave defaults

            double cx = 0.0, cy = 0.0, cz = 0.0;
            for (const auto& p : points) { cx += p[0]; cy += p[1]; cz += p[2]; }
            cx /= np; cy /= np; cz /= np;

            Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
            for (const auto& p : points)
            {
                Eigen::Vector3d d(p[0] - cx, p[1] - cy, p[2] - cz);
                cov += d * d.transpose();
            }
            cov /= static_cast<double>(np);

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
            if (solver.info() != Eigen::Success) return;

            // Eigen returns eigenvalues in increasing order.
            const Eigen::Vector3d ev = solver.eigenvalues();
            const double l2 = std::max(0.0, ev(0));   // smallest
            const double l1 = std::max(0.0, ev(1));
            const double l0 = std::max(0.0, ev(2));   // largest

            const Eigen::Vector3d v = solver.eigenvectors().col(0);   // smallest -> normal
            normal_pca = IRL::Normal(v(0), v(1), v(2));
            normal_pca.normalize();

            if (l0 > 1.0e-30)
            {
                linearity  = (l0 - l1) / l0;
                planarity  = (l1 - l2) / l0;
                sphericity = l2 / l0;
            }
        }

        std::vector<IRL::Pt> buildInterfacePoints(const std::vector<double>& moments) const
        {
            std::vector<IRL::Pt> points;
            points.reserve(NX * NY * NZ);
            for (int i = 0; i < NX; ++i)
            {
                for (int j = 0; j < NY; ++j)
                {
                    for (int k = 0; k < NZ; ++k)
                    {
                        const int base = get_idx(i, j, k, 0);
                        const double vf = moments[base];
                        if (vf > IRL::global_constants::VF_LOW)
                        {
                            points.push_back(
                                IRL::Pt(moments[base + 1], moments[base + 2], moments[base + 3]) +
                                IRL::Pt(gen->getStencil()->get_xm(i),
                                        gen->getStencil()->get_ym(j),
                                        gen->getStencil()->get_zm(k)));
                        }
                    }
                }
            }
            return points;
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

        void generate(double rota_l, double rota_h, double rotb_l, double rotb_h, double rotc_l, double rotc_h, double coa_l, double coa_h, double cob_l, double cob_h, double ox_l, double ox_h, double oy_l, double oy_h, double oz_l, double oz_h, bool disturb, bool normalize, std::string nam)
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
                if (n % 10000 == 0) { std::cout << n << std::endl; }
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
                        if (normalize || i % 7 == 0)
                        {
                            output << moments[i] << ",";
                        }
                        else if (!normalize)
                        {
                            output << moments[i]*moments[i - i%7] << ",";
                        }
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
                            }
                            else if (moments[i] + c < -0.5 && !full)
                            {
                                moments[i] = -0.5;
                            }
                            else if (!full)
                            {
                                moments[i] = moments[i] + c;
                            }

                            if (normalize)
                            {
                                output << moments[i] << ",";
                            }
                            else
                            {
                                output << moments[i]*moments[i - i%7] << ",";
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

        void generate_sheet(double coa_l, double coa_h, double cob_l, double cob_h, double t_l, double t_h, bool disturb, bool normalize, std::string nam)
        {
            // ----------------------------------------------------------------
            // MPI: samples are fully independent, so generation is split by
            // rank with no communication in the sample loop at all.
            //
            // Each rank writes its OWN files, suffixed with the rank, rather
            // than sharing handles. Several processes appending to one
            // std::ofstream is not synchronized and interleaves partial lines,
            // which would silently corrupt rows -- and a corrupt row here is
            // worse than a slow generator, because the moments/normals/shape/
            // coefficients files are matched line-for-line and one torn write
            // desynchronizes all five.
            //
            // The shards are merged back into the normal filenames by rank 0
            // at the end of this function and then deleted, so callers see
            // exactly the same output files as in a serial run.
            //
            // MPI_Initialized is checked rather than assumed, so calling this
            // from a non-MPI driver still works and keeps the original
            // un-suffixed filenames -- existing serial workflows are unchanged.
            // ----------------------------------------------------------------
            int mpi_ready = 0;
            MPI_Initialized(&mpi_ready);
            int rank = 0;
            int numranks = 1;
            if (mpi_ready)
            {
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                MPI_Comm_size(MPI_COMM_WORLD, &numranks);
            }
            const bool parallel = (mpi_ready && numranks > 1);
            const std::string tag = parallel ? ("_r" + std::to_string(rank)) : std::string("");

            // Split Ndata across ranks; the first (Ndata % numranks) ranks take
            // one extra so the total is exactly Ndata regardless of divisibility.
            const int base_n  = Ndata / numranks;
            const int extra_n = Ndata % numranks;
            const int my_Ndata = base_n + (rank < extra_n ? 1 : 0);

            bool flip;
            std::ofstream output;
            std::string data_name = "moments"+nam+tag+".txt";
            output.open(data_name, std::ios_base::app);
            std::ofstream normals;
            std::string normals_name = "normals"+nam+tag+".txt";
            normals.open(normals_name, std::ios_base::app);
            std::ofstream coefficients1;
            std::string name1 = "coefficients1"+nam+tag+".txt";
            coefficients1.open(name1, std::ios_base::app);
            std::ofstream coefficients2;
            std::string name2 = "coefficients2"+nam+tag+".txt";
            coefficients2.open(name2, std::ios_base::app);
            // Lightweight per-sample log of thickness vs. the PCA shape
            // descriptors (linearity, planarity, sphericity), so the
            // distribution can be histogrammed without loading the much
            // larger moments file just to pull out its last three columns.
            std::ofstream shape_log;
            std::string shape_name = "shape"+nam+tag+".txt";
            shape_log.open(shape_name, std::ios_base::app);

            // Seed per rank. random_device alone is not enough: on some
            // implementations it is deterministic, and even when it is not,
            // ranks launched simultaneously can draw correlated seeds. XOR-ing
            // in a rank-dependent odd constant guarantees distinct streams, so
            // ranks cannot silently generate duplicate samples.
            std::random_device rd;
            std::mt19937_64 a_eng(static_cast<unsigned long long>(rd())
                                  ^ (0x9E3779B97F4A7C15ULL * static_cast<unsigned long long>(rank + 1)));
            //std::uniform_real_distribution<double> noise(-0.025, 0.025);
            std::normal_distribution<double> noise(0.0,0.0125);

            const double t_l_safe = std::max(t_l, 1.0e-12);

            std::uniform_real_distribution<double> ang1(0, 2*M_PI);
            std::uniform_real_distribution<double> z_dist(-1.0, 1.0);
            std::uniform_real_distribution<double> translation(-0.5, 0.5);
            std::uniform_real_distribution<double> dist_a(coa_l, coa_h);
            std::uniform_real_distribution<double> dist_b(cob_l, cob_h);
            std::uniform_int_distribution<int> dist_sign(1, 2);
            std::uniform_real_distribution<double> unit01(0.0, 1.0);

            const double W = (gen->getStencil()->getNX() * gen->getStencil()->getDx()) / 2.0;
            const double DX = gen->getStencil()->getDx();

            // Half-width of the search bracket along dir1. 3W sweeps fully
            // across the domain from any start point drawn from
            // translation(-0.5,0.5).
            const double T_LIM = 3.0 * W;

            const auto cube = gen->getStencil()->getCell(gen->getStencil()->get_ic(), gen->getStencil()->get_jc(), gen->getStencil()->get_kc());

            auto random_unit_vector = [&]() -> IRL::Normal
            {
                double zz = z_dist(a_eng);
                double rr = std::sqrt(std::max(0.0, 1.0 - zz * zz));
                double aa = ang1(a_eng);
                IRL::Normal v(rr * std::cos(aa), rr * std::sin(aa), zz);
                v.normalize();
                return v;
            };

            std::bernoulli_distribution want_both_dist(0.5);
            std::bernoulli_distribution absent_is_2_dist(0.5);

            auto in_band = [](double v)
            {
                return v > IRL::global_constants::VF_LOW && v < IRL::global_constants::VF_HIGH;
            };

            auto vf_of = [&](const IRL::Paraboloid& p)
            {
                return IRL::getVolumeMoments<IRL::Volume>(cube, p);
            };

            // ----------------------------------------------------------------
            // Bracket-then-sample.
            //
            // The film is translated rigidly along its mean normal dir_m, so
            // as t increases each surface sweeps across the center cell and
            // its volume fraction runs monotonically from one extreme to the
            // other. The set of t for which a surface properly cuts the cell
            // is therefore a single interval, found by bisection rather than
            // discovered by blind sampling.
            //
            // Monotonicity is not exact: each surface sits theta/2 off dir_m,
            // so translating it also drifts its parabola sideways rather than
            // purely raising it. The tilt is bounded by
            // max_theta = atan(thickness/(2*sqrt(3)*W)), which keeps both
            // surfaces well inside the regime where the axial sweep dominates
            // the transverse drift. The endpoint straddle check below rejects
            // any case that falls outside it rather than returning a wrong
            // interval.
            //
            // Cost: ~2 endpoint evaluations + 2 bisections of BISECT_ITERS
            // cuts, per surface, after which every t drawn from the resulting
            // interval is valid by construction.
            // ----------------------------------------------------------------
            const int BISECT_ITERS = 14;   // 6W/2^14 ~ 5e-4 cell widths

            // Solve vf(t) == target on [ta,tb], given the values at the ends
            // straddle target. Monotonicity is not assumed beyond that: the
            // half that still straddles is always the one kept.
            auto solve_t = [&](auto&& cand_at, double target,
                               double ta, double va, double tb) -> double
            {
                for (int i = 0; i < BISECT_ITERS; ++i)
                {
                    const double tm = 0.5 * (ta + tb);
                    const double vm = vf_of(cand_at(tm));
                    if ((va - target) * (vm - target) <= 0.0) { tb = tm; }
                    else                                      { ta = tm; va = vm; }
                }
                return 0.5 * (ta + tb);
            };

            // Interval of t over which this surface properly cuts the center
            // cell (VF strictly between VF_LOW and VF_HIGH). Returns false if
            // no such interval exists within the bracket.
            auto band_interval = [&](auto&& cand_at, double& lo, double& hi) -> bool
            {
                const double ta = -T_LIM, tb = T_LIM;
                const double va = vf_of(cand_at(ta));
                const double vb = vf_of(cand_at(tb));

                // Require the sweep to run from "cell entirely on one side" to
                // "cell entirely on the other". If it does not, either the
                // surface never reaches the cell or the bracket is too narrow;
                // either way there is no reliable interval to hand back.
                const bool rising  = (va <= IRL::global_constants::VF_LOW &&
                                      vb >= IRL::global_constants::VF_HIGH);
                const bool falling = (va >= IRL::global_constants::VF_HIGH &&
                                      vb <= IRL::global_constants::VF_LOW);
                if (!rising && !falling) { return false; }

                const double t_at_low  = solve_t(cand_at, IRL::global_constants::VF_LOW,  ta, va, tb);
                const double t_at_high = solve_t(cand_at, IRL::global_constants::VF_HIGH, ta, va, tb);
                lo = std::min(t_at_low, t_at_high);
                hi = std::max(t_at_low, t_at_high);
                return (hi - lo) > 1.0e-9;
            };

            // Up to two disjoint t-intervals (set difference can split one).
            struct TSet
            {
                double lo[2], hi[2];
                int n = 0;
                void add(double a, double b)
                {
                    if (b - a > 1.0e-9 && n < 2) { lo[n] = a; hi[n] = b; ++n; }
                }
                double measure() const
                {
                    double m = 0.0;
                    for (int i = 0; i < n; ++i) { m += hi[i] - lo[i]; }
                    return m;
                }
            };

            // Report once if a requested mode turns out to be structurally
            // impossible for the (t_l, t_h) that were passed in, rather than
            // spinning forever in the shape loop looking for it.
            bool warned_infeasible = false;

            if (rank == 0)
            {
                std::cout << "# generate_sheet: " << Ndata << " samples over "
                          << numranks << " rank(s)";
                if (parallel) { std::cout << " -> files suffixed _r0.._r" << (numranks-1); }
                std::cout << std::endl;
            }

            for (int n = 0; n < my_Ndata; ++n) 
            {
                // Progress from rank 0 only, scaled to the global count so the
                // number printed means what it did before the split.
                if (rank == 0 && n % 100 == 0)
                {
                    std::cout << (static_cast<long>(n) * numranks) << std::endl;
                }
                // ------------------------------------------------------------
                // Presence mode, drawn ONCE per sample and held fixed across
                // every shape retry below (not redrawn per attempt: modes have
                // different success rates, so redrawing would let the easier
                // ones dominate the accepted data even though each draw was a
                // fair coin flip).
                // ------------------------------------------------------------
                const bool want_both = true;//want_both_dist(a_eng);
                const bool absent_is_2 = want_both ? false : absent_is_2_dist(a_eng);

                bool valid_pair = false;
                IRL::Paraboloid paraboloid1;
                IRL::Paraboloid paraboloid2;
                double thickness = 0.0;
                IRL::Normal dir1;
                IRL::Normal dir2;
                // The labels are computed during acceptance rather than after
                // it, and carried out of the loop -- see the consistency check
                // below for why.
                IRL::Normal accepted_normal1;
                IRL::Normal accepted_normal2;
                int countttt = 0;
                while (!valid_pair)
                {
                    // Safety cap. The loop below is a rejection sampler with no
                    // natural bound, so a predicate that can never be satisfied
                    // turns it into a silent infinite hang rather than an error
                    // (exactly what an inverted want1/want2 did). Abort loudly
                    // instead: at the observed acceptance rates this many
                    // consecutive failures is impossible by chance, so if it
                    // trips, something is wrong with the acceptance logic and
                    // not with luck.
                    if (++countttt > 1000)
                    {
                        std::cout << "# ERROR: generate_sheet could not satisfy mode "
                                  << (want_both ? "want_both" : (absent_is_2 ? "absent_is_2" : "absent_is_1"))
                                  << " after " << countttt << " shape draws; aborting."
                                  << " (rank " << rank << ")" << std::endl;
                        // MPI_Abort rather than exit(): one rank calling exit()
                        // leaves the others running to completion, or blocked,
                        // with no indication anything went wrong.
                        if (mpi_ready) { MPI_Abort(MPI_COMM_WORLD, 1); }
                        std::exit(1);
                    }

                    int sign1 = dist_sign(a_eng);
                    int sign2 = dist_sign(a_eng);
                    int sign3 = dist_sign(a_eng);
                    int sign4 = dist_sign(a_eng);

                    dir1 = random_unit_vector();

                    // --------------------------------------------------------
                    // Thickness.
                    //
                    // cell_extent is the width of the center cell projected
                    // onto dir1: how far the film can be translated along its
                    // own normal before it has swept the cell completely. Both
                    // surfaces can only cut the cell at once if they are
                    // closer together than that, so want_both is capped there.
                    //
                    // The single-plane modes get the FULL range: a very thin
                    // film grazing a corner of the cell puts one interface
                    // inside and the other outside perfectly well, so there is
                    // no lower bound to impose. The interval arithmetic finds
                    // those grazing configurations directly -- when the film is
                    // thin, the single-plane t-set is just the two narrow
                    // slivers at the ends of the other surface's band.
                    // --------------------------------------------------------
                    const double cell_extent =
                        DX * (std::abs(dir1[0]) + std::abs(dir1[1]) + std::abs(dir1[2]));

                    double tk_lo, tk_hi;
                    if (want_both)
                    {
                        tk_lo = t_l_safe;
                        tk_hi = std::min(t_h, cell_extent);
                    }
                    else
                    {
                        tk_lo = t_l_safe;
                        tk_hi = t_h;
                    }
                    if (tk_hi <= tk_lo)
                    {
                        // The caller's [t_l,t_h] cannot express this mode at
                        // this orientation. Fall back to the full range so the
                        // loop still terminates, and say so once.
                        if (!warned_infeasible)
                        {
                            std::cout << "# warning: thickness range [" << t_l << "," << t_h
                                      << "] cannot express "
                                      << (want_both ? "two-plane" : "single-plane")
                                      << " cells (cell extent ~" << cell_extent
                                      << "); falling back to the full range."
                                      << std::endl;
                            warned_infeasible = true;
                        }
                        tk_lo = t_l_safe;
                        tk_hi = t_h;
                    }
                    std::uniform_real_distribution<double> log_thick(std::log(tk_lo), std::log(tk_hi));
                    thickness = std::exp(log_thick(a_eng));

                    const double max_theta = std::atan(thickness / (2*std::sqrt(3)*W));
                    const double dot_min = std::cos(max_theta);
                    std::uniform_real_distribution<double> vec2_pruned(dot_min, 1.0);

                    double dot = vec2_pruned(a_eng);
                    double rot = ang1(a_eng);
                    double r = std::sqrt(std::max(0.0, 1.0 - dot * dot));
                    IRL::Normal a;
                    if (std::abs(dir1[0]) >= 0.97)
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
                    if (std::abs(dir2[0]) >= 0.97)
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

                    // Sweep direction: the film's MEAN normal, not either
                    // surface's own axis.
                    //
                    // Translating along dir1 makes surface 1's VF exactly
                    // monotonic in t (dir1 is frame1's axis, so the implicit
                    // function shifts by exactly t) but leaves surface 2 at
                    // the full tilt theta off the sweep axis. That exactness
                    // buys nothing here, because band_interval bisects rather
                    // than using the identity -- and bisection only needs the
                    // endpoints to straddle, not exact monotonicity.
                    //
                    // What it costs is real. vf(t) stops being monotonic once
                    // the parabola's transverse drift outruns the axial sweep,
                    // roughly when 2*a*R*tan(phi) > 1 for a surface at angle
                    // phi to the sweep direction. Sweeping along dir1 puts all
                    // of theta on surface 2; sweeping along the mean splits
                    // it, putting theta/2 on each. Since tan grows faster than
                    // linearly and the threshold is hard, halving the worst
                    // case is worth more than making one surface exact: at
                    // thickness 2.0 with curvature 0.6, sweeping along dir1
                    // puts surface 2 at ~1.2 (past the threshold, where
                    // band_interval can return a wrong interval) while the
                    // mean puts both at ~0.58.
                    //
                    // This also makes the construction symmetric in the two
                    // surfaces on its own, so no role-swapping coin is needed.
                    IRL::Normal dir_m = IRL::Normal(dir1[0] + dir2[0],
                                                    dir1[1] + dir2[1],
                                                    dir1[2] + dir2[2]);
                    dir_m.normalize();

                    double a1 = (2*sign1-3)*dist_a(a_eng);
                    double b1 = (2*sign2-3)*dist_b(a_eng);
                    double a2 = (2*sign3-3)*dist_a(a_eng);
                    double b2 = (2*sign4-3)*dist_b(a_eng);

                    // Which surface sits on which side of the datum. With the
                    // sweep along the mean normal both surfaces are treated
                    // identically -- each is tilted theta/2 off it -- so this
                    // is a fixed convention rather than something that needs
                    // randomizing.
                    const double side = -1.0;

                    struct TSet {
                        struct Interval { double lo, hi; };
                        std::vector<Interval> intervals;

                        void add(double lo, double hi) {
                            if (lo < hi) intervals.push_back({lo, hi});
                        }

                        // 1D Boolean Intersection
                        static TSet intersect(double lo1, double hi1, double lo2, double hi2) {
                            TSet s;
                            s.add(std::max(lo1, lo2), std::min(hi1, hi2));
                            return s;
                        }

                        // 1D Boolean Difference (A minus B)
                        static TSet difference(double loA, double hiA, double loB, double hiB) {
                            TSet s;
                            s.add(loA, std::min(hiA, loB));
                            s.add(std::max(loA, hiB), hiA);
                            return s;
                        }

                        double measure() const {
                            double len = 0;
                            for (const auto& i : intervals) len += (i.hi - i.lo);
                            return len;
                        }

                        bool sample(double u, double& out_t) const {
                            double len = measure();
                            if (len <= 0) return false;
                            u *= len; // Scale uniform [0,1] to total length
                            for (const auto& i : intervals) {
                                double ilen = i.hi - i.lo;
                                if (u <= ilen) { out_t = i.lo + u; return true; }
                                u -= ilen;
                            }
                            return false;
                        }
                    };

                    IRL::Pt P0(translation(a_eng), translation(a_eng), translation(a_eng));

                    auto cand1_at = [&](double t) {
                        return IRL::Paraboloid(P0 + t * dir_m + side * (thickness/2.0) * dir_m, frame1, a1, b1);
                    };
                    auto cand2_at = [&](double t) {
                        return IRL::Paraboloid(P0 + t * dir_m - side * (thickness/2.0) * dir_m, frame2, a2, b2);
                    };

                    double lo1, hi1, lo2, hi2;
                    const bool has1 = band_interval(cand1_at, lo1, hi1);
                    const bool has2 = band_interval(cand2_at, lo2, hi2);

                    TSet S;
                    if (want_both && has1 && has2) {
                        S = TSet::intersect(lo1, hi1, lo2, hi2);
                    } 
                    else if (absent_is_2 && has1) {
                        S = has2 ? TSet::difference(lo1, hi1, lo2, hi2) : TSet{{ {lo1, hi1} }};
                    } 
                    else if (!want_both && !absent_is_2 && has2) { // absent_is_1
                        S = has1 ? TSet::difference(lo2, hi2, lo1, hi1) : TSet{{ {lo2, hi2} }};
                    }

                    if (S.measure() > 0.0) 
                    {
                        // Because we've guaranteed valid bounds, we just draw T once or twice.
                        const int max_t_draws = 4; 
                        for (int k = 0; k < max_t_draws; ++k)
                        {
                            double t;
                            if (!S.sample(unit01(a_eng), t)) break; 

                            IRL::Paraboloid cand1 = cand1_at(t);
                            IRL::Paraboloid cand2 = cand2_at(t);

                            if (!intersect_in_domain(cand1, cand2, -W, W))
                            {
                                // ----------------------------------------
                                // Label-consistency check.
                                //
                                // The presence mode was decided from the VF
                                // band test, but the LABEL is decided by
                                // whether getAverageNormalNonAligned() finds
                                // a surface patch in the cell. Those two
                                // tests are not the same function, and near
                                // the band edges they disagree: a cell can
                                // satisfy VF > VF_LOW while the surface patch
                                // is too small to register, yielding a zero
                                // normal for a surface the mode says is
                                // present. Measured at ~0.42% of samples,
                                // concentrated in thin films (0.94% in the
                                // thinnest quintile, where the feasible t
                                // slivers are only about as wide as the
                                // thickness itself, so draws land hard
                                // against the band edge).
                                //
                                // Rather than tightening the VF inset and
                                // hoping, the fix is to make acceptance and
                                // labelling the SAME test: extract the
                                // normals here and require them to match the
                                // mode. A mismatch cannot survive, because
                                // the thing being checked IS the thing being
                                // written. The normals are then carried out
                                // of the loop and reused, so the extraction
                                // that used to happen downstream is not
                                // repeated -- this costs nothing on the
                                // ~99.6% that already agreed.
                                // ----------------------------------------
                                auto sm1 = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParaboloidParametrizedSurfaceOutput>>(cube, cand1);
                                IRL::Normal n1 = sm1.getSurface().getAverageNormalNonAligned();
                                auto sm2 = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParaboloidParametrizedSurfaceOutput>>(cube, cand2);
                                IRL::Normal n2 = sm2.getSurface().getAverageNormalNonAligned();

                                const bool have1 = n1.calculateMagnitude() > 1.0e-9;
                                const bool have2 = n2.calculateMagnitude() > 1.0e-9;

                                // absent_is_2 == true  means surface 2 is the
                                // absent one, so surface 1 is present. The
                                // TSet above is built the same way:
                                // difference(I1, I2) = in surface 1's band,
                                // outside surface 2's.
                                const bool want1 = want_both ? true :  absent_is_2;
                                const bool want2 = want_both ? true : !absent_is_2;

                                if (have1 == want1 && have2 == want2)
                                {
                                    paraboloid1 = cand1;
                                    paraboloid2 = cand2;
                                    accepted_normal1 = n1;
                                    accepted_normal2 = n2;
                                    valid_pair = true;
                                    break; // Success!
                                }
                            }
                        }
                    }
                }

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
                            }
                        }
                    }
                }

                // ----------------------------------------------------------
                // Noise goes here: BEFORE the fit, the PCA and reflectMoments.
                //
                // Everything downstream (the paraboloid fit, the PCA frame,
                // the canonicalizing reflection) is then derived from exactly
                // the data the deployed network will see. Previously the noise
                // was applied during output, i.e. after all of it, which meant
                // the canonical frame was picked using clean information that
                // does not exist at inference time.
                // ----------------------------------------------------------
                if (disturb)
                {
                    perturbMoments(moments, noise, a_eng);
                }

                // Points are rebuilt from the (possibly perturbed) moments,
                // and only from genuinely interfacial cells.
                points = buildInterfacePoints(moments);

                //points = smoothPoints(points,0.75);

                // PCA of the barycentre cloud, used directly. No paraboloid
                // fit here: its returned axis is the surface normal at the
                // datum, not the PCA axis, and the patch-averaged PCA axis is
                // the more stable canonicalizing direction.
                IRL::Normal normal_pca;
                double pca_shape[3] = {0.0, 0.0, 1.0};
                computePCA(points, normal_pca, pca_shape[0], pca_shape[1], pca_shape[2]);

                shape_log << thickness << "," << pca_shape[0] << "," << pca_shape[1] << "," << pca_shape[2]
                        << "," << (want_both ? 2 : (absent_is_2 ? 1 : -1))
                        << "," << std::acos(std::min(1.0, std::abs(IRL::dotProduct(dir1, dir2)))) << "\n";

                // Labels come from the acceptance stage, where they were
                // computed and validated against the presence mode.
                // Re-deriving them here would repeat two surface extractions,
                // and would also reintroduce the possibility of the written
                // label disagreeing with the mode recorded alongside it.
                auto normal1 = accepted_normal1;
                auto normal2 = accepted_normal2;

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



                //std::cout << normal1 << std::endl << std::endl;
                IRL::Pt center1 = get_global_centroid(moments);
                if (IRL::dotProduct(center1,normal_pca) < 0)
                {
                    //std::cout << fit.getAlignedParaboloid().a() << " " << fit.getAlignedParaboloid().b() << " " << fit.getReferenceFrame()[2] << std::endl;
                    //std::cout << normal1 << std::endl << std::endl;
                    normal_pca = -normal_pca;
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
                // std::cout << normal1 << " " << normal_pca << " " << surface_centroid << std::endl;
                // if (IRL::dotProduct(normal_pca, surface_centroid) < 0) {
                //     normal_pca = -normal_pca;
                // }
                // std::cout << normal1 << " " << normal_pca << std::endl << std::endl;


                const double eps = 1e-10;

                // if (normal_pca[0] < -eps) {
                //     normal_pca = -normal_pca;
                // } else if (std::abs(normal_pca[0]) <= eps) {
                //     if (normal_pca[1] < -eps) {
                //         normal_pca = -normal_pca;
                //     } else if (std::abs(normal_pca[1]) <= eps && normal_pca[2] < -eps) {
                //         normal_pca = -normal_pca;
                //     }
                // }

                // moments_fit was computed here and never used -- 27 cell
                // cuts per sample thrown away. Removed.

                IRL::Pt center = IRL::Pt(normal_pca[0],normal_pca[1],normal_pca[2]);
                reflectMoments(moments, direction, direction2, center);

                // Straight write-out: the noise is already baked into
                // `moments` above, so there is no longer a per-component
                // perturbation branch here.
                for (int i = 0; i < moments.size(); ++i)
                {
                    if (normalize || i % 7 == 0)
                    {
                        output << moments[i] << ",";
                    }
                    else
                    {
                        output << moments[i]*moments[i - i%7] << ",";
                    }
                }

                double tmp;
                switch (direction)
                {
                    case 1:
                        normal1[0] = -normal1[0];

                        normal2[0] = -normal2[0];

                        normal_pca[0] = -normal_pca[0];
                    break;
                    case 2:
                        normal1[1] = -normal1[1];

                        normal2[1] = -normal2[1];

                        normal_pca[1] = -normal_pca[1];
                    break;
                    case 3:
                        normal1[2] = -normal1[2];

                        normal2[2] = -normal2[2];

                        normal_pca[2] = -normal_pca[2];
                    break;
                    case 4:
                        normal1[0] = -normal1[0];
                        normal1[1] = -normal1[1];

                        normal2[0] = -normal2[0];
                        normal2[1] = -normal2[1];

                        normal_pca[0] = -normal_pca[0];
                        normal_pca[1] = -normal_pca[1];
                    break;
                    case 5:
                        normal1[0] = -normal1[0];
                        normal1[2] = -normal1[2];

                        normal2[0] = -normal2[0];
                        normal2[2] = -normal2[2];

                        normal_pca[0] = -normal_pca[0];
                        normal_pca[2] = -normal_pca[2];
                    break;
                    case 6:
                        normal1[1] = -normal1[1];
                        normal1[2] = -normal1[2];

                        normal2[1] = -normal2[1];
                        normal2[2] = -normal2[2];

                        normal_pca[1] = -normal_pca[1];
                        normal_pca[2] = -normal_pca[2];
                    break;
                    case 7:
                        normal1[0] = -normal1[0];
                        normal1[1] = -normal1[1];
                        normal1[2] = -normal1[2];

                        normal2[0] = -normal2[0];
                        normal2[1] = -normal2[1];
                        normal2[2] = -normal2[2];  

                        normal_pca[0] = -normal_pca[0];
                        normal_pca[1] = -normal_pca[1];
                        normal_pca[2] = -normal_pca[2];                     
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

                        tmp=normal_pca[0]; 
                        normal_pca[0]=normal_pca[1]; 
                        normal_pca[1]=tmp;                       
                    break;
                    case 2:
                        tmp=normal1[1]; 
                        normal1[1]=normal1[2]; 
                        normal1[2]=tmp;

                        tmp=normal2[1]; 
                        normal2[1]=normal2[2]; 
                        normal2[2]=tmp;    

                        tmp=normal_pca[1]; 
                        normal_pca[1]=normal_pca[2]; 
                        normal_pca[2]=tmp;                        
                    break;
                    case 3:
                        tmp=normal1[0]; 
                        normal1[0]=normal1[2]; 
                        normal1[2]=tmp;

                        tmp=normal2[0]; 
                        normal2[0]=normal2[2]; 
                        normal2[2]=tmp;     

                        tmp=normal_pca[0]; 
                        normal_pca[0]=normal_pca[2]; 
                        normal_pca[2]=tmp;                     
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

                        tmp=normal_pca[0]; 
                        normal_pca[0]=normal_pca[1]; 
                        normal_pca[1]=tmp;
                        tmp=normal_pca[1]; 
                        normal_pca[1]=normal_pca[2]; 
                        normal_pca[2]=tmp;                                            
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

                        tmp=normal_pca[0]; 
                        normal_pca[0]=normal_pca[1]; 
                        normal_pca[1]=tmp;
                        tmp=normal_pca[0]; 
                        normal_pca[0]=normal_pca[2]; 
                        normal_pca[2]=tmp;                    
                    break;
                }

                normal_pca.normalize();

                // if (!flip)
                // {
                //     normal_pca = -normal_pca;
                // }
                output << normal_pca[0] << ",";
                output << normal_pca[1] << ",";
                output << normal_pca[2] << ",";
                // Inputs 193-195: how trustworthy the PCA direction is.
                // Reflection-invariant, so no frame transform is applied.
                // output << pca_shape[0] << ",";
                // output << pca_shape[1] << ",";
                // output << pca_shape[2] << ",";
                // output << "\n";

                // std::cout << normal1 << std::endl;
                // std::cout << normal2 << std::endl;
                // std::cout << normal_pca << std::endl;

                normal1.normalize();
                normal2.normalize();
                // std::cout << normal1 << " " << normal2 << " " << normal_pca << std::endl;

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
                // ------------------------------------------------------------
                // Residual parameterization.
                //
                // The written pair is (n1, n2) = (normal1, -normal2). Instead
                // of writing those directly, write
                //
                //   H     = (n1 - n2)/2      delta = H - normal_pca
                //   s     = (n1 + n2)
                //
                // which inverts exactly as n1 = (delta+normal_pca) + s/2 and
                // n2 = -(delta+normal_pca) + s/2. It is a linear change of
                // variables, so nothing is lost, but the targets shrink where
                // it matters: the measured wedge is 0.75 deg median and n1.n2
                // sits at 179.4 deg, so both normals are almost entirely
                // determined by normal_pca -- which the network already gets as
                // an input. Writing them raw makes it spend most of its output
                // range re-encoding something it was handed. On the current
                // data |delta| has median 0.0073 for two-plane cells, against
                // unit-magnitude targets before, and delta = s = 0 is already a
                // sensible default.
                //
                // The single-plane modes give |delta| near 0.5 and 1.5, almost
                // constant: with one normal zero, H is half the surviving
                // normal, parallel to normal_pca in one mode and antiparallel
                // in the other. Those are mode-dependent constants rather than
                // small residuals, so the shrinkage benefit is confined to the
                // two-plane half. |s| doubles as a presence signal: ~0 when
                // both planes are present, ~1 when only one is.
                //
                // normal_pca is used AFTER the reflection switch above, so it
                // sits in the same canonical frame as the normals; deployment
                // reproduces it from inputs 189-191.
                // ------------------------------------------------------------
                const double n1c[3] = { normal1[0],  normal1[1],  normal1[2]};
                const double n2c[3] = {-normal2[0], -normal2[1], -normal2[2]};
                const double delta[3] = {0.5*(n1c[0]-n2c[0]) - normal_pca[0],
                                         0.5*(n1c[1]-n2c[1]) - normal_pca[1],
                                         0.5*(n1c[2]-n2c[2]) - normal_pca[2]};
                const double svec[3]  = {n1c[0]+n2c[0], n1c[1]+n2c[1], n1c[2]+n2c[2]};
                normals << delta[0] << "," << delta[1] << "," << delta[2] << "," << svec[0] << "," << svec[1] << "," << svec[2] << "\n";
           
           
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
            shape_log.close();

            // Every rank must have flushed and closed before rank 0 merges the
            // per-rank files, or it would read a file another rank is still
            // writing to.
            if (mpi_ready) { MPI_Barrier(MPI_COMM_WORLD); }

            if (parallel && rank == 0)
            {
                // Merge the per-rank shards back into the single files the rest
                // of the pipeline expects, then delete the shards.
                //
                // The rank loop is the INNER loop and runs in the same 0..P-1
                // order for every one of the five files. That is what preserves
                // the line-for-line correspondence between moments/normals/
                // shape/coefficients -- they are matched by row index, so any
                // file merged in a different rank order would silently pair
                // each stencil with the wrong label.
                //
                // Appending (not truncating) matches the ios_base::app the
                // shards themselves were opened with, so re-running without
                // clearing the outputs stacks runs exactly as it did serially.
                const std::string bases[5] = {"moments", "normals",
                                              "coefficients1", "coefficients2",
                                              "shape"};
                bool merge_ok = true;
                for (const auto& base : bases)
                {
                    const std::string dest_name = base + nam + ".txt";
                    std::ofstream dest(dest_name, std::ios_base::app | std::ios_base::binary);
                    if (!dest)
                    {
                        std::cout << "# ERROR: cannot open " << dest_name
                                  << " for merge; per-rank files left in place."
                                  << std::endl;
                        merge_ok = false;
                        continue;
                    }
                    for (int r = 0; r < numranks; ++r)
                    {
                        const std::string part_name = base + nam + "_r" + std::to_string(r) + ".txt";
                        std::ifstream part(part_name, std::ios_base::binary);
                        if (!part)
                        {
                            std::cout << "# ERROR: missing shard " << part_name
                                      << "; merge incomplete." << std::endl;
                            merge_ok = false;
                            continue;
                        }
                        // rdbuf() copies in bulk rather than line by line.
                        // Guarded because streaming an EMPTY file through
                        // operator<< sets failbit on some implementations, and
                        // a rank legitimately produces an empty shard whenever
                        // Ndata < numranks.
                        if (part.peek() != std::ifstream::traits_type::eof())
                        {
                            dest << part.rdbuf();
                        }
                        part.close();
                    }
                    dest.close();
                }

                // Only remove the shards once every merge succeeded -- if
                // anything went wrong the partial data is still recoverable
                // by hand.
                if (merge_ok)
                {
                    for (const auto& base : bases)
                    {
                        for (int r = 0; r < numranks; ++r)
                        {
                            std::remove((base + nam + "_r" + std::to_string(r) + ".txt").c_str());
                        }
                    }
                }
                else
                {
                    std::cout << "# per-rank files kept for manual recovery." << std::endl;
                }
            }

            if (rank == 0)
            {
                std::cout << "# generate_sheet done: " << Ndata
                          << " samples across " << numranks << " rank(s)";
                if (parallel) { std::cout << ", merged and shards removed"; }
                std::cout << "." << std::endl;
            }
        }; 

        void generate_sheet_single_phase(double coa_l, double coa_h, double cob_l, double cob_h, double t_l, double t_h, bool disturb, bool normalize, std::string nam)
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
            std::uniform_real_distribution<double> z_dist(-1.0, 1.0);
            std::uniform_real_distribution<double> translation(-1, 1);
            std::uniform_real_distribution<double> vec2(0.7, 1.0);///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            std::uniform_real_distribution<double> dist_a(coa_l, coa_h);
            std::uniform_real_distribution<double> dist_b(cob_l, cob_h);
            std::uniform_int_distribution<int> dist_sign(1, 2);
            for (int n = 0; n < Ndata; ++n) 
            {
                if (n % 10000 == 0) { std::cout << n << std::endl; }

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
                const double W = (gen->getStencil()->getNX() * gen->getStencil()->getDx()) / 2.0;
                const double s = std::min(1.0, thickness / (2.0 * W));
                const double dot_min = std::max(0.7, std::cos(2.0 * std::asin(s)));
                std::uniform_real_distribution<double> vec2_pruned(dot_min, 1.0);
                double z = z_dist(a_eng);
                double r = std::sqrt(1.0 - z * z);
                //dir1 = IRL::Normal(cos(angle1)*cos(angle2),sin(angle1)*cos(angle2),sin(angle2));
                dir1 = IRL::Normal(r * std::cos(angle1), r * std::sin(angle1), z);
                dir1.normalize();

                int max_it = 0;
                int max_it2 = 0;
                while (!valid_pair)
                {
                    if (max_it > 20)
                    {
                        sign1 = dist_sign(a_eng);
                        sign2 = dist_sign(a_eng);
                        sign3 = dist_sign(a_eng);
                        sign4 = dist_sign(a_eng);
                        //thickness = thick(a_eng);
                        max_it = 0;
                        if (max_it2 > 3)
                        {
                            thickness = thick(a_eng);
                            max_it2 = 0;
                        }
                        else
                        {
                            ++max_it2;
                        }
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

                    //double dot = vec2(a_eng);
                    double dot = vec2_pruned(a_eng);
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
                    normal_fit = -normal_fit;
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
                int current_ind = 0;
                for (int i = 0; i < moments.size(); ++i)
                {
                    if (i % 7 == 0)
                    {
                        current_ind = 0;
                    }
                    else
                    {
                        ++current_ind;
                    }
                    if (p == 0 || !disturb)
                    {
                        if (normalize || i % 7 == 0)
                        {
                            if (current_ind < 4)
                            {
                                output << moments[i] << ",";
                            }
                        }
                        else if (!normalize)
                        {
                            if (current_ind < 4)
                            {
                                output << moments[i]*moments[i - i%7] << ",";
                            }
                        }
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
                            }
                            else if (moments[i] + c < -0.5 && !full)
                            {
                                moments[i] = -0.5;
                            }
                            else if (!full)
                            {
                                moments[i] = moments[i] + c;
                            }

                            if (normalize)
                            {
                                if (current_ind < 4)
                                {
                                    output << moments[i] << ",";
                                }
                            }
                            else
                            {
                                if (current_ind < 4)
                                {
                                    output << moments[i]*moments[i - i%7] << ",";
                                }
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

        void generate_sheet_both(double coa_l, double coa_h, double cob_l, double cob_h, double t_l, double t_h, bool disturb, bool normalize, std::string nam)
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
            std::uniform_real_distribution<double> z_dist(-1.0, 1.0);
            std::uniform_real_distribution<double> translation(-1, 1);
            std::uniform_real_distribution<double> vec2(0.7, 1.0);///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            std::uniform_real_distribution<double> dist_a(coa_l, coa_h);
            std::uniform_real_distribution<double> dist_b(cob_l, cob_h);
            std::uniform_int_distribution<int> dist_sign(1, 2);
            for (int n = 0; n < Ndata; ++n) 
            {
                if (n % 10000 == 0) { std::cout << n << std::endl; }

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
                double z = z_dist(a_eng);
                double r = std::sqrt(1.0 - z * z);
                //dir1 = IRL::Normal(cos(angle1)*cos(angle2),sin(angle1)*cos(angle2),sin(angle2));
                dir1 = IRL::Normal(r * std::cos(angle1), r * std::sin(angle1), z);
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
                        thickness = thick(a_eng);
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

                auto cell = gen->getStencil()->getDomain();
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
                    normal_fit = -normal_fit;
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
                        if (normalize || i % 7 == 0)
                        {
                            output << moments[i] << ",";
                        }
                        else if (!normalize)
                        {
                            output << moments[i]*moments[i - i%7] << ",";
                        }
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
                            }
                            else if (moments[i] + c < -0.5 && !full)
                            {
                                moments[i] = -0.5;
                            }
                            else if (!full)
                            {
                                moments[i] = moments[i] + c;
                            }

                            if (normalize)
                            {
                                output << moments[i] << ",";
                            }
                            else
                            {
                                output << moments[i]*moments[i - i%7] << ",";
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

        void generate_sheet_plic(double coa_l, double coa_h, double cob_l, double cob_h, double t_l, double t_h, bool disturb, bool normalize, std::string nam)
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
            std::normal_distribution<double> noise(0.0,0.0125);

            std::uniform_real_distribution<double> thick(t_l, t_h);
            std::uniform_real_distribution<double> ang1(0, 2*M_PI);
            std::uniform_real_distribution<double> ang2(-M_PI/2.0, M_PI/2.0);
            std::uniform_real_distribution<double> z_dist(-1.0, 1.0);
            std::uniform_real_distribution<double> translation(-1, 1);
            std::uniform_real_distribution<double> vec2(0.7, 1.0);
            std::uniform_real_distribution<double> dist_a(coa_l, coa_h);
            std::uniform_real_distribution<double> dist_b(cob_l, cob_h);
            std::uniform_int_distribution<int> dist_sign(1, 2);

            for (int n = 0; n < Ndata; ++n) 
            {
                if (n % 10000 == 0) { std::cout << n << std::endl; }

                bool valid = false;
                IRL::Paraboloid paraboloid1;
                IRL::Normal dir1;

                int sign1 = dist_sign(a_eng);
                int sign2 = dist_sign(a_eng);
                double thickness = thick(a_eng);
                double angle1 = ang1(a_eng);
                double z = z_dist(a_eng);
                double r = std::sqrt(1.0 - z * z);
                dir1 = IRL::Normal(r * std::cos(angle1), r * std::sin(angle1), z);
                dir1.normalize();

                int max_it = 0;
                while (!valid)
                {
                    if (max_it > 100)
                    {
                        sign1 = dist_sign(a_eng);
                        sign2 = dist_sign(a_eng);
                        max_it = 0;
                    }
                    else
                    {
                        ++max_it;
                    }

                    double o_x = translation(a_eng);
                    double o_y = translation(a_eng);
                    double o_z = translation(a_eng);
                    IRL::Pt datum = IRL::Pt(o_x, o_y, o_z);

                    IRL::Normal a;
                    if (abs(dir1[0]) >= 0.97)
                    {
                        a = IRL::Normal(0,1,0);
                    }
                    else
                    {
                        a = IRL::Normal(1,0,0);
                    }
                    IRL::Normal t1 = IRL::crossProduct(dir1, a);
                    t1.normalize();
                    IRL::Normal t2 = IRL::crossProduct(dir1, t1);
                    t2.normalize();
                    IRL::ReferenceFrame frame1 = IRL::ReferenceFrame(t1, t2, dir1);

                    double a1 = (2*sign1-3)*dist_a(a_eng);
                    double b1 = (2*sign2-3)*dist_b(a_eng);

                    paraboloid1 = IRL::Paraboloid(datum, frame1, a1, b1);

                    const auto cube = gen->getStencil()->getCell(gen->getStencil()->get_ic(),gen->getStencil()->get_jc(),gen->getStencil()->get_kc());
                    const auto vol1 = IRL::getVolumeMoments<IRL::Volume>(cube, paraboloid1);

                    if (vol1 > IRL::global_constants::VF_LOW && vol1 < IRL::global_constants::VF_HIGH)
                    {
                        valid = true;
                    }
                }

                coefficients1 << paraboloid1.getDatum().x() << "," << paraboloid1.getDatum().y() << "," << paraboloid1.getDatum().z()
                << "," << paraboloid1.getReferenceFrame()[0][0] << "," << paraboloid1.getReferenceFrame()[0][1] << "," << paraboloid1.getReferenceFrame()[0][2]
                << "," << paraboloid1.getReferenceFrame()[1][0] << "," << paraboloid1.getReferenceFrame()[1][1] << "," << paraboloid1.getReferenceFrame()[1][2]
                << "," << paraboloid1.getReferenceFrame()[2][0] << "," << paraboloid1.getReferenceFrame()[2][1] << "," << paraboloid1.getReferenceFrame()[2][2]
                << "," << paraboloid1.getAlignedParaboloid().a() << "," << paraboloid1.getAlignedParaboloid().b() << "," << thickness << "\n";

                coefficients2 << 0 << "," << 0 << "," << 0
                << "," << 0 << "," << 0 << "," << 0
                << "," << 0 << "," << 0 << "," << 0
                << "," << 0 << "," << 0 << "," << 0
                << "," << 0 << "," << 0 << "," << thickness << "\n";

                auto moments1 = gen->get_moments(paraboloid1, 1, false, flip);
                std::vector<double> moments;
                std::vector<IRL::Pt> points;
                moments.resize(moments1.size());

                int direction = 0;
                int direction2 = 0;

                // Use moments1 directly — single paraboloid, no subtraction needed
                for (int i = 0; i < NX; ++i)
                {
                    for (int j = 0; j < NY; ++j)
                    {
                        for (int k = 0; k < NZ; ++k)
                        {
                            for (int m = 0; m < 7; ++m)
                            {
                                moments[get_idx(i, j, k, m)] = moments1[get_idx(i, j, k, m)];
                            }
                            if (moments[get_idx(i, j, k, 0)] > IRL::global_constants::VF_LOW)
                            {
                                points.push_back(IRL::Pt(moments[get_idx(i, j, k, 1)],moments[get_idx(i, j, k, 2)],moments[get_idx(i, j, k, 3)])+IRL::Pt(gen->getStencil()->get_xm(i),gen->getStencil()->get_ym(j),gen->getStencil()->get_zm(k)));
                            }
                        }
                    }
                }

                IRL::Paraboloid fit = fitConstrainedParaboloidFromPoints(points);

                auto cell = gen->getStencil()->getCell(gen->getStencil()->get_ic(),gen->getStencil()->get_jc(),gen->getStencil()->get_kc());
                auto surface_and_moments = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParaboloidParametrizedSurfaceOutput>>(cell, paraboloid1);
                auto surface = surface_and_moments.getSurface();
                auto normal1 = surface.getAverageNormalNonAligned();

                surface_and_moments = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParaboloidParametrizedSurfaceOutput>>(cell, fit);
                surface = surface_and_moments.getSurface();
                auto normal_fit = fit.getReferenceFrame()[2];

                IRL::Pt center1 = get_global_centroid(moments);
                if (IRL::dotProduct(center1, normal_fit) < 0)
                {
                    normal_fit = -normal_fit;
                }

                const double eps = 1e-10;

                auto moments_fit = gen->get_moments(fit, 1, true, flip);

                IRL::Pt center = IRL::Pt(normal_fit[0], normal_fit[1], normal_fit[2]);
                reflectMoments(moments, direction, direction2, center);

                int p = dis(a_eng);
                bool full = false;

                for (int i = 0; i < moments.size(); ++i)
                {
                    if (p == 0 || !disturb)
                    {
                        if (normalize || i % 7 == 0)
                        {
                            output << moments[i] << ",";
                        }
                        else if (!normalize)
                        {
                            output << moments[i]*moments[i - i%7] << ",";
                        }
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
                            }
                            else if (moments[i] + c < -0.5 && !full)
                            {
                                moments[i] = -0.5;
                            }
                            else if (!full)
                            {
                                moments[i] = moments[i] + c;
                            }

                            if (normalize)
                            {
                                output << moments[i] << ",";
                            }
                            else
                            {
                                output << moments[i]*moments[i - i%7] << ",";
                            }
                        }
                    }
                }

                double tmp;
                switch (direction)
                {
                    case 1:
                        normal1[0] = -normal1[0];
                        normal_fit[0] = -normal_fit[0];
                    break;
                    case 2:
                        normal1[1] = -normal1[1];
                        normal_fit[1] = -normal_fit[1];
                    break;
                    case 3:
                        normal1[2] = -normal1[2];
                        normal_fit[2] = -normal_fit[2];
                    break;
                    case 4:
                        normal1[0] = -normal1[0];
                        normal1[1] = -normal1[1];
                        normal_fit[0] = -normal_fit[0];
                        normal_fit[1] = -normal_fit[1];
                    break;
                    case 5:
                        normal1[0] = -normal1[0];
                        normal1[2] = -normal1[2];
                        normal_fit[0] = -normal_fit[0];
                        normal_fit[2] = -normal_fit[2];
                    break;
                    case 6:
                        normal1[1] = -normal1[1];
                        normal1[2] = -normal1[2];
                        normal_fit[1] = -normal_fit[1];
                        normal_fit[2] = -normal_fit[2];
                    break;
                    case 7:
                        normal1[0] = -normal1[0];
                        normal1[1] = -normal1[1];
                        normal1[2] = -normal1[2];
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
                        tmp=normal_fit[0]; 
                        normal_fit[0]=normal_fit[1]; 
                        normal_fit[1]=tmp;
                    break;
                    case 2:
                        tmp=normal1[1]; 
                        normal1[1]=normal1[2]; 
                        normal1[2]=tmp;
                        tmp=normal_fit[1]; 
                        normal_fit[1]=normal_fit[2]; 
                        normal_fit[2]=tmp;
                    break;
                    case 3:
                        tmp=normal1[0]; 
                        normal1[0]=normal1[2]; 
                        normal1[2]=tmp;
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
                        tmp=normal_fit[0]; 
                        normal_fit[0]=normal_fit[1]; 
                        normal_fit[1]=tmp;
                        tmp=normal_fit[0]; 
                        normal_fit[0]=normal_fit[2]; 
                        normal_fit[2]=tmp;
                    break;
                }

                normal_fit.normalize();
                output << normal_fit[0] << ",";
                output << normal_fit[1] << ",";
                output << normal_fit[2] << ",";
                output << "\n";

                normal1.normalize();

                // Write normal1 twice in place of normal1 + normal2 to preserve file format
                normals << -normal1[0] << "," << -normal1[1] << "," << -normal1[2] << "," << 0 << "," << 0 << "," << 0 << "\n";
            }  
            output.close();  
            normals.close(); 
            coefficients1.close(); 
        };

        // ============================================================================
        //  Sheet-edge data generation.
        //
        //  Drop these members into IRL::data_gen (data_gen.h), alongside
        //  generate_sheet / intersect_in_domain / fitConstrainedParaboloidFromPoints.
        //
        //  DIFFERENCE FROM generate_sheet:
        //
        //    The liquid region is, in both cases,
        //
        //        L = { phi_out < 0 } INTERSECT { phi_in > 0 }
        //
        //    where phi(P) = W + a*U^2 + b*V^2 is IRL's signed paraboloid function
        //    (see evaluateParaboloid()).  For a plain sheet the two surfaces never
        //    cross, so { phi_in < 0 } is nested inside { phi_out < 0 } cell by cell
        //    and the per-cell moments can be obtained by simple subtraction:
        //
        //        |L n C| = |P_out n C| - |P_in n C|
        //
        //    For a sheet EDGE the surfaces DO cross, the nesting fails in exactly the
        //    cells straddling the crossing curve, and that subtraction produces
        //    negative volume fractions and meaningless centroids.  Those cells are
        //    therefore integrated directly here.
        //
        //    The crossing location is random because the datum, thickness, tilt and
        //    curvature draws are random; requiring the crossing to fall inside the
        //    stencil is what turns a sheet into a sheet edge.
        // ============================================================================

        // ---------------------------------------------------------------------------
        //  Column-wise exact-in-z integration helpers
        // ---------------------------------------------------------------------------

        struct EdgeInterval { double lo, hi; };

        // Along the vertical line (x, y, t), phi(t) = A t^2 + B t + C.
        //
        //   shift = (x - dx, y - dy, t - dz)
        //   U = xi0  + rx t,  xi0  = f00*(x-dx) + f01*(y-dy) + f02*(-dz), rx = f[0][2]
        //   V = eta0 + ry t,  eta0 = f10*(x-dx) + f11*(y-dy) + f12*(-dz), ry = f[1][2]
        //   W = ze0  + rz t,  ze0  = f20*(x-dx) + f21*(y-dy) + f22*(-dz), rz = f[2][2]
        //
        //   phi = W + a U^2 + b V^2
        //       = (a rx^2 + b ry^2) t^2
        //       + (rz + 2 a xi0 rx + 2 b eta0 ry) t
        //       + (ze0 + a xi0^2 + b eta0^2)
        void columnQuadratic(const IRL::Paraboloid& p, double x, double y,
                            double& A, double& B, double& C) const
        {
            const auto& d = p.getDatum();
            const auto& f = p.getReferenceFrame();
            const double a = p.getAlignedParaboloid().a();
            const double b = p.getAlignedParaboloid().b();

            const double sx = x - d[0];
            const double sy = y - d[1];
            const double sz = -d[2];

            const double xi0  = f[0][0]*sx + f[0][1]*sy + f[0][2]*sz;
            const double eta0 = f[1][0]*sx + f[1][1]*sy + f[1][2]*sz;
            const double ze0  = f[2][0]*sx + f[2][1]*sy + f[2][2]*sz;

            const double rx = f[0][2];
            const double ry = f[1][2];
            const double rz = f[2][2];

            A = a*rx*rx + b*ry*ry;
            B = rz + 2.0*a*xi0*rx + 2.0*b*eta0*ry;
            C = ze0 + a*xi0*xi0 + b*eta0*eta0;
        }

        void pushClipped(std::vector<EdgeInterval>& v,
                        double lo, double hi, double z0, double z1) const
        {
            lo = std::max(lo, z0);
            hi = std::min(hi, z1);
            if (hi > lo) v.push_back({lo, hi});
        }

        // { t in [z0,z1] : sign * phi(t) < 0 }.  At most two disjoint intervals.
        void insideIntervals(const IRL::Paraboloid& p, double sign,
                            double x, double y, double z0, double z1,
                            std::vector<EdgeInterval>& out) const
        {
            out.clear();

            double A, B, C;
            columnQuadratic(p, x, y, A, B, C);
            A *= sign; B *= sign; C *= sign;

            const double tiny = 1.0e-14;

            if (std::abs(A) < tiny)
            {
                if (std::abs(B) < tiny)
                {
                    if (C < 0.0) pushClipped(out, z0, z1, z0, z1);
                }
                else
                {
                    const double root = -C / B;
                    if (B > 0.0) pushClipped(out, z0, root, z0, z1);   // phi < 0 below root
                    else         pushClipped(out, root, z1, z0, z1);   // phi < 0 above root
                }
                return;
            }

            const double disc = B*B - 4.0*A*C;

            if (A > 0.0)
            {
                if (disc <= 0.0) return;                       // phi >= 0 everywhere
                const double s = std::sqrt(disc);
                double r1 = (-B - s)/(2.0*A);
                double r2 = (-B + s)/(2.0*A);
                if (r1 > r2) std::swap(r1, r2);
                pushClipped(out, r1, r2, z0, z1);              // phi < 0 between the roots
            }
            else
            {
                if (disc <= 0.0) { pushClipped(out, z0, z1, z0, z1); return; }  // phi < 0 everywhere
                const double s = std::sqrt(disc);
                double r1 = (-B - s)/(2.0*A);
                double r2 = (-B + s)/(2.0*A);
                if (r1 > r2) std::swap(r1, r2);
                pushClipped(out, z0, r1, z0, z1);              // phi < 0 outside the roots
                pushClipped(out, r2, z1, z0, z1);
            }
        }

        // a \ b, both already clipped to the same segment.
        void subtractIntervals(const std::vector<EdgeInterval>& a,
                            const std::vector<EdgeInterval>& b,
                            std::vector<EdgeInterval>& out) const
        {
            out.clear();
            for (const auto& ia : a)
            {
                std::vector<EdgeInterval> pieces{ia};
                for (const auto& ib : b)
                {
                    std::vector<EdgeInterval> next;
                    for (const auto& pc : pieces)
                    {
                        if (ib.hi <= pc.lo || ib.lo >= pc.hi) { next.push_back(pc); continue; }
                        if (ib.lo > pc.lo) next.push_back({pc.lo, ib.lo});
                        if (ib.hi < pc.hi) next.push_back({ib.hi, pc.hi});
                    }
                    pieces.swap(next);
                    if (pieces.empty()) break;
                }
                for (const auto& pc : pieces) if (pc.hi > pc.lo) out.push_back(pc);
            }
        }

        // Volume fraction of { sign*phi < 0 } in one cell.  Used only to confirm the
        // orientation convention against IRL, never as a production quantity.
        double singleFraction(const IRL::Paraboloid& p, double sign,
                            double xc, double yc, double zc, double dx, int nq) const
        {
            const double h  = dx/nq;
            const double z0 = zc - 0.5*dx;
            const double z1 = zc + 0.5*dx;

            std::vector<EdgeInterval> in;
            double vol = 0.0;

            for (int i = 0; i < nq; ++i)
            {
                const double x = xc - 0.5*dx + (i + 0.5)*h;
                for (int j = 0; j < nq; ++j)
                {
                    const double y = yc - 0.5*dx + (j + 0.5)*h;
                    insideIntervals(p, sign, x, y, z0, z1, in);
                    for (const auto& s : in) vol += (s.hi - s.lo);
                }
            }
            return vol*h*h/(dx*dx*dx);
        }

        // Find the sign s such that { s*phi < 0 } is the region IRL reports as the
        // internal (liquid) phase.  IRL's convention is s = +1; this verifies it
        // against an actual cut cell rather than assuming it, and picks a cell with an
        // intermediate volume fraction so the comparison is unambiguous.
        double calibrateSign(const IRL::Paraboloid& p) const
        {
            const double dx = gen->getStencil()->getDx();

            int bi = gen->getStencil()->get_ic();
            int bj = gen->getStencil()->get_jc();
            int bk = gen->getStencil()->get_kc();
            double best_vf = -1.0;
            double best_gap = 1.0e30;

            for (int i = 0; i < NX; ++i)
            {
                for (int j = 0; j < NY; ++j)
                {
                    for (int k = 0; k < NZ; ++k)
                    {
                        const auto cell = gen->getStencil()->getCell(i, j, k);
                        const double vf =
                            IRL::getVolumeMoments<IRL::Volume>(cell, p)/(dx*dx*dx);
                        const double gap = std::abs(vf - 0.5);
                        if (gap < best_gap) { best_gap = gap; best_vf = vf; bi = i; bj = j; bk = k; }
                    }
                }
            }

            if (best_gap > 0.49) return 1.0;   // no cut cell anywhere: fall back to IRL convention

            const double vp = singleFraction(p, 1.0,
                                            gen->getStencil()->get_xm(bi),
                                            gen->getStencil()->get_ym(bj),
                                            gen->getStencil()->get_zm(bk),
                                            dx, 12);

            return (std::abs(vp - best_vf) <= std::abs((1.0 - vp) - best_vf)) ? 1.0 : -1.0;
        }

        // Moments of  { s_out*phi_out < 0 } \ { s_in*phi_in < 0 }  over one cell.
        // vf is the volume fraction; (cx,cy,cz) is the liquid centroid RELATIVE to the
        // cell center, matching the convention used by moments_gen::get_moments.
        void differenceCellMoments(const IRL::Paraboloid& p_out, const IRL::Paraboloid& p_in,
                                double s_out, double s_in,
                                double xc, double yc, double zc, double dx, int nq,
                                double& vf, double& cx, double& cy, double& cz) const
        {
            const double h  = dx/nq;
            const double z0 = zc - 0.5*dx;
            const double z1 = zc + 0.5*dx;

            std::vector<EdgeInterval> a, b, r;
            double vol = 0.0, mx = 0.0, my = 0.0, mz = 0.0;

            for (int i = 0; i < nq; ++i)
            {
                const double x = xc - 0.5*dx + (i + 0.5)*h;
                for (int j = 0; j < nq; ++j)
                {
                    const double y = yc - 0.5*dx + (j + 0.5)*h;

                    insideIntervals(p_out, s_out, x, y, z0, z1, a);
                    if (a.empty()) continue;
                    insideIntervals(p_in, s_in, x, y, z0, z1, b);
                    subtractIntervals(a, b, r);

                    for (const auto& s : r)
                    {
                        const double len = s.hi - s.lo;
                        vol += len;
                        mx  += len*x;
                        my  += len*y;
                        mz  += 0.5*(s.hi*s.hi - s.lo*s.lo);   // exact integral of t dt
                    }
                }
            }

            vf = vol*h*h/(dx*dx*dx);
            if (vol > 0.0)
            {
                cx = mx/vol - xc;
                cy = my/vol - yc;
                cz = mz/vol - zc;
            }
            else
            {
                cx = cy = cz = 0.0;
            }
        }

        // Total liquid volume of the edge configuration over the whole stencil, for a
        // given outer/inner assignment.  Cheap screening quantity.
        double totalEdgeVolume(const IRL::Paraboloid& p_out, const IRL::Paraboloid& p_in,
                            double s_out, double s_in, int nq) const
        {
            const double dx = gen->getStencil()->getDx();
            double total = 0.0, vf, cx, cy, cz;

            for (int i = 0; i < NX; ++i)
            {
                for (int j = 0; j < NY; ++j)
                {
                    for (int k = 0; k < NZ; ++k)
                    {
                        differenceCellMoments(p_out, p_in, s_out, s_in,
                                            gen->getStencil()->get_xm(i),
                                            gen->getStencil()->get_ym(j),
                                            gen->getStencil()->get_zm(k),
                                            dx, nq, vf, cx, cy, cz);
                        total += vf;
                    }
                }
            }
            return total;
        }

        // Number of midpoint columns per direction used in doubly-cut cells.
        static constexpr int kEdgeQuadScreen = 12;   // acceptance screening
        static constexpr int kEdgeQuad       = 128;   // production moments


        // ---------------------------------------------------------------------------
        //  generate_sheet_edge
        // ---------------------------------------------------------------------------
        void generate_sheet_edge(double coa_l, double coa_h, double cob_l, double cob_h,
                                double t_l, double t_h, bool disturb, bool normalize,
                                std::string nam)
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
            std::normal_distribution<double> noise(0.0,0.0125);

            std::uniform_real_distribution<double> thick(t_l, t_h);
            std::uniform_real_distribution<double> ang1(0, 2*M_PI);
            std::uniform_real_distribution<double> ang2(-M_PI/2.0, M_PI/2.0);
            std::uniform_real_distribution<double> z_dist(-1.0, 1.0);
            std::uniform_real_distribution<double> translation(-1, 1);
            std::uniform_real_distribution<double> vec2(0.7, 1.0);
            std::uniform_real_distribution<double> dist_a(coa_l, coa_h);
            std::uniform_real_distribution<double> dist_b(cob_l, cob_h);
            std::uniform_int_distribution<int> dist_sign(1, 2);

            const double dxc  = gen->getStencil()->getDx();
            const double half = (gen->getStencil()->getNX()*dxc)/2.0;

            for (int n = 0; n < Ndata; ++n)
            {
                if (n % 10000 == 0) { std::cout << n << std::endl; }

                bool valid_pair = false;
                IRL::Paraboloid paraboloid1;
                IRL::Paraboloid paraboloid2;
                IRL::Normal dir1;
                IRL::Normal dir2;

                // Orientation signs and outer/inner assignment, decided during
                // acceptance and reused when the moments are assembled.
                double sign_p1 = 1.0;
                double sign_p2 = 1.0;
                bool one_is_outer = true;

                int sign1 = dist_sign(a_eng);
                int sign2 = dist_sign(a_eng);
                int sign3 = dist_sign(a_eng);
                int sign4 = dist_sign(a_eng);
                double thickness = thick(a_eng);
                double angle1 = ang1(a_eng);
                double angle2 = ang2(a_eng);
                double z = z_dist(a_eng);
                double r = std::sqrt(1.0 - z * z);
                dir1 = IRL::Normal(r * std::cos(angle1), r * std::sin(angle1), z);
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
                        thickness = thick(a_eng);
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

                    paraboloid1 = IRL::Paraboloid(datum1,frame1,a1,b1);
                    paraboloid2 = IRL::Paraboloid(datum2,frame2,a2,b2);

                    // ---- Sheet EDGE requirement -------------------------------
                    // The two surfaces must cross somewhere inside the stencil.
                    // That crossing curve IS the edge; its location is random
                    // because datum, thickness, tilt and curvature are random.
                    // checkIntersectionRecursive demands that both signed
                    // functions bracket zero in a common box, so this genuinely
                    // tests for the intersection curve.  It is an interval test
                    // with a depth cap, so it can return true spuriously and is
                    // used only as a screen -- the volume test below is the real
                    // gate.
                    if (!intersect_in_domain(paraboloid1, paraboloid2, -half, half))
                    {
                        continue;
                    }

                    sign_p1 = calibrateSign(paraboloid1);
                    sign_p2 = calibrateSign(paraboloid2);

                    // Which paraboloid's interior is the liquid side.  The datum
                    // offsets above make one assignment the thin sheet and the
                    // other its (near-empty or near-full) complement, so pick by
                    // total liquid volume rather than assuming.
                    const double tv_a = totalEdgeVolume(paraboloid1, paraboloid2,
                                                        sign_p1, sign_p2, kEdgeQuadScreen);
                    const double tv_b = totalEdgeVolume(paraboloid2, paraboloid1,
                                                        sign_p2, sign_p1, kEdgeQuadScreen);
                    one_is_outer = (tv_a >= tv_b);

                    const IRL::Paraboloid& p_out = one_is_outer ? paraboloid1 : paraboloid2;
                    const IRL::Paraboloid& p_in  = one_is_outer ? paraboloid2 : paraboloid1;
                    const double s_out = one_is_outer ? sign_p1 : sign_p2;
                    const double s_in  = one_is_outer ? sign_p2 : sign_p1;

                    // The center cell must actually straddle the liquid surface,
                    // otherwise there is nothing to learn from this sample.
                    double vf_c, cx_c, cy_c, cz_c;
                    differenceCellMoments(p_out, p_in, s_out, s_in,
                                        gen->getStencil()->get_xm(gen->getStencil()->get_ic()),
                                        gen->getStencil()->get_ym(gen->getStencil()->get_jc()),
                                        gen->getStencil()->get_zm(gen->getStencil()->get_kc()),
                                        dxc, kEdgeQuadScreen, vf_c, cx_c, cy_c, cz_c);

                    if (vf_c > IRL::global_constants::VF_LOW &&
                        vf_c < IRL::global_constants::VF_HIGH)
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

                // ---- Per-cell moment assembly ---------------------------------
                // L = { s_out*phi_out < 0 } \ { s_in*phi_in < 0 }.
                //
                // Three cases are exactly resolvable from the IRL moments:
                //   * inner body fills the cell, or outer body absent -> empty
                //   * inner body absent                               -> outer moments
                //   * outer body fills the cell                       -> complement of inner
                // Everything else is a doubly-cut cell.  This is where the old
                // subtraction |P_out| - |P_in| is invalid (the inner region is no
                // longer contained in the outer one), so those cells are
                // integrated directly.
                {
                    const IRL::Paraboloid& p_out = one_is_outer ? paraboloid1 : paraboloid2;
                    const IRL::Paraboloid& p_in  = one_is_outer ? paraboloid2 : paraboloid1;
                    const double s_out = one_is_outer ? sign_p1 : sign_p2;
                    const double s_in  = one_is_outer ? sign_p2 : sign_p1;
                    const auto& m_out  = one_is_outer ? moments1 : moments2;
                    const auto& m_in   = one_is_outer ? moments2 : moments1;

                    for (int i = 0; i < NX; ++i)
                    {
                        for (int j = 0; j < NY; ++j)
                        {
                            for (int k = 0; k < NZ; ++k)
                            {
                                const double vfo = m_out[get_idx(i, j, k, 0)];
                                const double vfi = m_in [get_idx(i, j, k, 0)];

                                const bool out_empty = (vfo <= IRL::global_constants::VF_LOW);
                                const bool out_full  = (vfo >= IRL::global_constants::VF_HIGH);
                                const bool in_empty  = (vfi <= IRL::global_constants::VF_LOW);
                                const bool in_full   = (vfi >= IRL::global_constants::VF_HIGH);

                                if (out_empty || in_full)
                                {
                                    for (int m = 0; m < 7; ++m)
                                    {
                                        moments[get_idx(i, j, k, m)] = 0.0;
                                    }
                                }
                                else if (in_empty)
                                {
                                    // Nothing subtracted: outer moments are exact.
                                    for (int m = 0; m < 7; ++m)
                                    {
                                        moments[get_idx(i, j, k, m)] = m_out[get_idx(i, j, k, m)];
                                    }
                                }
                                else if (out_full)
                                {
                                    // vfi is intermediate here, so m_in's centroids
                                    // are populated (get_moments zeroes them only for
                                    // full/empty cells).  Liquid is exactly the
                                    // complement of the inner body.
                                    moments[get_idx(i, j, k, 0)] = 1.0 - vfi;
                                    moments[get_idx(i, j, k, 1)] = m_in[get_idx(i, j, k, 4)];
                                    moments[get_idx(i, j, k, 2)] = m_in[get_idx(i, j, k, 5)];
                                    moments[get_idx(i, j, k, 3)] = m_in[get_idx(i, j, k, 6)];
                                    moments[get_idx(i, j, k, 4)] = m_in[get_idx(i, j, k, 1)];
                                    moments[get_idx(i, j, k, 5)] = m_in[get_idx(i, j, k, 2)];
                                    moments[get_idx(i, j, k, 6)] = m_in[get_idx(i, j, k, 3)];
                                }
                                else
                                {
                                    // Doubly-cut cell: the edge region.
                                    double vf, cx, cy, cz;
                                    differenceCellMoments(p_out, p_in, s_out, s_in,
                                                        gen->getStencil()->get_xm(i),
                                                        gen->getStencil()->get_ym(j),
                                                        gen->getStencil()->get_zm(k),
                                                        dxc, kEdgeQuad, vf, cx, cy, cz);

                                    moments[get_idx(i, j, k, 0)] = vf;

                                    if (vf > IRL::global_constants::VF_LOW &&
                                        vf < IRL::global_constants::VF_HIGH)
                                    {
                                        moments[get_idx(i, j, k, 1)] = cx;
                                        moments[get_idx(i, j, k, 2)] = cy;
                                        moments[get_idx(i, j, k, 3)] = cz;
                                        // Centroids are stored relative to the cell
                                        // center, so the whole-cell centroid is 0 and
                                        //   vf*c_liq + (1-vf)*c_gas = 0.
                                        moments[get_idx(i, j, k, 4)] = -vf*cx/(1.0 - vf);
                                        moments[get_idx(i, j, k, 5)] = -vf*cy/(1.0 - vf);
                                        moments[get_idx(i, j, k, 6)] = -vf*cz/(1.0 - vf);
                                    }
                                    else
                                    {
                                        // Match get_moments: full/empty cells carry
                                        // zero centroids.
                                        for (int m = 1; m < 7; ++m)
                                        {
                                            moments[get_idx(i, j, k, m)] = 0.0;
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

                IRL::Paraboloid fit = fitConstrainedParaboloidFromPoints(points);

                auto cell = gen->getStencil()->getCell(gen->getStencil()->get_ic(),gen->getStencil()->get_jc(),gen->getStencil()->get_kc());
                auto surface_and_moments = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParaboloidParametrizedSurfaceOutput>>(cell, paraboloid1);
                auto surface = surface_and_moments.getSurface();
                auto normal1 = surface.getAverageNormalNonAligned();

                surface_and_moments = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParaboloidParametrizedSurfaceOutput>>(cell, paraboloid2);
                surface = surface_and_moments.getSurface();
                auto normal2 = surface.getAverageNormalNonAligned();

                surface_and_moments = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParaboloidParametrizedSurfaceOutput>>(cell, fit);
                surface = surface_and_moments.getSurface();
                auto normal_fit = fit.getReferenceFrame()[2];

                IRL::Pt center1 = get_global_centroid(moments);
                if (IRL::dotProduct(center1,normal_fit) < 0)
                {
                    normal_fit = -normal_fit;
                }

                const double eps = 1e-10;

                auto moments_fit = gen->get_moments(fit, 1, true, flip);

                IRL::Pt center = IRL::Pt(normal_fit[0],normal_fit[1],normal_fit[2]);
                reflectMoments(moments, direction, direction2, center);

                int p = dis(a_eng);
                bool full = false;

                for (int i = 0; i < moments.size(); ++i)
                {
                    if (p == 0 || !disturb)
                    {
                        if (normalize || i % 7 == 0)
                        {
                            output << moments[i] << ",";
                        }
                        else if (!normalize)
                        {
                            output << moments[i]*moments[i - i%7] << ",";
                        }
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
                            }
                            else if (moments[i] + c < -0.5 && !full)
                            {
                                moments[i] = -0.5;
                            }
                            else if (!full)
                            {
                                moments[i] = moments[i] + c;
                            }

                            if (normalize)
                            {
                                output << moments[i] << ",";
                            }
                            else
                            {
                                output << moments[i]*moments[i - i%7] << ",";
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
                output << normal_fit[0] << ",";
                output << normal_fit[1] << ",";
                output << normal_fit[2] << ",";
                output << "\n";

                normal1.normalize();
                normal2.normalize();

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
                    auto tmp = normal1;
                    normal1 = -normal2;
                    normal2 = -tmp;
                }

                normals << normal1[0] << "," << normal1[1] << "," << normal1[2] << "," << -normal2[0] << "," << -normal2[1] << "," << -normal2[2] << "\n";
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

            if (std::abs(center[0]) <= 1e-8) center[0] = 0;
            if (std::abs(center[1]) <= 1e-8) center[1] = 0;
            if (std::abs(center[2]) <= 1e-8) center[2] = 0;

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

            if (std::abs(center[0] - center[1]) <= 1e-8 && (center[0] - center[2]) > 1e-8) 
            {
                dir2 = 0;
            } 
            else if (std::abs(center[1] - center[2]) <= 1e-8 && (center[0] - center[1]) > 1e-8) 
            {
                dir2 = 0;
            } 
            else if (std::abs(center[0] - center[1]) <= 1e-8 && (center[2] - center[0]) > 1e-8) 
            {
                dir2 = 3;
                this->reflectMomentsXZ(moments);
                temp = center[0];
                center[0] = center[2];
                center[2] = temp;
            } 
            else if (std::abs(center[0] - center[2]) <= 1e-8 && (center[1] - center[0]) > 1e-8) 
            {
                dir2 = 1;
                this->reflectMomentsXY(moments);
                temp = center[0];
                center[0] = center[1];
                center[1] = temp;
            } 
            else if (std::abs(center[0] - center[2]) <= 1e-8 && (center[0] - center[1]) > 1e-8) 
            {
                dir2 = 2;
                this->reflectMomentsYZ(moments);
                temp = center[1];
                center[1] = center[2];
                center[2] = temp;
            } 
            else if (std::abs(center[1] - center[2]) <= 1e-8 && (center[1] - center[0]) > 1e-8) 
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