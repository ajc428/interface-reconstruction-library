// #include <mpi.h>
// #include <chrono>
// #include <iostream>
// #include <string>

// #include "examples/cylinder_advector/deformation_3d.h"
// #include "examples/cylinder_advector/reconstruction_types.h"
// #include "examples/cylinder_advector/solver.h"
// #include "examples/cylinder_advector/translation_3d.h"
// #include "examples/cylinder_advector/vof_advection.h"

// static int startSimulation(const std::string& a_simulation_type,
//                            const std::string& a_advection_method,
//                            const std::string& a_reconstruction_method,
//                            const double a_time_step_size,
//                            const double a_time_duration,
//                            const int a_viz_frequency);

// int main(int argc, char* argv[]) {
//   if (argc != 7) {
//     std::cout << "Incorrect amount of command line arguments supplied. \n";
//     std::cout << "Arguments should be \n";
//     std::cout << "Simulation to run. Options: Deformation3D, Translation3D\n";
//     std::cout << "Advection method. Options: SemiLagrangian, "
//                  "SemiLagrangianCorrected, FullLagrangian\n";
//     std::cout << "Reconstruction method. Options: PLIC, CentroidFit, Jibben\n";
//     std::cout << "Time step size, dt (double)\n";
//     std::cout << "Simulation duration(double)\n";
//     std::cout
//         << "Amount of time steps between visualization output (integer)\n";
//     std::exit(-1);
//   }

//   MPI_Init(&argc, &argv);

//   int rank, size;
//   MPI_Comm_size(MPI_COMM_WORLD, &size);
//   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

//   std::string simulation_type = argv[1];
//   std::string advection_method = argv[2];
//   std::string reconstruction_method = argv[3];
//   double time_step_size = std::stod(argv[4]);
//   double time_duration = std::stod(argv[5]);
//   int viz_frequency = atoi(argv[6]);

//   auto start = std::chrono::system_clock::now();
//   startSimulation(simulation_type, advection_method, reconstruction_method,
//                   time_step_size, time_duration, viz_frequency);
//   auto end = std::chrono::system_clock::now();
//   std::chrono::duration<double> runtime = end - start;
//   if (rank == 0) {
//     printf("Total run time: %20f \n\n", runtime.count());
//   }

//   MPI_Finalize();

//   return 0;
// }

// static int startSimulation(const std::string& a_simulation_type,
//                            const std::string& a_advection_method,
//                            const std::string& a_reconstruction_method,
//                            const double a_time_step_size,
//                            const double a_time_duration,
//                            const int a_viz_frequency) {
//   if (a_simulation_type == "Deformation3D") {
//     // return runSimulation<Deformation3D>(
//     //     a_advection_method, a_reconstruction_method, a_time_step_size,
//     //     a_time_duration, a_viz_frequency);
//     return runSimulation2<Deformation3D>(
//         a_advection_method, a_reconstruction_method, a_time_step_size,
//         a_time_duration, a_viz_frequency);
//   } else if (a_simulation_type == "Translation3D") {
//     // return runSimulation<Translation3D>(
//     //     a_advection_method, a_reconstruction_method, a_time_step_size,
//     //     a_time_duration, a_viz_frequency);
//     return runSimulation2<Translation3D>(
//         a_advection_method, a_reconstruction_method, a_time_step_size,
//         a_time_duration, a_viz_frequency);
//   } else {
//     std::cout << "Unknown simulation type of : " << a_simulation_type << '\n';
//     std::cout << "Value entries are: Deformation3D, Translation3D. \n";
//     std::exit(-1);
//   }
//   return -1;
// }

#include <iostream>
#include <random>
#include <chrono>
#include <cmath>
#include <vector>
#include <algorithm>
#include <mpi.h>

#include "examples/cylinder_advector/reconstruction_types.h"
#include "examples/cylinder_advector/data.h"
#include "examples/cylinder_advector/basic_mesh.h"
#include "irl/cylinder_reconstruction/cylinder.h"
#include "irl/moments/volume_moments.h"
#include "irl/generic_cutting/generic_cutting.h"
#include "examples/cylinder_advector/vtk.h"
#include "examples/cylinder_advector/solver.h"

// --- Test Configuration ---
constexpr int NX = 5;
constexpr int NY = 5;
constexpr int NZ = 5;
constexpr int GC = 5; // Using a larger ghost cell layer for the global reconstruction method
constexpr IRL::Pt lower_domain(0.0, 0.0, 0.0);
constexpr IRL::Pt upper_domain(1.0, 1.0, 1.0);

constexpr int NUM_RADII = 13;
constexpr int NUM_CASES = 10000;

/**
 * @brief Generates a random, normalized 3D vector.
 */
IRL::Normal generateRandomNormal(std::mt19937& gen) {
    std::uniform_real_distribution<double> d(0.0, 1.0);
    
    double u = d(gen);
    double v = d(gen);
    
    double theta = 2.0 * std::acos(-1.0) * u; // Azimuth: 0 to 2pi
    double z = 2.0 * v - 1.0;                 // Cosine of zenith uniformly distributed
    
    double r = std::sqrt(1.0 - z*z);
    double x = r * std::cos(theta);
    double y = r * std::sin(theta);
    
    // Applying the reversed operational sign convention
    return IRL::Normal(-x, -y, -z);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // ========================================================================
    // 1. SETUP
    // ========================================================================

    // Each rank gets a unique seed so RNG streams don't overlap.
    unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count()
                    + static_cast<unsigned>(rank) * 1000003u;
    std::mt19937 gen(seed);

    BasicMesh mesh(NX, NY, NZ, GC);
    mesh.setCellBoundaries(lower_domain, upper_domain);
    IRL::setVolumeFractionBounds(1.0e-14);
    load();

    Data<double> liquid_volume_fraction(&mesh);
    Data<IRL::Pt> liquid_centroid(&mesh);
    Data<IRL::Pt> gas_centroid(&mesh);
    Data<IRL::Cylinder> reconstructed_interface(&mesh);
    Data<IRL::PlanarSeparator> reconstructed_interface2(&mesh);
    Data<IRL::PlanarSeparator> reconstructed_interface3(&mesh);

    // ========================================================================
    // 2. DISTRIBUTE WORK: each rank owns a contiguous slice of the case loop.
    //    The radius (rad) loop is kept serial on every rank — it is only 13
    //    iterations and the synchronisation point is naturally at the end of
    //    each radius value anyway.  The inner case loop (100 cases) is the
    //    hot loop and is split across ranks.
    // ========================================================================
    // Work division for the case loop
    int local_cases = NUM_CASES / size;
    int remainder   = NUM_CASES % size;
    // Give the first `remainder` ranks one extra case
    int case_start  = rank * local_cases + std::min(rank, remainder);
    int case_end    = case_start + local_cases + (rank < remainder ? 1 : 0) - 1;
    // local_cases is now the actual number of cases this rank processes
    local_cases     = case_end - case_start + 1;

    // ========================================================================
    // 3. OPEN PER-RADIUS ERROR FILES (rank 0 only)
    // ========================================================================
    std::ofstream total_error_file;
    if (rank == 0) {
        total_error_file.open("reconstruction_errors_total.txt");
        total_error_file
            << "radius,"
            << "L1_radius_error,L2_radius_error,Lm_radius_error,P975_radius_error,"
            << "L1_orientation_error,L2_orientation_error,Lm_orientation_error,P975_orientation_error,"
            << "L1_datum_error,L2_datum_error,Lm_datum_error,P975_datum_error,"
            << "L1_bary_error,L2_bary_error,Lm_bary_error,P975_bary_error,"
            << "L1_bary_error_plane,L2_bary_error_plane,Lm_bary_error_plane,P975_bary_error_plane\n";
    }

    for (int rad = 0; rad < NUM_RADII; ++rad) {
        double ground_truth_radius = 0.04 / std::pow(4.0, rad);
        std::uniform_real_distribution<double> distrib(0.4-(4*ground_truth_radius), 0.6+(4*ground_truth_radius));

        // Per-radius output file (rank 0 only — written after gather)
        std::ofstream error_file;
        if (rank == 0) {
            error_file.open("reconstruction_errors_" +
                            std::to_string(10.0 * std::sqrt(ground_truth_radius)) + ".txt");
            error_file
                << "gt_radius,gt_datum_x,gt_datum_y,gt_datum_z,gt_dir_x,gt_dir_y,gt_dir_z,"
                << "radius_error,orientation_error,datum_error,bary_error,bary_error_plane\n";
        }

        // --- Local accumulators ---
        double local_L1_radius_error      = 0.0, local_L1_orientation_error = 0.0;
        double local_L1_datum_error       = 0.0, local_L1_bary_error        = 0.0;
        double local_L1_bary_error_plane  = 0.0;
        double local_L2_radius_error      = 0.0, local_L2_orientation_error = 0.0;
        double local_L2_datum_error       = 0.0, local_L2_bary_error        = 0.0;
        double local_L2_bary_error_plane  = 0.0;
        double local_Lm_radius_error      = 0.0, local_Lm_orientation_error = 0.0;
        double local_Lm_datum_error       = 0.0, local_Lm_bary_error        = 0.0;
        double local_Lm_bary_error_plane  = 0.0;

        // For percentile computation we need per-case scalars gathered on rank 0.
        // Pack them into a single flat array: [radius_err, ori_err, dat_err, bary_err, bary_pl_err]
        constexpr int METRICS_PER_CASE = 5;
        // local storage for valid cases only
        std::vector<double> local_case_data; // grows as cases are processed
        // We also need the ground-truth parameters for the per-radius CSV
        // Pack: [gt_r, gt_dx, gt_dy, gt_dz, gt_nx, gt_ny, gt_nz,
        //        r_err, ori_err, dat_err, bary_err, bary_pl_err]  (12 doubles)
        constexpr int PARAMS_PER_CASE = 12;
        std::vector<double> local_csv_data;

        int local_valid_count = 0;

        // ====================================================================
        // 4. CASE LOOP (this rank's slice)
        // ====================================================================
        for (int case_num = case_start; case_num <= case_end; ++case_num) {
                std::cout << rank << " rad " << rad << "  case " << case_num + 1 << std::endl;

            IRL::Cylinder   ground_truth_cylinder;
            IRL::Pt         ground_truth_datum;
            IRL::Normal     ground_truth_direction;

            // --- Generate ground truth ---
            ground_truth_direction  = generateRandomNormal(gen);
            ground_truth_direction.normalize();
            bool flag = true;
            do {
                do {
                    double dx = distrib(gen) - 0.5;
                    double dy = distrib(gen) - 0.5;
                    double dz = distrib(gen) - 0.5;
                    
                    // Only accept the datum if it falls inside the spherical volume
                    if (dx*dx + dy*dy + dz*dz <= 1.5*1.5) {
                        ground_truth_datum = IRL::Pt(dx + 0.5, dy + 0.5, dz + 0.5);
                        break; // Valid spherical datum found
                    }
                } while (true);

                IRL::Normal v1;
                if (std::abs(ground_truth_direction[0]) > 1.0e-6)
                    v1 = IRL::Normal(-ground_truth_direction[1], ground_truth_direction[0], 0.0);
                else
                    v1 = IRL::Normal(0.0, ground_truth_direction[2], -ground_truth_direction[1]);
                v1.normalize();
                IRL::Normal v2 = crossProduct(ground_truth_direction, v1);
                v2.normalize();
                IRL::ReferenceFrame ground_truth_frame(ground_truth_direction, v1, v2);

                ground_truth_cylinder = IRL::Cylinder(ground_truth_datum, ground_truth_frame,
                                                      1.0, ground_truth_radius);

                auto cell = IRL::RectangularCuboid::fromBoundingPts(
                    IRL::Pt(mesh.x(2), mesh.y(2), mesh.z(2)),
                    IRL::Pt(mesh.x(3), mesh.y(3), mesh.z(3)));
                auto moments = IRL::getNormalizedVolumeMoments<IRL::VolumeMoments>(
                    cell, ground_truth_cylinder);
                double vf = moments.volume() / cell.calculateVolume();
                if (vf > IRL::global_constants::VF_LOW && vf < IRL::global_constants::VF_HIGH)
                    flag = false;
            } while (flag);

            // --- Exact volume moments ---
            for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) {
                for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) {
                    for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) {
                        auto cell = IRL::RectangularCuboid::fromBoundingPts(
                            IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)),
                            IRL::Pt(mesh.x(i+1), mesh.y(j+1), mesh.z(k+1)));
                        auto moments = IRL::getNormalizedVolumeMoments<IRL::VolumeMoments>(
                            cell, ground_truth_cylinder);
                        double vf = moments.volume() / cell.calculateVolume();

                        liquid_volume_fraction(i,j,k) = vf;
                        liquid_centroid(i,j,k)        = moments.centroid();

                        gas_centroid(i,j,k)[0] = (mesh.xm(i) - moments.centroid()[0]*moments.volume()) / (1.0 - moments.volume());
                        gas_centroid(i,j,k)[1] = (mesh.ym(j) - moments.centroid()[1]*moments.volume()) / (1.0 - moments.volume());
                        gas_centroid(i,j,k)[2] = (mesh.zm(k) - moments.centroid()[2]*moments.volume()) / (1.0 - moments.volume());

                        if (vf < IRL::global_constants::VF_LOW) {
                            liquid_volume_fraction(i,j,k) = 0.0;
                            liquid_centroid(i,j,k) = IRL::Pt(mesh.xm(i), mesh.ym(j), mesh.zm(k));
                            gas_centroid(i,j,k)    = IRL::Pt(mesh.xm(i), mesh.ym(j), mesh.zm(k));
                        } else if (vf > IRL::global_constants::VF_HIGH) {
                            liquid_volume_fraction(i,j,k) = 1.0;
                            liquid_centroid(i,j,k) = IRL::Pt(mesh.xm(i), mesh.ym(j), mesh.zm(k));
                            gas_centroid(i,j,k)    = IRL::Pt(mesh.xm(i), mesh.ym(j), mesh.zm(k));
                        }
                    }
                }
            }

            liquid_volume_fraction.updateBorder();
            liquid_centroid.updateBorder();
            gas_centroid.updateBorder();

            // --- Reconstruct ---
            Data<double> U(&mesh), V(&mesh), W(&mesh);
            Cylinder_Curve_Global::getReconstruction(
                liquid_volume_fraction, liquid_centroid, gas_centroid,
                0.0, U, V, W, &reconstructed_interface);
            PLIC_NET::getReconstruction(
                liquid_volume_fraction, liquid_centroid, gas_centroid,
                0.0, U, V, W, &reconstructed_interface2);

            // --- Errors for cell (2,2,2) ---
            const double vf_center = liquid_volume_fraction(2,2,2);
            if (vf_center > IRL::global_constants::VF_LOW &&
                vf_center < IRL::global_constants::VF_HIGH)
            {
                local_valid_count++;
                const IRL::Cylinder& reconstructed_cyl = reconstructed_interface(2,2,2);

                auto cell = IRL::RectangularCuboid::fromBoundingPts(
                    IRL::Pt(mesh.x(2), mesh.y(2), mesh.z(2)),
                    IRL::Pt(mesh.x(3), mesh.y(3), mesh.z(3)));

                // Ground-truth bary
                auto moments = IRL::getNormalizedVolumeMoments<IRL::VolumeMoments>(
                    cell, ground_truth_cylinder);
                IRL::Pt bary = moments.centroid();

                // Reconstructed bary (cylinder)
                moments = IRL::getNormalizedVolumeMoments<IRL::VolumeMoments>(
                    cell, reconstructed_cyl);
                IRL::Pt bary2 = moments.centroid();

                double bary_error = std::sqrt(
                    std::pow(bary[0]-bary2[0],2.0) +
                    std::pow(bary[1]-bary2[1],2.0) +
                    std::pow(bary[2]-bary2[2],2.0));

                // if (rad > 10 && bary_error > 30*0.05/std::pow(4.0, rad))
                // {
                //     --case_num;
                //     std::cout << "SKIP" << std::endl;
                // }
                // else
                // {
                    // Reconstructed bary (plane)
                    const IRL::PlanarSeparator& reconstructed_plane = reconstructed_interface2(2,2,2);
                    auto moments2 = IRL::getNormalizedVolumeMoments<IRL::VolumeMoments>(
                        cell, reconstructed_plane);
                    IRL::Pt bary3 = moments2.centroid();

                    double bary_error_plane = std::sqrt(
                        std::pow(bary[0]-bary3[0],2.0) +
                        std::pow(bary[1]-bary3[1],2.0) +
                        std::pow(bary[2]-bary3[2],2.0));

                    // Radius error
                    const double r_recon = reconstructed_cyl.getAlignedCylinder().r();
                    double radius_error  = std::pow(
                        (r_recon - ground_truth_cylinder.getAlignedCylinder().r()) /
                        ground_truth_cylinder.getAlignedCylinder().r(), 2.0);

                    // Orientation error
                    const IRL::Normal& n_recon = reconstructed_cyl.getReferenceFrame()[0];
                    const IRL::Normal& n_truth = ground_truth_cylinder.getReferenceFrame()[0];
                    double dot_product = n_recon[0] * n_truth[0] + 
                                        n_recon[1] * n_truth[1] + 
                                        n_recon[2] * n_truth[2];
                    double cos_theta = std::abs(dot_product);
                    double orientation_error = std::acos(std::min(1.0, cos_theta));

                    // Datum error
                    const IRL::Pt& datum_recon   = reconstructed_cyl.getDatum();
                    IRL::Vec3<double> vec_to_datum = datum_recon - ground_truth_cylinder.getDatum();
                    IRL::Vec3<double> cross_prod   = crossProduct(
                        vec_to_datum,
                        static_cast<IRL::Vec3<double>>(ground_truth_cylinder.getReferenceFrame()[0]));
                    double datum_error = magnitude(cross_prod);

                    // Local accumulation
                    local_L1_radius_error     += radius_error;
                    local_L2_radius_error     += radius_error * radius_error;
                    local_Lm_radius_error      = std::max(local_Lm_radius_error, radius_error);

                    local_L1_orientation_error += orientation_error;
                    local_L2_orientation_error += orientation_error * orientation_error;
                    local_Lm_orientation_error  = std::max(local_Lm_orientation_error, orientation_error);

                    local_L1_datum_error      += datum_error;
                    local_L2_datum_error      += datum_error * datum_error;
                    local_Lm_datum_error       = std::max(local_Lm_datum_error, datum_error);

                    local_L1_bary_error       += bary_error;
                    local_L2_bary_error       += bary_error * bary_error;
                    local_Lm_bary_error        = std::max(local_Lm_bary_error, bary_error);

                    local_L1_bary_error_plane += bary_error_plane;
                    local_L2_bary_error_plane += bary_error_plane * bary_error_plane;
                    local_Lm_bary_error_plane  = std::max(local_Lm_bary_error_plane, bary_error_plane);

                    // Store per-case data for percentile gather
                    local_case_data.push_back(radius_error);
                    local_case_data.push_back(orientation_error);
                    local_case_data.push_back(datum_error);
                    local_case_data.push_back(bary_error);
                    local_case_data.push_back(bary_error_plane);

                    // Store CSV row data
                    local_csv_data.push_back(ground_truth_cylinder.getAlignedCylinder().r());
                    local_csv_data.push_back(ground_truth_cylinder.getDatum()[0]);
                    local_csv_data.push_back(ground_truth_cylinder.getDatum()[1]);
                    local_csv_data.push_back(ground_truth_cylinder.getDatum()[2]);
                    local_csv_data.push_back(ground_truth_cylinder.getReferenceFrame()[0][0]);
                    local_csv_data.push_back(ground_truth_cylinder.getReferenceFrame()[0][1]);
                    local_csv_data.push_back(ground_truth_cylinder.getReferenceFrame()[0][2]);
                    local_csv_data.push_back(radius_error);
                    local_csv_data.push_back(orientation_error);
                    local_csv_data.push_back(datum_error);
                    local_csv_data.push_back(bary_error);
                    local_csv_data.push_back(bary_error_plane);
                //}
            }
        } // end case loop

        // ====================================================================
        // 5. REDUCE ACCUMULATORS TO RANK 0
        // ====================================================================
        // Sum-based quantities
        double global_L1_radius_error     = 0.0, global_L1_orientation_error = 0.0;
        double global_L1_datum_error      = 0.0, global_L1_bary_error        = 0.0;
        double global_L1_bary_error_plane = 0.0;
        double global_L2_radius_error     = 0.0, global_L2_orientation_error = 0.0;
        double global_L2_datum_error      = 0.0, global_L2_bary_error        = 0.0;
        double global_L2_bary_error_plane = 0.0;
        double global_Lm_radius_error     = 0.0, global_Lm_orientation_error = 0.0;
        double global_Lm_datum_error      = 0.0, global_Lm_bary_error        = 0.0;
        double global_Lm_bary_error_plane = 0.0;
        int    global_valid_count         = 0;

        MPI_Reduce(&local_L1_radius_error,     &global_L1_radius_error,     1, MPI_DOUBLE, MPI_SUM,  0, MPI_COMM_WORLD);
        MPI_Reduce(&local_L1_orientation_error,&global_L1_orientation_error,1, MPI_DOUBLE, MPI_SUM,  0, MPI_COMM_WORLD);
        MPI_Reduce(&local_L1_datum_error,      &global_L1_datum_error,      1, MPI_DOUBLE, MPI_SUM,  0, MPI_COMM_WORLD);
        MPI_Reduce(&local_L1_bary_error,       &global_L1_bary_error,       1, MPI_DOUBLE, MPI_SUM,  0, MPI_COMM_WORLD);
        MPI_Reduce(&local_L1_bary_error_plane, &global_L1_bary_error_plane, 1, MPI_DOUBLE, MPI_SUM,  0, MPI_COMM_WORLD);

        MPI_Reduce(&local_L2_radius_error,     &global_L2_radius_error,     1, MPI_DOUBLE, MPI_SUM,  0, MPI_COMM_WORLD);
        MPI_Reduce(&local_L2_orientation_error,&global_L2_orientation_error,1, MPI_DOUBLE, MPI_SUM,  0, MPI_COMM_WORLD);
        MPI_Reduce(&local_L2_datum_error,      &global_L2_datum_error,      1, MPI_DOUBLE, MPI_SUM,  0, MPI_COMM_WORLD);
        MPI_Reduce(&local_L2_bary_error,       &global_L2_bary_error,       1, MPI_DOUBLE, MPI_SUM,  0, MPI_COMM_WORLD);
        MPI_Reduce(&local_L2_bary_error_plane, &global_L2_bary_error_plane, 1, MPI_DOUBLE, MPI_SUM,  0, MPI_COMM_WORLD);

        MPI_Reduce(&local_Lm_radius_error,     &global_Lm_radius_error,     1, MPI_DOUBLE, MPI_MAX,  0, MPI_COMM_WORLD);
        MPI_Reduce(&local_Lm_orientation_error,&global_Lm_orientation_error,1, MPI_DOUBLE, MPI_MAX,  0, MPI_COMM_WORLD);
        MPI_Reduce(&local_Lm_datum_error,      &global_Lm_datum_error,      1, MPI_DOUBLE, MPI_MAX,  0, MPI_COMM_WORLD);
        MPI_Reduce(&local_Lm_bary_error,       &global_Lm_bary_error,       1, MPI_DOUBLE, MPI_MAX,  0, MPI_COMM_WORLD);
        MPI_Reduce(&local_Lm_bary_error_plane, &global_Lm_bary_error_plane, 1, MPI_DOUBLE, MPI_MAX,  0, MPI_COMM_WORLD);

        MPI_Reduce(&local_valid_count,         &global_valid_count,         1, MPI_INT,    MPI_SUM,  0, MPI_COMM_WORLD);

        // ====================================================================
        // 6. GATHER PER-CASE DATA FOR PERCENTILES AND CSV (variable-length)
        // ====================================================================
        // --- Metric data (METRICS_PER_CASE doubles per valid case) ---
        int local_metric_count = static_cast<int>(local_case_data.size()); // = valid_count * METRICS_PER_CASE
        std::vector<int> all_metric_counts(size, 0);
        MPI_Gather(&local_metric_count, 1, MPI_INT,
                   all_metric_counts.data(), 1, MPI_INT,
                   0, MPI_COMM_WORLD);

        std::vector<double> all_case_data;
        std::vector<int>    metric_displs(size, 0);
        if (rank == 0) {
            for (int r = 1; r < size; ++r)
                metric_displs[r] = metric_displs[r-1] + all_metric_counts[r-1];
            int total = metric_displs[size-1] + all_metric_counts[size-1];
            all_case_data.resize(total);
        }
        MPI_Gatherv(local_case_data.data(), local_metric_count, MPI_DOUBLE,
                    all_case_data.data(), all_metric_counts.data(), metric_displs.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // --- CSV data (PARAMS_PER_CASE doubles per valid case) ---
        int local_csv_count = static_cast<int>(local_csv_data.size());
        std::vector<int> all_csv_counts(size, 0);
        MPI_Gather(&local_csv_count, 1, MPI_INT,
                   all_csv_counts.data(), 1, MPI_INT,
                   0, MPI_COMM_WORLD);

        std::vector<double> all_csv_data;
        std::vector<int>    csv_displs(size, 0);
        if (rank == 0) {
            for (int r = 1; r < size; ++r)
                csv_displs[r] = csv_displs[r-1] + all_csv_counts[r-1];
            int total = csv_displs[size-1] + all_csv_counts[size-1];
            all_csv_data.resize(total);
        }
        MPI_Gatherv(local_csv_data.data(), local_csv_count, MPI_DOUBLE,
                    all_csv_data.data(), all_csv_counts.data(), csv_displs.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // ====================================================================
        // 7. RANK 0: FINALISE STATISTICS AND WRITE FILES
        // ====================================================================
        if (rank == 0) {
            size_t valid_cases = (global_valid_count > 0) ? static_cast<size_t>(global_valid_count) : 1;

            // --- Averages ---
            double L1_rad = global_L1_radius_error      / valid_cases;
            double L2_rad = std::sqrt(global_L2_radius_error / valid_cases);

            double L1_ori = global_L1_orientation_error / valid_cases;
            double L2_ori = std::sqrt(global_L2_orientation_error / valid_cases);

            double L1_dat = global_L1_datum_error       / valid_cases;
            double L2_dat = std::sqrt(global_L2_datum_error / valid_cases);

            double L1_bar = global_L1_bary_error        / valid_cases;
            double L2_bar = std::sqrt(global_L2_bary_error / valid_cases);

            double L1_bpl = global_L1_bary_error_plane  / valid_cases;
            double L2_bpl = std::sqrt(global_L2_bary_error_plane / valid_cases);

            // --- Unpack interleaved metric vectors ---
            int n_valid = global_valid_count;
            std::vector<double> rad_err_list(n_valid), ori_err_list(n_valid),
                                dat_err_list(n_valid), bary_err_list(n_valid),
                                bary_pl_err_list(n_valid);
            for (int c = 0; c < n_valid; ++c) {
                int base = c * METRICS_PER_CASE;
                rad_err_list[c]     = all_case_data[base + 0];
                ori_err_list[c]     = all_case_data[base + 1];
                dat_err_list[c]     = all_case_data[base + 2];
                bary_err_list[c]    = all_case_data[base + 3];
                bary_pl_err_list[c] = all_case_data[base + 4];
            }

            // --- 97.5th-percentile relative differences ---
            double P975_rad = 0.0, P975_ori = 0.0, P975_dat = 0.0,
                   P975_bar = 0.0, P975_bpl = 0.0;
            if (n_valid > 0) {
                size_t idx = static_cast<size_t>(n_valid * 0.975);
                if (idx >= static_cast<size_t>(n_valid)) idx = n_valid - 1;

                std::nth_element(rad_err_list.begin(), rad_err_list.begin() + idx, rad_err_list.end());
                P975_rad = std::max(0.0, rad_err_list[idx] - L1_rad);

                std::nth_element(ori_err_list.begin(), ori_err_list.begin() + idx, ori_err_list.end());
                P975_ori = std::max(0.0, ori_err_list[idx] - L1_ori);

                std::nth_element(dat_err_list.begin(), dat_err_list.begin() + idx, dat_err_list.end());
                P975_dat = std::max(0.0, dat_err_list[idx] - L1_dat);

                std::nth_element(bary_err_list.begin(), bary_err_list.begin() + idx, bary_err_list.end());
                P975_bar = std::max(0.0, bary_err_list[idx] - L1_bar);

                std::nth_element(bary_pl_err_list.begin(), bary_pl_err_list.begin() + idx, bary_pl_err_list.end());
                P975_bpl = std::max(0.0, bary_pl_err_list[idx] - L1_bpl);
            }

            // --- Total error file ---
            total_error_file
                << 10.0 * std::sqrt(ground_truth_radius) << ","
                << L1_rad << "," << L2_rad << "," << global_Lm_radius_error      << "," << P975_rad << ","
                << L1_ori << "," << L2_ori << "," << global_Lm_orientation_error << "," << P975_ori << ","
                << L1_dat << "," << L2_dat << "," << global_Lm_datum_error       << "," << P975_dat << ","
                << L1_bar << "," << L2_bar << "," << global_Lm_bary_error        << "," << P975_bar << ","
                << L1_bpl << "," << L2_bpl << "," << global_Lm_bary_error_plane  << "," << P975_bpl << "\n";

            // --- Per-radius CSV ---
            int n_csv_rows = static_cast<int>(all_csv_data.size()) / PARAMS_PER_CASE;
            for (int c = 0; c < n_csv_rows; ++c) {
                int base = c * PARAMS_PER_CASE;
                error_file
                    << all_csv_data[base +  0] << ","  // gt_radius
                    << all_csv_data[base +  1] << ","  // gt_datum_x
                    << all_csv_data[base +  2] << ","  // gt_datum_y
                    << all_csv_data[base +  3] << ","  // gt_datum_z
                    << all_csv_data[base +  4] << ","  // gt_dir_x
                    << all_csv_data[base +  5] << ","  // gt_dir_y
                    << all_csv_data[base +  6] << ","  // gt_dir_z
                    << all_csv_data[base +  7] << ","  // radius_error
                    << all_csv_data[base +  8] << ","  // orientation_error
                    << all_csv_data[base +  9] << ","  // datum_error
                    << all_csv_data[base + 10] << ","  // bary_error
                    << all_csv_data[base + 11] << "\n"; // bary_error_plane
            }

            if (n_valid == 0)
                std::cout << "rad " << rad << ": No mixed-phase cells found." << std::endl;
        } // end rank 0 block
    } // end rad loop

    MPI_Finalize();
    return 0;
}


















































































// // =============================================================================
// // test_torus_reconstruction.cpp
// //
// // Static unit test for curved interface reconstruction on a Torus.
// //
// // PURPOSE
// // -------
// // Verifies that Cylinder_Curve_Global::getReconstruction recovers a curved
// // ring (a torus) whose minor radius is *smaller than one grid cell*.
// //
// // GROUND TRUTH ACCURACY (LOCAL CYLINDER TANGENTS)
// // -----------------------------------------------
// // Because IRL expects perfectly realizable moments (or the Newton solver fails)
// // and lacks a native Torus polyhedral cutter, we approximate the Torus 
// // dynamically: For each cell, we compute the closest point on the Torus core, 
// // build an exact local IRL::Cylinder tangent to the Torus, and use IRL's 
// // HalfEdgeCutting. This globally forms a Torus while ensuring 100% valid 
// // moment pairs per cell.
// // =============================================================================

// #include <cmath>
// #include <cstdio>
// #include <fstream>
// #include <iostream>
// #include <string>
// #include <vector>

// // ---- IRL headers -----------------------------------------------------------
// #include "irl/cylinder_reconstruction/cylinder.h"
// #include "irl/cylinder_reconstruction/cylinder_parametrized_surface.h"
// #include "irl/generic_cutting/generic_cutting.h"
// #include "irl/geometry/general/normal.h"
// #include "irl/geometry/general/pt.h"
// #include "irl/geometry/polyhedrons/rectangular_cuboid.h"
// #include "irl/moments/volume_moments.h"
// #include "irl/parameters/constants.h"
// #include "irl/planar_reconstruction/planar_localizer.h"

// // ---- Project headers -------------------------------------------------------
// #include "examples/cylinder_advector/basic_mesh.h"
// #include "examples/cylinder_advector/data.h"
// #include "examples/cylinder_advector/reconstruction_types.h"
// #include "examples/cylinder_advector/solver.h"
// #include "examples/cylinder_advector/vtk.h"

// // =============================================================================
// // Helper: Torus Geometry
// // =============================================================================
// struct Torus {
//     IRL::Pt center;
//     IRL::Normal axis;
//     double R_major;
//     double r_minor;

//     Torus(const IRL::Pt& c, const IRL::Normal& a, double R, double r)
//         : center(c), axis(a), R_major(R), r_minor(r) {
//         axis.normalize();
//     }
// };

// // =============================================================================
// // Helper: Write Exact Analytical Torus to Wavefront OBJ for ParaView
// // =============================================================================
// static void writeExactTorusOBJ(const Torus& t, const std::string& filename) {
//     std::ofstream out(filename);
//     out << "# Exact Torus Ground Truth\n";
//     const int n_theta = 200; 
//     const int n_phi = 100;   
    
//     IRL::Normal u(1, 0, 0);
//     if (std::abs(t.axis[0]) > 0.9) u = IRL::Normal(0, 1, 0);
//     u = IRL::crossProduct(t.axis, u); u.normalize();
//     IRL::Normal v = IRL::crossProduct(t.axis, u); v.normalize();
    
//     // Vertices
//     for (int i = 0; i < n_theta; ++i) {
//         double theta = 2.0 * M_PI * i / n_theta;
//         IRL::Normal ring_dir(
//             std::cos(theta)*u[0] + std::sin(theta)*v[0],
//             std::cos(theta)*u[1] + std::sin(theta)*v[1],
//             std::cos(theta)*u[2] + std::sin(theta)*v[2]
//         );
        
//         IRL::Pt C(
//             t.center[0] + t.R_major * ring_dir[0],
//             t.center[1] + t.R_major * ring_dir[1],
//             t.center[2] + t.R_major * ring_dir[2]
//         );
        
//         for (int j = 0; j < n_phi; ++j) {
//             double phi = 2.0 * M_PI * j / n_phi;
//             IRL::Pt P(
//                 C[0] + t.r_minor * (std::cos(phi)*ring_dir[0] + std::sin(phi)*t.axis[0]),
//                 C[1] + t.r_minor * (std::cos(phi)*ring_dir[1] + std::sin(phi)*t.axis[1]),
//                 C[2] + t.r_minor * (std::cos(phi)*ring_dir[2] + std::sin(phi)*t.axis[2])
//             );
//             out << "v " << P[0] << " " << P[1] << " " << P[2] << "\n";
//         }
//     }
    
//     // Faces (OBJ is 1-indexed)
//     for (int i = 0; i < n_theta; ++i) {
//         int i_next = (i + 1) % n_theta;
//         for (int j = 0; j < n_phi; ++j) {
//             int j_next = (j + 1) % n_phi;
//             int p1 = i * n_phi + j + 1;
//             int p2 = i_next * n_phi + j + 1;
//             int p3 = i_next * n_phi + j_next + 1;
//             int p4 = i * n_phi + j_next + 1;
//             out << "f " << p1 << " " << p2 << " " << p3 << " " << p4 << "\n";
//         }
//     }
// }

// // =============================================================================
// // Helper: writeInterfaceVTK
// // =============================================================================
// static void writeInterfaceVTK(const Data<double>& liquid_vf,
//                               const Data<IRL::Cylinder>& interface_field,
//                               double time,
//                               VTKOutput& vtk_out) {
//     const BasicMesh& mesh = liquid_vf.getMesh();
//     std::vector<IRL::CylinderParametrizedSurfaceOutput> surfaces;

//     for (int i = mesh.imin(); i <= mesh.imax(); ++i) {
//         for (int j = mesh.jmin(); j <= mesh.jmax(); ++j) {
//             for (int k = mesh.kmin(); k <= mesh.kmax(); ++k) {
//                 const double alpha = liquid_vf(i, j, k);
//                 if (alpha < IRL::global_constants::VF_LOW ||
//                     alpha > IRL::global_constants::VF_HIGH) {
//                     continue;
//                 }
//                 const auto cell = IRL::RectangularCuboid::fromBoundingPts(
//                     IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)),
//                     IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));

//                 auto vol_and_surf = IRL::getVolumeMoments<
//                     IRL::AddSurfaceOutput<IRL::Volume,
//                                           IRL::CylinderParametrizedSurfaceOutput>>(
//                     cell, interface_field(i, j, k));

//                 if (vol_and_surf.getMoments() > -DBL_MAX) {
//                     auto surf = vol_and_surf.getSurface();
//                     surf.setLengthScale(1.0e-2);
//                     if (surf.getSurfaceArea() > 1.0e-6 * 1.0e-3 * 1.0e-3) {
//                         surfaces.push_back(surf);
//                     }
//                 }
//             }
//         }
//     }
//     vtk_out.writeVTKInterface(time, surfaces, /*print_info=*/false);
// }

// // =============================================================================
// // Main test
// // =============================================================================
// int main(int argc, char* argv[]) {
//     MPI_Init(&argc, &argv);

//     IRL::setVolumeFractionBounds(1.0e-14);
//     IRL::setVolumeFractionTolerance(1.0e-13);
//     load();

//     constexpr int NX = 40, NY = 40, NZ = 40, GC = 3;
//     const IRL::Pt domain_lo(0.0, 0.0, 0.0);
//     const IRL::Pt domain_hi(1.0, 1.0, 1.0);

//     BasicMesh mesh(NX, NY, NZ, GC);
//     mesh.setCellBoundaries(domain_lo, domain_hi);
//     const double dx = mesh.dx(); 

//     const IRL::Pt torus_center(0.5, 0.5, 0.5);
//     IRL::Normal axis_dir(std::sin(M_PI / 6.0) * std::cos(M_PI / 9.0),
//                          std::sin(M_PI / 6.0) * std::sin(M_PI / 9.0),
//                          std::cos(M_PI / 6.0));
//     axis_dir.normalize();

//     const double R_major = 0.25;
//     const double r_minor = 0.1 * dx; // strictly sub-grid
//     Torus gt_torus(torus_center, axis_dir, R_major, r_minor);

//     std::cout << "============================================================\n";
//     std::cout << "  Static Torus Reconstruction Test (Local Tangent Cylinders)\n";
//     std::cout << "============================================================\n";

//     Data<double>       liquid_vf(&mesh);
//     Data<IRL::Pt>      liquid_centroid(&mesh);
//     Data<IRL::Pt>      gas_centroid(&mesh);
//     Data<IRL::Cylinder> reconstructed(&mesh);
//     Data<IRL::Cylinder> gt_field(&mesh);

//     Data<double> U(&mesh), V(&mesh), W(&mesh);
//     for (int i = mesh.imino(); i <= mesh.imaxo(); ++i)
//         for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j)
//             for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) {
//                 U(i, j, k) = V(i, j, k) = W(i, j, k) = 0.0;
//             }

//     const double cell_volume = mesh.dx() * mesh.dy() * mesh.dz();

//     std::cout << "  Computing perfectly realizable moments via Local Tangents...\n";

//     for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) {
//         for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) {
//             for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) {
//                 const IRL::Pt cell_center(mesh.xm(i), mesh.ym(j), mesh.zm(k));
                
//                 // 1. Find the local tangent frame to the Torus relative to this cell
//                 IRL::Normal v(cell_center[0] - gt_torus.center[0],
//                               cell_center[1] - gt_torus.center[1],
//                               cell_center[2] - gt_torus.center[2]);
                
//                 double h = v[0]*gt_torus.axis[0] + v[1]*gt_torus.axis[1] + v[2]*gt_torus.axis[2];
//                 IRL::Normal v_proj(v[0] - h*gt_torus.axis[0],
//                                    v[1] - h*gt_torus.axis[1],
//                                    v[2] - h*gt_torus.axis[2]);
//                 double d = std::sqrt(v_proj[0]*v_proj[0] + v_proj[1]*v_proj[1] + v_proj[2]*v_proj[2]);

//                 if (d < 1e-12) { d = 1e-12; v_proj = IRL::Normal(1, 0, 0); }

//                 IRL::Normal n_proj(v_proj[0]/d, v_proj[1]/d, v_proj[2]/d);
//                 IRL::Pt P_core(
//                     gt_torus.center[0] + gt_torus.R_major * n_proj[0],
//                     gt_torus.center[1] + gt_torus.R_major * n_proj[1],
//                     gt_torus.center[2] + gt_torus.R_major * n_proj[2]
//                 );

//                 IRL::Normal local_axis = IRL::crossProduct(n_proj, gt_torus.axis);
//                 local_axis.normalize();

//                 IRL::ReferenceFrame local_frame(local_axis, n_proj, gt_torus.axis);
//                 IRL::Cylinder local_cyl(P_core, local_frame, 1.0, gt_torus.r_minor * gt_torus.r_minor);

//                 // 2. Cut the cell exactly using IRL's native polyhedral cutter
//                 auto cell = IRL::RectangularCuboid::fromBoundingPts(
//                     IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)),
//                     IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));

//                 auto moments = IRL::getNormalizedVolumeMoments<IRL::VolumeMoments>(cell, local_cyl);

//                 const double liq_vol = moments.volume();
//                 double alpha = liq_vol / cell_volume;

//                 if (alpha < IRL::global_constants::VF_LOW) {
//                     liquid_vf(i, j, k)       = 0.0;
//                     liquid_centroid(i, j, k) = cell_center; 
//                     gas_centroid(i, j, k)    = cell_center;
//                     gt_field(i, j, k)        = IRL::Cylinder::createAlwaysBelow();
//                 } else if (alpha > IRL::global_constants::VF_HIGH) {
//                     liquid_vf(i, j, k)       = 1.0;
//                     liquid_centroid(i, j, k) = cell_center; 
//                     gas_centroid(i, j, k)    = cell_center;
//                     gt_field(i, j, k)        = IRL::Cylinder::createAlwaysAbove();
//                 } else {
//                     liquid_vf(i, j, k)       = alpha;
//                     liquid_centroid(i, j, k) = moments.centroid();
//                     gt_field(i, j, k)        = local_cyl;

//                     const IRL::Pt x_liq = moments.centroid() / liq_vol;
//                     const double beta = 1.0 - alpha;
//                     gas_centroid(i, j, k)[0] = (cell_center[0] - alpha * x_liq[0]) / beta;
//                     gas_centroid(i, j, k)[1] = (cell_center[1] - alpha * x_liq[1]) / beta;
//                     gas_centroid(i, j, k)[2] = (cell_center[2] - alpha * x_liq[2]) / beta;
//                 }
//             }
//         }
//     }

//     liquid_vf.updateBorder();
//     liquid_centroid.updateBorder();
//     gas_centroid.updateBorder();

//     std::cout << "  Running Cylinder_Curve_Global reconstruction...\n";
//     Cylinder_Curve_Global::getReconstruction(
//         liquid_vf, liquid_centroid, gas_centroid, 0.0, U, V, W, &reconstructed);

//     std::cout << "  Evaluating barycenter error in mixed cells...\n\n";

//     int    n_mixed      = 0;
//     double sum_bary_err   = 0.0;
//     double max_bary_err   = 0.0;
//     double sum_liq_vol    = 0.0;

//     IRL::Pt gt_global_bary(0.0, 0.0, 0.0);
//     IRL::Pt re_global_bary(0.0, 0.0, 0.0);

//     for (int i = mesh.imin(); i <= mesh.imax(); ++i) {
//         for (int j = mesh.jmin(); j <= mesh.jmax(); ++j) {
//             for (int k = mesh.kmin(); k <= mesh.kmax(); ++k) {
//                 const double alpha = liquid_vf(i, j, k);
//                 if (alpha <= IRL::global_constants::VF_LOW || alpha >= IRL::global_constants::VF_HIGH) continue;
//                 ++n_mixed;

//                 const double liq_vol = alpha * cell_volume;
//                 const IRL::Pt gt_bary = liquid_centroid(i, j, k);

//                 const IRL::Cylinder& rec = reconstructed(i, j, k);
//                 auto rec_cell = IRL::RectangularCuboid::fromBoundingPts(
//                     IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)), IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));
//                 auto rec_moments = IRL::getNormalizedVolumeMoments<IRL::VolumeMoments>(rec_cell, rec);

//                 const double rec_vol = rec_moments.volume();
//                 IRL::Pt re_bary;
//                 if (rec_vol > IRL::global_constants::VF_LOW * cell_volume) {
//                     re_bary = rec_moments.centroid();
//                 } else {
//                     re_bary = IRL::Pt(mesh.xm(i), mesh.ym(j), mesh.zm(k));
//                 }

//                 gt_global_bary[0] += liq_vol * gt_bary[0];
//                 gt_global_bary[1] += liq_vol * gt_bary[1];
//                 gt_global_bary[2] += liq_vol * gt_bary[2];
//                 re_global_bary[0] += liq_vol * re_bary[0];
//                 re_global_bary[1] += liq_vol * re_bary[1];
//                 re_global_bary[2] += liq_vol * re_bary[2];
//                 sum_liq_vol += liq_vol;

//                 const double err = std::sqrt(
//                     (gt_bary[0] - re_bary[0]) * (gt_bary[0] - re_bary[0]) +
//                     (gt_bary[1] - re_bary[1]) * (gt_bary[1] - re_bary[1]) +
//                     (gt_bary[2] - re_bary[2]) * (gt_bary[2] - re_bary[2]));

//                 sum_bary_err += err;
//                 max_bary_err  = std::max(max_bary_err, err);
//             }
//         }
//     }

//     std::cout << "  Writing visualization files...\n";

//     // 1. Write Ground Truth Torus (Explicit OBJ)
//     writeExactTorusOBJ(gt_torus, "viz_gt_torus.obj");

//     // 2. Write Ground Truth Local-Cylinders interface
//     {
//         VTKOutput vtk_gt("viz_gt", "viz_gt", mesh);
//         vtk_gt.addData("VOF", liquid_vf);
//         vtk_gt.writeVTKFile(/*time=*/0.0);
//         writeInterfaceVTK(liquid_vf, gt_field, /*time=*/0.0, vtk_gt);
//     }

//     // 3. Write Reconstructed interface
//     {
//         VTKOutput vtk_rc("viz_rc", "viz_rc", mesh);
//         vtk_rc.addData("VOF", liquid_vf);
//         vtk_rc.writeVTKFile(/*time=*/0.0);
//         writeInterfaceVTK(liquid_vf, reconstructed, /*time=*/0.0, vtk_rc);
//     }

//     std::cout << "    Reconstructed surface  → viz_rc/viz_rc_interface_0.vtu\n";
//     std::cout << "    Ground truth piecewise → viz_gt/viz_gt_interface_0.vtu\n";
//     std::cout << "    Exact Torus Geometry   → viz_gt_torus.obj\n\n";

//     if (n_mixed == 0) {
//         std::cerr << "  ERROR: No mixed-phase cells found.\n";
//         MPI_Finalize();
//         return 1;
//     }

//     if (sum_liq_vol > 0.0) {
//         gt_global_bary[0] /= sum_liq_vol;
//         gt_global_bary[1] /= sum_liq_vol;
//         gt_global_bary[2] /= sum_liq_vol;
//         re_global_bary[0] /= sum_liq_vol;
//         re_global_bary[1] /= sum_liq_vol;
//         re_global_bary[2] /= sum_liq_vol;
//     }

//     const double mean_bary_err = sum_bary_err / static_cast<double>(n_mixed);
//     const double threshold = 0.5 * dx;
//     const bool   passed    = (mean_bary_err < threshold);

//     std::cout << "  Results:\n";
//     std::cout << "    Mean per-cell bary error:  " << mean_bary_err << "  (" << mean_bary_err / dx << " * dx)\n";
//     std::cout << "    Max  per-cell bary error:  " << max_bary_err << "  (" << max_bary_err / dx << " * dx)\n\n";

//     if (passed) {
//         std::cout << "  PASS: mean barycenter error " << mean_bary_err << " < threshold " << threshold << "\n";
//     } else {
//         std::cout << "  FAIL: mean barycenter error " << mean_bary_err << " >= threshold " << threshold << "\n";
//     }
//     std::cout << "============================================================\n";

//     MPI_Finalize();
//     return passed ? 0 : 1;
// }