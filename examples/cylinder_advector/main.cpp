#include <mpi.h>
#include <chrono>
#include <iostream>
#include <string>

#include "examples/cylinder_advector/deformation_3d.h"
#include "examples/cylinder_advector/reconstruction_types.h"
#include "examples/cylinder_advector/solver.h"
#include "examples/cylinder_advector/translation_3d.h"
#include "examples/cylinder_advector/vof_advection.h"

static int startSimulation(const std::string& a_simulation_type,
                           const std::string& a_advection_method,
                           const std::string& a_reconstruction_method,
                           const double a_time_step_size,
                           const double a_time_duration,
                           const int a_viz_frequency);

int main(int argc, char* argv[]) {
  if (argc != 7) {
    std::cout << "Incorrect amount of command line arguments supplied. \n";
    std::cout << "Arguments should be \n";
    std::cout << "Simulation to run. Options: Deformation3D, Translation3D\n";
    std::cout << "Advection method. Options: SemiLagrangian, "
                 "SemiLagrangianCorrected, FullLagrangian\n";
    std::cout << "Reconstruction method. Options: PLIC, CentroidFit, Jibben\n";
    std::cout << "Time step size, dt (double)\n";
    std::cout << "Simulation duration(double)\n";
    std::cout
        << "Amount of time steps between visualization output (integer)\n";
    std::exit(-1);
  }

  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::string simulation_type = argv[1];
  std::string advection_method = argv[2];
  std::string reconstruction_method = argv[3];
  double time_step_size = std::stod(argv[4]);
  double time_duration = std::stod(argv[5]);
  int viz_frequency = atoi(argv[6]);

  auto start = std::chrono::system_clock::now();
  startSimulation(simulation_type, advection_method, reconstruction_method,
                  time_step_size, time_duration, viz_frequency);
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> runtime = end - start;
  if (rank == 0) {
    printf("Total run time: %20f \n\n", runtime.count());
  }

  MPI_Finalize();

  return 0;
}

static int startSimulation(const std::string& a_simulation_type,
                           const std::string& a_advection_method,
                           const std::string& a_reconstruction_method,
                           const double a_time_step_size,
                           const double a_time_duration,
                           const int a_viz_frequency) {
  if (a_simulation_type == "Deformation3D") {
    // return runSimulation<Deformation3D>(
    //     a_advection_method, a_reconstruction_method, a_time_step_size,
    //     a_time_duration, a_viz_frequency);
    return runSimulation2<Deformation3D>(
        a_advection_method, a_reconstruction_method, a_time_step_size,
        a_time_duration, a_viz_frequency);
  } else if (a_simulation_type == "Translation3D") {
    // return runSimulation<Translation3D>(
    //     a_advection_method, a_reconstruction_method, a_time_step_size,
    //     a_time_duration, a_viz_frequency);
    return runSimulation2<Translation3D>(
        a_advection_method, a_reconstruction_method, a_time_step_size,
        a_time_duration, a_viz_frequency);
  } else {
    std::cout << "Unknown simulation type of : " << a_simulation_type << '\n';
    std::cout << "Value entries are: Deformation3D, Translation3D. \n";
    std::exit(-1);
  }
  return -1;
}

// #include <iostream>
// #include <random>
// #include <chrono>
// #include <cmath>

// #include "examples/cylinder_advector/reconstruction_types.h"
// #include "examples/cylinder_advector/data.h"
// #include "examples/cylinder_advector/basic_mesh.h"
// #include "irl/cylinder_reconstruction/cylinder.h"
// #include "irl/moments/volume_moments.h"
// #include "irl/generic_cutting/generic_cutting.h"
// #include "examples/cylinder_advector/vtk.h"
// #include "examples/cylinder_advector/solver.h"

// // --- Test Configuration ---
// constexpr int NX = 5;
// constexpr int NY = 5;
// constexpr int NZ = 5;
// constexpr int GC = 5; // Using a larger ghost cell layer for the global reconstruction method
// constexpr IRL::Pt lower_domain(0.0, 0.0, 0.0);
// constexpr IRL::Pt upper_domain(1.0, 1.0, 1.0);

// /**
//  * @brief Generates a random, normalized 3D vector.
//  *
//  * This is used to create a random orientation for the cylinder. It uses a
//  * robust method of sampling from a normal distribution to ensure a uniform
//  * spherical distribution.
//  *
//  * @param gen The random number generator engine.
//  * @return A normalized IRL::Normal vector.
//  */
// IRL::Normal generateRandomNormal(std::mt19937& gen) {
//     std::uniform_real_distribution<> d(0.0, 1.0);
//     IRL::Normal n(d(gen), d(gen), d(gen));
//     n.normalize();
//     return n;
// }

// int main(int argc, char* argv[]) {
//     MPI_Init(&argc, &argv);
//     int rank, size;
//     MPI_Comm_size(MPI_COMM_WORLD, &size);
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     // ========================================================================
//     // 1. SETUP
//     // ========================================================================
//     // --- Initialize a random number generator ---
//     unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
//     std::mt19937 gen(seed);
//     std::uniform_real_distribution<double> distrib(-1.0, 2.0); // For datum

//     // --- Setup the computational mesh ---
//     BasicMesh mesh(NX, NY, NZ, GC);
//     mesh.setCellBoundaries(lower_domain, upper_domain);
//     IRL::setVolumeFractionBounds(1.0e-14);
//     load();

//     // --- Prepare data containers ---
//     Data<double> liquid_volume_fraction(&mesh);
//     Data<IRL::Pt> liquid_centroid(&mesh);
//     Data<IRL::Pt> gas_centroid(&mesh); // Not used by this method, but required by the function signature
//     Data<IRL::Cylinder> reconstructed_interface(&mesh);
//     Data<IRL::PlanarSeparator> reconstructed_interface2(&mesh);

//     std::ofstream error_file;
//     error_file.open("reconstruction_errors.txt");
//     error_file << "gt_radius,gt_datum_x,gt_datum_y,gt_datum_z,gt_dir_x,gt_dir_y,gt_dir_z,"
//                    << "radius_error,orientation_error,datum_error,bary_error,bary_error_plane,normal_error_plane\n";

//     IRL::Cylinder ground_truth_cylinder;
//     double ground_truth_radius;
//     IRL::Pt ground_truth_datum;
//     IRL::Normal ground_truth_direction;
//     for (int case_num = 0; case_num < 10000; ++case_num) {
//       std::cout << case_num + 1 << std::endl;
//       // ========================================================================
//       // 2. DEFINE GROUND TRUTH
//       // ========================================================================
//       //std::cout << "Generating ground truth cylinder..." << std::endl;
      
//       bool flag = true;
//       do {
//         // --- Generate a cylinder with random properties ---
//         const double ground_truth_radius = 0.16;//0.16/(4*(case_num+1));//0.04;
//         //if (case_num == 0)
//         {
//           ground_truth_datum = IRL::Pt(distrib(gen), distrib(gen), distrib(gen));//IRL::Pt(distrib(gen), distrib(gen), distrib(gen));
//           ground_truth_direction = generateRandomNormal(gen);
//         }

//         // --- Create the reference frame for the cylinder ---
//         IRL::Normal v1;
//         if (std::abs(ground_truth_direction[0]) > 1.0e-6) {
//             v1 = IRL::Normal(-ground_truth_direction[1], ground_truth_direction[0], 0.0);
//         } else {
//             v1 = IRL::Normal(0.0, ground_truth_direction[2], -ground_truth_direction[1]);
//         }
//         v1.normalize();
//         IRL::Normal v2 = crossProduct(ground_truth_direction, v1);
//         v2.normalize();
//         IRL::ReferenceFrame ground_truth_frame(ground_truth_direction, v1, v2);

//         // --- Construct the ground truth cylinder object ---
//         ground_truth_cylinder = IRL::Cylinder(ground_truth_datum, ground_truth_frame, 1.0, ground_truth_radius);
//         auto cell = IRL::RectangularCuboid::fromBoundingPts(
//                         IRL::Pt(mesh.x(2), mesh.y(2), mesh.z(2)),
//                         IRL::Pt(mesh.x(2 + 1), mesh.y(2 + 1), mesh.z(2 + 1)));
//                     auto moments = IRL::getNormalizedVolumeMoments<IRL::VolumeMoments>(
//                         cell, ground_truth_cylinder);
//         if (moments.volume() / cell.calculateVolume() > IRL::global_constants::VF_LOW && moments.volume() / cell.calculateVolume() < IRL::global_constants::VF_HIGH)
//         {
//           flag = false;
//         }
//       } while (flag);
      
//       // std::cout << "\n--- Ground Truth Parameters ---" << std::endl;
//       // std::cout << "Radius:      " << ground_truth_cylinder.getAlignedCylinder().r() << std::endl;
//       // std::cout << "Datum:       " << ground_truth_cylinder.getDatum() << std::endl;
//       // std::cout << "Orientation: " << ground_truth_cylinder.getReferenceFrame()[0] << std::endl;
//       // std::cout << "-----------------------------\n" << std::endl;


//       // ========================================================================
//       // 3. CALCULATE EXACT VOLUME MOMENTS FROM GROUND TRUTH
//       // ========================================================================
//       //std::cout << "Calculating exact volume moments from ground truth..." << std::endl;
      
//       // Loop over the entire mesh including ghost cells to provide neighborhood info
//       for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) {
//           for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) {
//               for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) {
//                   auto cell = IRL::RectangularCuboid::fromBoundingPts(
//                       IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)),
//                       IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));
//                   auto moments = IRL::getNormalizedVolumeMoments<IRL::VolumeMoments>(
//                       cell, ground_truth_cylinder);
//                   (liquid_volume_fraction)(i, j, k) =
//                       moments.volume() / cell.calculateVolume();
//                   (liquid_centroid)(i, j, k) = moments.centroid();

//                   (gas_centroid)(i, j, k)[0] = (mesh.xm(i) - moments.centroid()[0]*moments.volume()) / (1-moments.volume());
//                   (gas_centroid)(i, j, k)[1] = (mesh.ym(j) - moments.centroid()[1]*moments.volume()) / (1-moments.volume());
//                   (gas_centroid)(i, j, k)[2] = (mesh.zm(k) - moments.centroid()[2]*moments.volume()) / (1-moments.volume());

//                   if ((liquid_volume_fraction)(i, j, k) < IRL::global_constants::VF_LOW) {
//                     (liquid_volume_fraction)(i, j, k) = 0.0;
//                     (liquid_centroid)(i, j, k) = IRL::Pt(mesh.xm(i),mesh.ym(j),mesh.zm(k));
//                     (gas_centroid)(i, j, k) = IRL::Pt(mesh.xm(i),mesh.ym(j),mesh.zm(k));
//                   } else if ((liquid_volume_fraction)(i, j, k) > IRL::global_constants::VF_HIGH) {
//                     (liquid_volume_fraction)(i, j, k) = 1.0;
//                     (liquid_centroid)(i, j, k) = IRL::Pt(mesh.xm(i),mesh.ym(j),mesh.zm(k));
//                     (gas_centroid)(i, j, k) = IRL::Pt(mesh.xm(i),mesh.ym(j),mesh.zm(k));
//                   }
//               }
//           }
//       }

//       liquid_volume_fraction.updateBorder();
//       liquid_centroid.updateBorder();
//       gas_centroid.updateBorder();

//       // ========================================================================
//       // 4. RECONSTRUCT THE INTERFACE
//       // ========================================================================
//       //std::cout << "Reconstructing interface using 'Cylinder_Curve_Global'..." << std::endl;

//       // Use one of your reconstruction functions to generate an interface
//       Data<double> U(&mesh);
//       Data<double> V(&mesh);
//       Data<double> W(&mesh);
//       Cylinder_Curve_Global::getReconstruction(
//           liquid_volume_fraction, liquid_centroid, gas_centroid, 
//           0.0, // dt (not used)
//           U, V, W, // Velocities (not used)
//           &reconstructed_interface
//       );
//       PLIC_NET::getReconstruction(
//           liquid_volume_fraction, liquid_centroid, gas_centroid, 
//           0.0, // dt (not used)
//           U, V, W, // Velocities (not used)
//           &reconstructed_interface2
//       );

//       // ========================================================================
//       // 5. CALCULATE AND REPORT ERRORS
//       // ========================================================================
//       //std::cout << "Calculating reconstruction errors..." << std::endl;

//       double total_radius_error = 0.0;
//       double total_orientation_error = 0.0;
//       double total_datum_error = 0.0;
//       int mixed_cell_count = 0;
//       double bary_error = 0;
//       double bary_error_plane = 0;
//       double normal_error_plane = 0;

//       // Loop over the inner domain (non-ghost cells)
//       //for (int i = mesh.imin(); i <= mesh.imax(); ++i) {
//           //for (int j = mesh.jmin(); j <= mesh.jmax(); ++j) {
//               //for (int k = mesh.kmin(); k <= mesh.kmax(); ++k) {
//                   const double vf = liquid_volume_fraction(2, 2, 2);

//                   // Only evaluate error in cells that are actually cut by the interface
//                   if (vf > IRL::global_constants::VF_LOW && vf < IRL::global_constants::VF_HIGH) {
//                       mixed_cell_count++;
//                       const IRL::Cylinder& reconstructed_cyl = reconstructed_interface(2, 2, 2);
//                       auto cell = IRL::RectangularCuboid::fromBoundingPts(
//                         IRL::Pt(mesh.x(2), mesh.y(2), mesh.z(2)),
//                         IRL::Pt(mesh.x(2 + 1), mesh.y(2 + 1), mesh.z(2 + 1)));
//                       auto moments = IRL::getVolumeMoments<IRL::AddSurfaceOutput<
//                           IRL::VolumeMoments, IRL::CylinderParametrizedSurfaceOutput>>(
//                       cell, ground_truth_cylinder);
//                       auto n = moments.getSurface().getAverageNormalNonAligned();
//                       IRL::Pt bary = moments.getMoments().centroid()/moments.getMoments().volume();
//                       moments = IRL::getVolumeMoments<IRL::AddSurfaceOutput<
//                           IRL::VolumeMoments, IRL::CylinderParametrizedSurfaceOutput>>(
//                       cell, reconstructed_cyl);
//                       IRL::Pt bary2 = moments.getMoments().centroid()/moments.getMoments().volume();
//                       bary_error = sqrt(pow(bary[0]-bary2[0],2.0) + pow(bary[1]-bary2[1],2.0) + pow(bary[2]-bary2[2],2.0));

//                       const IRL::PlanarSeparator& reconstructed_plane = reconstructed_interface2(2, 2, 2);
//                       auto moments2 = IRL::getNormalizedVolumeMoments<IRL::VolumeMoments>(
//                       cell, reconstructed_plane);
//                       bary2 = moments2.centroid();
//                       bary_error_plane = sqrt(pow(bary[0]-bary2[0],2.0) + pow(bary[1]-bary2[1],2.0) + pow(bary[2]-bary2[2],2.0));
//                       normal_error_plane = sqrt(pow(n[0]-reconstructed_plane[0].normal()[0],2.0) + pow(n[1]-reconstructed_plane[0].normal()[1],2.0) + pow(n[2]-reconstructed_plane[0].normal()[2],2.0));

//                       // --- Radius Error ---
//                       const double r_recon = reconstructed_cyl.getAlignedCylinder().r();
//                       total_radius_error += std::pow((r_recon - ground_truth_cylinder.getAlignedCylinder().r())/ground_truth_cylinder.getAlignedCylinder().r(),2.0);
                      
//                       // --- Orientation Error (angle between vectors in radians) ---
//                       const IRL::Normal& n_recon = reconstructed_cyl.getReferenceFrame()[0];
//                       // Use abs() because the direction can be flipped (-n is same as n)
//                       const double cos_theta = std::abs(std::abs(n_recon[0])*ground_truth_cylinder.getReferenceFrame()[0][0] + std::abs(n_recon[1])*ground_truth_cylinder.getReferenceFrame()[0][1] + std::abs(n_recon[2])*ground_truth_cylinder.getReferenceFrame()[0][2]);
//                       // Clamp the value to avoid domain errors with acos
//                       total_orientation_error += std::acos(std::min(1.0, cos_theta));

//                       // --- Datum Error (perpendicular distance to the ground truth axis) ---
//                       const IRL::Pt& datum_recon = reconstructed_cyl.getDatum();
//                       IRL::Vec3<double> vec_to_datum = datum_recon - ground_truth_cylinder.getDatum();
//                       IRL::Vec3<double> cross_prod = crossProduct(vec_to_datum, static_cast<IRL::Vec3<double>>(ground_truth_cylinder.getReferenceFrame()[0]));
//                       total_datum_error += magnitude(cross_prod);
//                   }
//               //}
//           //}
//       //}

//       // --- Final Report ---
//       //std::cout << "\n--- Reconstruction Error Report ---" << std::endl;
//       if (mixed_cell_count > 0) {
//         error_file << ground_truth_cylinder.getAlignedCylinder().r() << ","
//                        << ground_truth_cylinder.getDatum()[0] << "," << ground_truth_cylinder.getDatum()[1] << "," << ground_truth_cylinder.getDatum()[2] << ","
//                        << ground_truth_cylinder.getReferenceFrame()[0][0] << "," << ground_truth_cylinder.getReferenceFrame()[0][1] << "," << ground_truth_cylinder.getReferenceFrame()[0][2] << ","
//                        << total_radius_error << "," << total_orientation_error << "," << total_datum_error << "," << bary_error << "," << bary_error_plane << "," << normal_error_plane << "\n";
//           // std::cout << "Radius " << reconstructed_interface(2,2,2).getAlignedCylinder().r() << std::endl;
//           // std::cout << "Orientation " << reconstructed_interface(2,2,2).getReferenceFrame()[0] << std::endl;
//           // std::cout << "Datum " << reconstructed_interface(2,2,2).getDatum() << std::endl;
//           // std::cout << "Average Radius Error:      " << total_radius_error / mixed_cell_count << std::endl;
//           // std::cout << "Average Orientation Error: " << total_orientation_error / mixed_cell_count << std::endl;
//           // std::cout << "Average Datum Error:       " << total_datum_error / mixed_cell_count << std::endl;
//       } else {
//           //std::cout << "No mixed-phase cells found in the domain. The cylinder may be too small or outside the domain." << std::endl;
//       }
//       //std::cout << "---------------------------------" << std::endl;
//       for (int i = mesh.imin(); i <= mesh.imax(); ++i) {
//           for (int j = mesh.jmin(); j <= mesh.jmax(); ++j) {
//               for (int k = mesh.kmin(); k <= mesh.kmax(); ++k) {
//                 if (i!=2||j!=2||k!=2) 
//                 {
//                   reconstructed_interface(i,j,k) = IRL::Cylinder();
//                   //reconstructed_interface2(i,j,k).fromOnePlane(IRL::Plane(IRL::Normal(0.0,0.0,0.0),10000));
//                 }
//               }
//           }
//       }
//         VTKOutput vtk_io("viz_out", "viz", mesh);
//         vtk_io.addData("VOF", liquid_volume_fraction);
//         vtk_io.addData("VelocityX", U);
//         vtk_io.addData("VelocityY", V);
//         vtk_io.addData("VelocityZ", W);
//         double simulation_time = 0.0;
//         int iteration = 0;
//         vtk_io.writeVTKFile(simulation_time);
//         writeInterfaceToFile(liquid_volume_fraction, reconstructed_interface, simulation_time, &vtk_io);
//         //writeInterfaceToFile(liquid_volume_fraction, reconstructed_interface2, simulation_time, &vtk_io);
//         simulation_time = 1.0;
//         iteration = 1;
//             for (int i = mesh.imin(); i <= mesh.imax(); ++i) {
//           for (int j = mesh.jmin(); j <= mesh.jmax(); ++j) {
//               for (int k = mesh.kmin(); k <= mesh.kmax(); ++k) {
//                 reconstructed_interface(i,j,k) = ground_truth_cylinder;
//               }
//           }
//       }
//         vtk_io.writeVTKFile(simulation_time);
//         writeInterfaceToFile(liquid_volume_fraction, reconstructed_interface, simulation_time,
//           &vtk_io);
//     }

//     return 0;
// }