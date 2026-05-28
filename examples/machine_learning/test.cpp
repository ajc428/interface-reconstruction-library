#include "mpi.h"
#include <math.h>
#include <iostream>
#include "irl/machine_learning_reconstruction/trainer.h"
#include "irl/machine_learning_reconstruction/data_gen.h"

using namespace std;

void create_surface(string name, double x, double y, double z, double alpha, double beta, double gamma, double a, double b)
{
    const auto bottom_corner = IRL::Pt(-1.5, -1.5, -1.5);
    const auto top_corner = IRL::Pt(1.5, 1.5, 1.5);
    const auto cell = IRL::StoredRectangularCuboid<IRL::Pt>::fromBoundingPts(bottom_corner, top_corner);
    IRL::moments_gen *gen = new IRL::moments_gen(1, 1, 1, 3, 3, 3, -1.5, -1.5, -1.5);
    IRL::Paraboloid p1 = gen->new_paraboloid(x,y,z,alpha,beta,gamma,a,b);

    const auto first_moments_and_surface = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParaboloidParametrizedSurfaceOutput>, IRL::HalfEdgeCutting>(cell, p1);
    auto surface = first_moments_and_surface.getSurface();
    const double length_scale = 0.05;
    IRL::TriangulatedSurfaceOutput triangulated_surface = first_moments_and_surface.getSurface().triangulate(length_scale);
    triangulated_surface.write(name);
}

void data_generate(int num, double rota_l, double rota_h, double rotb_l, double rotb_h, double rotc_l, double rotc_h, double coa_l, double coa_h, double cob_l, double cob_h, double ox_l, double ox_h, double oy_l, double oy_h, double oz_l, double oz_h)
{
    IRL::data_gen gen(num,3,3,3,1,1,1,-1.5,-1.5,-1.5);
    gen.generate(rota_l, rota_h, rotb_l, rotb_h, rotc_l, rotc_h, coa_l, coa_h, cob_l, cob_h, ox_l, ox_h, oy_l, oy_h, oz_l, oz_h, true);
}

/***********************
trainer(epochs, data size, learning rate, OPTION)
Trainer options:
0: PLIC-Net
1: Binary Classification
2: Classification
************************/

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank;
    int numranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numranks);      

    //create_surface("test_surface_", -0.12341,-0.411455,-0.246806,3.33333,3.55152,0.615339,484230,0);
    //create_surface("test_surface2_", 0, 0, 0, M_PI/2, M_PI/2, 0, 1, 1);
    //create_surface("test_surface2",0,0,0,0,0,0,0.11,0.11);

    //create_surface_cylinder("test_surface",0,0,0,0,0,0,0.0625,1);

    int select;
    if (rank == 0)
    {
        std::cout << "Generate data: Enter 1.  Train NN: Enter 2 " << std::endl;
        std::cin >> select;
    }
    MPI_Bcast(&select, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (select == 1)
    {
        int num;
        if (rank == 0)
        {
            std::cout << "Enter number of data to generate: " << std::endl;
            std::cin >> num;
        }
        MPI_Bcast(&num, 1, MPI_INT, 0, MPI_COMM_WORLD);
        data_generate(num,0,2*M_PI,0,2*M_PI,0,2*M_PI,-2,2,-2,2,-0.5,0.5,-0.5,0.5,-0.5,0.5);
    }
    else
    {
        int ep;
        int num;
        double lr;
        int ty;
        int lo;
        int val;
        if (rank == 0)
        {
            std::cout << "Enter number of epochs: " << std::endl;
            std::cin >> ep;
            std::cout << "Enter number of training data: " << std::endl;
            std::cin >> num;
            std::cout << "Enter number of validation data: " << std::endl;
            std::cin >> val;
            std::cout << "Enter learning rate: " << std::endl;
            std::cin >> lr;
            std::cout << "Enter type: " << std::endl;
            std::cin >> ty;
            std::cout << "Reload: Enter 1 for true, 0 for false: " << std::endl;
            std::cin >> lo;
        }
        MPI_Bcast(&ep, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&num, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&val, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&lr, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&ty, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&lo, 1, MPI_INT, 0, MPI_COMM_WORLD);
        bool load = false;
        if (lo == 1) load = true;

        auto t = IRL::trainer(ep, num, lr, ty);
        t.load_train_data("moments.txt", "normals.txt");
        t.load_validation_data("moments_val.txt", "normals_val.txt", val);
        t.load_test_data("moments_test.txt", "normals_test.txt");
        t.train_model(load, "model.pt", "model.pt");
        //t.load_model("model.pt");
        t.test_model();
    }

    MPI_Finalize();
}