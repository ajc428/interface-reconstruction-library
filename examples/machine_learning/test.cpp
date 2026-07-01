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

void data_generate(int num, double rota_l, double rota_h, double rotb_l, double rotb_h, double rotc_l, double rotc_h, double coa_l, double coa_h, double cob_l, double cob_h, double ox_l, double ox_h, double oy_l, double oy_h, double oz_l, double oz_h, bool disturb, std::string name)
{
    IRL::data_gen gen(num,3,3,3,1,1,1,-1.5,-1.5,-1.5);
    gen.generate(rota_l, rota_h, rotb_l, rotb_h, rotc_l, rotc_h, coa_l, coa_h, cob_l, cob_h, ox_l, ox_h, oy_l, oy_h, oz_l, oz_h, disturb, name);
}

void data_generate_sheet(int num, double coa_l, double coa_h, double cob_l, double cob_h, double t_l, double t_h, bool disturb, std::string name)
{
    IRL::data_gen gen(num,3,3,3,1,1,1,-1.5,-1.5,-1.5);
    gen.generate_sheet(coa_l, coa_h, cob_l, cob_h, t_l, t_h, disturb, name);
}

/***********************
trainer(epochs, data size, learning rate, OPTION)
Trainer options:
0: PLIC-Net
1: R2P-Net
2: Binary Classification
3: Classification
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
        double alpha;
        double beta;
        double alpha1;
        double beta1;
        if (rank == 0)
        {
            std::cout << "Enter number of data to generate: " << std::endl;
            std::cin >> num;
            std::cout << "Enter alpha low: " << std::endl;
            std::cin >> alpha;
            std::cout << "Enter alpha high: " << std::endl;
            std::cin >> alpha1;
            std::cout << "Enter beta low: " << std::endl;
            std::cin >> beta;
            std::cout << "Enter beta high: " << std::endl;
            std::cin >> beta1;
        }
        MPI_Bcast(&num, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        data_generate_sheet(num*0.15,alpha,alpha1,beta,beta1,0.01,1,false,"_test");
        data_generate_sheet(num*0.15,alpha,alpha1,beta,beta1,0.01,1,false,"_val");
        data_generate_sheet(num*0.7,alpha,alpha1,beta,beta1,0.01,1,false,"");
        // data_generate(num,0,2*M_PI,0,2*M_PI,0,2*M_PI,alpha,alpha1,beta,beta1,-0.5,0.5,-0.5,0.5,-0.5,0.5,true,"_test");
        // data_generate(num,0,2*M_PI,0,2*M_PI,0,2*M_PI,alpha,alpha1,beta,beta1,-0.5,0.5,-0.5,0.5,-0.5,0.5,true,"_val");
        // data_generate(num*14.0/3.0,0,2*M_PI,0,2*M_PI,0,2*M_PI,alpha,alpha1,beta,beta1,-0.5,0.5,-0.5,0.5,-0.5,0.5,true,"");
        
        // data_generate(num,0,2*M_PI,0,2*M_PI,0,2*M_PI,2,2,2,2,-0.5,0.5,-0.5,0.5,-0.5,0.5,false,"_2");
        // data_generate(num,0,2*M_PI,0,2*M_PI,0,2*M_PI,1,1,1,1,-0.5,0.5,-0.5,0.5,-0.5,0.5,false,"_1");
        // data_generate(num,0,2*M_PI,0,2*M_PI,0,2*M_PI,0.5,0.5,0.5,0.5,-0.5,0.5,-0.5,0.5,-0.5,0.5,false,"_05");
        // data_generate(num,0,2*M_PI,0,2*M_PI,0,2*M_PI,0.25,0.25,0.25,0.25,-0.5,0.5,-0.5,0.5,-0.5,0.5,false,"_025");
        // data_generate(num,0,2*M_PI,0,2*M_PI,0,2*M_PI,0.125,0.125,0.125,0.125,-0.5,0.5,-0.5,0.5,-0.5,0.5,false,"_0125");
        // data_generate(num,0,2*M_PI,0,2*M_PI,0,2*M_PI,0.0625,0.0625,0.0625,0.0625,-0.5,0.5,-0.5,0.5,-0.5,0.5,false,"_00625");
        // data_generate(num,0,2*M_PI,0,2*M_PI,0,2*M_PI,0.03125,0.03125,0.03125,0.03125,-0.5,0.5,-0.5,0.5,-0.5,0.5,false,"_003125");
        // data_generate(num,0,2*M_PI,0,2*M_PI,0,2*M_PI,0.015625,0.015625,0.015625,0.015625,-0.5,0.5,-0.5,0.5,-0.5,0.5,false,"_0015625");
        // data_generate(num,0,2*M_PI,0,2*M_PI,0,2*M_PI,0.0078125,0.0078125,0.0078125,0.0078125,-0.5,0.5,-0.5,0.5,-0.5,0.5,false,"_00078125");
        // data_generate(num,0,2*M_PI,0,2*M_PI,0,2*M_PI,0.00390625,0.00390625,0.00390625,0.00390625,-0.5,0.5,-0.5,0.5,-0.5,0.5,false,"_000390625");
        // data_generate(num,0,2*M_PI,0,2*M_PI,0,2*M_PI,0.001953125,0.001953125,0.001953125,0.001953125,-0.5,0.5,-0.5,0.5,-0.5,0.5,false,"_0001953125");
        // data_generate(num,0,2*M_PI,0,2*M_PI,0,2*M_PI,0.0009765625,0.0009765625,0.0009765625,0.0009765625,-0.5,0.5,-0.5,0.5,-0.5,0.5,false,"_00009765625");
        // data_generate(num,0,2*M_PI,0,2*M_PI,0,2*M_PI,0,0,0,0,-0.5,0.5,-0.5,0.5,-0.5,0.5,false,"_flat");
    }
    else
    {
        int ep;
        int num;
        double lr;
        int ty;
        int lo;
        int val;
        int test;

        if (rank == 0)
        {
            std::cout << "Enter 1 to train, 2 to test: " << std::endl;
            std::cin >> test;
        }
        MPI_Bcast(&test, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (test == 1)
        {
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
            t.test_model("result_ex.txt","result_pr.txt");
        }
        else
        {
            if (rank == 0)
            {
                std::cout << "Enter number of data: " << std::endl;
                std::cin >> num;
            }
            MPI_Bcast(&num, 1, MPI_INT, 0, MPI_COMM_WORLD);

            auto t = IRL::trainer(1,num,1,0);
            t.load_test_data("moments.txt", "normals.txt");
            t.load_model("model.pt");
            t.test_model("result_ex.txt","result_pr.txt");
            // t.load_model("model.pt");
            // t.load_test_data("moments_2.txt", "normals_2.txt");
            // t.test_model("result_2_ex.txt","result_2_pr.txt");

            // t.load_test_data("moments_1.txt", "normals_1.txt");
            // t.test_model("result_1_ex.txt","result_1_pr.txt");

            // t.load_test_data("moments_05.txt", "normals_05.txt");
            // t.test_model("result_05_ex.txt","result_05_pr.txt");

            // t.load_test_data("moments_025.txt", "normals_025.txt");
            // t.test_model("result_025_ex.txt","result_025_pr.txt");

            // t.load_test_data("moments_0125.txt", "normals_0125.txt");
            // t.test_model("result_0125_ex.txt","result_0125_pr.txt");

            // t.load_test_data("moments_00625.txt", "normals_00625.txt");
            // t.test_model("result_00625_ex.txt","result_00625_pr.txt");

            // t.load_test_data("moments_003125.txt", "normals_003125.txt");
            // t.test_model("result_003125_ex.txt","result_003125_pr.txt");

            // t.load_test_data("moments_0015625.txt", "normals_0015625.txt");
            // t.test_model("result_0015625_ex.txt","result_0015625_pr.txt");

            // t.load_test_data("moments_00078125.txt", "normals_00078125.txt");
            // t.test_model("result_00078125_ex.txt","result_00078125_pr.txt");

            // t.load_test_data("moments_000390625.txt", "normals_000390625.txt");
            // t.test_model("result_000390625_ex.txt","result_000390625_pr.txt");
            
            // t.load_test_data("moments_0001953125.txt", "normals_0001953125.txt");
            // t.test_model("result_0001953125_ex.txt","result_0001953125_pr.txt");

            // t.load_test_data("moments_00009765625.txt", "normals_00009765625.txt");
            // t.test_model("result_00009765625_ex.txt","result_00009765625_pr.txt");

            // t.load_test_data("moments_flat.txt", "normals_flat.txt");
            // t.test_model("result_flat_ex.txt","result_flat_pr.txt");
        }
    }

    MPI_Finalize();
}