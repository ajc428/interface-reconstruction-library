#include "mpi.h"
#include <math.h>
#include <iostream>
#include "irl/machine_learning_reconstruction/trainer.h"
#include "irl/machine_learning_reconstruction/trainer_cuda.h"
#include "irl/machine_learning_reconstruction/data_gen.h"

using namespace std;

void create_surface(string name, double x, double y, double z, double alpha, double beta, double gamma, double a, double b)
{
    //for (int i = 0; i < 3; ++i)
    {
        //for (int j = 0; j < 3; ++j)
        {
            //for (int k = 0; k < 3; ++k)
            {
                const auto bottom_corner = IRL::Pt(-1.5, -1.5, -1.5);
                const auto top_corner = IRL::Pt(1.5, 1.5, 1.5);
                const auto cell = IRL::StoredRectangularCuboid<IRL::Pt>::fromBoundingPts(bottom_corner, top_corner);
                IRL::fractions *gen = new IRL::fractions(3);
                IRL::Paraboloid p1 = gen->new_parabaloid(x,y,z,alpha,beta,gamma,a,b);
                //IRL::ReferenceFrame frame = IRL::ReferenceFrame(IRL::Pt(-0.861205976379694, -0.153148365815364, -0.484633721789954), IRL::Pt(-0.217406642015315, -0.750885257422074, 0.623623028915554), IRL::Pt(-0.459411164704912, 0.64243046957182, 0.613371399325244));
                //IRL::Paraboloid p1 = gen->new_parabaloid(x,y,z,frame,a,b);

                const auto first_moments_and_surface = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParametrizedSurfaceOutput>, IRL::HalfEdgeCutting>(cell, p1);
                auto surface = first_moments_and_surface.getSurface();
                const double length_scale = 0.05;
                IRL::TriangulatedSurfaceOutput triangulated_surface = first_moments_and_surface.getSurface().triangulate(length_scale);
                //string name2 = name + std::to_string(i)+std::to_string(j)+std::to_string(k);
                triangulated_surface.write(name);
            }
        }
    }
}

void data_generate(int num, double rota_l, double rota_h, double rotb_l, double rotb_h, double rotc_l, double rotc_h, double coa_l, double coa_h, double cob_l, double cob_h, double ox_l, double ox_h, double oy_l, double oy_h, double oz_l, double oz_h)
{
    IRL::data_gen gen(3,num);
    //gen.generate(rota_l, rota_h, rotb_l, rotb_h, rotc_l, rotc_h, coa_l, coa_h, cob_l, cob_h, ox_l, ox_h, oy_l, oy_h, oz_l, oz_h, true);
    //gen.generate_with_disturbance2(rota_l, rota_h, rotb_l, rotb_h, rotc_l, rotc_h, coa_l, coa_h, cob_l, cob_h, ox_l, ox_h, oy_l, oy_h, oz_l, oz_h);
    gen.generate_with_disturbance(rota_l, rota_h, rotb_l, rotb_h, rotc_l, rotc_h, coa_l, coa_h, cob_l, cob_h, ox_l, ox_h, oy_l, oy_h, oz_l, oz_h, true);
    //gen.generate_two_paraboloids(rota_l, rota_h, rotb_l, rotb_h, rotc_l, rotc_h, coa_l, coa_h, cob_l, cob_h, ox_l, ox_h, oy_l, oy_h, oz_l, oz_h, true);
    //gen.generate_two_paraboloids_with_disturbance(rota_l, rota_h, rotb_l, rotb_h, rotc_l, rotc_h, coa_l, coa_h, cob_l, cob_h, ox_l, ox_h, oy_l, oy_h, oz_l, oz_h, true);
    //gen.generate_two_paraboloids_in_cell(rota_l, rota_h, rotb_l, rotb_h, rotc_l, rotc_h, coa_l, coa_h, cob_l, cob_h, ox_l, ox_h, oy_l, oy_h, oz_l, oz_h, true);
    //gen.generate_noise(rota_l, rota_h, rotb_l, rotb_h, rotc_l, rotc_h);
    //gen.generate_paraboloid_with_plane(rota_l, rota_h, rotb_l, rotb_h, rotc_l, rotc_h, coa_l, coa_h, cob_l, cob_h, ox_l, ox_h, oy_l, oy_h, oz_l, oz_h);
    //gen.generate_two_planes(rota_l, rota_h, rotb_l, rotb_h, rotc_l, rotc_h, ox_l, ox_h, oy_l, oy_h, oz_l, oz_h);
    //gen.generate_plane_with_paraboloid(rota_l, rota_h, rotb_l, rotb_h, rotc_l, rotc_h, coa_l, coa_h, cob_l, cob_h, ox_l, ox_h, oy_l, oy_h, oz_l, oz_h);
}

void data_generate_planes(int num, double rota1_l, double rota1_h, double rotb1_l, double rotb1_h, double rota2_l, double rota2_h, double rotb2_l, double rotb2_h, double d1_l, double d1_h, double d2_l, double d2_h, bool R2P, bool inter, bool same)
{
    IRL::data_gen gen(3,num);
    //gen.generate_plane(rota1_l, rota1_h, rotb1_l, rotb1_h, d1_l, d1_h, R2P);
    //gen.generate_plane_with_disturbance(rota1_l, rota1_h, rotb1_l, rotb1_h, d1_l, d1_h, R2P);
    gen.generate_R2P(rota1_l, rota1_h, rotb1_l, rotb1_h, rota2_l, rota2_h, rotb2_l, rotb2_h, d1_l, d1_h, d2_l, d2_h, inter, same);
    //gen.generate_R2P_with_disturbance(rota1_l, rota1_h, rotb1_l, rotb1_h, rota2_l, rota2_h, rotb2_l, rotb2_h, d1_l, d1_h, d2_l, d2_h, inter, same);
}

/***********************
trainer(epochs, data size, learning rate, OPTION)
Trainer options:
0: Predict paraboloid, training with coefficients
1: Predict surface normal
2: Predict R2P
3: Classification
************************/

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    //create_surface("test_surface_", -22.7285,23.6557,-33.7081,0.608029,3.67144,5.9367,27.3659,25.7638);
    //create_surface("test_surface2_", 0, 0, 0, M_PI/2, M_PI/2, 0, 1, 1);
    //create_surface("test_surface2",0,0,0,0,0,0,0.11,0.11);
    
    //data_generate(140000,0,2*M_PI,0,2*M_PI,0,2*M_PI,-2,2,-2,2,-0.5,0.5,-0.5,0.5,-0.5,0.5);
    //data_generate_planes(1000,0,2*M_PI,0,2*M_PI,0,2*M_PI,0,2*M_PI,-1,1,-1,1,true, false, true);
    //data_generate_planes(1000,0,2*M_PI,0,2*M_PI,0,2*M_PI,0,2*M_PI,-1,1,-1,1,true, true, true);

    auto t = IRL::cnn_trainer(10000, 2000, 0.0001, 6);
    t.load_train_data("fractions.txt", "type.txt");
    t.load_validation_data("fractions.txt", "type.txt", 2000);
    t.load_test_data("fractions.txt", "type.txt");
    t.train_model(false, "model.pt", "model.pt");
    //t.load_model("model.pt");
    t.test_model(3);
}