#include "mpi.h"
#include <iostream>
#include "irl/machine_learning_reconstruction/trainer.h"
#include "irl/machine_learning_reconstruction/data_gen.h"

using namespace std;

void create_surface(string name, double x, double y, double z, double alpha, double beta, double gamma, double a, double b)
{
    const auto bottom_corner = IRL::Pt(-0.5, -0.5, -0.5);
    const auto top_corner = IRL::Pt(0.5, 0.5, 0.5);
    const auto cell = IRL::StoredRectangularCuboid<IRL::Pt>::fromBoundingPts(bottom_corner, top_corner);
    IRL::fractions *gen = new IRL::fractions(3);
    IRL::Paraboloid p1 = gen->new_parabaloid(x,y,z,alpha,beta,gamma,a,b);
    const auto first_moments_and_surface = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParametrizedSurfaceOutput>, IRL::HalfEdgeCutting>(cell, p1);
    const double length_scale = 0.05;
    IRL::TriangulatedSurfaceOutput triangulated_surface = first_moments_and_surface.getSurface().triangulate(length_scale);
    triangulated_surface.write(name);
}

void data_generate(int num, int type, double rota_l, double rota_h, double rotb_l, double rotb_h, double rotc_l, double rotc_h, double coa_l, double coa_h, double cob_l, double cob_h, double ox_l, double ox_h, double oy_l, double oy_h, double oz_l, double oz_h)
{
    IRL::data_gen gen(3,num, type);
    gen.generate(rota_l, rota_h, rotb_l, rotb_h, rotc_l, rotc_h, coa_l, coa_h, cob_l, cob_h, ox_l, ox_h, oy_l, oy_h, oz_l, oz_h);
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    //create_surface("test_surface",0.399235,0.0907319,0.275013,1.07142,3.87079,5.85515,2.37561,0.01);
    
    //data_generate(2000,1,0,2*3.1415,0,2*3.1415,0,2*3.1415,2,10,0.01,0.1,-10,10,-10,10,-10,10);

    auto t = IRL::trainer(10000, 8000, 0.0001, 4);
    t.load_train_data("fractions.txt", "type.txt");
    t.load_test_data("fractions_test.txt", "type_test.txt");
    t.train_model(true, "model.pt", "model.pt");
    t.test_model(4);
}