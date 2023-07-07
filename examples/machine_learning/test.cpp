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

    /*IRL::Pt datum;
    IRL::ReferenceFrame frame;
    auto t = IRL::trainer(10000, 2000, 0.0001, 6);
    t.load_train_data("fractions.txt", "coefficients.txt");
    t.load_test_data("fractions.txt", "coefficients.txt");
    frame = t.getFrame(1);
    datum = IRL::Pt(x,y,z);
    IRL::Paraboloid p1 = IRL::Paraboloid(datum, frame, a, b);*/

    const auto first_moments_and_surface = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParametrizedSurfaceOutput>, IRL::HalfEdgeCutting>(cell, p1);
    const double length_scale = 0.05;
    IRL::TriangulatedSurfaceOutput triangulated_surface = first_moments_and_surface.getSurface().triangulate(length_scale);
    triangulated_surface.write(name);
}

void data_generate(int num, double rota_l, double rota_h, double rotb_l, double rotb_h, double rotc_l, double rotc_h, double coa_l, double coa_h, double cob_l, double cob_h, double ox_l, double ox_h, double oy_l, double oy_h, double oz_l, double oz_h)
{
    IRL::data_gen gen(3,num);
    gen.generate(rota_l, rota_h, rotb_l, rotb_h, rotc_l, rotc_h, coa_l, coa_h, cob_l, cob_h, ox_l, ox_h, oy_l, oy_h, oz_l, oz_h);
}

/***********************
trainer(epochs, data size, learning rate, OPTION)
Trainer options:
0: Predict paraboloid, training with coefficients
1: Predict paraboloid, training with volume fractions, analytical gradients (broken)
2: Predict paraboloid, training with volume fractions, Finite Difference (This is the default) 
3: Classification, sheet vs. ligament, uses 3 invariants
4: Predict paraboloid curvature, training with principal curvatures
************************/

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    //create_surface("test_surface2",-0.128248,0.444811,0.232889,5.07745,3.27663,5.71901,6.08245,3.45158);
    
    //data_generate(2000,0,2*3.1415,0,2*3.1415,0,2*3.1415,0.5,10,0.5,10,-0.5,0.5,-0.5,0.5,-0.5,0.5);

    auto t = IRL::trainer(10000, 2000, 0.0001, 6);
    t.load_train_data("fractions.txt", "coefficients.txt");
    t.load_test_data("fractions.txt", "coefficients.txt");
    t.train_model(false, "model.pt", "model.pt");
    t.test_model(3);
    /*IRL::ReferenceFrame f = t.getFrame(1);
    std::cout << f[0] << std::endl << f[1] << std::endl << f[2] << std::endl << std::endl;

    IRL::ReferenceFrame frame;
    std::array<double, 3> angle;
    frame = IRL::ReferenceFrame(IRL::Normal(1.0, 0.0, 0.0), IRL::Normal(0.0, 1.0, 0.0), IRL::Normal(0.0, 0.0, 1.0));
    angle = {5.07745,3.27663,5.71901};
    IRL::UnitQuaternion x_rotation(angle[0], frame[0]);
    IRL::UnitQuaternion y_rotation(angle[1], frame[1]);
    IRL::UnitQuaternion z_rotation(angle[2], frame[2]);
    frame = x_rotation * y_rotation * z_rotation * frame;
    std::cout << frame[0] << std::endl << frame[1] << std::endl << frame[2] << std::endl << std::endl;*/
}