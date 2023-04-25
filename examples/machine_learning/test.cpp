#include "mpi.h"
#include <iostream>
#include "irl/machine_learning_reconstruction/trainer.h"
#include "irl/machine_learning_reconstruction/data_gen.h"

using namespace std;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    
    /*IRL::data_gen gen(3,10000);
    gen.generate();*/
    auto t = IRL::trainer(1000, 1000, 0.001);
    t.load_train_data("fractions.txt", "coefficients.txt");
    t.load_test_data("fractions.txt", "coefficients.txt");
    t.train_model(2, false, "model.pt", "model.pt");
    t.test_model(2);
}