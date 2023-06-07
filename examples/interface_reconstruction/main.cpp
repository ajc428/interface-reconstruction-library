#include "reconstruct.h"
#include <fstream>
#include <iostream>
using namespace std;

BasicMesh initializeMesh(const int a_number_of_cells) 
{
    constexpr const int a_number_of_ghost_cells = 1;
    BasicMesh mesh(a_number_of_cells, a_number_of_cells, a_number_of_cells, a_number_of_ghost_cells);
    auto nx = mesh.getNx();
    auto ny = mesh.getNy();
    auto nz = mesh.getNz();
    IRL::Pt lower_domain(-0.5 * nx, -0.5 * ny, -0.5 * nz);
    IRL::Pt upper_domain(0.5 * nx, 0.5 * ny, 0.5 * nz);
    mesh.setCellBoundaries(lower_domain, upper_domain);
    return mesh;
}

vector<vector<double>> read_data(string file, int data_size)
{
    ifstream indata;
    vector<vector<double>> output;
    vector <double> num;
    indata.open(file);
    string line, value;
    int i = 0;
    while (getline(indata, line) && i < data_size)
    {
        num.clear();
        stringstream str(line);
        while (getline(str, value, ','))
        {
            num.push_back(stod(value));
        }
        output.push_back(num);
        ++i;
    }
    indata.close();

    return output;
}

int main(int argc, char* argv[])
{
    int data_size = 100000;
    vector<vector<double>> fractions;
    vector<vector<double>> coefficients;
    fractions = read_data("fractions.txt", data_size);
    coefficients = read_data("coefficients.txt", data_size);
    BasicMesh mesh = initializeMesh(5);
    Data<double> liquid_volume_fraction(mesh);
    Data<IRL::Paraboloid> a_interface(mesh);
    ofstream results;
    ofstream results1;
    results.open("result_expect.txt");
    results1.open("result_pred.txt");
    for (int n = 0; n < data_size; ++n)
    {
        int j = 0;
        int k = 0;
        for (int i = 0; i < 5; ++i)
        {
            for (int j = 0; j < 5; ++j)
            {
                for (int k = 0; k < 5; ++k)
                {
                    liquid_volume_fraction(i, j, k) = fractions[n][i*25+j*5+k];
                }
            }
        }
        getReconstruction("Jibben", liquid_volume_fraction, &a_interface);
        results1 << (a_interface)(2,2,2).getDatum()[0] << " " << (a_interface)(2,2,2).getDatum()[1] << " " << (a_interface)(2,2,2).getDatum()[2] << " " << (a_interface)(2,2,2).getAlignedParaboloid().a() << " " << (a_interface)(2,2,2).getAlignedParaboloid().b();
        for (int j = 0; j < 8; ++j)
        {
            if (j < 3 || j > 5)
            {
                results << coefficients[n][j] << " ";
            }
        }
        results1 << "\n";
        results << "\n";
    }
}