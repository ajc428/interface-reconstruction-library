#include <math.h>
#include <iostream>
#include "data_gen.h"

using namespace std;

void data_generate(int num, double rota_l, double rota_h, double rotb_l, double rotb_h, double rotc_l, double rotc_h, double coa_l, double coa_h, double cob_l, double cob_h, double ox_l, double ox_h, double oy_l, double oy_h, double oz_l, double oz_h)
{
    IRL::data_gen gen(3,num);
    gen.generate(rota_l, rota_h, rotb_l, rotb_h, rotc_l, rotc_h, coa_l, coa_h, cob_l, cob_h, ox_l, ox_h, oy_l, oy_h, oz_l, oz_h);
}

int main(int argc, char* argv[])
{
    data_generate(1000,0,2*M_PI,0,2*M_PI,0,2*M_PI,-1,1,-1,1,-1,1,-0.5,0.5,-0.5,0.5);
}