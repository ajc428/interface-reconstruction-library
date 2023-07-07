// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2023 Andrew Cahaly <andrew.cahaly@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef IRL_MACHINE_LEARNING_RECONSTRUCTION_DATA_GEN_H_
#define IRL_MACHINE_LEARNING_RECONSTRUCTION_DATA_GEN_H_

#include "irl/machine_learning_reconstruction/fractions.h"

namespace IRL 
{
    class data_gen
    {
    private:
        int number_of_cells;     
        int Ntests;  

        IRL::fractions *gen;
        std::array<double, 3> angles;
        IRL::spatial_moments *sm;

    public:
        data_gen(int x, int y)
        {
            number_of_cells = x;
            Ntests = y;
            gen = new IRL::fractions(number_of_cells);
            sm = new IRL::spatial_moments();
        };

        ~data_gen()
        {
            delete gen;
            delete sm;
        };

        void generate(double rota_l, double rota_h, double rotb_l, double rotb_h, double rotc_l, double rotc_h, double coa_l, double coa_h, double cob_l, double cob_h, double ox_l, double ox_h, double oy_l, double oy_h, double oz_l, double oz_h)
        {
            for (int n = 0; n < Ntests; ++n) 
            {
                std::cout << n << endl;
                IRL::Paraboloid paraboloid = gen->new_random_parabaloid(rota_l, rota_h, rotb_l, rotb_h, rotc_l, rotc_h, coa_l, coa_h, cob_l, cob_h, ox_l, ox_h, oy_l, oy_h, oz_l, oz_h);
                angles = gen->getAngles();

                std::ofstream coefficients;
                std::string name = "coefficients.txt";
                coefficients.open(name, std::ios_base::app);
                coefficients << paraboloid.getDatum().x() << "," << paraboloid.getDatum().y() << "," << paraboloid.getDatum().z()
                << "," << angles[0] << "," << angles[1] << "," << angles[2]
                << "," << paraboloid.getAlignedParaboloid().a() << "," << paraboloid.getAlignedParaboloid().b() << "\n";
                coefficients.close();

                std::ofstream curvatures;
                std::string curv_name = "curvatures.txt";
                curvatures.open(curv_name, std::ios_base::app);
                curvatures << paraboloid.getAlignedParaboloid().a() << "," << paraboloid.getAlignedParaboloid().b() << "\n";
                coefficients.close();

                std::ofstream classification;
                std::string data_name = "type.txt";
                classification.open(data_name, std::ios_base::app);
                int x;
                if ((abs(paraboloid.getAlignedParaboloid().a() - paraboloid.getAlignedParaboloid().b()) > 1) && (paraboloid.getAlignedParaboloid().a() < 0.2 || paraboloid.getAlignedParaboloid().b() < 0.2))
                {
                    x = 0;
                }
                else
                {
                    x = 1;
                }
                classification << x << " \n";
                classification.close();

                auto result = gen->get_fractions(paraboloid, true);
                std::ofstream output;
                data_name = "fractions.txt";
                output.open(data_name, std::ios_base::app);

                for (int i = 0; i < result.sizes()[0]; ++i)
                {
                    output << result[i].item<double>() << ",";
                }
                output << "\n";
                output.close();                    
            }       
        };
    };
}

#endif