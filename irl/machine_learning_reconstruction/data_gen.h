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

    public:
        data_gen(int x, int y)
        {
            number_of_cells = x;
            Ntests = y;
            gen = new IRL::fractions(number_of_cells);
        };

        ~data_gen()
        {
            delete gen;
        };

        void generate()
        {
            for (int n = 0; n < Ntests; ++n) 
            {
                std::cout << n << endl;
                IRL::Paraboloid paraboloid = gen->new_random_parabaloid();
                angles = gen->getAngles();

                std::ofstream coefficients;
                std::string data_name = "coefficients.txt";
                coefficients.open(data_name, std::ios_base::app);
                coefficients << paraboloid.getDatum().x() << "," << paraboloid.getDatum().y() << "," << paraboloid.getDatum().z()
                << "," << angles[0] << "," << angles[1] << "," << angles[2]
                << "," << paraboloid.getAlignedParaboloid().a() << "," << paraboloid.getAlignedParaboloid().b() << "\n";
                coefficients.close();

                auto result = gen->get_fractions(paraboloid, true);
                std::ofstream output;
                data_name = "fractions.txt";
                output.open(data_name, std::ios_base::app);

                for (int i = 0; i < 108; ++i)
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