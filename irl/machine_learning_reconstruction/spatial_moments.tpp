// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2023 Andrew Cahaly <andrew.cahaly@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef IRL_MACHINE_LEARNING_RECONSTRUCTION_SPATIAL_MOMENTS_TPP_
#define IRL_MACHINE_LEARNING_RECONSTRUCTION_SPATIAL_MOMENTS_TPP_

using namespace std;

namespace IRL
{
    spatial_moments::spatial_moments()
    {
        m000 = 0;
        m100 = 0;
        m010 = 0;
        m001 = 0;
        m200 = 0;
        m020 = 0;
        m002 = 0;
        xc = 0;
        yc = 0;
        zc = 0;
        mu100 = 0;
        mu010 = 0;
        mu001 = 0;
        mu101 = 0;
        mu011 = 0;
        mu110 = 0;
        mu111 = 0;
        mu200 = 0;
        mu020 = 0;
        mu002 = 0;
    }

    torch::Tensor spatial_moments::calculate_moments(const DataMesh<double>& a_liquid_volume_fraction, Mesh mesh)
    {
        vector<double> temp;
        m000 = 0;
        m100 = 0;
        m010 = 0;
        m001 = 0;
        m200 = 0;
        m020 = 0;
        m002 = 0;
        xc = 0;
        yc = 0;
        zc = 0;
        mu100 = 0;
        mu010 = 0;
        mu001 = 0;
        mu101 = 0;
        mu011 = 0;
        mu110 = 0;
        mu111 = 0;
        mu200 = 0;
        mu020 = 0;
        mu002 = 0;
        for(int i = 0; i < 3; ++i)
        {
            for(int j = 0; j < 3; ++j)
            {
                for(int k = 0; k < 3; ++k)
                {
                    m000 = m000 + a_liquid_volume_fraction(i,j,k);
                    m100 = m100 + mesh.x(i)*a_liquid_volume_fraction(i,j,k);
                    m010 = m010 + mesh.y(j)*a_liquid_volume_fraction(i,j,k);
                    m001 = m001 + mesh.z(k)*a_liquid_volume_fraction(i,j,k);
                    m200 = m200 + mesh.x(i)*mesh.x(i)*a_liquid_volume_fraction(i,j,k);
                    m020 = m020 + mesh.y(j)*mesh.y(j)*a_liquid_volume_fraction(i,j,k);
                    m002 = m002 + mesh.z(k)*mesh.z(k)*a_liquid_volume_fraction(i,j,k);
                }
            }
        }
        xc = m100 / m000;
        yc = m010 / m000;
        zc = m001 / m000;
        
        for(int i = 0; i < 3; ++i)
        {
            for(int j = 0; j < 3; ++j)
            {
                for(int k = 0; k < 3; ++k)
                {
                    mu100 = mu100 + (mesh.x(i)-xc)*a_liquid_volume_fraction(i,j,k);
                    mu010 = mu010 + (mesh.y(j)-yc)*a_liquid_volume_fraction(i,j,k);
                    mu001 = mu001 + (mesh.z(k)-zc)*a_liquid_volume_fraction(i,j,k);
                    mu101 = mu101 + (mesh.x(i)-xc)*(mesh.z(k)-zc)*a_liquid_volume_fraction(i,j,k);
                    mu011 = mu011 + (mesh.y(j)-yc)*(mesh.z(k)-zc)*a_liquid_volume_fraction(i,j,k);
                    mu110 = mu110 + (mesh.x(i)-xc)*(mesh.y(j)-yc)*a_liquid_volume_fraction(i,j,k);
                    mu111 = mu111 + (mesh.x(i)-xc)*(mesh.y(j)-yc)*(mesh.z(k)-zc)*a_liquid_volume_fraction(i,j,k);
                    mu200 = mu200 + pow((mesh.x(i)-xc),2.0)*a_liquid_volume_fraction(i,j,k);
                    mu020 = mu020 + pow((mesh.y(j)-yc),2.0)*a_liquid_volume_fraction(i,j,k);
                    mu002 = mu002 + pow((mesh.z(k)-zc),2.0)*a_liquid_volume_fraction(i,j,k);
                }
            }
        }
        mu100 = mu100/pow(m000,(1+0+0+3)/3.0);
        mu010 = mu010/pow(m000,(0+1+0+3)/3.0);
        mu001 = mu001/pow(m000,(0+0+1+3)/3.0);
        mu101 = mu101/pow(m000,(1+0+1+3)/3.0);
        mu011 = mu011/pow(m000,(0+1+1+3)/3.0);
        mu110 = mu110/pow(m000,(1+1+0+3)/3.0);
        mu111 = mu111/pow(m000,(1+1+1+3)/3.0);
        mu200 = mu200/pow(m000,(2+0+0+3)/3.0);
        mu020 = mu020/pow(m000,(0+2+0+3)/3.0);
        mu002 = mu002/pow(m000,(0+0+2+3)/3.0);

        double J1 = mu200 + mu020 + mu002;
        double J2 = mu200*mu020 + mu200*mu002 + mu020*mu002 - mu110*mu110 - mu101*mu101 - mu011*mu011;
        double J3 = mu200*mu020*mu002 + 2*mu110*mu101*mu011 - mu002*mu110*mu110 - mu020*mu101*mu101 - mu200*mu011*mu011;

        temp.push_back(J1);
        temp.push_back(J2);
        temp.push_back(J3);

        return torch::tensor(temp);
    }

    torch::Tensor spatial_moments::getMoments()
    {
        return moments;
    }
}

#endif