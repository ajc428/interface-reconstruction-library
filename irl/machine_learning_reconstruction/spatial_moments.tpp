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
        xc = 0;
        yc = 0;
        zc = 0;
        mu101 = 0;
        mu011 = 0;
        mu110 = 0;
        mu200 = 0;
        mu020 = 0;
        mu002 = 0;

        mx000 = 0;
        my000 = 0;
        mz000 = 0;
        mx100 = 0;
        my100 = 0;
        mz100 = 0;
        mx010 = 0;
        my010 = 0;
        mz010 = 0;
        mx001 = 0;
        my001 = 0;
        mz001 = 0;
        x_xc = 0;
        x_yc = 0;
        x_zc = 0;
        y_xc = 0;
        y_yc = 0;
        y_zc = 0;
        z_xc = 0;
        z_yc = 0;
        z_zc = 0;
    }

    torch::Tensor spatial_moments::calculate_moments(const DataMesh<double>& a_liquid_volume_fraction, DataMesh<IRL::Pt>& a_liquid_centroid, Mesh mesh)
    {
        vector<double> temp;
        m000 = 0;
        m100 = 0;
        m010 = 0;
        m001 = 0;
        xc = 0;
        yc = 0;
        zc = 0;
        mu101 = 0;
        mu011 = 0;
        mu110 = 0;
        mu200 = 0;
        mu020 = 0;
        mu002 = 0;

        mx000 = 0;
        my000 = 0;
        mz000 = 0;
        mx100 = 0;
        my100 = 0;
        mz100 = 0;
        mx010 = 0;
        my010 = 0;
        mz010 = 0;
        mx001 = 0;
        my001 = 0;
        mz001 = 0;
        x_xc = 0;
        x_yc = 0;
        x_zc = 0;
        y_xc = 0;
        y_yc = 0;
        y_zc = 0;
        z_xc = 0;
        z_yc = 0;
        z_zc = 0;

        centroid_x.clear();
        centroid_y.clear();
        centroid_z.clear();

        for (int i = 0; i < 6; ++i)
        {
            centroid_x.push_back(0);
            centroid_y.push_back(0);
            centroid_z.push_back(0);
        }

        for(int i = 0; i < 3; ++i)
        {
            for(int j = 0; j < 3; ++j)
            {
                for(int k = 0; k < 3; ++k)
                {
                    m000 = m000 + a_liquid_volume_fraction(i,j,k);
                    m100 = m100 + a_liquid_centroid(i,j,k)[0]*a_liquid_volume_fraction(i,j,k);
                    m010 = m010 + a_liquid_centroid(i,j,k)[1]*a_liquid_volume_fraction(i,j,k);
                    m001 = m001 + a_liquid_centroid(i,j,k)[2]*a_liquid_volume_fraction(i,j,k);

                    mx000 = mx000 + a_liquid_centroid(i,j,k)[0];
                    mx100 = mx100 + ((mesh.x(i)+mesh.x(i+1))/2)*a_liquid_centroid(i,j,k)[0];
                    mx010 = mx010 + ((mesh.y(j)+mesh.y(j+1))/2)*a_liquid_centroid(i,j,k)[0];
                    mx001 = mx001 + ((mesh.z(k)+mesh.z(k+1))/2)*a_liquid_centroid(i,j,k)[0];

                    my000 = my000 + a_liquid_centroid(i,j,k)[1];
                    my100 = my100 + ((mesh.x(i)+mesh.x(i+1))/2)*a_liquid_centroid(i,j,k)[1];
                    my010 = my010 + ((mesh.y(j)+mesh.y(j+1))/2)*a_liquid_centroid(i,j,k)[1];
                    my001 = my001 + ((mesh.z(k)+mesh.z(k+1))/2)*a_liquid_centroid(i,j,k)[1];

                    mz000 = mz000 + a_liquid_centroid(i,j,k)[2];
                    mz100 = mz100 + ((mesh.x(i)+mesh.x(i+1))/2)*a_liquid_centroid(i,j,k)[2];
                    mz010 = mz010 + ((mesh.y(j)+mesh.y(j+1))/2)*a_liquid_centroid(i,j,k)[2];
                    mz001 = mz001 + ((mesh.z(k)+mesh.z(k+1))/2)*a_liquid_centroid(i,j,k)[2];
                }
            }
        }
        xc = m100 / m000;
        yc = m010 / m000;
        zc = m001 / m000;

        x_xc = mx100 / mx000;
        x_yc = mx010 / mx000;
        x_zc = mx001 / mx000;

        y_xc = my100 / my000;
        y_yc = my010 / my000;
        y_zc = my001 / my000;

        z_xc = mz100 / mz000;
        z_yc = mz010 / mz000;
        z_zc = mz001 / mz000;
        
        for(int i = 0; i < 3; ++i)
        {
            for(int j = 0; j < 3; ++j)
            {
                for(int k = 0; k < 3; ++k)
                {
                    mu101 = mu101 + (a_liquid_centroid(i,j,k)[0]-xc)*(a_liquid_centroid(i,j,k)[2]-zc)*a_liquid_volume_fraction(i,j,k);
                    mu011 = mu011 + (a_liquid_centroid(i,j,k)[1]-yc)*(a_liquid_centroid(i,j,k)[2]-zc)*a_liquid_volume_fraction(i,j,k);
                    mu110 = mu110 + (a_liquid_centroid(i,j,k)[0]-xc)*(a_liquid_centroid(i,j,k)[1]-yc)*a_liquid_volume_fraction(i,j,k);
                    mu200 = mu200 + pow((a_liquid_centroid(i,j,k)[0]-xc),2.0)*a_liquid_volume_fraction(i,j,k);
                    mu020 = mu020 + pow((a_liquid_centroid(i,j,k)[1]-yc),2.0)*a_liquid_volume_fraction(i,j,k);
                    mu002 = mu002 + pow((a_liquid_centroid(i,j,k)[2]-zc),2.0)*a_liquid_volume_fraction(i,j,k);

                    centroid_x[0] = centroid_x[0] + ((mesh.x(i)+mesh.x(i+1))/2-x_xc)*((mesh.z(k)+mesh.z(k+1))/2-x_zc)*a_liquid_centroid(i,j,k)[0];
                    centroid_x[1] = centroid_x[1] + ((mesh.y(j)+mesh.y(j+1))/2-x_yc)*((mesh.z(k)+mesh.z(k+1))/2-x_zc)*a_liquid_centroid(i,j,k)[0];
                    centroid_x[2] = centroid_x[2] + ((mesh.x(i)+mesh.x(i+1))/2-x_xc)*((mesh.y(j)+mesh.y(j+1))/2-x_yc)*a_liquid_centroid(i,j,k)[0];
                    centroid_x[3] = centroid_x[3] + pow(((mesh.x(i)+mesh.x(i+1))/2-x_xc),2.0)*a_liquid_centroid(i,j,k)[0];
                    centroid_x[4] = centroid_x[4] + pow(((mesh.y(j)+mesh.y(j+1))/2-x_yc),2.0)*a_liquid_centroid(i,j,k)[0];
                    centroid_x[5] = centroid_x[5] + pow(((mesh.z(k)+mesh.z(k+1))/2-x_zc),2.0)*a_liquid_centroid(i,j,k)[0];
        
                    centroid_y[0] = centroid_y[0] + ((mesh.x(i)+mesh.x(i+1))/2-y_xc)*((mesh.z(k)+mesh.z(k+1))/2-y_zc)*a_liquid_centroid(i,j,k)[1];
                    centroid_y[1] = centroid_y[1] + ((mesh.y(j)+mesh.y(j+1))/2-y_yc)*((mesh.z(k)+mesh.z(k+1))/2-y_zc)*a_liquid_centroid(i,j,k)[1];
                    centroid_y[2] = centroid_y[2] + ((mesh.x(i)+mesh.x(i+1))/2-y_xc)*((mesh.y(j)+mesh.y(j+1))/2-y_yc)*a_liquid_centroid(i,j,k)[1];
                    centroid_y[3] = centroid_y[3] + pow(((mesh.x(i)+mesh.x(i+1))/2-y_xc),2.0)*a_liquid_centroid(i,j,k)[1];
                    centroid_y[4] = centroid_y[4] + pow(((mesh.y(j)+mesh.y(j+1))/2-y_yc),2.0)*a_liquid_centroid(i,j,k)[1];
                    centroid_y[5] = centroid_y[5] + pow(((mesh.z(k)+mesh.z(k+1))/2-y_zc),2.0)*a_liquid_centroid(i,j,k)[1];

                    centroid_z[0] = centroid_z[0] + ((mesh.x(i)+mesh.x(i+1))/2-z_xc)*((mesh.z(k)+mesh.z(k+1))/2-z_zc)*a_liquid_centroid(i,j,k)[2];
                    centroid_z[1] = centroid_z[1] + ((mesh.y(j)+mesh.y(j+1))/2-z_yc)*((mesh.z(k)+mesh.z(k+1))/2-z_zc)*a_liquid_centroid(i,j,k)[2];
                    centroid_z[2] = centroid_z[2] + ((mesh.x(i)+mesh.x(i+1))/2-z_xc)*((mesh.y(j)+mesh.y(j+1))/2-z_yc)*a_liquid_centroid(i,j,k)[2];
                    centroid_z[3] = centroid_z[3] + pow(((mesh.x(i)+mesh.x(i+1))/2-z_xc),2.0)*a_liquid_centroid(i,j,k)[2];
                    centroid_z[4] = centroid_z[4] + pow(((mesh.y(j)+mesh.y(j+1))/2-z_yc),2.0)*a_liquid_centroid(i,j,k)[2];
                    centroid_z[5] = centroid_z[5] + pow(((mesh.z(k)+mesh.z(k+1))/2-z_zc),2.0)*a_liquid_centroid(i,j,k)[2];
                }
            }
        }

        double J1 = (mu200 + mu020 + mu002);
        double J2 = (mu200*mu020 + mu200*mu002 + mu020*mu002 - mu110*mu110 - mu101*mu101 - mu011*mu011);
        double J3 = (mu200*mu020*mu002 + 2*mu110*mu101*mu011 - mu200*mu011*mu011 - mu020*mu101*mu101 - mu002*mu110*mu110);

        double J1_x = (centroid_x[3] + centroid_x[4] + centroid_x[5]);
        double J2_x = (centroid_x[3]*centroid_x[4] + centroid_x[3]*centroid_x[5] + centroid_x[4]*centroid_x[5] - centroid_x[2]*centroid_x[2] - centroid_x[0]*centroid_x[0] - centroid_x[1]*centroid_x[1]);
        double J3_x = (centroid_x[3]*centroid_x[3]*centroid_x[5] + 2*centroid_x[2]*centroid_x[0]*centroid_x[1] - centroid_x[3]*centroid_x[1]*centroid_x[1] - centroid_x[4]*centroid_x[0]*centroid_x[0] - centroid_x[5]*centroid_x[2]*centroid_x[2]);

        double J1_y = (centroid_y[3] + centroid_y[4] + centroid_y[5]);
        double J2_y = (centroid_y[3]*centroid_y[4] + centroid_y[3]*centroid_y[5] + centroid_y[4]*centroid_y[5] - centroid_y[2]*centroid_y[2] - centroid_y[0]*centroid_y[0] - centroid_y[1]*centroid_y[1]);
        double J3_y = (centroid_y[3]*centroid_y[3]*centroid_y[5] + 2*centroid_y[2]*centroid_y[0]*centroid_y[1] - centroid_y[3]*centroid_y[1]*centroid_y[1] - centroid_y[4]*centroid_y[0]*centroid_y[0] - centroid_y[5]*centroid_y[2]*centroid_y[2]);

        double J1_z = (centroid_z[3] + centroid_z[4] + centroid_z[5]);
        double J2_z = (centroid_z[3]*centroid_z[4] + centroid_z[3]*centroid_z[5] + centroid_z[4]*centroid_z[5] - centroid_z[2]*centroid_z[2] - centroid_z[0]*centroid_z[0] - centroid_z[1]*centroid_z[1]);
        double J3_z = (centroid_z[3]*centroid_z[3]*centroid_z[5] + 2*centroid_z[2]*centroid_z[0]*centroid_z[1] - centroid_z[3]*centroid_z[1]*centroid_z[1] - centroid_z[4]*centroid_z[0]*centroid_z[0] - centroid_z[5]*centroid_z[2]*centroid_z[2]);

        temp.push_back(J1);
        temp.push_back(J2);
        temp.push_back(J3);
        /*temp.push_back(J1_x);
        temp.push_back(J2_x);
        temp.push_back(J3_x);
        temp.push_back(J1_y);
        temp.push_back(J2_y);
        temp.push_back(J3_y);
        temp.push_back(J1_z);
        temp.push_back(J2_z);
        temp.push_back(J3_z);*/

        return torch::tensor(temp);
    }

    vector<double> spatial_moments::get_mass_centers(vector<double> fractions)
    {
        m000 = 0;
        m100 = 0;
        m010 = 0;
        m001 = 0;
        for(int i = 0; i < 3; ++i)
        {
            for(int j = 0; j < 3; ++j)
            {
                for(int k = 0; k < 3; ++k)
                {
                    m000 = m000 + fractions[7*(i*9+j*3+k)+0];
                    m100 = m100 + (fractions[7*(i*9+j*3+k)+1]+(i-1))*fractions[7*(i*9+j*3+k)+0];
                    m010 = m010 + (fractions[7*(i*9+j*3+k)+2]+(j-1))*fractions[7*(i*9+j*3+k)+0];
                    m001 = m001 + (fractions[7*(i*9+j*3+k)+3]+(k-1))*fractions[7*(i*9+j*3+k)+0];
                }
            }
        }
        xc = m100 / m000;
        yc = m010 / m000;
        zc = m001 / m000;
        vector<double> centers;
        centers.push_back(xc);
        centers.push_back(yc);
        centers.push_back(zc);
        return centers;
    }
}

#endif