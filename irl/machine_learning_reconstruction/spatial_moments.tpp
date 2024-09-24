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
        mu300 = 0;
        mu030 = 0;
        mu003 = 0;
        mu210 = 0;
        mu201 = 0;
        mu120 = 0;
        mu102 = 0;
        mu021 = 0;
        mu012 = 0;
        mu111 = 0;
    }

    torch::Tensor spatial_moments::calculate_moments(vector<double> fractions, IRL::Normal dir, int num_cells)
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
        mu300 = 0;
        mu030 = 0;
        mu003 = 0;
        mu210 = 0;
        mu201 = 0;
        mu120 = 0;
        mu102 = 0;
        mu021 = 0;
        mu012 = 0;
        mu111 = 0;

        int count = 0;
        double c000 = 0;
        double c100 = 0;
        double c010 = 0;
        double c001 = 0;
        double cxc = 0;
        double cyc = 0;
        double czc = 0;
        double c101 = 0;
        double c011 = 0;
        double c110 = 0;
        double c200 = 0;
        double c020 = 0;
        double c002 = 0;

        int lim = num_cells;

        for(int i = 0; i < lim; ++i)
        {
            for(int j = 0; j < lim; ++j)
            {
                for(int k = 0; k < lim; ++k)
                {
                    if (fractions[7*(i*lim*lim+j*lim+k)+0] > IRL::global_constants::VF_LOW)
                    {
                        ++count;
                    }
                    m000 = m000 + fractions[7*(i*lim*lim+j*lim+k)+0];
                    m100 = m100 + (fractions[7*(i*lim*lim+j*lim+k)+1]+(i-lim/2))*fractions[7*(i*lim*lim+j*lim+k)+0];
                    m010 = m010 + (fractions[7*(i*lim*lim+j*lim+k)+2]+(j-lim/2))*fractions[7*(i*lim*lim+j*lim+k)+0];
                    m001 = m001 + (fractions[7*(i*lim*lim+j*lim+k)+3]+(k-lim/2))*fractions[7*(i*lim*lim+j*lim+k)+0];
                }
            }
        }
        xc = m100 / m000;
        yc = m010 / m000;
        zc = m001 / m000;

        double thr = 0.01*(m000 / count);
        for(int i = 0; i < lim; ++i)
        {
            for(int j = 0; j < lim; ++j)
            {
                for(int k = 0; k < lim; ++k)
                {
                    if (fractions[7*(i*lim*lim+j*lim+k)+0] > thr)
                    {
                        c000 = c000 + 1;
                        c100 = c100 + (0.5+(i-lim/2));
                        c010 = c010 + (0.5+(j-lim/2));
                        c001 = c001 + (0.5+(k-lim/2));
                    }
                }
            }
        }
        cxc = c100 / c000;
        cyc = c010 / c000;
        czc = c001 / c000;

        bool gaussian = false;
        double scale = 2;
        
        for(int i = 0; i < lim; ++i)
        {
            for(int j = 0; j < lim; ++j)
            {
                for(int k = 0; k < lim; ++k)
                {
                    if (!gaussian)
                    {
                        mu101 = mu101 + ((fractions[7*(i*lim*lim+j*lim+k)+1]+(i-lim/2))-xc)*((fractions[7*(i*lim*lim+j*lim+k)+3]+(k-lim/2))-zc)*fractions[7*(i*lim*lim+j*lim+k)+0];
                        mu011 = mu011 + ((fractions[7*(i*lim*lim+j*lim+k)+2]+(j-lim/2))-yc)*((fractions[7*(i*lim*lim+j*lim+k)+3]+(k-lim/2))-zc)*fractions[7*(i*lim*lim+j*lim+k)+0];
                        mu110 = mu110 + ((fractions[7*(i*lim*lim+j*lim+k)+1]+(i-lim/2))-xc)*((fractions[7*(i*lim*lim+j*lim+k)+2]+(j-lim/2))-yc)*fractions[7*(i*lim*lim+j*lim+k)+0];
                        mu200 = mu200 + pow(((fractions[7*(i*lim*lim+j*lim+k)+1]+(i-lim/2))-xc),2.0)*fractions[7*(i*lim*lim+j*lim+k)+0];
                        mu020 = mu020 + pow(((fractions[7*(i*lim*lim+j*lim+k)+2]+(j-lim/2))-yc),2.0)*fractions[7*(i*lim*lim+j*lim+k)+0];
                        mu002 = mu002 + pow(((fractions[7*(i*lim*lim+j*lim+k)+3]+(k-lim/2))-zc),2.0)*fractions[7*(i*lim*lim+j*lim+k)+0];
                        
                        mu300 = mu300 + pow(((fractions[7*(i*lim*lim+j*lim+k)+1]+(i-lim/2))-xc),3.0)*fractions[7*(i*lim*lim+j*lim+k)+0];
                        mu030 = mu030 + pow(((fractions[7*(i*lim*lim+j*lim+k)+2]+(j-lim/2))-yc),3.0)*fractions[7*(i*lim*lim+j*lim+k)+0];
                        mu003 = mu003 + pow(((fractions[7*(i*lim*lim+j*lim+k)+3]+(k-lim/2))-zc),3.0)*fractions[7*(i*lim*lim+j*lim+k)+0];
                        mu210 = mu210 + pow(((fractions[7*(i*lim*lim+j*lim+k)+1]+(i-lim/2))-xc),2.0)*((fractions[7*(i*lim*lim+j*lim+k)+2]+(j-lim/2))-yc)*fractions[7*(i*lim*lim+j*lim+k)+0];
                        mu201 = mu201 + pow(((fractions[7*(i*lim*lim+j*lim+k)+1]+(i-lim/2))-xc),2.0)*((fractions[7*(i*lim*lim+j*lim+k)+3]+(k-lim/2))-zc)*fractions[7*(i*lim*lim+j*lim+k)+0];
                        mu120 = mu120 + ((fractions[7*(i*lim*lim+j*lim+k)+1]+(i-lim/2))-xc)*pow(((fractions[7*(i*lim*lim+j*lim+k)+2]+(j-lim/2))-yc),2.0)*fractions[7*(i*lim*lim+j*lim+k)+0];
                        mu102 = mu102 + ((fractions[7*(i*lim*lim+j*lim+k)+1]+(i-lim/2))-xc)*pow(((fractions[7*(i*lim*lim+j*lim+k)+3]+(k-lim/2))-zc),2.0)*fractions[7*(i*lim*lim+j*lim+k)+0];
                        mu021 = mu021 + pow(((fractions[7*(i*lim*lim+j*lim+k)+2]+(j-lim/2))-yc),2.0)*((fractions[7*(i*lim*lim+j*lim+k)+3]+(k-lim/2))-zc)*fractions[7*(i*lim*lim+j*lim+k)+0];
                        mu012 = mu012 + ((fractions[7*(i*lim*lim+j*lim+k)+2]+(j-lim/2))-yc)*pow(((fractions[7*(i*lim*lim+j*lim+k)+3]+(k-lim/2))-zc),2.0)*fractions[7*(i*lim*lim+j*lim+k)+0];
                        mu111 = mu111 + ((fractions[7*(i*lim*lim+j*lim+k)+1]+(i-lim/2))-xc)*((fractions[7*(i*lim*lim+j*lim+k)+2]+(j-lim/2))-yc)*((fractions[7*(i*lim*lim+j*lim+k)+3]+(k-lim/2))-zc)*fractions[7*(i*lim*lim+j*lim+k)+0];

                        if (fractions[7*(i*lim*lim+j*lim+k)+0] > thr)
                        {
                            c101 = c101 + ((0.5+(i-lim/2))-cxc)*((0.5+(k-lim/2))-czc);
                            c011 = c011 + ((0.5+(j-lim/2))-cyc)*((0.5+(k-lim/2))-czc);
                            c110 = c110 + ((0.5+(i-lim/2))-cxc)*((0.5+(j-lim/2))-cyc);
                            c200 = c200 + pow(((0.5+(i-lim/2))-cxc),2.0);
                            c020 = c020 + pow(((0.5+(j-lim/2))-cyc),2.0);
                            c002 = c002 + pow(((0.5+(k-lim/2))-czc),2.0);
                        }
                    }
                    else
                    {
                        mu101 = mu101 + (((fractions[7*(i*lim*lim+j*lim+k)+1]+(i-lim/2))-xc)/scale)*(((fractions[7*(i*lim*lim+j*lim+k)+3]+(k-lim/2))-zc)/scale)*exp(-(pow((fractions[7*(i*lim*lim+j*lim+k)+1]+(i-lim/2))-xc,2.0)+pow(((fractions[7*(i*lim*lim+j*lim+k)+2]+(i-lim/2))-yc),2.0)+pow(((fractions[7*(i*lim*lim+j*lim+k)+3]+(i-lim/2))-zc),2.0))/(2*scale*scale))*fractions[7*(i*lim*lim+j*lim+k)+0];
                        mu011 = mu011 + (((fractions[7*(i*lim*lim+j*lim+k)+2]+(i-lim/2))-yc)/scale)*(((fractions[7*(i*lim*lim+j*lim+k)+3]+(k-lim/2))-zc)/scale)*exp(-(pow((fractions[7*(i*lim*lim+j*lim+k)+1]+(i-lim/2))-xc,2.0)+pow(((fractions[7*(i*lim*lim+j*lim+k)+2]+(i-lim/2))-yc),2.0)+pow(((fractions[7*(i*lim*lim+j*lim+k)+3]+(i-lim/2))-zc),2.0))/(2*scale*scale))*fractions[7*(i*lim*lim+j*lim+k)+0];
                        mu110 = mu110 + (((fractions[7*(i*lim*lim+j*lim+k)+1]+(i-lim/2))-xc)/scale)*(((fractions[7*(i*lim*lim+j*lim+k)+2]+(i-lim/2))-yc)/scale)*exp(-(pow((fractions[7*(i*lim*lim+j*lim+k)+1]+(i-lim/2))-xc,2.0)+pow(((fractions[7*(i*lim*lim+j*lim+k)+2]+(i-lim/2))-yc),2.0)+pow(((fractions[7*(i*lim*lim+j*lim+k)+3]+(i-lim/2))-zc),2.0))/(2*scale*scale))*fractions[7*(i*lim*lim+j*lim+k)+0];
                        mu200 = mu200 + pow(((fractions[7*(i*lim*lim+j*lim+k)+1]+(i-lim/2))-xc)/scale,2.0)*exp(-(pow((fractions[7*(i*lim*lim+j*lim+k)+1]+(i-lim/2))-xc,2.0)+pow(((fractions[7*(i*lim*lim+j*lim+k)+2]+(i-lim/2))-yc),2.0)+pow(((fractions[7*(i*lim*lim+j*lim+k)+3]+(i-lim/2))-zc),2.0))/(2*scale*scale))*fractions[7*(i*lim*lim+j*lim+k)+0];
                        mu020 = mu020 + pow(((fractions[7*(i*lim*lim+j*lim+k)+2]+(i-lim/2))-yc)/scale,2.0)*exp(-(pow((fractions[7*(i*lim*lim+j*lim+k)+1]+(i-lim/2))-xc,2.0)+pow(((fractions[7*(i*lim*lim+j*lim+k)+2]+(i-lim/2))-yc),2.0)+pow(((fractions[7*(i*lim*lim+j*lim+k)+3]+(i-lim/2))-zc),2.0))/(2*scale*scale))*fractions[7*(i*lim*lim+j*lim+k)+0];
                        mu002 = mu002 + pow(((fractions[7*(i*lim*lim+j*lim+k)+3]+(k-lim/2))-zc)/scale,2.0)*exp(-(pow((fractions[7*(i*lim*lim+j*lim+k)+1]+(i-lim/2))-xc,2.0)+pow(((fractions[7*(i*lim*lim+j*lim+k)+2]+(i-lim/2))-yc),2.0)+pow(((fractions[7*(i*lim*lim+j*lim+k)+3]+(i-lim/2))-zc),2.0))/(2*scale*scale))*fractions[7*(i*lim*lim+j*lim+k)+0];   
                        
                        mu300 = mu300 + pow(((fractions[7*(i*lim*lim+j*lim+k)+1]+(i-lim/2))-xc)/scale,3.0)*fractions[7*(i*lim*lim+j*lim+k)+0];
                        mu030 = mu030 + pow(((fractions[7*(i*lim*lim+j*lim+k)+2]+(j-lim/2))-yc)/scale,3.0)*fractions[7*(i*lim*lim+j*lim+k)+0];
                        mu003 = mu003 + pow(((fractions[7*(i*lim*lim+j*lim+k)+3]+(k-lim/2))-zc)/scale,3.0)*fractions[7*(i*lim*lim+j*lim+k)+0];
                        mu210 = mu210 + pow(((fractions[7*(i*lim*lim+j*lim+k)+1]+(i-lim/2))-xc)/scale,2.0)*(((fractions[7*(i*lim*lim+j*lim+k)+2]+(j-lim/2))-yc)/scale)*fractions[7*(i*lim*lim+j*lim+k)+0];
                        mu201 = mu201 + pow(((fractions[7*(i*lim*lim+j*lim+k)+1]+(i-lim/2))-xc)/scale,2.0)*(((fractions[7*(i*lim*lim+j*lim+k)+3]+(k-lim/2))-zc)/scale)*fractions[7*(i*lim*lim+j*lim+k)+0];
                        mu120 = mu120 + (((fractions[7*(i*lim*lim+j*lim+k)+1]+(i-lim/2))-xc)/scale)*pow(((fractions[7*(i*lim*lim+j*lim+k)+2]+(j-lim/2))-yc)/scale,2.0)*fractions[7*(i*lim*lim+j*lim+k)+0];
                        mu102 = mu102 + (((fractions[7*(i*lim*lim+j*lim+k)+1]+(i-lim/2))-xc)/scale)*pow(((fractions[7*(i*lim*lim+j*lim+k)+3]+(k-lim/2))-zc)/scale,2.0)*fractions[7*(i*lim*lim+j*lim+k)+0];
                        mu021 = mu021 + pow(((fractions[7*(i*lim*lim+j*lim+k)+2]+(j-lim/2))-yc)/scale,2.0)*(((fractions[7*(i*lim*lim+j*lim+k)+3]+(k-lim/2))-zc)/scale)*fractions[7*(i*lim*lim+j*lim+k)+0];
                        mu012 = mu012 + (((fractions[7*(i*lim*lim+j*lim+k)+2]+(j-lim/2))-yc)/scale)*pow(((fractions[7*(i*lim*lim+j*lim+k)+3]+(k-lim/2))-zc)/scale,2.0)*fractions[7*(i*lim*lim+j*lim+k)+0];
                        mu111 = mu111 + (((fractions[7*(i*lim*lim+j*lim+k)+1]+(i-lim/2))-xc)/scale)*(((fractions[7*(i*lim*lim+j*lim+k)+2]+(j-lim/2))-yc)/scale)*(((fractions[7*(i*lim*lim+j*lim+k)+3]+(k-lim/2))-zc)/scale)*fractions[7*(i*lim*lim+j*lim+k)+0];
                    }
                }
            }
        }

        double ang = M_PI/2 - acos(dir[2]/sqrt(dir[0]*dir[0] + dir[1]*dir[1] + dir[2]*dir[2]));
        //sqrt(1+pow(tan(ang),2.0))

        mu101 = mu101/m000;
        mu011 = mu011/m000;
        mu110 = mu110/m000;
        mu200 = mu200/m000;
        mu020 = mu020/m000;
        mu002 = mu002/m000;
        mu300 = mu300/m000;
        mu030 = mu030/m000;
        mu003 = mu003/m000;
        mu210 = mu210/m000;
        mu201 = mu201/m000;
        mu120 = mu120/m000;
        mu102 = mu102/m000;
        mu021 = mu021/m000;
        mu012 = mu012/m000;
        mu111 = mu111/m000;

        // c101 = c101/c000;
        // c011 = c011/c000;
        // c110 = c110/c000;
        // c200 = c200/c000;
        // c020 = c020/c000;
        // c002 = c002/c000; 

        double J1 = (mu200 + mu020 + mu002);///pow(m000,5.0/3.0);
        //double J2 = (mu200*mu020 + mu200*mu002 + mu020*mu002 - mu110*mu110 - mu101*mu101 - mu011*mu011);///pow(m000,10.0/3.0);
        double J2 = (mu200*mu200 + mu020*mu020 + mu002*mu002 + 2*mu110*mu110 + 2*mu101*mu101 + 2*mu011*mu011);///pow(m000,10.0/3.0);
        //double J3 = (mu200*mu020*mu002 + 2*mu110*mu101*mu011 - mu200*mu011*mu011 - mu020*mu101*mu101 - mu002*mu110*mu110);///pow(m000,5.0);
        double J3 = (mu200*mu200*mu200 + 3*mu200*mu110*mu110 + 3*mu200*mu101*mu101 + 3*mu110*mu110*mu020 + 3*mu101*mu101*mu002 + mu020*mu020*mu020 + 3*mu020*mu011*mu011 + 3*mu011*mu011*mu002 + mu002*mu002*mu002 + 6*mu110*mu101*mu011);///pow(m000,5.0);
        double J4 = (mu300*mu300 + mu030*mu030 + mu003*mu003 + 3*mu210*mu210 + 3*mu201*mu201 + 3*mu120*mu120 + 3*mu102*mu102 + 3*mu021*mu021 + 3*mu012*mu012 + 6*mu111*mu111);///pow(m000,5.0);
        double C1 = (c200 + c020 + c002)/pow(c000,5.0/3.0);

        temp.push_back(m000);
        // temp.push_back(J2);
        // temp.push_back(J3);
        //temp.push_back(J4);
        double mag = sqrt(xc*xc + yc*yc + zc*zc);
        //temp.push_back(mag);
        temp.push_back(max(abs(dir[0]), max(abs(dir[1]), abs(dir[2]))));
        if (mag == 0)
        {
            //temp.push_back(0);
        }
        else
        {
            //temp.push_back((1-pow((xc/mag)*dir[0]+(yc/mag)*dir[1]+(zc/mag)*dir[2],4.0))*mag);
        }

        //temp.push_back(sqrt(pow(xc-fractions[(fractions.size()-7)/2+1],2.0) + pow(yc-fractions[((fractions.size()-7)/2)+2],2.0) + pow(zc-fractions[((fractions.size()-7)/2)+3],2.0)));
        temp.push_back(mag-abs((xc)*dir[0]+(yc)*dir[1]+(zc)*dir[2]));

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
                    m000 = m000 + fractions[/*7*/4*(i*9+j*3+k)+0];
                    m100 = m100 + (fractions[/*7*/4*(i*9+j*3+k)+1]+(i-1))*fractions[/*7*/4*(i*9+j*3+k)+0];
                    m010 = m010 + (fractions[/*7*/4*(i*9+j*3+k)+2]+(j-1))*fractions[/*7*/4*(i*9+j*3+k)+0];
                    m001 = m001 + (fractions[/*7*/4*(i*9+j*3+k)+3]+(k-1))*fractions[/*7*/4*(i*9+j*3+k)+0];
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

    vector<double> spatial_moments::get_mass_centers_all(vector<double>* fractions)
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
                    m000 = m000 + fractions[0][7*(i*9+j*3+k)+0];
                    m100 = m100 + (fractions[0][7*(i*9+j*3+k)+1]+(i-1))*fractions[0][7*(i*9+j*3+k)+0];
                    m010 = m010 + (fractions[0][7*(i*9+j*3+k)+2]+(j-1))*fractions[0][7*(i*9+j*3+k)+0];
                    m001 = m001 + (fractions[0][7*(i*9+j*3+k)+3]+(k-1))*fractions[0][7*(i*9+j*3+k)+0];
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

    vector<double> spatial_moments::get_moment_of_intertia(vector<double>* fractions)
    {
        double Ixx = 0;
        double Iyy = 0;
        double Izz = 0;
        double Ixy = 0;
        double Iyz = 0;
        double Ixz = 0;
        for(int i = 0; i < 3; ++i)
        {
            for(int j = 0; j < 3; ++j)
            {
                for(int k = 0; k < 3; ++k)
                {
                    Ixx = Ixx + ((fractions[0][7*(i*9+j*3+k)+2]+(j-1))*(fractions[0][7*(i*9+j*3+k)+2]+(j-1))+(fractions[0][7*(i*9+j*3+k)+3]+(k-1))*(fractions[0][7*(i*9+j*3+k)+3]+(k-1)))*fractions[0][7*(i*9+j*3+k)+0];
                    Iyy = Iyy + ((fractions[0][7*(i*9+j*3+k)+1]+(i-1))*(fractions[0][7*(i*9+j*3+k)+1]+(i-1))+(fractions[0][7*(i*9+j*3+k)+3]+(k-1))*(fractions[0][7*(i*9+j*3+k)+3]+(k-1)))*fractions[0][7*(i*9+j*3+k)+0];
                    Izz = Izz + ((fractions[0][7*(i*9+j*3+k)+1]+(i-1))*(fractions[0][7*(i*9+j*3+k)+1]+(i-1))+(fractions[0][7*(i*9+j*3+k)+2]+(j-1))*(fractions[0][7*(i*9+j*3+k)+2]+(j-1)))*fractions[0][7*(i*9+j*3+k)+0];
                    Ixy = Ixy + (fractions[0][7*(i*9+j*3+k)+1]+(i-1))*(fractions[0][7*(i*9+j*3+k)+2]+(j-1))*fractions[0][7*(i*9+j*3+k)+0];
                    Iyz = Iyz + (fractions[0][7*(i*9+j*3+k)+2]+(j-1))*(fractions[0][7*(i*9+j*3+k)+3]+(k-1))*fractions[0][7*(i*9+j*3+k)+0];
                    Ixz = Ixz + (fractions[0][7*(i*9+j*3+k)+1]+(i-1))*(fractions[0][7*(i*9+j*3+k)+3]+(k-1))*fractions[0][7*(i*9+j*3+k)+0];
                }
            }
        }
        Ixy = -Ixy;
        Iyz = -Iyz;
        Ixz = -Ixz;
        Eigen::MatrixXd I(3,3);
        I(0,0) = Ixx;
        I(1,0) = Ixy;
        I(3,0) = Ixz;
        I(0,1) = Ixy;
        I(1,1) = Iyy;
        I(2,1) = Iyz;
        I(0,2) = Ixz;
        I(1,2) = Iyz;
        I(2,2) = Izz;
        Eigen::EigenSolver<Eigen::MatrixXd> es(I);
        vector<double> eigenvectors;
        eigenvectors.push_back(es.eigenvalues()[0].real());
        eigenvectors.push_back(es.eigenvectors().col(0)[0].real());
        eigenvectors.push_back(es.eigenvectors().col(0)[1].real());
        eigenvectors.push_back(es.eigenvectors().col(0)[2].real());
        eigenvectors.push_back(es.eigenvalues()[1].real());
        eigenvectors.push_back(es.eigenvectors().col(1)[0].real());
        eigenvectors.push_back(es.eigenvectors().col(1)[1].real());
        eigenvectors.push_back(es.eigenvectors().col(1)[2].real());
        eigenvectors.push_back(es.eigenvalues()[2].real());
        eigenvectors.push_back(es.eigenvectors().col(2)[0].real());
        eigenvectors.push_back(es.eigenvectors().col(2)[1].real());
        eigenvectors.push_back(es.eigenvectors().col(2)[2].real());
        //std::cout << eigenvectors << std::endl;
        return eigenvectors;
    }
}

#endif