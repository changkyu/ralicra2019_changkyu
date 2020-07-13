
#include <iostream>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


#include "segmentation/graph/spectral_clustering.hpp"
#include "segmentation/meanshift/MeanShift.h"

using namespace std;
using namespace cv;

MeanShift ms;

void SpectralClustering(Mat &W, Mat &D, vector<uint32_t> &labels_out, int num_of_objects)
{    
    // Laplacian Matrix
    Mat L = D - W;

    // Normalized Laplacian Matrix
    Mat D_rsqr = Mat::zeros(D.rows, D.cols, CV_32FC1); // D^-1/2
    for( int r=0; r<D.rows; r++ )
    {        
        float d = D.at<float>(r,r);
        if (d>0) D_rsqr.at<float>(r,r) = 1/sqrt(d);
        else     D_rsqr.at<float>(r,r) = 0;        
    }
    //Mat L_sym = Mat::eye(W.rows,W.cols,CV_32FC1) - D_rsqr * W * D_rsqr;
    Mat L_sym = D_rsqr*L*D_rsqr;

    //Mat u, U, Vt;
    //SVD::compute(L_sym, u, U, Vt);
    Mat u, U;
    eigen(L_sym, u, U);
    U = U.t();
    
    int k = 0;
    for( int r = u.rows-1; r >= 0; r-- )
    {   
        if( u.at<float>(r,0) > 1.0e-3 ) break;
        //if( u.at<float>(r,0)*u.at<float>(r,0) > 0.001 ) break;
        //if( u.at<float>(r,0)*u.at<float>(r,0) > 0.01 ) break;

        k++;
    }    

    if( num_of_objects < 0 )
    {
        num_of_objects = k;
    }

    if( k<=1 )
    {
        labels_out.resize(W.rows);
        for( int r=0; r<W.rows; r++ )
        {
            labels_out[r] = 1;
        }
        return;
    }

#if 1 // use kmeans
    // normalized wrt rows    
    Mat T = Mat::zeros(U.rows,k,CV_32FC1);
    for( int r=0; r<U.rows; r++ )
    {        
        float norm = 0;
        for( int c=0; c<k; c++ )        
        {            
            norm += (U.at<float>(r,U.cols-1-c) * U.at<float>(r,U.cols-1-c));
        }
        norm = sqrt(norm);        
        
        for( int c=0; c<k; c++ )
        {            
            if( norm != 0 ) T.at<float>(r,c) = U.at<float>(r,U.cols-1-c) / norm;
            else            T.at<float>(r,c) = 0;
        }
    }

    // kmean    
    Mat labels_tmp;
    kmeans(T, num_of_objects, labels_tmp, 
        TermCriteria(TermCriteria::EPS+TermCriteria::COUNT,1000,0.0000001),
        1, KMEANS_PP_CENTERS);

    labels_out.resize(labels_tmp.rows);
    for( int r=0; r<labels_tmp.rows; r++ )
    {
        labels_out[r] = labels_tmp.at<int>(r,0);
    }
#else
    vector<double> T_mean(k,0);
    vector<vector<double> > T;
    T.resize(U.rows);
    for( int r=0; r<U.rows; r++ )
    {
        T[r].resize(k);

        float norm = 0;
        for( int c=0; c<k; c++ )        
        {            
            norm += (U.at<float>(r,U.cols-1-c) * U.at<float>(r,U.cols-1-c));
        }
        norm = sqrt(norm);        
        
        for( int c=0; c<k; c++ )
        {
            #if 1
            if( norm != 0 ) T[r][c] = U.at<float>(r,U.cols-1-c) / norm;
            else            T[r][c] = 0;
            #else
            T[r][c] = U.at<float>(r,U.cols-1-c);
            #endif

            T_mean[c] += T[r][c];
        }
    }

    double T_std_mean = 0;
    std::vector<double> T_std(k,0);
    for( int c=0; c<k; c++ )
    {
        T_mean[c] = T_mean[c] / U.rows;

        double sum=0;
        for( int r=0; r<U.rows; r++ )
        {
            sum += ((T[r][c] - T_mean[c]) * (T[r][c] - T_mean[c]));
        }
        sum = sum / U.rows;
        T_std[c] = sqrt(sum);
        T_std_mean += T_std[c];
    }
    T_std_mean = T_std_mean / k;
    cout << T_std_mean << endl;

    vector<Cluster> cs;
    vector<int> idxes;
    //ms.cluster(T, T_std_mean*1.1, cs, idxes);
    ms.cluster(T, 0.35, cs, idxes);

    labels_out.resize(T.size());
    for( int r=0; r<idxes.size(); r++ )
    {
        labels_out[r] = idxes[r];
    }

#endif

#if 0
    ofstream f("/home/cs1080/tmp_Lsym.m");
    f << "W = [" << endl;
    for( int r=0; r<W.rows; r++ )
    {
        for( int c=0; c<W.cols; c++ )
        {
            f << W.at<float>(r,c) << " ";
        }
        f << endl;
    }
    f << "];" << endl;

    f << "D2 = [" << endl;
    for( int r=0; r<D.rows; r++ )
    {
        for( int c=0; c<D.cols; c++ )
        {
            f << D.at<float>(r,c) << " ";
        }
        f << endl;
    }
    f << "];" << endl;

    f << "L_sym2 = [" << endl;
    for( int r=0; r<L_sym.rows; r++ )
    {
        for( int c=0; c<L_sym.cols; c++ )
        {
            f << L_sym.at<float>(r,c) << " ";
        }
        f << endl;
    }
    f << "];" << endl;

    f << "U2 = [" << endl;
    for( int r=0; r<U.rows; r++ )
    {
        for( int c=0; c<U.cols; c++ )
        {
            f << U.at<float>(r,c) << " ";
        }
        f << endl;
    }
    f << "];" << endl;

    f << "T2 = [" << endl;
    for( int r=0; r<T.rows; r++ )
    {
        for( int c=0; c<T.cols; c++ )
        {
            f << T.at<float>(r,c) << " ";
        }
        f << endl;
    }
    f << "];" << endl;

    f << "u2 = [" << endl;    
    /*
    for( int i=0; i<u.size(); i++ )
    {
        f << u[i] << ";";
    }
    */
    for( int i=0; i<u.rows; i++ )
    {
        f << u.at<float>(i,0) << ";";
    }
    f << "];" << endl;    

    f << "IDX2 = [" << endl;
    for( int r=0; r<(*labels_out).rows; r++ )
    {
        for( int c=0; c<(*labels_out).cols; c++ )
        {
            f << (*labels_out).at<int>(r,c) << " ";
        }
        f << endl;
    }
    f << "];" << endl;    
    f.close();
#endif
}

void SpectralClustering(Mat &W, vector<uint32_t> &labels_out, int num_of_objects)
{
    Mat D = Mat::zeros(W.rows, W.cols, CV_32FC1);
    for( int r=0; r<W.rows; r++ )
    {
        float degree = 0;
        for( int c=0; c<W.cols; c++ )
        {       
            degree += W.at<float>(r,c);
        }
        D.at<float>(r,r) = degree;
    }

    SpectralClustering(W,D,labels_out, num_of_objects);
}