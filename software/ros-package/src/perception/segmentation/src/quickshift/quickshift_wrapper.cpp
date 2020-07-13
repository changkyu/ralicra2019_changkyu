#include <fstream>
#include <map>

#include "Image.h"
#include "quickshift_common.h"

#include "segmentation/quickshift/quickshift_wrapper.hpp"

int * map_to_flatmap(float * map, unsigned int size)
{
  /********** Flatmap **********/
  int *flatmap      = (int *) malloc(size*sizeof(int)) ;
  for (unsigned int p = 0; p < size; p++)
  {
    flatmap[p] = map[p];
  }

  bool changed = true;
  while (changed)
  {
    changed = false;
    for (unsigned int p = 0; p < size; p++)
    {
      changed = changed || (flatmap[p] != flatmap[flatmap[p]]);
      flatmap[p] = flatmap[flatmap[p]];
    }
  }

  /* Consistency check */
  for (unsigned int p = 0; p < size; p++)
    assert(flatmap[p] == flatmap[flatmap[p]]);

  return flatmap;
}

void cvimage_to_matlab(cv::Mat& IMG, image_t & im)
{
  /********** Convert image to MATLAB style representation **********/
  im.N1 = IMG.rows;
  im.N2 = IMG.cols;
  im.K  = IMG.channels();
  im.I = (float *) calloc(im.N1*im.N2*im.K, sizeof(float));
  for(int k = 0; k < im.K; k++)
    for(int col = 0; col < im.N2; col++)
      for(int row = 0; row < im.N1; row++)
      {
        unsigned char * pt = ((unsigned char *)IMG.data) + im.K * (row*im.N2 + col);
        im.I[row + col*im.N1 + k*im.N1*im.N2] = 32. * pt[k] / 255.; // Scale 0-32
      }
}

void Segmentation_QuickShift(cv::Mat &image, cv::Mat &image_seg,float sigma, float tau)
{
  image_t im;
  cvimage_to_matlab(image,im);
  //normalize_image(im);

  float *map  = (float *) calloc(im.N1*im.N2, sizeof(float)) ;
  float *gaps = (float *) calloc(im.N1*im.N2, sizeof(float)) ;
  float *E    = (float *) calloc(im.N1*im.N2, sizeof(float)) ;

  quickshift_gpu(im, sigma, tau, map, gaps, E);
  int* flatmap = map_to_flatmap(map, im.N1*im.N2);

  image_seg = cv::Mat(image.rows,image.cols,CV_16UC1);  
  int label_add = 0;
  std::map<int,int> old2new;  
  for( int r=0; r<im.N1; r++ )
  {
    for( int c=0; c<im.N2; c++ )
    {
      int label = flatmap[r + im.N1*c];
      int label_new = -1;
      std::map<int,int>::iterator it_old2new = old2new.find(label);
      if( it_old2new == old2new.end() )
      {
        old2new.insert(std::pair<int,int>(label,label_add));
        label_new = label_add;
        label_add++;
      }
      else
      {
        label_new = it_old2new->second;
      }
      image_seg.at<unsigned short>(r,c) = label_new;
    }
  }

  free(flatmap);
  free(E);
  free(gaps);
  free(map);
  free(im.I);
}