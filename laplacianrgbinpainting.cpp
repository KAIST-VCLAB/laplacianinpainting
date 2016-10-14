/****************************************************************************

- Codename: Laplacian Patch-Based Image Synthesis (CVPR 2016)

- Writers:   Joo Ho Lee(jhlee@vclab.kaist.ac.kr), Min H. Kim (minhkim@kaist.ac.kr)

- Institute: KAIST Visual Computing Laboratory

- Bibtex:
	
@InProceedings{LeeChoiKim:CVPR:2016,
  author  = {Joo Ho Lee and Inchang Choi and Min H. Kim},
  title   = {Laplacian Patch-Based Image Synthesis},
  booktitle = {Proc. IEEE Computer Vision and Pattern Recognition (CVPR 2016)},
  publisher = {IEEE},  
  address = {Las Vegas, USA},
  year = {2016},
  pages = {2727--2735},
}


- License:  GNU General Public License Usage
  Alternatively, this file may be used under the terms of the GNU General
  Public License version 3.0 as published by the Free Software Foundation
  and appearing in the file LICENSE.GPL included in the packaging of this
  file. Please review the following information to ensure the GNU General
  Public License version 3.0 requirements will be met:
  http://www.gnu.org/copyleft/gpl.html.

- Warranty: KAIST-VCLAB MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE 
  SUITABILITY OF THE SOFTWARE, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT 
  LIMITED TO THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A 
  PARTICULAR PURPOSE, OR NON-INFRINGEMENT. KAIST-VCLAB SHALL NOT BE LIABLE FOR ANY 
  DAMAGES SUFFERED BY LICENSEE AS A RESULT OF USING, MODIFYING OR DISTRIBUTING
  THIS SOFTWARE OR ITS DERIVATIVES

*****************************************************************************/

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>
#include <memory>
#include <limits.h>
#include <algorithm>
#include <time.h>

#include "lapinpainting.h"

FILE *logout;

int main(int argc, char **argv){
   
   //Argument description
   //argv[1]   project name
   //argv[2]   rgb image name
   //argv[3]   mask image name
   //argv[4]   patch size
   //argv[5]   distance weight parameter. [Wexler et al. 2007]
   //argv[6]   minimum size for resizing
   //argv[7]   distance metric parameter
   //argv[8]   the number of EM iteration
   //argv[9]   decrease factor of EM iteration
   //argv[10]  minimum EM iteration
   //argv[11]  random search iteration

   // You have to give at least three arguments: a project name, a rgb image and an mask image name.
   // Then, you can run our algorithm with default setting.
   // If you want to change parameters, just give -1 for unchanged varaibles and give a number which you want.
   // EX) bungee bungee.png bungee_mask.png  

	cv::Mat maskmat, colormat, origcolormat, rgbmat; 
	double *colorptr, *maskptr;
	int dheight, dwidth;
	int height, width;
	unsigned char *mask;


	int decrease_factor;
	int min_iter;
	double *scales;
	char *outputfilename, *fname, *processfilename, *dirname;
	time_t timer;

	//inpainting parameter
  	double start_scale;
  	int num_scale;
	int num_em;
	int psz;
	int min_size;
   int rs_iter;
	double gamma;
	double lambda;

   //pyramid
   //gpyr - Gaussian pyramid
   //upyr - upsampled Gaussian pyramid
   //fpyr - Laplacian pyramid
	std::vector<std::pair<int,int> > pyr_size;
	std::vector<cv::Mat> mask_gpyr, color_gpyr;
	std::vector<cv::Mat> mask_upyr, color_upyr;
	std::vector<cv::Mat> mask_fpyr, color_fpyr;
	std::vector<cv::Mat> rgb_gpyr,rgb_fpyr,rgb_upyr;

   //Laplacian inpainting object
	LaplacianInpainting inpainting;

	//logout = fopen("output.txt","w");//for debug

	processfilename = (char*)malloc(sizeof(char) * 200);
	dirname = (char*)malloc(sizeof(char) *200);
	fname = (char*)malloc(sizeof(char) *200);
	outputfilename = (char*)malloc(sizeof(char) *200);
 
   ////////////////////////
	//*Step 1: read input*//
	////////////////////////
   
   if(argc > 3) {
      colormat = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);     //read a rgb image
	   maskmat = cv::imread(argv[3], CV_LOAD_IMAGE_GRAYSCALE);  //read a mask image
   }
   else{
//      printf("No image and no mask.\n"); 
	   printf("Laplacian Patch-Based Image Synthesis (CVPR 2016) ver. 1.0\n");
	   printf("Copyright (c) 2016 Joo Ho Lee, Min H. Kim\n"); 
	   printf("\n"); 
	   printf("Syntax example: lapinpainting bungee bungee.png bungee_mask.png [opt1] [opt2] [opt3] [opt4] [opt5] [opt6] [opt7]\n"); 
	   printf("\n"); 
	   printf("[opt1] patch size: (default) 7\n");
	   printf("[opt2] gamma: (default) 1.3\n");
	   printf("[opt3] minimum size in percentages: (default) 20\n");
	   printf("[opt4] number of EM: (default) 50\n");
	   printf("[opt5] decrease factor in EM: (default) 10\n");
	   printf("[opt6] minimum iteration: (default) 10\n");
	   printf("[opt7] random search iteration: (default) 1\n"); 
	   printf("\n"); 
       return 1;
   }

   psz            = (argc<5 || atoi(argv[4])  == -1) ? 7  : atoi(argv[4]);  //patch size
	gamma          = (argc<6 || atof(argv[5])  == -1) ? 1.3: atof(argv[5]);//gamma
	min_size       = (argc<7 || atoi(argv[6])  == -1) ? 20 : atoi(argv[6]);  //minimum size
	lambda         = (argc<8 || atoi(argv[7])  == -1) ? 0.4: atof(argv[7]);//lambda - ratio btw lab Laplacian patch distance and lab upsampled Gaussian patch distance         
	num_em         = (argc<9 || atoi(argv[8])  == -1) ? 50 : atoi(argv[8]); //the number of EM iteration
	decrease_factor= (argc<10|| atoi(argv[9])  == -1) ? 10 : atoi(argv[9]); //decrease_factor
	min_iter       = (argc<11|| atoi(argv[10]) == -1) ? 10 : atoi(argv[10]);//minimum iteration
   rs_iter        = (argc<12|| atoi(argv[11]) == -1) ? 1  : atoi(argv[11]); //random search iteration
     
	width = colormat.cols;  //image width
	height = colormat.rows; //image height

   int tmp_width = width,tmp_height = height;
   int tmp = 1;
   for(int i=0;;i++){
      tmp_width  >>= 1;
      tmp_height >>= 1;
      if(min_size > tmp_width || min_size > tmp_height)
         break;
      tmp <<= 1;
   }
   printf("%d\n",tmp);

	if(width%tmp) width=width-(width%tmp);
	if(height%tmp) height=height-(height%tmp);
   
   origcolormat = colormat.clone();

	colormat = colormat(cv::Rect(0,0,width, height));  //crop the image
	maskmat = maskmat(cv::Rect(0,0,width, height));


	colormat.convertTo(colormat, CV_32FC3);   //convert an uchar image to a float image (Input of cvtColor function should be a single precision )
	maskmat.convertTo(maskmat,CV_64FC1);      //double mask

	colormat/=255.0;	//255 -> 1.0
	maskmat/=255.0;

	colormat.convertTo(rgbmat, CV_64FC3); 

	//convert rgb to CIEL*a*b*
	cvtColor(colormat, colormat, CV_RGB2Lab); //RGB to Lab
	colormat.convertTo(colormat, CV_64FC3);   //single -> double
	
	//values in mask region should be zero.
	colorptr = (double*) colormat.data;
	maskptr = (double*) maskmat.data;
   
   //refine mask and color image
   for(int i=0;i<height;i++){
		for(int j=0;j<width;j++){
			int ndx = i*width + j;
			if(maskptr[ndx]>0){
				colorptr[3*ndx] = 0;
				colorptr[3*ndx+1] = 0;
				colorptr[3*ndx+2] = 0;
				maskptr[ndx]=1;
			}
			else maskptr[ndx]=0;
		}
	}
   
   ///////////////////////////////////
	//*step 2: set parameters       *//
   ///////////////////////////////////

	inpainting.gamma_=gamma;            //parameter for voting
	inpainting.lambda_=lambda;          //ratio between Laplacian patch distance metric and upsampled Gaussian patch distance metric. 
	inpainting.minsize_=min_size;       //minimum scale
	inpainting.psz_=psz;                //patch size
	inpainting.highconfidence_= 1.0f;   //confidence for non-mask region
	inpainting.patchmatch_iter_ = 10;   //EM iteration
	inpainting.siminterval_ = 3.0f;     //parameter for voting
   inpainting.rs_iter_ = rs_iter;      //random search itertation

	sprintf(fname,"%s_lap_psz%02d_gamma%.2f_lambda%.2f_minsize%02d_simint_%.1f", argv[1], inpainting.psz_, inpainting.gamma_,inpainting.lambda_,inpainting.minsize_,inpainting.siminterval_); 

   ///////////////////////////////////
	//*step 3: generate pyramid     *//
   ///////////////////////////////////

   inpainting.constructLaplacianPyr(rgb_gpyr, rgb_upyr, rgb_fpyr, rgbmat);

	//construct Laplacian pyramid
	inpainting.constructLaplacianPyr(color_gpyr, color_upyr, color_fpyr, colormat);
	inpainting.constructLaplacianPyr(mask_gpyr, mask_upyr, mask_fpyr, maskmat);

	//reverse order (from low-res to high-res)
	std::reverse(color_gpyr.begin(), color_gpyr.end());
	std::reverse(color_upyr.begin(), color_upyr.end());
	std::reverse(color_fpyr.begin(), color_fpyr.end());
	std::reverse(mask_gpyr.begin(), mask_gpyr.end());
	std::reverse(mask_upyr.begin(), mask_upyr.end());
	std::reverse(mask_fpyr.begin(), mask_fpyr.end());

	//compute pyr_size
   pyr_size.clear();
	
   //set size
   for(int i=0;i<color_gpyr.size();i++){
		pyr_size.push_back(std::pair<int,int>(color_gpyr[i].rows, color_gpyr[i].cols));
		printf("%dth image size: %d %d\n", i,color_gpyr[i].rows,color_gpyr[i].cols);
	}

   //refine mask
	fixDownsampledMaskMatColorMat(mask_gpyr[0],color_gpyr[0]);
	
   for(int i=0;i<mask_upyr.size();i++){
		fixDownsampledMaskMatColorMat(mask_upyr[i],color_upyr[i]);
		fixDownsampledMaskMatColorMat(mask_gpyr[i+1],color_gpyr[i+1]);
		color_fpyr[i]=color_gpyr[i+1]-color_upyr[i];
		
      mask_upyr[i]=mask_gpyr[i+1]+mask_upyr[i];
		fixDownsampledMaskMat(mask_upyr[i]);
		fixDownsampledMaskMatColorMat(mask_upyr[i],color_upyr[i]);
		fixDownsampledMaskMatColorMat(mask_upyr[i],color_gpyr[i+1]);
		
      //	displayMat<double>(mask_upyr[i]-mask_gpyr[i+1],"gpyr",cv::Rect(0,0,mask_gpyr[i+1].cols,mask_gpyr[i+1].rows));
      //	displayMat<double>(mask_upyr[i],"upyr",cv::Rect(0,0,mask_upyr[i].cols,mask_upyr[i].rows));
		//		cv::waitKey();
		//		if(i<mask_upyr.size()-1)	
		//			cv::pyrUp(mask_upyr[i],mask_upyr[i+1],cv::Size(mask_upyr[i+1].cols,mask_upyr[i+1].rows));
	}	

	//dilate mask
	int pyrlevel = color_gpyr.size();

   /////////////////////////////////////////////
	//*step 4: initialize the zero level image*//
   /////////////////////////////////////////////

	cv::Mat color8u, mask8u, feature8u;
	cv::Mat repmask;
	cv::Mat trg_color;
	cv::Mat trg_feature;

	double featuremin, featuremax;
	cv::minMaxLoc(color_fpyr[0], &featuremin, &featuremax);

	color_upyr[0].convertTo(color8u,CV_32FC3);
	cvtColor(color8u, color8u, CV_Lab2RGB);
	color8u = color8u*255.;
	mask8u = mask_upyr[0]*255.;
	//	cv::imshow("asdf",mask8u);
	//	cv::waitKey();

	feature8u = (color_fpyr[0]-featuremin)/(featuremax-featuremin) * 255.;

	color8u.convertTo(color8u, CV_8U);
	mask8u.convertTo(mask8u, CV_8U);
	feature8u.convertTo(feature8u, CV_8U);

   //initialization
   //We use a Navier-Stokes based method [Navier et al. 01] only for initialization.
   //http://www.math.ucla.edu/~bertozzi/papers/cvpr01.pdf
	cv::inpaint(color8u, mask8u, color8u, 10, 0);
	cv::inpaint(feature8u, mask8u, feature8u, 10, 0);

	color8u.convertTo(color8u,CV_32FC3);
	color8u=color8u/255.f;
	cvtColor(color8u, color8u, CV_RGB2Lab);
	color8u.convertTo(color_upyr[0],CV_64FC3);
	feature8u.convertTo(color_fpyr[0],CV_64FC3);
	color_fpyr[0] = color_fpyr[0]/255.0 * (featuremax-featuremin) + featuremin;
	//	depth8u.convertTo(depth_gpyr[0],CV_64FC1);

	trg_color = color_upyr[0].clone();
	trg_feature = color_fpyr[0].clone();

	//displayMat<double>(trg_feature,"feature",cv::Rect(0,0,trg_feature.cols, trg_feature.rows));
	//displayLABMat(trg_color,"color",cv::Rect(0,0,trg_color.cols, trg_color.rows));
	int cur_iter = num_em;

   /////////////////////////////////
	//*Step 5: Do image completion*//
   /////////////////////////////////


	cv::Mat nnf, nnff;
	cv::Mat nnferr;
	cv::Mat nxt_color;
	bool *patch_type = NULL;

	nnf = cv::Mat::zeros(pyr_size[1].first, pyr_size[1].second, CV_32SC2); // H x W x 2 int

	clock_t t;
	clock_t recont,accumt;
	int f;
	accumt=0;
	t=clock();

	for(int ilevel = 0; ilevel < color_upyr.size(); ilevel++){
		printf("Processing %dth scale image\n", ilevel);

		if(ilevel){
			
			//resize trg_color, trg_depth, trg_feature
			recont = clock();
			nxt_color = trg_color + trg_feature; //Gaussian = upsampled Gaussian + Laplacian
			recont = clock()-recont;
			accumt+=recont;

			cv::pyrUp(nxt_color, trg_color, cv::Size(trg_color.cols*2, trg_color.rows*2)); // upsample a low-level Gaussian image
			cv::pyrUp(trg_feature, trg_feature, cv::Size(trg_feature.cols*2, trg_feature.rows*2)); //upsample a Laplacian image (we will reset a initial laplacian image later)

			double *trgcptr = (double*) trg_color.data;
			double *trgfptr = (double*) trg_feature.data;
			double *maskptr = (double*) mask_upyr[ilevel].data;
			int *nnfptr = (int*) nnf.data;

			//initialize
			for(int i=0;i<pyr_size[ilevel+1].first;i++){
				for(int j=0;j<pyr_size[ilevel+1].second;j++){
					int ndx = i * pyr_size[ilevel+1].second + j;
					if(maskptr[ndx]<0.1){
						trgcptr[3*ndx] = ((double*)(color_upyr[ilevel].data))[3*ndx];
						trgcptr[3*ndx+1] = ((double*)(color_upyr[ilevel].data))[3*ndx+1];
						trgcptr[3*ndx+2] = ((double*)(color_upyr[ilevel].data))[3*ndx+2]; 
						trgfptr[3*ndx] = ((double*)(color_fpyr[ilevel].data))[3*ndx];
						trgfptr[3*ndx+1] = ((double*)(color_fpyr[ilevel].data))[3*ndx+1];
						trgfptr[3*ndx+2] = ((double*)(color_fpyr[ilevel].data))[3*ndx+2]; 
					}
				}
			}

         //NNF propagation
			recont = clock();
         inpainting.upscaleImages(nnf, nnferr, patch_type, trg_feature, mask_upyr[ilevel-1].clone(), mask_upyr[ilevel].clone());
			recont = clock() - recont;
			accumt += recont;


         //upscale NNF field
			nnf.convertTo(nnff,CV_64FC2);
			cv::resize(nnff, nnff, cv::Size(pyr_size[ilevel+1].second, pyr_size[ilevel+1].first),cv::INTER_LINEAR);
			nnff.convertTo(nnf,CV_32SC2);
			nnff = nnf * 2;
		}

      if(patch_type != NULL)
   		free(patch_type);
		patch_type = (bool*) malloc(sizeof(bool) * pyr_size[ilevel+1].first * pyr_size[ilevel+1].second); 

		nnferr = cv::Mat::ones(pyr_size[ilevel+1].first, pyr_size[ilevel+1].second, CV_64FC1); // H x W x 1 double

		//do EM iteration
		sprintf(processfilename, "%s_scale%02d", fname, ilevel); 
		inpainting.doEMIterLap(nnf, nnferr, patch_type, trg_color, trg_feature, mask_upyr[ilevel].clone(), pyr_size[ilevel+1], cur_iter, cv::Size(width, height), processfilename);

		//compute next iteration
		cur_iter -= decrease_factor;
		if(cur_iter<min_iter)
			cur_iter=min_iter;
	}

   //print final result
	cv::Mat tmpimg;	
	sprintf(outputfilename,"%s_final.png", fname); 

	tmpimg = trg_color.clone() + trg_feature.clone();
	tmpimg.convertTo(tmpimg, CV_32FC3);
	cvtColor(tmpimg, tmpimg, CV_Lab2RGB);
	tmpimg=255*tmpimg;
	tmpimg.convertTo(tmpimg, CV_8UC3);
	//		displayMat<double>(tmpimg,outputfilename,cv::Rect(0,0,tmpimg.cols, tmpimg.rows));
	cv::imwrite(outputfilename, tmpimg);

	t = clock() - t;
	//printf ("It took %d clicks (%f seconds, (%d,%f) for reconstruction).\n",(int)t,((float)t)/CLOCKS_PER_SEC, (int)accumt, (float)accumt/CLOCKS_PER_SEC);
	float total_secs = ((float)t)/((float)CLOCKS_PER_SEC);
	int mins = (int)floor(total_secs/60.f);
	int secs = (int)total_secs - (mins*60);
	printf ("It took %d:%d (minutes:seconds).\n", mins, secs);

    free(processfilename);
	free(dirname);
	free(fname);
	free(outputfilename);

   return 0;
}
