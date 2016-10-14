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

#include "lapinpainting.h"

void displayLABMat(cv::Mat a, char *title, cv::Rect ROI){
   cv::Mat tmp;
   a.convertTo(tmp, CV_32FC3);
   cvtColor(tmp, tmp, CV_Lab2RGB);
   tmp=tmp*255;
   tmp.convertTo(tmp,CV_8UC3);
   cv::imshow(title, tmp);
   cv::waitKey();
}

__inline double computePatchError(double *patcha, double *patchb, int psz){
   int pixeln = psz*psz*3; // 3 channels
   double sum = 0;
   for(int i = 0 ; i < pixeln; i++)
      sum+=(patcha[i]-patchb[i])*(patcha[i]-patchb[i]);
   return sum;
}

__inline double computePatchErrorLap(std::vector<double*> &colorpatches, std::vector<double*> &colorfpatches,int x, int y, int psz, double lambda){
   int pixeln = psz*psz*3; // 3 channels

   double *patcha, *patchb, *patchfa, *patchfb;

   patcha = (double*)colorpatches[x];
   patchb = (double*)colorpatches[y];
   patchfa = (double*)colorfpatches[x];
   patchfb = (double*)colorfpatches[y];

   double sum = 0;
   for(int i = 0 ; i < pixeln; i++)
      sum += (1-lambda)*(patcha[i]-patchb[i])*(patcha[i]-patchb[i]);
   for(int i = 0 ; i < pixeln; i++)
      sum += (lambda)*(patchfa[i]-patchfb[i])*(patchfa[i]-patchfb[i]);
   return sum;
}

void fixDownsampledMaskMat(cv::Mat mask){
   double TT = 0.6;	
   double *maskptr = (double*) mask.data;

   for(int i=0;i<mask.rows;i++){
      for(int j=0;j<mask.cols;j++){
         int ndx = i*mask.cols+j;
         if(maskptr[ndx]>TT){
            maskptr[ndx]=1;
         }
         else{
            maskptr[ndx]=0;
         }
      }
   }
}

void fixDownsampledMaskMatColorMat(cv::Mat mask,cv::Mat color){
   double TT = 0.6;	
   double *maskptr = (double*) mask.data;
   double *colorptr = (double*) color.data;

   for(int i=0;i<mask.rows;i++){
      for(int j=0;j<mask.cols;j++){
         int ndx = i*mask.cols+j;
         if(maskptr[ndx]>TT){
            maskptr[ndx]=1;
            colorptr[3*ndx]=0;
            colorptr[3*ndx+1]=0;
            colorptr[3*ndx+2]=0;
         }
         else{
            colorptr[3*ndx]=colorptr[3*ndx]/(1-maskptr[ndx]);
            colorptr[3*ndx+1]=colorptr[3*ndx+1]/(1-maskptr[ndx]);
            colorptr[3*ndx+2]=colorptr[3*ndx+2]/(1-maskptr[ndx]);
            maskptr[ndx]=0;
         }
      }
   }
}

void LaplacianInpainting::constructLaplacianPyrMask(std::vector<cv::Mat> &gpyr, std::vector<cv::Mat> &upyr, std::vector<cv::Mat> &fpyr,cv::Mat mask,cv::Mat &img){

   cv::Mat prvimg, curimg, curfimg, upimg;
   cv::Mat prvmask, curmask, upmask;
   gpyr.push_back(img);
   prvimg = img;
   prvmask = mask;

   for(;prvimg.cols>=2*minsize_&&prvimg.rows>=2*minsize_;){
      cv::pyrDown(prvimg, curimg, cv::Size(prvimg.cols/2, prvimg.rows/2));
      cv::pyrUp(curimg, upimg, cv::Size(curimg.cols*2, curimg.rows*2));
      cv::pyrDown(prvmask, curmask, cv::Size(prvimg.cols/2, prvimg.rows/2));
      cv::pyrUp(curmask, upmask, cv::Size(curimg.cols*2, curimg.rows*2));

      //		fixDownsampledMaskColorMat(upmask, upimg);
      //		fixDownsampledMaskColorMat(curmask, curimg);

      curfimg = prvimg - upimg;
      gpyr.push_back(curimg);
      upyr.push_back(upimg);
      fpyr.push_back(curfimg);
      //displayLABMat(curimg, "gaussian", cv::Rect(0, 0, curimg.rows, curimg.cols));
      //displayLABMat(upimg, "up gaussian", cv::Rect(0, 0, curimg.rows, curimg.cols));
      //displayLABMat(curfimg, "lap", cv::Rect(0, 0, curimg.rows, curimg.cols));
      prvimg=curimg;
      prvmask=curmask;
   }
}

void LaplacianInpainting::constructLaplacianPyr(std::vector<cv::Mat> &gpyr, std::vector<cv::Mat> &upyr, std::vector<cv::Mat> &fpyr,cv::Mat &img){

   cv::Mat prvimg, curimg, curfimg, upimg;
   gpyr.push_back(img.clone());
   prvimg = img.clone();

   for(;prvimg.cols>=2*minsize_&&prvimg.rows>=2*minsize_;){
      cv::pyrDown(prvimg, curimg, cv::Size(prvimg.cols/2, prvimg.rows/2));
      cv::pyrUp(curimg, upimg, cv::Size(curimg.cols*2, curimg.rows*2));
      curfimg = prvimg - upimg;
      gpyr.push_back(curimg);
      upyr.push_back(upimg);
      fpyr.push_back(curfimg);
      //displayLABMat(curimg, "gaussian", cv::Rect(0, 0, curimg.rows, curimg.cols));
      //displayLABMat(upimg, "up gaussian", cv::Rect(0, 0, curimg.rows, curimg.cols));
      //displayLABMat(curfimg, "lap", cv::Rect(0, 0, curimg.rows, curimg.cols));
      prvimg=curimg;
   }
}

void LaplacianInpainting::findNearestNeighborLap(cv::Mat nnf,cv::Mat nnferr, bool *patch_type, cv::Mat colormat, cv::Mat colorfmat, cv::Mat maskmat, std::pair<int, int> size,int emiter){

   /*Patch preparation*/		
   std::vector<double*> colorpatches,colorfpatches;

   srand(time(NULL));

   double *maskptr = (double*) maskmat.data;
   double *colorptr = (double*) colormat.data;
   double *colorfptr = (double*) colorfmat.data;

   double errmin, errmax;

   int tmph = size.first - psz_ + 1; //we ignore 
   int tmpw = size.second - psz_ + 1;
   int randomcnt=0, propagationcnt=0;
   int lurow,lucol,rdrow, rdcol;

   lurow = tmph;
   lucol = tmpw;
   rdrow = 0;
   rdcol = 0;


   double *colorpatch, *colorfpatch;

   //collect patches
   for(int i=0;i<tmph;i++){
      for(int j=0;j<tmpw;j++){
         int ndx = i * size.second + j;

         int flag=0;
         colorpatch  = (double*)malloc(sizeof(double) * psz_ * psz_ * 3);		
         colorfpatch  = (double*)malloc(sizeof(double) * psz_ * psz_ * 3);		

         //copy patch
         for(int i2=0;i2<psz_;i2++){
            for(int j2=0;j2<psz_;j2++){
               int ndx2 = (i+i2) * size.second + (j+j2);
               int pndx = i2 * psz_ + j2;
               if(maskptr[ndx2]>0.00)
                  flag=1;

               colorpatch [3 * pndx	] = colorptr[3 * ndx2];
               colorpatch [3 * pndx + 1] = colorptr[3 * ndx2 + 1];
               colorpatch [3 * pndx + 2] = colorptr[3 * ndx2 + 2];
               colorfpatch [3 * pndx	] = colorfptr[3 * ndx2];
               colorfpatch [3 * pndx + 1] = colorfptr[3 * ndx2 + 1];
               colorfpatch [3 * pndx + 2] = colorfptr[3 * ndx2 + 2];
            }
         }

         if(flag){ // find bounding box
            rdrow = std::max(rdrow, i);
            rdcol = std::max(rdcol, j);
            lurow = std::min(lurow, i);
            lucol = std::min(lucol, j);
         }

         patch_type [ndx] = flag;//If variable flag is one, there is a mask pixel in the patch.
         colorpatches.push_back(colorpatch);//Note that index for patches is i*tmpw + j since we only take inner patches.
         colorfpatches.push_back(colorfpatch);//feature patch
      }
   }

   rdrow = std::min(rdrow+2*psz_, tmph-1);
   rdcol = std::min(rdcol+2*psz_, tmpw-1);
   lurow = std::max(lurow-2*psz_, 0);
   lucol = std::max(lucol-2*psz_, 0);		

   nnfcount_=(rdcol-lucol)*(rdrow-lurow);

   int* nnfptr = (int*)nnf.data;
   double* nnferrptr = (double*)nnferr.data;

   /*Initialize NNF*/
   for(int i=lurow;i<=rdrow;i++){
      for(int j=lucol;j<=rdcol;j++){
         int ndx = i * size.second + j;
         int newrow, newcol;
         double newerr;

         //nnferrptr[ndx] = computePatchErrorLap(colorpatches, colorfpatches, i*tmpw+j, nnfptr[ndx*2] * tmpw +nnfptr[ndx*2 + 1], psz_, lambda_); 
         nnferrptr[ndx] = 50000;

         do{
            newrow = rand() % tmph;//row
            newcol = rand() % tmpw;//col
         }while(patch_type[newrow*size.second+newcol]||(newrow==i&&newcol==j));//Until patch is from a source patch. If the pointed patch is a target, reset values.

         newerr = computePatchErrorLap(colorpatches, colorfpatches, i*tmpw+j, newrow * tmpw + newcol, psz_, lambda_); 

         if(emiter == -1){
            nnfptr[ndx*2	]=newrow;//row
            nnfptr[ndx*2 + 1]=newcol;//col
            nnferrptr[ndx] = newerr;
         }
         else{
            if(nnferrptr[ndx] >newerr || patch_type[nnfptr[ndx*2]*size.second+nnfptr[ndx*2+1]]){
               nnfptr[ndx*2	]=newrow;//row
               nnfptr[ndx*2 + 1]=newcol;//col
               nnferrptr[ndx] = newerr;
            }
         }
      }
   }

   //cv::minMaxLoc(nnferr, &errmin, &errmax);
   //printf("max error: %lf\n", errmax);
   //cv::imshow("nnf error", nnferr/errmax);
   //cv::waitKey();
   /*Patchmatch start*/

   for(int patchiter = 0; patchiter < patchmatch_iter_; patchiter++){

      /*random search*/
      for(int i=lurow;i<=rdrow;i++){
         for(int j=lucol;j<=rdcol;j++){

            int vrow, vcol;
            int ndx = i*size.second + j;
            int w_row = tmph, w_col = tmpw;
            double alpha = 0.5;
            int cur_row, cur_col;
            double newerr;
            int row1, row2, col1, col2;
            int ranr, ranc;

            vrow = nnfptr[ndx*2];
            vcol = nnfptr[ndx*2+1];

            cur_row = w_row;
            cur_col = w_col;

            for(int h=0;cur_row>=1&&cur_col>=1;h++){
               //
               row1 = vrow - cur_row;
               row2 = vrow + cur_row+1;
               col1 = vcol - cur_col;
               col2 = vcol + cur_col+1;

               //cropping
               if(row1<0) row1 = 0;
               if(row2>tmph) row2 = tmph;
               if(col1<0) col1 = 0;
               if(col2>tmpw) col2 = tmpw;

               for(int k = 0 ; k < rs_iter_;k++){

                  do{
                     ranr = (rand() % (row2 - row1)) + row1;//2~4 2,5	 3	0,1,2 + 2
                     ranc = (rand() % (col2 - col1)) + col1;
                  }while(patch_type[ranr * size.second + ranc]);

                  newerr =  computePatchErrorLap(colorpatches, colorfpatches, i*tmpw+j, ranr * tmpw + ranc, psz_, lambda_); 

                  if(newerr < nnferrptr[ndx]){
                     randomcnt++;
                     nnfptr[ndx*2	] = ranr;//row
                     nnfptr[ndx*2 + 1] = ranc;//col
                     nnferrptr[ndx]	= newerr;						
                  }
               }
               //shrink a window size
               cur_row >>= 1;
               cur_col >>= 1;
            }
         }
      }

      if(patchiter&1){//odd leftup order

         for(int i=rdrow;i>=lurow;i--){
            for(int j=rdcol;j>=lucol;j--){
               //			for(int i=tmph-1;i>=0;i--){
               //				for(int j=tmpw-1;j>=0;j--){

               int vrow, vcol;
               int ndx = i*size.second + j;
               int w_row = tmph, w_col = tmpw;
               double alpha = 0.5;
               int cur_row, cur_col;
               double newerr;
               int row1, row2, col1, col2;
               int ranr, ranc;

               int vrowright, vcolright;
               int vrowdown, vcoldown;

               /*propagation*/
               if(j<rdcol){//left
                  vrowright = nnfptr[ndx * 2 + 2];
                  vcolright = nnfptr[ndx * 2 + 3];
                  if(vcolright>0)
                     --vcolright;

                  if(!patch_type[vrowright*size.second + vcolright]){

                     newerr = computePatchErrorLap(colorpatches, colorfpatches, i*tmpw+j, vrowright * tmpw + vcolright, psz_, lambda_); 

                     if(newerr < nnferrptr[ndx]){
                        propagationcnt++;
                        nnfptr[ndx*2	] = vrowright;//row
                        nnfptr[ndx*2 + 1] = vcolright;//col
                        nnferrptr[ndx]	= newerr;
                     }
                  }
               }

               if(i<rdrow){//right
                  vrowdown = nnfptr[(ndx+size.second) * 2	];
                  vcoldown = nnfptr[(ndx+size.second) * 2 + 1];
                  if(vrowdown>0)
                     --vrowdown;
                  if(!patch_type[vrowdown*size.second + vcoldown]){

                     newerr = computePatchErrorLap(colorpatches, colorfpatches, i*tmpw+j, vrowdown * tmpw + vcoldown, psz_, lambda_); 

                     if(newerr < nnferrptr[ndx]){
                        propagationcnt++;
                        nnfptr[ndx*2	] = vrowdown;//row
                        nnfptr[ndx*2 + 1] = vcoldown;//col
                        nnferrptr[ndx]	= newerr;
                     }
                  }
               }
            }
         }
      }
      else{//even

         for(int i=lurow;i<=rdrow;i++){
            for(int j=lucol;j<=rdcol;j++){
               //			for(int i=0;i<tmph;i++){//right down order
               //				for(int j=0;j<tmpw;j++){

               int vrow, vcol;
               int ndx = i*size.second + j;
               int w_row = tmph, w_col = tmpw;
               double alpha = 0.5;
               int cur_row, cur_col;
               double newerr;
               int row1, row2, col1, col2;
               int ranr, ranc;

               int vrowleft, vcolleft;
               int vrowup, vcolup;

               /*propagation*/
               if(j>lucol){//left
                  vrowleft = nnfptr[ndx * 2 - 2];
                  vcolleft = nnfptr[ndx * 2 - 1];
                  if(vcolleft<tmpw-1)
                     ++vcolleft;

                  if(!patch_type[vrowleft*size.second + vcolleft]){

                     newerr = computePatchErrorLap(colorpatches, colorfpatches, i*tmpw+j, vrowleft * tmpw + vcolleft, psz_, lambda_); 

                     if(newerr < nnferrptr[ndx]){
                        propagationcnt++;
                        nnfptr[ndx*2	] = vrowleft;//row
                        nnfptr[ndx*2 + 1] = vcolleft;//col
                        nnferrptr[ndx]	= newerr;
                     }
                  }
               }

               if(i>lurow){//up
                  vrowup = nnfptr[(ndx-size.second) * 2	];
                  vcolup = nnfptr[(ndx-size.second) * 2 + 1];
                  if(vrowup<tmph-1)
                     ++vrowup;
                  if(!patch_type[vrowup*size.second + vcolup]){

                     newerr = computePatchErrorLap(colorpatches, colorfpatches, i*tmpw+j, vrowup * tmpw + vcolup, psz_, lambda_); 

                     if(newerr < nnferrptr[ndx]){
                        propagationcnt++;
                        nnfptr[ndx*2	] = vrowup;//row
                        nnfptr[ndx*2 + 1] = vcolup;//col
                        nnferrptr[ndx]	= newerr;
                     }
                  }
               }
            }
         }
      }
   }

   while(!colorpatches.empty()){
      free(colorpatches.back());
      colorpatches.pop_back();
   }
   while(!colorfpatches.empty()){
      free(colorfpatches.back());
      colorfpatches.pop_back();
   }

   printf("(pu: %d, ru: %d)", propagationcnt, randomcnt); 
}

void LaplacianInpainting::colorVoteLap(cv::Mat nnf, cv::Mat nnferr, bool *patch_type, cv::Mat colormat, cv::Mat colorfmat, cv::Mat maskmat, std::pair<int, int> size){

   int tmph = size.first - psz_ + 1;
   int tmpw = size.second - psz_ + 1;

   cv::Mat weight;
   cv::Mat colorsum, colorfsum;
   cv::Mat dist;
   cv::Mat similarity;
   cv::Mat squarednnferr;
   double nnfavg, nnfsqavg, variance;
   double *nnferrptr = (double*) nnferr.data;
   double maskcnt = 0;

   nnfavg = 0;
   nnfsqavg = 0;

   nnfavg = cv::sum(nnferr).val[0];
   cv::multiply(nnferr,nnferr,squarednnferr);
   nnfsqavg = cv::sum(squarednnferr).val[0];

   nnfavg /= nnfcount_;
   nnfsqavg /= nnfcount_;
   variance = nnfsqavg - nnfavg * nnfavg;

   //	printf("variance: %lf\n", variance);

   //Wexler's similarity function
   //cv::exp( - nnferr / (2.0 * (nnfavg + 0.68 * sqrt(variance)) * (nnfavg + 0.68 * sqrt(variance)) * siminterval_), similarity);//0.68 percentile
  
   //ours
   cv::exp( - nnferr / (2.0 * (nnfavg + 0.68 * sqrt(variance))  * siminterval_), similarity);//0.68 percentile

   double *colorptr = (double*) colormat.data;
   double *colorfptr = (double*) colorfmat.data;
   double *similarityptr = (double*) similarity.data;

   maskmat.convertTo(maskmat, CV_8UC1);
   cv::distanceTransform(maskmat, dist, CV_DIST_L1, 3);

   //Wexler's distance-weight function
   //dist = dist * log(gamma_) * -1;
   //cv::exp(dist, dist);

   //ours
   pow(dist, -1 * gamma_, dist);
   
   weight = cv::Mat::zeros(size.first, size.second, CV_64FC1);
   colorsum = cv::Mat::zeros(size.first, size.second, CV_64FC3);
   colorfsum = cv::Mat::zeros(size.first, size.second, CV_64FC3);

   double *weightptr = (double*) weight.data;
   double *colorsumptr = (double*) colorsum.data;
   double *colorfsumptr = (double*) colorfsum.data;
   float *distptr = (float*) dist.data;
   int *nnfptr = (int*) nnf.data;
   unsigned char *maskptr = (unsigned char*) maskmat.data;

   for(int i=0;i<size.first;i++){
      for(int j=0;j<size.second;j++){
         int ndx = i*size.second + j;
         if(maskptr[ndx]==0)
            distptr[ndx]=highconfidence_;
      }
   }

   for(int i=0;i<tmph;i++){
      for(int j=0;j<tmpw;j++){
         int ndx= i*size.second+j;
         int patchcenter_ndx= (i+(psz_>>1))*size.second+j+(psz_>>1);
         double alpha = 0.0;
#ifdef CENTERINMASK
         if(maskptr[patchcenter_ndx]>0.0){
#else
         if(patch_type[ndx]){//If a patch is a target patch
#endif
            alpha = distptr[patchcenter_ndx];
            //pixel by pixel
            for(int i2=0;i2<psz_;i2++){
               for(int j2=0;j2<psz_;j2++){
                  int ndx2 = (i+i2)*size.second + (j+j2);
                  int ndx3 = (nnfptr[2*ndx]+i2) * size.second + nnfptr[2*ndx+1]+j2; 

                  weightptr[ndx2] += alpha * similarityptr[ndx];
                  colorsumptr[3*ndx2] += alpha * similarityptr[ndx] * colorptr[3*ndx3	];
                  colorsumptr[3*ndx2+1] += alpha * similarityptr[ndx] * colorptr[3*ndx3+1];
                  colorsumptr[3*ndx2+2] += alpha * similarityptr[ndx] * colorptr[3*ndx3+2];
                  colorfsumptr[3*ndx2] += alpha * similarityptr[ndx] * colorfptr[3*ndx3	];
                  colorfsumptr[3*ndx2+1] += alpha * similarityptr[ndx] * colorfptr[3*ndx3+1];
                  colorfsumptr[3*ndx2+2] += alpha * similarityptr[ndx] * colorfptr[3*ndx3+2];
               }
            }
         }
      }
   }


   //normalize
   for(int i=0;i<size.first;i++){
      for(int j=0;j<size.second;j++){
         int ndx = i*size.second +j;
         if(maskptr[ndx]>0.0){
            colorptr [3*ndx]   = colorsumptr[3*ndx] / weightptr[ndx];
            colorptr [3*ndx+1] = colorsumptr[3*ndx+1] / weightptr[ndx];
            colorptr [3*ndx+2] = colorsumptr[3*ndx+2] / weightptr[ndx];
            colorfptr [3*ndx]   = colorfsumptr[3*ndx] / weightptr[ndx];
            colorfptr [3*ndx+1] = colorfsumptr[3*ndx+1] / weightptr[ndx];
            colorfptr [3*ndx+2] = colorfsumptr[3*ndx+2] / weightptr[ndx];
         }
      }
   }
   //	cv::imshow("color", colormat);	
   //	cv::waitKey();

}

void LaplacianInpainting::upscaleImages(cv::Mat nnf, cv::Mat nnferr, bool *patch_type, cv::Mat colorfmat,  cv::Mat dmaskmat,  cv::Mat umaskmat){

   std::pair<int, int> dsize (nnf.rows, nnf.cols), usize(colorfmat.rows, colorfmat.cols);

   int dtmph = dsize.first - psz_ + 1;
   int dtmpw = dsize.second - psz_ + 1;

   cv::Mat weight;
   cv::Mat colorfsum;
   cv::Mat dist;
   cv::Mat similarity;
   cv::Mat squarednnferr;
   double nnfavg, nnfsqavg, variance;
   double *nnferrptr = (double*) nnferr.data;
   double maskcnt = 0;

   nnfavg = 0;
   nnfsqavg = 0;

   nnfavg = cv::sum(nnferr).val[0];
   cv::multiply(nnferr,nnferr,squarednnferr);
   nnfsqavg = cv::sum(squarednnferr).val[0];

   nnfavg /= nnfcount_;
   nnfsqavg /= nnfcount_;
   variance = nnfsqavg - nnfavg * nnfavg;


   //Wexler's similarity function
   //cv::exp( - nnferr / (2.0 * (nnfavg + 0.68 * sqrt(variance)) * (nnfavg + 0.68 * sqrt(variance)) * siminterval_), similarity);//0.68 percentile
   
   //ours
   cv::exp( - nnferr / (2.0 * (nnfavg + 0.68 * sqrt(variance))  * siminterval_), similarity);//0.68 percentile

   double *colorfptr = (double*) colorfmat.data;
   double *similarityptr = (double*) similarity.data;

   dmaskmat.convertTo(dmaskmat, CV_8UC1);
   cv::distanceTransform(dmaskmat, dist, CV_DIST_L1, 3);
   //Wexler's distance-weight function
   //dist = dist * log(gamma_) * -1;
   //cv::exp(dist, dist);

   //ours
   pow(dist, -1 * gamma_, dist);

   weight = cv::Mat::zeros(usize.first, usize.second, CV_64FC1);
   colorfsum = cv::Mat::zeros(usize.first, usize.second, CV_64FC3);

   double *weightptr = (double*) weight.data;
   double *colorfsumptr = (double*) colorfsum.data;
   float *distptr = (float*) dist.data;
   int *nnfptr = (int*) nnf.data;
   unsigned char *dmaskptr = (unsigned char*) dmaskmat.data;
   double *umaskptr = (double*) umaskmat.data;

   for(int i=0;i<dsize.first;i++){
      for(int j=0;j<dsize.second;j++){
         int ndx = i*dsize.second + j;
         if(!dmaskptr[ndx])
            distptr[ndx]=highconfidence_;
      }
   }

   for(int i=0;i<dtmph;i++){
      for(int j=0;j<dtmpw;j++){
         int dndx= i*dsize.second+j;
         int undx= (2*i) * usize.second + 2*j;
         int patchcenter_dndx= (i+(psz_>>1))*dsize.second+j+(psz_>>1);
         double alpha = 0.0;
#ifdef CENTERINMASK
         if(dmaskptr[patchcenter_dndx]>0.0){
#else
         if(patch_type[dndx]){//If a patch is a target patch
#endif
            alpha = distptr[patchcenter_dndx];
            //pixel by pixel
            for(int i2=0;i2<psz_*2;i2++){
               for(int j2=0;j2<psz_*2;j2++){
                  int undx2 = (2*i+i2)*usize.second + (2*j+j2);
                  int undx3 = (2*nnfptr[2*dndx]+i2) * usize.second + 2 * nnfptr[2*dndx+1]+j2; 

                     weightptr[undx2] += alpha * similarityptr[dndx];
                     colorfsumptr[3*undx2] += alpha * similarityptr[dndx] * colorfptr[3*undx3	];
                     colorfsumptr[3*undx2+1] += alpha * similarityptr[dndx] * colorfptr[3*undx3+1];
                     colorfsumptr[3*undx2+2] += alpha * similarityptr[dndx] * colorfptr[3*undx3+2];
                 
               }
            }
         }
      }
   }

   //normalize
   for(int i=0;i<usize.first;i++){
      for(int j=0;j<usize.second;j++){
         int undx = i*usize.second +j;
         if(umaskptr[undx]>0.0){
            colorfptr [3*undx]   = colorfsumptr[3*undx] / weightptr[undx];
            colorfptr [3*undx+1] = colorfsumptr[3*undx+1] / weightptr[undx];
            colorfptr [3*undx+2] = colorfsumptr[3*undx+2] / weightptr[undx];
         }
      }
   }
}

void LaplacianInpainting::doEMIterLap(cv::Mat nnf, cv::Mat nnferr, bool *patch_type, cv::Mat colormat, cv::Mat colorfmat, cv::Mat maskmat, std::pair<int, int> size, int num_emiter, cv::Size orig_size, char *processfilename){
   double errmin, errmax;
   char *outputfilename;

   outputfilename = (char*) malloc(sizeof(char) * 300);

   cv::Mat a, tmpimg,tmpimg2,tmpimg1;

   for(int emiter = 0; emiter< num_emiter; emiter++){
      //compute the nearest neighbor fields
      printf("computing %dth NNF", emiter);
      findNearestNeighborLap(nnf, nnferr, patch_type, colormat, colorfmat, maskmat.clone(), size, emiter);

      //show err results
      //cv::minMaxLoc(nnferr, &errmin, &errmax);
      //printf("max error: %lf\n", errmax);
      //cv::imshow("nnf error", nnferr/errmax);	
      //cv::waitKey();

      //update a color image
      printf("-> voting");
      colorVoteLap(nnf, nnferr, patch_type, colormat,  colorfmat, maskmat.clone(), size);


      printf("-> finish\n");
   }

   //	free(patch_type);
   free(outputfilename);
}
