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

#pragma once


#include <iostream>
#include <math.h>
/*OPENCV library*/
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videostab/inpainting.hpp>
#include <time.h>

#define MASK_THD 0.1
//#define DISPLAYMODE
//#define PRINT_MIDDLERESULTS 1
#define CENTERINMASK 1

void displayLABMat(cv::Mat a, char *title, cv::Rect ROI);

template <typename T>
void displayMat(cv::Mat a, char *title, cv::Rect ROI){
	T amin, amax;
	cv::minMaxLoc(a(ROI), &amin, &amax);
	cv::imshow(title, (a(ROI)-amin)/(amax-amin));
	//cv::waitKey();
}

template <typename T>
void displayMatres(cv::Mat a, char *title, cv::Rect ROI,int width, int height){
	T amin, amax;
	cv::Mat tmp;
	cv::minMaxLoc(a(ROI), &amin, &amax);
	cv::resize((a(ROI)-amin)/(amax-amin), tmp,cv::Size(width, height));
	cv::imshow(title, tmp);
	//cv::waitKey();
}

void fixDownsampledMaskMat(cv::Mat mask);
void fixDownsampledMaskMatColorMat(cv::Mat mask,cv::Mat color);

class LaplacianInpainting{
public:
	void findNearestNeighborLap(cv::Mat nnf,cv::Mat nnferr, bool *patch_type, cv::Mat colormat, cv::Mat colorfmat, cv::Mat maskmat, std::pair<int, int> size,int emiter);
	void colorVoteLap(cv::Mat nnf, cv::Mat nnferr, bool *patch_type, cv::Mat colormat, cv::Mat colorfmat, cv::Mat maskmat, std::pair<int, int> size);
	void doEMIterLap(cv::Mat nnf, cv::Mat nnferr, bool *patch_type, cv::Mat colormat, cv::Mat featuremat, cv::Mat maskmat, std::pair<int, int> size,int num_emiter, cv::Size orig_size, char *processfilename);
	void constructLaplacianPyr(std::vector<cv::Mat> &gpyr, std::vector<cv::Mat> &upyr, std::vector<cv::Mat> &fpyr,cv::Mat &img);
	void constructLaplacianPyrMask(std::vector<cv::Mat> &gpyr, std::vector<cv::Mat> &upyr, std::vector<cv::Mat> &fpyr,cv::Mat mask,cv::Mat &img);
	void upscaleImages(cv::Mat nnf, cv::Mat nnferr, bool *patch_type,  cv::Mat colorfmat,  cv::Mat dmaskmat,  cv::Mat umaskmat);
public:
	int psz_, minsize_;
	double gamma_, highconfidence_, lambda_;
	double siminterval_; 
	int patchmatch_iter_;
   int rs_iter_;
	int nnfcount_;
};
