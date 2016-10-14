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


--------------------------------------------------------------------------------------------------
Background
--------------------------------------------------------------------------------------------------

For information please see the paper:

 - Laplacian Patch-Based Image Synthesis
   CVPR 2016, Joo Ho Lee, Inchang Choi, Min H. Kim
   http://vclab.kaist.ac.kr/cvpr2016p2/index.html


Please cite this paper if you use this code in an academic publication.


--------------------------------------------------------------------------------------------------
Dependency
--------------------------------------------------------------------------------------------------

OpenCV >= 2.4.10 required.

--------------------------------------------------------------------------------------------------
Contents
--------------------------------------------------------------------------------------------------
1. laplacianrgbinpainting.cpp
The main function is in this code. The whole process of image inpainting are written here.

2. lapinpainting.cpp, lapinpainting.h
functions needed for image inpainting are implemented here.

--------------------------------------------------------------------------------------------------
How to compile it
--------------------------------------------------------------------------------------------------s
>> mkdir build
>> cd build
>> ccmake -DCMAKE_BUILD_TYPE=Release ../
>> make

--------------------------------------------------------------------------------------------------
How to use it
--------------------------------------------------------------------------------------------------

First, a user have to create a mask image as a white-black png image.
Then, run a program with arguments. For detail of arguments, please see the code and the paper.

Syntax: Syntax example: lapinpainting bungee bungee.png bungee_mask.png [opt1] [opt2] [opt3] [opt4] [opt5] [opt6] [opt7]
[opt1] patch size: (default) 7
[opt2] gamma: (default) 1.3
[opt3] minimum size in percentages: (default) 20
[opt4] number of EM: (default) 50
[opt5] decrease factor in EM: (default) 10
[opt6] minimum iteration: (default) 10
[opt7] random search iteration: (default) 1