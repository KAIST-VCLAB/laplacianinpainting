# /****************************************************************************
#
# - Codename: Laplacian Patch-Based Image Synthesis (CVPR 2016)
#
# - Writers:   Joo Ho Lee(jhlee@vclab.kaist.ac.kr), Min H. Kim (minhkim@kaist.ac.kr)
#
# - Institute: KAIST Visual Computing Laboratory
#
# - Bibtex:
#
# @InProceedings{LeeChoiKim:CVPR:2016,
#   author  = {Joo Ho Lee and Inchang Choi and Min H. Kim},
#   title   = {Laplacian Patch-Based Image Synthesis},
#   booktitle = {Proc. IEEE Computer Vision and Pattern Recognition (CVPR 2016)},
#   publisher = {IEEE},
#   address = {Las Vegas, USA},
#   year = {2016},
#   pages = {2727--2735},
# }
#
#
# - License:  GNU General Public License Usage
#   Alternatively, this file may be used under the terms of the GNU General
#   Public License version 3.0 as published by the Free Software Foundation
#   and appearing in the file LICENSE.GPL included in the packaging of this
#   file. Please review the following information to ensure the GNU General
#   Public License version 3.0 requirements will be met:
#   http://www.gnu.org/copyleft/gpl.html.
#
# - Warranty: KAIST-VCLAB MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE
#   SUITABILITY OF THE SOFTWARE, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT
#   LIMITED TO THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
#   PARTICULAR PURPOSE, OR NON-INFRINGEMENT. KAIST-VCLAB SHALL NOT BE LIABLE FOR ANY
#   DAMAGES SUFFERED BY LICENSEE AS A RESULT OF USING, MODIFYING OR DISTRIBUTING
#   THIS SOFTWARE OR ITS DERIVATIVES
#
# *****************************************************************************/
#cmake entry point
cmake_minimum_required(VERSION 2.8)
project(lapinpainting)
#==================================================
# src
add_executable(lapinpainting
	lapinpainting.cpp
	lapinpainting.h
	laplacianrgbinpainting.cpp 
)
#==================================================
#find packages
find_package(OpenCV)
if (OpenCV_FOUND)
  include_directories( ${OpenCV_INCLUDE_DIRS} )
  target_link_libraries (lapinpainting ${OpenCV_LIBS})
endif (OpenCV_FOUND)
#==================================================
add_definitions(
	-DTW_STATIC
	-DTW_NO_LIB_PRAGMA
	-DTW_NO_DIRECT3D
	-DGLEW_STATIC
	-D_CRT_SECURE_NO_WARNING
)
#==================================================

