######################################################################
# A Unified Approach of Multi-scale Deep and Hand-crafted Features
# for Defocus Estimation
#
# Jinsun Park, Yu-Wing Tai, Donghyeon Cho and In So Kweon
#
# CVPR 2017
#
# Please feel free to contact if you have any problems.
# 
# E-mail : Jinsun Park (zzangjinsun@gmail.com)
# Project Page : https://github.com/zzangjinsun/DHDE_CVPR17/
######################################################################

# Change TOOLS to your caffe path
TOOLS=/home/jinsun/caffe-master/build/tools

$TOOLS/caffe train --solver=DHDE_solver.prototxt --log_dir=logs
