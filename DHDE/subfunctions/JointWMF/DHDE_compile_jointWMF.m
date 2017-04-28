%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A Unified Approach of Multi-scale Deep and Hand-crafted Features
% for Defocus Estimation
%
% Jinsun Park, Yu-Wing Tai, Donghyeon Cho and In So Kweon
%
% CVPR 2017
%
% Please feel free to contact if you have any problems.
% 
% E-mail : Jinsun Park (zzangjinsun@gmail.com)
% Project Page : https://github.com/zzangjinsun/DHDE_CVPR17/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;

% Include and Library path
optIncludePath = '-I/usr/local/include';
optLibraryPath = '-L/usr/local/lib';
optLibrary = '-lopencv_core';

% For 32-bit compatibility
optEtc = '-DMX_COMPAT_32';

% Compile
mex -setup C++

mex('mexJointWMF.cpp', optIncludePath, optLibraryPath, optLibrary, optEtc);