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
%
%
%
% Name   : ReadFromDB
% Input  : dirDB   - path to db
%          type    - type of data. one of [dataDCT, dataGRD, dataSVD, dataIMG]
%          params  - global parameters
% Output : feature - feature data
%          labels  - label data
%          info    - db info
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [features, labels, info] = ReadFromDB(dirDB, type, params)
    % Parsing Parameters
    dbScale = params.dbScale;
    
    % Read from DB
    info = h5info(dirDB);
    
    features = h5read(dirDB, sprintf('/%s',type));
    labels = h5read(dirDB, '/label');
    
    features = double(features)/dbScale;
    labels = double(labels);
    
end