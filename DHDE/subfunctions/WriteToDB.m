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
% Name   : WriteToDB
% Input  : dirDB    - path to db
%          type     - type of data. one of [dataDCT, dataGRD, dataSVD, dataIMG]
%          features - feature data
%          labels   - label data
%          dims     - dimensions of the features
%          location - start location in db (if location = 1, create new db)
%          params   - global parameters
% Output : None
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function WriteToDB(dirDB, type, features, labels, dims, location, params)
    % Parsing Parameters
    chunkSize = params.chunkSize;
    dbScale = params.dbScale;
    
    width = dims(1);
    height = dims(2);
    ch = dims(3);
    
    features = uint8(dbScale*features);
    labels = uint8(labels);
    
    
    
    % Delete existing DB and create new one
    if(location == 1)
        if(exist(dirDB, 'file'))
            delete(dirDB);
        end
        
        h5create(dirDB, sprintf('/%s',type), [width, height, ch, Inf], 'Datatype', 'uint8', 'ChunkSize', [width, height, ch, chunkSize]);
        h5create(dirDB, '/label', [1, Inf], 'Datatype', 'uint8', 'ChunkSize', [1, chunkSize]);
    end
    
    
    
    % Write to DB
    startDat = [1, 1, 1, location];
    startLbl = [1, location];
    
    h5write(dirDB, sprintf('/%s',type), features, startDat, size(features));
    h5write(dirDB, '/label', labels, startLbl, size(labels));
    
end