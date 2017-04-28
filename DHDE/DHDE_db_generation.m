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

addpath(genpath('subfunctions'));



% Global Parameter Setting
params = ParameterSetting();

rList = params.rList;
wList = params.wList;
CH = params.CH;

rRef = params.rRef;
wRef = params.wRef;

nScale = params.nScale;

nLabel = params.nLabel;

nBinsDCT = params.nBinsDCT;
nBinsGRD = params.nBinsGRD;
nBinsSVD = params.nBinsSVD;

nMerge = params.nMerge;



% Parameters for dataset
dirRoot = 'data/training';

% Training set
dirDst = 'data/db/train';

iStart = 1;
iEnd = 2;

nDB = 7680;

% Validation set
% dirDst = 'data/db/validation';
% 
% iStart = 3;
% iEnd = 3;
% 
% nDB = 128;



% Check existance
if(isdir(dirDst))
    flag = input('DB already exists. [0 : Cancel, otherwise : Delete]\n');
    
    if(flag == 0)
        return;
    end
    
    rmdir(dirDst,'s');
    mkdir(dirDst);
end



% Check features
nTotal = 0;
for i=iStart:iEnd
    dirSrc = sprintf('%s/%04d',dirRoot,i);
    
    dirLog = sprintf('%s/log.txt',dirSrc);

    fLog = fopen(dirLog, 'r');
    
    iLog = fscanf(fLog, '%d\n%f');
    
    nLog = iLog(1);
    
    nTotal = nTotal + nLog;
    
    fclose(fLog);
end

fprintf(1,'nDB : %d, nTotal : %d\n', nDB, nTotal);

if(nTotal < nDB)
    fprintf(1,'Insufficient features.\n');
    return;
end



% DB Generation
tt0 = clock;

iShuffle = randperm(nTotal);
iValid = iShuffle <= nDB;

dstDCT = sprintf('%s/dbDCT',dirDst);
dstGRD = sprintf('%s/dbGRD',dirDst);
dstSVD = sprintf('%s/dbSVD',dirDst);
dstIMG = sprintf('%s/dbIMG',dirDst);

logDCT = sprintf('%s/logDCT.txt',dstDCT);
logGRD = sprintf('%s/logGRD.txt',dstGRD);
logSVD = sprintf('%s/logSVD.txt',dstSVD);
logIMG = sprintf('%s/logIMG.txt',dstIMG);

mkdir(dstDCT);
mkdir(dstGRD);
mkdir(dstSVD);
mkdir(dstIMG);

fLogDCT = fopen(logDCT, 'w');
fLogGRD = fopen(logGRD, 'w');
fLogSVD = fopen(logSVD, 'w');
fLogIMG = fopen(logIMG, 'w');

cntSrc = 0;
cntDst = 0;
cntMerge = 0;
cntDB = 0;

fDCT = cell(nMerge,1);
fGRD = cell(nMerge,1);
fSVD = cell(nMerge,1);
fIMG = cell(nMerge,1);
labels = cell(nMerge,1);

for i=iStart:iEnd
    dirSrc = sprintf('%s/%04d',dirRoot,i);
    
    fprintf(1,'%s\n',dirSrc);
    
    
    
    srcDCT = sprintf('%s/dbDCT',dirSrc);
    srcGRD = sprintf('%s/dbGRD',dirSrc);
    srcSVD = sprintf('%s/dbSVD',dirSrc);
    srcIMG = sprintf('%s/dbIMG',dirSrc);
    
    [fD, l, ~] = ReadFromDB(srcDCT, 'dataDCT', params);
    [fG, ~, ~] = ReadFromDB(srcGRD, 'dataGRD', params);
    [fS, ~, ~] = ReadFromDB(srcSVD, 'dataSVD', params);
    [fI, ~, ~] = ReadFromDB(srcIMG, 'dataIMG', params);
    
    nFeatures = numel(l);
    
    idxPerm = randperm(nFeatures);
    
    idxStart = cntSrc+1;
    idxEnd = cntSrc+nFeatures;
    
    idx = iValid(idxStart:idxEnd);
    
    nValid = sum(idx(:));
    
    fD = fD(:,:,:,idxPerm(idx(:)));
    fG = fG(:,:,:,idxPerm(idx(:)));
    fS = fS(:,:,:,idxPerm(idx(:)));
    fI = fI(:,:,:,idxPerm(idx(:)));
    l = l(:,idxPerm(idx(:)));
    
    cntMerge = cntMerge + 1;
    
    fDCT{cntMerge} = fD;
    fGRD{cntMerge} = fG;
    fSVD{cntMerge} = fS;
    fIMG{cntMerge} = fI;
    labels{cntMerge} = l;
    
    if(cntMerge == nMerge)
         location = 1;
        
        for k=1:cntMerge
            nCurrent = numel(labels{k});
            
            WriteToDB(sprintf('%s/%04d',dstDCT,cntDB), 'dataDCT', fDCT{k}, labels{k}, [1, sum(nBinsDCT(:))+nScale, 1], location, params);
            WriteToDB(sprintf('%s/%04d',dstGRD,cntDB), 'dataGRD', fGRD{k}, labels{k}, [1, sum(nBinsGRD(:))+nScale, 1], location, params);
            WriteToDB(sprintf('%s/%04d',dstSVD,cntDB), 'dataSVD', fSVD{k}, labels{k}, [1, sum(nBinsSVD(:))+nScale, 1], location, params);
            WriteToDB(sprintf('%s/%04d',dstIMG,cntDB), 'dataIMG', fIMG{k}, labels{k}, [wRef, wRef, CH], location, params);
            
            location = location + nCurrent;
        end
        
        fprintf(fLogDCT,'%s/%04d\n',dstDCT,cntDB);
        fprintf(fLogGRD,'%s/%04d\n',dstGRD,cntDB);
        fprintf(fLogSVD,'%s/%04d\n',dstSVD,cntDB);
        fprintf(fLogIMG,'%s/%04d\n',dstIMG,cntDB);
        
        cntMerge = 0;
        
        cntDB = cntDB + 1;
        
        fDCT = cell(nMerge,1);
        fGRD = cell(nMerge,1);
        fSVD = cell(nMerge,1);
        fIMG = cell(nMerge,1);
        labels = cell(nMerge,1);
    end
    
    cntSrc = cntSrc + nFeatures;
    cntDst = cntDst + nValid;
    
end



if(cntMerge ~= 0)
    location = 1;

    for k=1:cntMerge
        nCurrent = numel(labels{k});

        WriteToDB(sprintf('%s/%04d',dstDCT,cntDB), 'dataDCT', fDCT{k}, labels{k}, [1, sum(nBinsDCT(:))+nScale, 1], location, params);
        WriteToDB(sprintf('%s/%04d',dstGRD,cntDB), 'dataGRD', fGRD{k}, labels{k}, [1, sum(nBinsGRD(:))+nScale, 1], location, params);
        WriteToDB(sprintf('%s/%04d',dstSVD,cntDB), 'dataSVD', fSVD{k}, labels{k}, [1, sum(nBinsSVD(:))+nScale, 1], location, params);
        WriteToDB(sprintf('%s/%04d',dstIMG,cntDB), 'dataIMG', fIMG{k}, labels{k}, [wRef, wRef, CH], location, params);

        location = location + nCurrent;
    end
    
    cntDB = cntDB + 1;
    
    fprintf(fLogDCT,'%s/%04d\n',dstDCT,cntDB);
    fprintf(fLogGRD,'%s/%04d\n',dstGRD,cntDB);
    fprintf(fLogSVD,'%s/%04d\n',dstSVD,cntDB);
    fprintf(fLogIMG,'%s/%04d\n',dstIMG,cntDB);
    
end



tt1 = clock;

fclose(fLogDCT);
fclose(fLogGRD);
fclose(fLogSVD);
fclose(fLogIMG);

fprintf(1,'nDB : %d, count : %d\n',nDB,cntDst);
fprintf(1,'Total elapsed time : %7.2f sec.\n',etime(tt1,tt0));


fLog = fopen(sprintf('%s/log.txt',dirDst),'w');
fprintf(fLog,'%d\n',cntDst);
fprintf(fLog,'%d\n',cntDB);
fprintf(fLog,'%f',etime(tt1, tt0));
fclose(fLog);
