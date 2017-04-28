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

edgeThLowFE = params.edgeThLowFE;
edgeThHighFE = params.edgeThHighFE;

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

sMin = params.sMin;
sMax = params.sMax;
sList = params.sList;

rKernel = params.rKernel;
wKernel = params.wKernel;



% Parameters for dataset
dirRoot = 'data/training';

iStart = 1;
iEnd = 3;



for i=iStart:iEnd
    t0 = clock;

    % Source directory
    dirSrc = sprintf('%s/%04d',dirRoot,i);
    
    fprintf(1,'%s\n',dirSrc);
    
    % Image loading
    dirImg = sprintf('%s/image.jpg',dirSrc);
    
    rgbImg = imread(dirImg);
    
    rgbImg = im2double(rgbImg);
    
    hsvImg = rgb2hsv(rgbImg);
    
    % Use V channel as a gray image
    gryImg = hsvImg(:,:,3);
    
    % Extract edge map
    [edgMap, nPatches] = EdgeExtractor(gryImg, edgeThLowFE, edgeThHighFE, 1, params);
    
    
    
    fDCT = zeros(1, sum(nBinsDCT(:)) + nScale, 1, nScale*nLabel*nPatches);
    fGRD = zeros(1, sum(nBinsGRD(:)) + nScale, 1, nScale*nLabel*nPatches);
    fSVD = zeros(1, sum(nBinsSVD(:)) + nScale, 1, nScale*nLabel*nPatches);
    fIMG = zeros(wRef, wRef, CH, nScale*nLabel*nPatches);
    labels = zeros(1,nScale*nLabel*nPatches);
    
    
    % Feature Extraction
    fprintf(1,'Feature Extraction...');
    tFE0 = clock;
    
    for s=1:nScale
        rPatch = rList(s);
        wPatch = wList(s);
        
        [rows, cols] = find(edgMap == s);
        
        rgbImgPad = padarray(rgbImg, [rPatch, rPatch], 'replicate', 'both');

        [R, C, ~] = size(rgbImgPad);

        rgbImgList = cell(nLabel, 1);
        gryImgList = cell(nLabel, 1);
        grdImgList = cell(nLabel, 1);

        for k=1:nLabel
            sigma = sList(k);

            kernel = fspecial('gaussian', [wKernel, wKernel], sigma);

            rgbImgList{k} = imfilter(rgbImgPad, kernel, 'replicate', 'same');

            hsvTmp = rgb2hsv(rgbImgList{k});

            gryImgList{k} = hsvTmp(:,:,3);

            gX = [gryImgList{k}(:,2:C) - gryImgList{k}(:,1:C-1), gryImgList{k}(:,C) - gryImgList{k}(:,C-1)];
            gY = [gryImgList{k}(2:R,:) - gryImgList{k}(1:R-1,:); gryImgList{k}(R,:) - gryImgList{k}(R-1,:)];

            grdImgList{k} = sqrt(gX.^2 + gY.^2)/sqrt(2);
        end

        % Pre-calculate offset indices
        [offX, offY] = meshgrid(0:wPatch-1, 0:wPatch-1);
        offIdx = R*offX(:) + offY(:);

        for k=1:nLabel
            pRgb = zeros(wPatch, wPatch, CH, nPatches);
            pGry = zeros(wPatch, wPatch, nPatches);
            pGrd = zeros(wPatch*wPatch, nPatches);

            for n=1:nPatches
                pRgb(:,:,:,n) = rgbImgList{k}(rows(n):rows(n)+wPatch-1, cols(n):cols(n)+wPatch-1,:);

                pGry(:,:,n) = gryImgList{k}(rows(n):rows(n)+wPatch-1, cols(n):cols(n)+wPatch-1);

                pGrd(:,n) = grdImgList{k}((rows(n)+(cols(n)-1)*R) + offIdx);                
            end

            [fD, fG, fS] = FeatureExtractor(pGry, pGrd, s, params);

            idxStart = (s-1)*nPatches*nLabel + (k-1)*nPatches+1;
            idxEnd = (s-1)*nPatches*nLabel + k*nPatches;

            fDCT(1,sum(nBinsDCT(1:s-1))+s:sum(nBinsDCT(1:s))+s,1,idxStart:idxEnd) = fD;
            fGRD(1,sum(nBinsDCT(1:s-1))+s:sum(nBinsDCT(1:s))+s,1,idxStart:idxEnd) = fG;
            fSVD(1,sum(nBinsDCT(1:s-1))+s:sum(nBinsDCT(1:s))+s,1,idxStart:idxEnd) = fS;
            
            if(rPatch < rRef)
                rPad = rRef - rPatch;
                
                pRgb = padarray(pRgb, [rPad, rPad], 'replicate', 'both');
            end
            
            fIMG(:,:,:,idxStart:idxEnd) = pRgb;

            labels(1,idxStart:idxEnd) = (k-1);
        end
    end
    
    tFE1 = clock;
    fprintf(1,'  (%5.2f sec.)\n', etime(tFE1, tFE0));
    
    
    
    % Write to DB
    fprintf(1,'Write to DB...');
    tDB0 = clock;
    
    dirDCT = sprintf('%s/dbDCT',dirSrc);
    dirGRD = sprintf('%s/dbGRD',dirSrc);
    dirSVD = sprintf('%s/dbSVD',dirSrc);
    dirIMG = sprintf('%s/dbIMG',dirSrc);
    
    WriteToDB(dirDCT, 'dataDCT', fDCT, labels, [1, sum(nBinsDCT(:)) + nScale, 1], 1, params);
    WriteToDB(dirGRD, 'dataGRD', fGRD, labels, [1, sum(nBinsGRD(:)) + nScale, 1], 1, params);
    WriteToDB(dirSVD, 'dataSVD', fSVD, labels, [1, sum(nBinsSVD(:)) + nScale, 1], 1, params);
    WriteToDB(dirIMG, 'dataIMG', fIMG, labels, [wRef, wRef, CH], 1, params);
    
    tDB1 = clock;
    fprintf(1,'  (%5.2f sec.)\n', etime(tDB1, tDB0));
    
    
    
    nTotal = nScale*nLabel*nPatches;
    
    
    
    t1 = clock;
    fprintf(1,'Extracted features : %d  (%5.2f sec.)\n',nTotal,etime(t1,t0));
    
    
    
    % Write to Log
    dirLog = sprintf('%s/log.txt',dirSrc);
    fLog = fopen(dirLog,'w');
    fprintf(fLog,'%d\n',nTotal);
    fprintf(fLog,'%f',etime(t1,t0));
    fclose(fLog);
    
    
    
end
