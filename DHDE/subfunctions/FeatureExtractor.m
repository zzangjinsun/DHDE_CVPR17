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
%          Robotics and Computer Vision Lab., EE,
%          KAIST, Republic of Korea
% Project Page : https://github.com/zzangjinsun/DHDE_CVPR17/
%
%
%
% Name   : FeatureExtractor
% Input  : pGry   - gray image patches
%          pGrd   - gradient image patches
%          s      - index of current scale
%          params - parameters
% Output : fDCT   - DCT features
%          fGRD   - Gradient features
%          fSVD   - SVD features
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [fDCT, fGRD, fSVD] = FeatureExtractor(pGry, pGrd, s, params)
    % Parsing Parameters
    nBinsDCT = params.nBinsDCT(s);
    nSizeDCT = params.nSizeDCT(s);
    matDCT = params.matDCT{s};
    maskDCT = params.maskDCT(s,:);
    areaDCT = params.areaDCT(s,:);
    
    nBinsGRD = params.nBinsGRD(s);
    bndGrd = linspace(0, 1, nBinsGRD+1);
    bndGrd = (bndGrd(1:nBinsGRD) + bndGrd(2:nBinsGRD+1))/2;
    
    nBinsSVD = params.nBinsSVD(s);
    
    wPatch = params.wList(s);
    
    nPadDCT = nSizeDCT - wPatch;
    
    nFeatures = size(pGry, 3);
    
    fDCT = zeros(1, nBinsDCT, 1, nFeatures);
    fGRD = zeros(1, nBinsGRD, 1, nFeatures);
    fSVD = zeros(1, nBinsSVD, 1, nFeatures);
    
    if(nPadDCT > 0)
        pGryPad = padarray(pGry, [nPadDCT, nPadDCT, 0], 0, 'post');
    else
        pGryPad = pGry;
    end
    
    for k=1:nFeatures
        
        % DCT Feature Extraction
        pDCT = abs(matDCT*pGryPad(:,:,k)*matDCT');
    
        for p=1:nBinsDCT
            fDCT(1,p,1,k) = sum(pDCT(maskDCT{p}(:)))/areaDCT(p);
        end
        
        % SVD Feature Extraction
        S = svd(pGry(:,:,k));
        fSVD(1,1:nBinsSVD,1,k) = S(1:nBinsSVD);
        
    end
    
    fDCT = log(1+fDCT);
    fDCT = fDCT./repmat(sum(fDCT,2), [1, nBinsDCT, 1, 1]);
        
    fSVD = log(1+fSVD);
    fSVD = fSVD./repmat(sum(fSVD,2), [1, nBinsSVD, 1, 1]);
    
    
    
    % GRD Feature Extraction
    fGRD(1,:,1,:) = hist(pGrd, bndGrd);
    fGRD = log(1+fGRD);
    fGRD = fGRD./repmat(sum(fGRD,2), [1, nBinsGRD, 1, 1]);
    
    
    
    % Add biases
    fDCT = cat(2, fDCT, ones(1, 1, 1, nFeatures));
    fGRD = cat(2, fGRD, ones(1, 1, 1, nFeatures));
    fSVD = cat(2, fSVD, ones(1, 1, 1, nFeatures));
    
end