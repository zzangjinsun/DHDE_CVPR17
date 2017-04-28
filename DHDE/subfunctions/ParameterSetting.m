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
% Name   : ParameterSetting
% Input  : None
% Output : params - a structure containing all of the parameters
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function params = ParameterSetting()

params = struct();

% Parameters for Patch Extraction
params.rList = [13, 7];
params.wList = 2*params.rList+1;
params.CH = 3;

params.rRef = 13;
params.wRef = 2*params.rRef+1;

params.nScale = numel(params.rList);

% Parameters for Rolling Guidance Filter
params.sRGFSpt = 1.5;
params.sRGFRng = 1.5;
params.iterRGF = 3;

% Parameters for Testing Edge Extraction
params.edgeThLowCL = [0.117, 0.234];
params.edgeThHighCL = [0.176, 0.351];

% Parameters for Training Edge Extraction
params.edgeThLowFE = [0.234, 0.468];
params.edgeThHighFE = [0.351, 0.702];

% Parameters for Synthetic Blur Kernels
params.sMin = 0.5;
params.sMax = 2.0;
params.sInter = 0.15;
params.sList = params.sMin:params.sInter:params.sMax;

% Number of Labels
params.nLabel = numel(params.sList);

params.rKernel = params.rRef;
params.wKernel = 2*params.rKernel+1;

% Parameters for Feature Extraction
params.nBinsDCT = [25, 13];
params.nSizeDCT = [32, 16];
params.rOverlapDCT = 0.1;

params.matDCT = cell(params.nScale, 1);
params.maskDCT = cell(params.nScale, params.nLabel);
params.areaDCT = zeros(params.nScale, params.nLabel);

for k=1:params.nScale
    params.matDCT{k} = dctmtx(params.nSizeDCT(k));
    
    % Masks for fDCT
    [XX, YY] = meshgrid(0:params.nSizeDCT(k)-1, 0:params.nSizeDCT(k)-1);  
    rho = sqrt(XX.^2 + YY.^2);
    bnds = linspace(0, params.nSizeDCT(k)-1, params.nBinsDCT(k)+1);
    bndInter = (params.nSizeDCT(k)-1)/params.nBinsDCT(k);

    for b=1:params.nBinsDCT(k)
        bndLower = bnds(b)-params.rOverlapDCT*bndInter;
        bndUpper = bnds(b+1)+params.rOverlapDCT*bndInter;

        params.maskDCT{k, b} = (rho >= bndLower) & (rho <= bndUpper);
        params.areaDCT(k, b) = sum(params.maskDCT{k, b}(:));
    end
end

params.nBinsGRD = [25, 13];

params.nBinsSVD = [25, 13];

% Parameters for Joint Bilateral Filter
params.rJBF = params.rRef;
params.wJBF = 2*params.rJBF+1;

params.sJBFSpt = 100.0;
params.sJBFRng = 100.0;
params.sJBFPrb = 1.0;

% Parameters for DB
params.dbScale = 255.0;
params.chunkSize = 512;
params.nMerge = 2;

% Parameters for Matting Laplacian
params.propLambda = 0.005;
params.propEps = 0.00001;
params.propRadius = 1;

% Parameters for Weighted Median Filter
params.rWMF = 5;
params.sWMFRng = 0.25;
params.nIWMF = 256;
params.nFWMF = 256;
params.iterWMF = 3;
params.wtypeWMF = 'exp';

% Parameters for classification
params.nBatch = 128;

% Miscellaneous
params.pMin = 5;
params.pMax = 255;

params.thRndSeed = 1;

end