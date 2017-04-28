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



% Global Parameters Setting
params = ParameterSetting();

edgeThLowCL = params.edgeThLowCL;
edgeThHighCL = params.edgeThHighCL;

rList = params.rList;
wList = params.wList;
CH = params.CH;

rRef = params.rRef;
wRef = 2*rRef+1;

nScale = params.nScale;

nLabel = params.nLabel;

nBinsDCT = params.nBinsDCT;
nBinsGRD = params.nBinsGRD;
nBinsSVD = params.nBinsSVD;

sMin = params.sMin;
sMax = params.sMax;
sList = params.sList;

dbScale = params.dbScale;

propLambda = params.propLambda;
propEps = params.propEps;

sRGFSpt = params.sRGFSpt;
sRGFRng = params.sRGFRng;
iterRGF = params.iterRGF;

rWMF = params.rWMF;
sWMFRng = params.sWMFRng;
nIWMF = params.nIWMF;
nFWMF = params.nFWMF;
iterWMF = params.iterWMF;
wtypeWMF = params.wtypeWMF;

nBatch = params.nBatch;

rKernel = params.rKernel;
wKernel = params.wKernel;

pMin = params.pMin;
pMax = params.pMax;

thRndSeed = params.thRndSeed;



% Parameters
iStart = 1;
iEnd = 2;

dirSrc = 'data/defocus';

dirDeploy = 'DHDE_test.prototxt';
dirModel = 'DHDE_model.caffemodel';
phase = 'test';

gpu_id = 0;


% Network Initialization
caffe.set_mode_gpu();
caffe.set_device(gpu_id);

net = caffe.Net(dirDeploy, dirModel, phase);
net.reshape();

% parpool;



for i=iStart:iEnd
    fprintf(1,'image%04d\n',i);
    tt0 = clock;
    
    if(~isdir(sprintf('%s/%04d/multiscale', dirSrc, i)))
        mkdir(sprintf('%s/%04d/multiscale', dirSrc, i));
    end
    
    fLog = fopen(sprintf('%s/%04d/multiscale/log.txt', dirSrc, i), 'w');
    
    % Image Loading
    rgbImg = imread(sprintf('%s/%04d/image.jpg', dirSrc, i));
    
    rgbImg = im2double(rgbImg);
    
    [R, C, ~] = size(rgbImg);
    
    hsvImg = rgb2hsv(rgbImg);
    
    gryImg = hsvImg(:,:,3);
    
    %% Edge Extraction
    fprintf(1,'Edge Extraction...');
    fprintf(fLog,'Edge Extraction...');
    t0 = clock;
    
    [edgMap, nPatches] = EdgeExtractor(gryImg, edgeThLowCL, edgeThHighCL, 0, params);
    
    % Add Random Seed Points
    rSeed = rand(R, C);
    rSeed = rSeed > thRndSeed;
    rSeed(edgMap(:) ~= 0) = 0;
    
    edgMap = max(edgMap, rSeed);
    
    iRndSeed = find(rSeed);
    
    nPatches = nPatches + numel(iRndSeed);
    
    t1 = clock;
    fprintf(1,' (%5.2f sec.)\n',etime(t1,t0));
    fprintf(fLog,' (%5.2f sec.)\n',etime(t1,t0));
    
    %% Multi-scale Patch Extraction
    fprintf(1,'Multi-scale Patch Extraction... ');
    fprintf(fLog,'Multi-scale Patch Extraction... ');
    t0 = clock;
    
    fDCT = zeros(1, sum(nBinsDCT(:)) + nScale, 1, nPatches);
    fGRD = zeros(1, sum(nBinsGRD(:)) + nScale, 1, nPatches);
    fSVD = zeros(1, sum(nBinsSVD(:)) + nScale, 1, nPatches);
    fIMG = zeros(wRef, wRef, CH, nPatches);
    pos = zeros(nPatches, 2);
    
    nEdges = zeros(nScale, 1);
    
    for s=1:nScale
        
        edgImg = (edgMap == s);
        
        [rows, cols] = find(edgImg);

        nEdges(s) = numel(rows);
        
        rPatch = rList(s);
        wPatch = wList(s);
        
        rgbImgPad = padarray(rgbImg, [rPatch, rPatch], 'replicate', 'both');
        
        gryImgPad = padarray(gryImg, [rPatch, rPatch], 'replicate', 'both');
        
        [Rpad, Cpad, ~] = size(rgbImgPad);
        
        gX = [gryImgPad(:, 2:Cpad) - gryImgPad(:, 1:Cpad-1), gryImgPad(:, Cpad) - gryImgPad(:, Cpad-1)];
        gY = [gryImgPad(2:Rpad, :) - gryImgPad(1:Rpad-1, :); gryImgPad(Rpad, :) - gryImgPad(Rpad-1, :)];
        
        grdImgPad = sqrt(gX.^2 + gY.^2)/sqrt(2);
        
        % Pre-calculate offset indices
        [offX, offY] = meshgrid(0:wPatch-1, 0:wPatch-1);
        offIdx = Rpad*offX(:) + offY(:);
        
        pGry = zeros(wPatch, wPatch, nEdges(s));
        pGrd = zeros(wPatch*wPatch, nEdges(s));
        pRgb = zeros(wPatch, wPatch, CH, nEdges(s));
        
        for k=1:nEdges(s)
            pRgb(:,:,:,k) = rgbImgPad(rows(k):rows(k)+wPatch-1, cols(k):cols(k)+wPatch-1, :);
            pGry(:,:,k) = gryImgPad(rows(k):rows(k)+wPatch-1, cols(k):cols(k)+wPatch-1);
            pGrd(:,k) = grdImgPad(rows(k)+(cols(k)-1)*Rpad + offIdx);
        end
        
        [fD, fG, fS] = FeatureExtractor(pGry, pGrd, s, params);
        
        if(rRef > rPatch)
            rDiff = rRef - rPatch;
            
            pRgb = padarray(pRgb, [rDiff, rDiff], 'replicate', 'both');
        end
        
        idxS = sum(nEdges(1:s-1))+1;
        idxE = sum(nEdges(1:s));
        
        fDCT(1, sum(nBinsDCT(1:s-1))+s:sum(nBinsDCT(1:s))+s, 1, idxS:idxE) = fD;
        fGRD(1, sum(nBinsGRD(1:s-1))+s:sum(nBinsGRD(1:s))+s, 1, idxS:idxE) = fG;
        fSVD(1, sum(nBinsSVD(1:s-1))+s:sum(nBinsSVD(1:s))+s, 1, idxS:idxE) = fS;
        
        fIMG(:, :, :, idxS:idxE) = pRgb;
        
        pos(idxS:idxE, :) = [rows(:), cols(:)];
    end
    
    t1 = clock;
    fprintf(1,' (%5.2f sec.)\n',etime(t1,t0));
    fprintf(fLog,' (%5.2f sec.)\n',etime(t1,t0));
    
    %% Classification
    fprintf(1,'Classification...');
    fprintf(fLog,'Classification...');
    t0 = clock;
    
    label = zeros(1, nPatches);
    confidence = zeros(1, nPatches);

    nDiv = floor(nPatches/nBatch);

    for d=1:nDiv
        iBatchStart = (d-1)*nBatch+1;
        iBatchEnd = d*nBatch;

        feature = {uint8(dbScale*fDCT(:,:,:,iBatchStart:iBatchEnd)), ...
                   uint8(dbScale*fGRD(:,:,:,iBatchStart:iBatchEnd)), ...
                   uint8(dbScale*fSVD(:,:,:,iBatchStart:iBatchEnd)), ...
                   uint8(dbScale*fIMG(:,:,:,iBatchStart:iBatchEnd))};

        output = net.forward(feature);

        prob = output{1};
        [vConf, vLabel] = max(prob,[],1);

        label(iBatchStart:iBatchEnd) = (vLabel-1);
        confidence(iBatchStart:iBatchEnd) = vConf;
    end

    if(iBatchEnd < nPatches)
        iBatchStart = iBatchEnd + 1;
        iBatchEnd = nPatches;

        nBatchPad = nBatch - (iBatchEnd - iBatchStart + 1);
        
        fD = fDCT(:,:,:,iBatchStart:iBatchEnd);
        fG = fGRD(:,:,:,iBatchStart:iBatchEnd);
        fS = fSVD(:,:,:,iBatchStart:iBatchEnd);
        fI = fIMG(:,:,:,iBatchStart:iBatchEnd);
        
        fD = padarray(fD, [0, 0, 0, nBatchPad], 0, 'post');
        fG = padarray(fG, [0, 0, 0, nBatchPad], 0, 'post');
        fS = padarray(fS, [0, 0, 0, nBatchPad], 0, 'post');
        fI = padarray(fI, [0, 0, 0, nBatchPad], 0, 'post');
        
        feature = {dbScale*fD, dbScale*fG, dbScale*fS, dbScale*fI};

        output = net.forward(feature);

        prob = output{1};
        [vConf, vLabel] = max(prob,[],1);

        label(iBatchStart:iBatchEnd) = (vLabel(1:nBatch-nBatchPad)-1);
        confidence(iBatchStart:iBatchEnd) = vConf(1:nBatch-nBatchPad);
    end 

    t1 = clock;
    fprintf(1,' (%5.2f sec.)\n',etime(t1,t0));
    fprintf(fLog,' (%5.2f sec.)\n',etime(t1,t0));

    %% Sparse Defocus Map Generation
    fprintf(1,'Sparse Defocus Map Generation...');
    fprintf(fLog,'Sparse Defocus Map Generation...');
    t0 = clock;
    
    edgIdx = sub2ind([R, C], pos(:,1), pos(:,2));
    
    spsImg = zeros(R, C);
    prbImg = zeros(R, C);
    
    spsImg(edgIdx(:)) = sList(label(:) + 1);
    prbImg(edgIdx(:)) = confidence(:);
    
    % Give lowest confidence to the random seed points
    prbImg(iRndSeed) = 1/nLabel;

    t1 = clock;
    fprintf(1,' (%5.2f sec.)\n',etime(t1,t0));
    fprintf(fLog,' (%5.2f sec.)\n',etime(t1,t0));

    %% Sparse Probability-Joint Bilateral Filtering
    fprintf(1,'Probability-Joint Bilateral Filtering...');
    fprintf(fLog,'Probability-Joint Bilateral Filtering...');
    t0 = clock;
    
    rgfImg = RollingGuidanceFilter(rgbImg, sRGFSpt, sRGFRng, iterRGF);

    jbfImg = ProbJBF(spsImg, prbImg, rgfImg, params);

    t1 = clock;
    fprintf(1,' (%5.2f sec.)\n',etime(t1,t0));
    fprintf(fLog,' (%5.2f sec.)\n',etime(t1,t0));
    
    %% Matting Laplacian Propagation
    fprintf(1,'Matting Laplacian Propagation...');
    fprintf(fLog,'Matting Laplacian Propagation...');
    t0 = clock;
    
    edgImg = edgMap ~= 0;
    
    L = GetLaplacian(rgfImg, params);
    
    D = spdiags(edgImg(:), 0, R*C, R*C);
    
    dnsImg = (L + propLambda*D)\(propLambda*edgImg(:).*jbfImg(:));
    
    dnsImg = reshape(dnsImg, [R, C]);
    
    t1 = clock;
    fprintf(1,' (%5.2f sec.)\n',etime(t1,t0));
    fprintf(fLog,' (%5.2f sec.)\n',etime(t1,t0));
    
    %% Weighted Median Filtering
    fprintf(1,'Weighted Median Filtering...');
    fprintf(fLog,'Weighted Median Filtering...');
    t0 = clock;
    
    rstImg = jointWMF(dnsImg, rgfImg, rWMF, sWMFRng, nIWMF, nFWMF, iterWMF, wtypeWMF);
    
    t1 = clock;
    fprintf(1,' (%5.2f sec.)\n',etime(t1,t0));
    fprintf(fLog,' (%5.2f sec.)\n',etime(t1,t0));
    
    
    
    % Scale Images
    spsImg(edgIdx(:)) = (pMax - pMin)*(spsImg(edgIdx(:)) - sMin)/(sMax - sMin)+pMin;
    jbfImg(edgIdx(:)) = (pMax - pMin)*(jbfImg(edgIdx(:)) - sMin)/(sMax - sMin)+pMin;
    dnsImg = (pMax - pMin)*(dnsImg - sMin)/(sMax - sMin)+pMin;
    rstImg = (pMax - pMin)*(rstImg - sMin)/(sMax - sMin)+pMin;
    
    
    
    imwrite(rgfImg, sprintf('%s/%04d/multiscale/rgf.png', dirSrc, i), 'png');
    imwrite(mat2gray(edgMap, [0, nScale]), sprintf('%s/%04d/multiscale/edg.png', dirSrc, i), 'png');
    imwrite(mat2gray(spsImg, [0, pMax]), sprintf('%s/%04d/multiscale/sps.png', dirSrc, i), 'png');
    imwrite(prbImg, sprintf('%s/%04d/multiscale/prb.png', dirSrc, i), 'png');
    imwrite(mat2gray(jbfImg, [0, pMax]), sprintf('%s/%04d/multiscale/jbf.png', dirSrc, i), 'png');
    imwrite(mat2gray(dnsImg, [0, pMax]), sprintf('%s/%04d/multiscale/dns.png', dirSrc, i), 'png');
    imwrite(mat2gray(rstImg, [0, pMax]), sprintf('%s/%04d/multiscale/rst.png', dirSrc, i), 'png');
    imwrite(mat2gray(rstImg), sprintf('%s/%04d/multiscale/result.png', dirSrc, i), 'png');
    
    tt1 = clock;
    fprintf(1,'Total elapsed time : %5.2f sec.\n\n', etime(tt1, tt0));
    fprintf(fLog,'Total elapsed time : %5.2f sec.\n\n', etime(tt1, tt0));
    
    fclose(fLog);
    
    
    
end
