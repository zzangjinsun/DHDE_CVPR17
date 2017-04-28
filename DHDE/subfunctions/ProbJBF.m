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
% Name   : ProbJBF
% Input  : spsImg - sparse defocus map
%          prbImg - confidence map
%          rgfImg - rolling guidance filtered image
%          params - parameters
% Output : jbfImg - filtered sparse defocus map
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function jbfImg = ProbJBF(spsImg, prbImg, rgfImg, params)
    % Parsing Parameters
    rJBF = params.rJBF;
    wJBF = params.wJBF;
    
    sJBFSpt = params.sJBFSpt;
    sJBFRng = params.sJBFRng;
    sJBFPrb = params.sJBFPrb;
    
    denomSpt = 2*(sJBFSpt^2);
    denomRng = 2*(sJBFRng^2);
    denomPrb = 2*(sJBFPrb^2);
    
    % Edge Extraction
    [R, C] = size(spsImg);
    jbfImg = zeros(R, C);
    
    [rows, cols] = find(spsImg);
    
    spsImgPad = padarray(spsImg, [rJBF, rJBF], 0, 'both');
    rgfImgPad = padarray(rgfImg, [rJBF, rJBF], 0, 'both');
    prbImgPad = padarray(prbImg, [rJBF, rJBF], 0, 'both');
    
    nEdges = numel(rows);
    
    % Spatial Weight
    [XX, YY] = meshgrid(-rJBF:rJBF, -rJBF:rJBF);
    rhoSqr = XX.^2 + YY.^2;
    wSpt = exp(-rhoSqr/denomSpt);
    
    
    
    for k=1:nEdges
        spsRoi = spsImgPad(rows(k):rows(k)+wJBF-1, cols(k):cols(k)+wJBF-1);
        rgfRoi = rgfImgPad(rows(k):rows(k)+wJBF-1, cols(k):cols(k)+wJBF-1);
        prbRoi = prbImgPad(rows(k):rows(k)+wJBF-1, cols(k):cols(k)+wJBF-1);
        
        wRng = exp((-(rgfRoi - rgfRoi(rJBF+1, rJBF+1)).^2)/denomRng);
        wPrb = exp((-(1 - prbRoi).^2)/denomPrb);
        
        weight = wSpt.*wRng.*wPrb.*(spsRoi ~= 0);
        val = weight.*spsRoi;
        
        vSum = sum(val(:));
        wSum = sum(weight(:));
        
        jbfImg(rows(k), cols(k)) = vSum/wSum;
    end
    
end