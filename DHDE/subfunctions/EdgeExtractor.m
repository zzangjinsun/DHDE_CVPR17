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
% Name   : EdgeExtractor
% Input  : gryImg     - grayscale image
%          edgeThLow  - low thresholds for edge extraction
%          edgeThHigh - high thresholds for edge extraction
%          fBalance   - balancing flag
%          params     - parameters
% Output : edgMap     - extracted edges with labels
%          nPatches   - balanced number of patches
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [edgMap, nPatches] = EdgeExtractor(gryImg, edgeThLow, edgeThHigh, fBalance, params)

    % Parsing Parameters
    nScale = params.nScale;

    [R, C] = size(gryImg);

    edgIdx = zeros(R, C);

    for k=1:nScale
        edgTmp = edge(gryImg, 'canny', [edgeThLow(k), edgeThHigh(k)]);

        edgIdx = max(edgIdx, k*edgTmp);
    end
    
    if(fBalance == 0)
        edgMap = edgIdx;
        
        nPatches = sum(edgMap(:) ~= 0);
    else
        % Balance number of edges
        nEdges = zeros(nScale, 1);

        for k=1:nScale
            nEdges(k) = sum(edgIdx(:) == k);
        end

        nPatches = min(nEdges);

        edgMap = zeros(R, C);

        for k=1:nScale
            [rows, cols] = find(edgIdx == k);

            if(nEdges(k) > nPatches)
                iPerm = randperm(nEdges(k), nPatches);

                rows = rows(iPerm(:));
                cols = cols(iPerm(:));
            end

            idx = sub2ind([R, C], rows, cols);

            edgMap(idx(:)) = k;
        end
    end
    
end