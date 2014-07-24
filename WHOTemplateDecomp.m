function [ WHOTemplate, HOGTemplate ] = WHOTemplateDecomp( im, Mu, Gamma, nCellLimit, lambda, padding, hog_cell_threshold)
%WHOTEMPLATEDECOMP Summary of this function goes here
%   Detailed explanation goes here
% Nrow = N1

if nargin < 7
  hog_cell_threshold = 10^-4;
end

if nargin < 6
  padding = 50;
end

%%%%%%%% Get HOG template

% create white background padding
paddedIm = padarray(im2double(im), [padding, padding, 0]);
paddedIm(:,1:padding,:) = 1;
paddedIm(:,end-padding+1 : end, :) = 1;
paddedIm(1:padding,:,:) = 1;
paddedIm(end-padding+1 : end, :, :) = 1;

% bounding box coordinate x1, y1, x2, y2
bbox = [1 1 size(im,2) size(im,1)] + padding;

% TODO replace it
model = esvm_initialize_goalsize_exemplar_ncell(paddedIm, bbox, nCellLimit);
HOGTemplate = model.x;

%%%%%%%% WHO conversion using matrix decomposition

sz = size(HOGTemplate);
wHeight = sz(1);
wWidth = sz(2);
HOGDim = sz(3);

Sigma = zeros(prod(sz));
for i = 1:wHeight
    for j = 1:wWidth
        rowIdx = i + wHeight * (j - 1); % sub2ind([wHeight, wWidth],i,j);
        for k = 1:wHeight
            for l = 1:wWidth
                colIdx = k + wHeight * (l - 1); % sub2ind([wHeight, wWidth],k,l);
                gammaRowIdx = abs(i - k) + 1;
                gammaColIdx = abs(j - l) + 1;
                Sigma((rowIdx-1)*HOGDim + 1:rowIdx * HOGDim, (colIdx-1)*HOGDim + 1:colIdx*HOGDim) = ...
                    Gamma((gammaRowIdx-1)*HOGDim + 1 : gammaRowIdx*HOGDim , (gammaColIdx - 1)*HOGDim + 1 : gammaColIdx*HOGDim);
            end
        end
    end
end

success = false;
firstTry = true;
while ~success
  if firstTry
      Sigma = Sigma + lambda * eye(size(Sigma));
  else
      Sigma = Sigma + 0.01 * eye(size(Sigma));
  end
  firstTry = false;
  [R, p] = chol(Sigma);
  if p == 0
  	success = true;
  else
  	display('trying decomposition again');
  end
end

muSwapDim = zeros(1,1,HOGDim);
muSwapDim(1,1,:) = Mu;

%%%%%%%%%%%%%%%%%%%%%%%%%%
nonEmptyCells = (sum(HOGTemplate,3) > hog_cell_threshold);
%%%%%%%%%%%%%%%%%%%%%%%%%%

centeredWs = bsxfun(@times,bsxfun(@minus,HOGTemplate,muSwapDim),nonEmptyCells);
centeredWs = permute(centeredWs,[3 1 2]); % [HOGDim, Nrow, Ncol] = HOGDim, N1, N2

sigInvCenteredWs = R\(R'\centeredWs(:));
sigInvCenteredWs = reshape(sigInvCenteredWs,[HOGDim, wHeight, wWidth]);
WHOTemplate = permute(sigInvCenteredWs,[2,3,1]);

end