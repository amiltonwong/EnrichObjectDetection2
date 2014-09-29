function [bbsNMS_clip, tp, fp, detScore, gt] = dwot_compute_positives(bbsNMS_clip, gt, param)

nDet = size(bbsNMS_clip,1);
tp = zeros(1,nDet);
fp = zeros(1,nDet);
detScore = zeros(1,nDet);
for bbsIdx = 1:nDet
  ovmax=-inf;
  % search over all objects in the image
  for j=1:size(gt.BB,2)
      bbgt=gt.BB(:,j);
      bi=[max(bbsNMS_clip(bbsIdx,1),bbgt(1)) ; max(bbsNMS_clip(bbsIdx,2),bbgt(2)) ; min(bbsNMS_clip(bbsIdx,3),bbgt(3)) ; min(bbsNMS_clip(bbsIdx,4),bbgt(4))];
      iw=bi(3)-bi(1)+1;
      ih=bi(4)-bi(2)+1;
      if iw>0 && ih>0                
          % compute overlap as area of intersection / area of union
          ua=(bbsNMS_clip(bbsIdx,3)-bbsNMS_clip(bbsIdx,1)+1)*(bbsNMS_clip(bbsIdx,4)-bbsNMS_clip(bbsIdx,2)+1)+...
             (bbgt(3)-bbgt(1)+1)*(bbgt(4)-bbgt(2)+1)-...
             iw*ih;
          ov=iw*ih/ua;

          if ov > ovmax
              ovmax = ov;
              jmax = j;
          end
      end
  end

  % assign detection as true positive/don't care/false positive
  if ovmax >= param.min_overlap
      if ~gt.diff(jmax)
          if ~gt.det(jmax)
              tp(bbsIdx)=1;            % true positive
              gt.det(jmax)=true;
          else
              fp(bbsIdx)=1;            % false positive (multiple detection)
          end
      end
  else
      fp(bbsIdx)=1;                    % false positive
  end
  
  detScore(bbsIdx) = bbsNMS_clip(bbsIdx,end);
  bbsNMS_clip(bbsIdx, 9) = ovmax;
end

