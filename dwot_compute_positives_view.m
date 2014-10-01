function [bbsNMS_clip, tp, fp, detScore, gt] = dwot_compute_positives_view(bbsNMS_clip, gt, detectors, param)
% gt has BB field for bounding box location, x1 y1 x2 y2 format
% gt also has azimuth, elevation, yaw fields.

nDet = size(bbsNMS_clip,1);
tp = zeros(1,nDet);
fp = zeros(1,nDet);
detScore = zeros(1,nDet);

for bbsIdx = 1:nDet
  ovmax_view = -inf;
  template_idx = bbsNMS_clip(bbsIdx,11);
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

          % Compute viewpoint angle difference
          view_difference = min([abs(gt.azimuth - detectors{template_idx}.az),...
               abs(gt.azimuth + 360 - detectors{template_idx}.az),...
               abs(gt.azimuth - 360 - detectors{template_idx}.az)]);
             
          % Find the ground truth bounding box that has the highest overlap
          % with the current detection bounding box
          if ov > ovmax_view && view_difference < param.max_view_difference
            ovmax_view = ov;
            jmax_view = j;
          end
          
          if ov > param.min_overlap
            % For debugging purpose
            bbsNMS_clip(bbsIdx, 10) =  j; 
          end
      end
  end

  % assign detection as true positive/don't care/false positive
  if ovmax_view >= param.min_overlap
      if ~gt.diff(jmax_view)
          if ~gt.det(jmax_view)
              tp(bbsIdx)=1;            % true positive
              gt.det(jmax_view)=true;
          else
              fp(bbsIdx)=1;            % false positive (multiple detection)
          end
      end
   else
      fp(bbsIdx)=1;                    % false positive
   end
   detScore(bbsIdx) = bbsNMS_clip(bbsIdx,end);
   bbsNMS_clip(bbsIdx, 9) = ovmax_view;
end
