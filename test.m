% run detector on test images
function out = test(VOCopts,cls,detector)

% load test set ('val' for development kit)
[ids,gt]=textread(sprintf(VOCopts.imgsetpath,VOCopts.testset),'%s %d');

% create results file
fid=fopen(sprintf(VOCopts.detrespath,'comp3',cls),'w');

% apply detector to each image
tic;
for i=1:length(ids)
    % display progress
    if toc>1
        fprintf('%s: test: %d/%d\n',cls,i,length(ids));
        drawnow;
        tic;
    end
    
    try
        % try to load features
        load(sprintf(VOCopts.exfdpath,ids{i}),'fd');
    catch
        % compute and save features
        I=imread(sprintf(VOCopts.imgpath,ids{i}));
        fd=extractfd(VOCopts,I);
        save(sprintf(VOCopts.exfdpath,ids{i}),'fd');
    end

    % compute confidence of positive classification and bounding boxes
    [c,BB]=detect(VOCopts,detector,fd);

    % write to results file
    for j=1:length(c)
        fprintf(fid,'%s %f %d %d %d %d\n',ids{i},c(j),BB(:,j));
    end
end

% close results file
fclose(fid);

% trivial feature extractor: compute mean RGB
function fd = extractfd(VOCopts,I)

fd=squeeze(sum(sum(double(I)))/(size(I,1)*size(I,2)));

% trivial detector: confidence is computed as in example_classifier, and
% bounding boxes of nearest positive training image are output
function [c,BB] = detect(VOCopts,detector,fd)

% compute confidence
d=sum(fd.*fd)+sum(detector.FD.*detector.FD)-2*fd'*detector.FD;
dp=min(d(detector.gt>0));
dn=min(d(detector.gt<0));
c=dn/(dp+eps);

% copy bounding boxes from nearest positive image
pinds=find(detector.gt>0);
[dp,di]=min(d(pinds));
pind=pinds(di);
BB=detector.bbox{pind};

% replicate confidence for each detection
c=ones(size(BB,2),1)*c;