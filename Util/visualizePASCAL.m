% Addpath doesn't work pwd in the code uses current directory so move to
% the directory.
clear
DATA_SET = 'PASCAL12';
dwot_set_datapath;

LOWER_CASE_CLASS = 'car';
TEST_TYPE = 'train';

[gtids,t] = textread(sprintf(VOCopts.imgsetpath,[LOWER_CASE_CLASS '_' TEST_TYPE]),'%s %d');

N_IMAGE = length(gtids);
for imgIdx=1:N_IMAGE
    % read annotation
    recs(imgIdx)=PASreadrecord(sprintf(VOCopts.annopath,gtids{imgIdx}));

    
    clsinds = strmatch('car',{recs(imgIdx).objects(:).class},'exact');
    if isempty(clsinds)
        continue;
    end
    % if dwot_skip_criteria(recs(imgIdx).objects(clsinds), skip_criteria); continue; end
    fprintf('%d/%d  %s\n', imgIdx, N_IMAGE, gtids{imgIdx});

    im = imread([VOCopts.datadir, recs(imgIdx).imgname]);
    imshow(im)
    waitforbuttonpress
end
