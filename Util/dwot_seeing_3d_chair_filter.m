% Read url file

CHAIR_PATH = '/home/chrischoy/chair_models_compressed';
CHAIR_DEST = '/home/chrischoy/chair_filtered';

if ~exist(CHAIR_DEST,'dir'); mkdir(CHAIR_DEST); end;

fid = fopen(fullfile(CHAIR_PATH, 'url_filtered.txt'),'r');
urls = textscan(fid, '%s');
fclose(fid);

urls = urls{1};

for url = urls'
    url = url{1};
    
    name = regexp(url,'mid=(\w+)','tokens');
    name = name{1}{1};
    
    system(['cp -r ' CHAIR_PATH '/' name ' ' CHAIR_DEST '/' name]);
end