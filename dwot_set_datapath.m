[~, sys_host_name] = system('hostname');
server_id = regexp(sys_host_name, '^napoli(?<num>\d+).*','names');
if strcmp(DATA_SET,'PASCAL')
    if isempty(server_id)
        VOC_PATH = '/home/chrischoy/Dataset/VOCdevkit/';
    else
        VOC_PATH = '/scratch/chrischoy/Dataset/VOCdevkit/';
    end
    if ismac
        VOC_PATH = '~/dataset/VOCdevkit/';
    end
    addpath(VOC_PATH);
    addpath([VOC_PATH, 'VOCcode']);

    curDir = pwd;
    eval(['cd ' VOC_PATH]);
    VOCinit;
    eval(['cd ' curDir]);

elseif strcmp(DATA_SET,'PASCAL12')
    if isempty(server_id)
        VOC_PATH = '/home/chrischoy/Dataset/PASCAL3D+_release1.1/PASCAL12/VOCdevkit/';
    else
        VOC_PATH = '/scratch/chrischoy/Dataset/PASCAL12/VOCdevkit/';
    end
    if ismac
        VOC_PATH = '~/dataset/VOCdevkit/';
    end
    addpath(VOC_PATH);
    addpath([VOC_PATH, 'VOCcode']);
    
    curDir = pwd;
    eval(['cd ' VOC_PATH]);
    VOCinit;
    eval(['cd ' curDir]);

elseif strcmp(DATA_SET,'3DObject')
    if isempty(server_id)
      DATA_PATH = '/home/chrischoy/Dataset/3DObject/';
    else
      DATA_PATH = '/scratch/chrischoy/Dataset/3DObject/';
    end

    if ismac
      DATA_PATH = '~/dataset/3DObject';
    end
    addpath(DATA_PATH);
end

