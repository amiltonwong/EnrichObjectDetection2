function dwot_compile()
%EOD_COMPILE compile c++ and cuda codes required for the package

% Create directory if it does not exist
if ~exist('bin', 'dir')
    mkdir('bin');
end

% Remove all compiled binary files
!rm bin/*

% Set paths
CUDA_PATH = '/usr/local/cuda-6.5/';
MATLAB_PATH = '/afs/cs/package/matlab-r2013b/matlab/r2013b/';

% CUDA_PATH = '/usr/local/cuda/';
% MATLAB_PATH = '/Applications/MATLAB_R2014a.app/';


% ------------------------------------------------------------------------------
%                                                        Compile whitening codes
% ------------------------------------------------------------------------------

eval(sprintf(['!nvcc ',...
    ' -v -O3 -DNDEBUG -Xcompiler -fPIC -gencode=arch=compute_30,code=sm_30',...
    ' -c ./CUDA/whiten_features.cu',...
    ' -I./CUDA/ ',...
    ' -I%s/extern/include ',...
    ' -I%s/toolbox/distcomp/gpu/extern/include ',...
    ' --output-file bin/whiten_features.o'],...
    MATLAB_PATH, MATLAB_PATH));

eval(sprintf(['mex ',...
    '-v -largeArrayDims bin/whiten_features.o -L%s/lib64 ',...
    ' -outdir ./bin ',...
    '-L%s/bin/glnxa64 -lcudart -lcublas -lmwgpu'], CUDA_PATH, MATLAB_PATH));


!nvcc -ptx CUDA/scramble_gamma_to_sigma.cu --output-file bin/scramble_gamma_to_sigma.ptx

% ------------------------------------------------------------------------------
%                                           Compile HoG feature extraction codes
% ------------------------------------------------------------------------------
% resize
mex CXX=gcc CXXOPTIMFLAGS='-O3 -DNDEBUG -funroll-all-loops' HoG/resizeMex.cc ...
    -outdir ./bin

% Compile Felzenszwalb's 31D features
% mex -O HoG/features_pedro.cc -outdir ./bin

% Compile a variant of Felzenszwalb's features which doesn't do normalization
% mex -O HoG/features_raw.cc -outdir ./bin

% mulththreaded convolution without blas (see voc-release-4)
% This convolution gets slower everytime it runs. Probably some thread issues
%mex -O HoG/fconvblas.cc -lmwblas -o fconvblas -outdir ./bin
%mex CC=gcc LD=gcc COPTIMFLAGS='-O3 -DNDEBUG' fconvblas.cc -lmwblas ...
%          -o ./bin/fconvblas -outdir ./bin

% Compile blas
% mex CXX=gcc CXXOPTIMFLAGS='-O3 -DNDEBUG -funroll-all-loops' fconvblas.cc ...
%       -lmwblas

% Use the following if error occurs
% Floating point version
mex CXXOPTIMFLAGS='-O3 -DNDEBUG -funroll-all-loops' HoG/fconvblasfloat.cc -lmwblas ...
    -outdir ./bin/

% Mac error fix example
% if ismac
%   eval(sprintf(['!nvcc -v -O3 -DNDEBUG  -gencode=arch=compute_30,code=sm_30 ',...
%   '-Xcompiler -fPIC -I%s/extern/include -I%s/toolbox/distcomp/gpu/extern/include ',...
%   '-c cudaDecorrelateFeature.cu'], MATLAB_PATH, MATLAB_PATH));
% 
%   % Use default cuda in MATLABROOT/bin/maci64/
%   eval(sprintf(['mex -v -largeArrayDims bin/whiten_features.o -I%s/include -L%s/bin/maci64  ',...
%   '-lcudart -lcublas -lmwgpu'], CUDA_PATH, MATLAB_PATH));
% 
%   % If we use CUDA lib, it fails to find rpath.
%   % mex -v mexGPUExample.o  -L/usr/local/cuda/lib -L/Applications/MATLAB_R2014a.app/bin/maci64 -lcudart -lcufft -lmwgpu
%   % !install_name_tool -change @rpath/libcufft.6.0.dylib /usr/local/cuda/libcufft.dylib mexGPUExample.mexmaci64
%   % !install_name_tool -change @rpath/libcudart.6.0.dylib /usr/local/cuda/libcudart.dylib mexGPUExample.mexmaci64
% end
