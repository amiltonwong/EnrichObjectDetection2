% demo create detectors
addpath('../MatlabRenderer/');

if ~exist('renderer','var')  
  model_file = 'Mesh/Bicycle/road_bike';
  renderer = Renderer();
  if ~renderer.initialize([model_file '.3ds'], 700, 700, 0, 0, 0, 0, 25)
    error('fail to load model');
    renderer.delete();
    clear renderer;
  end
end

renderer.setViewpoint(10,10,0,0,25);
[im, depth] = renderer.renderCrop();
