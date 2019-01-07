% demo for generating HDF5 data
clc;
clear;

data_folder = 'path to ShapeNet category folder';

save_folder = 'path to save the data';
if ~isdir(save_folder)
    mkdir(save_folder)
end

list = dir(data_folder);
data = zeros(1000, 32, 32, 32, 'int8');

cur_num = 0;
for i=1:length(list)
    model_file = [data_folder '/' list(i).name '/model.obj'];
    
    if ~exist(model_file, 'file')
        continue;
    end
    
    volume = obj2vox(model_file, 24, 4, 0);
    cur_num = cur_num + 1;
    data(cur_num,:,:,:) = volume;
end

data = data(1:cur_num, :, :, :);
data = data(:, :, end:-1:1, :);
data = permute(data, [2, 3, 4, 1]);

save_name = [save_folder '/data.h5'];
if exist(save_name, 'file')
    delete(save_name);
end
h5create(save_name, '/data', size(data), 'Datatype', 'int8');
h5write(save_name, '/data', data);
