function [  ] = create_AMLhdf5_S( SS )
%CREATE_AMLHDF5 Summary of this function goes here
%   Detailed explanation goes here

rows_dataset = 30000;

% Number of individuals
n_ID = length(SS);
myd = zeros(30000,7,1,n_ID);
for cur_ID = 1 :n_ID
    [rows,columns] = size(SS(cur_ID).Data);
    rows_missing = rows_dataset - rows;
    
    %More missing than the set value (rows_dataset)
    if(rows_missing > 0 ) 
        append_zeros = zeros(rows_missing,columns);
        SS(cur_ID).Data = vertcat(SS(cur_ID).Data, append_zeros);
    %More data than set value (rows_dataset);
    %Throw away the rest
    elseif( rows_missing < 0)
            %keep_this = zeros(rows,1);
            %keep_this(1:rows_dataset) = 1;
            SS(cur_ID).Data = SS(cur_ID).Data((1:rows_dataset),:);
    end
    disp(cur_ID);
    myd(:,:,1,cur_ID) = SS(cur_ID).Data;
end

myl = vertcat(SS.Labels)';
% Loop over all individuals

% HDF5 create folder
h5create('hdf5_AML6.h5', '/data', size(myd));

% HDF5 push data to folder
h5write('hdf5_AML6.h5', '/data', myd);

% HDF5 create folder
h5create('hdf5_AML6.h5', '/label', size(myl));

% HDF5 push data to folder
h5write('hdf5_AML6.h5', '/label', myl);

end

