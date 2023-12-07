function SVs = load_process_scipy_io_mat(filename, flip_arrays_ud)
    SVs_raw = load(filename);  % this is a structure array 
    top_struct_fields = fieldnames(SVs_raw);
    SVs_raw = SVs_raw.(top_struct_fields{1});
    
    % convert to a structure of 2d arrays
    SVs = struct();    
    fields = fieldnames(SVs_raw);
    
    fld = fields{1};
    nz = size(SVs_raw)(1);
    nx = size(SVs_raw)(2);
    
    % initialize the arrays
    
    for ifield = 1:numel(fields)
        fld = fields{ifield};
        SVs.(fld) = zeros(nz, nx);
    end
    
    % fill the arrays
    for iz = 1:nz
        for ix = 1:nx
            for ifield = 1:numel(fields)
                fld = fields{ifield};
                val = SVs_raw(iz, ix).(fld);
                SVs.(fld)(iz, ix) = val;
            end
        end
    end           
    
    if flip_arrays_ud == 1
        for ifield = 1:numel(fields)
            fld = fields{ifield};
            SVs.(fld) = flipud(SVs.(fld));
        end
    end
end