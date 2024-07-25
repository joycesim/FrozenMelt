function run_VBRc(vbr_filename)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reads the intermediate datafiles and starts off the VBRc in various ways.
    % this gets started from the Python methods, not meant to be called directly
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    vbr_filename_out = [vbr_filename, "_VBRc_output.mat"];

    % first check the local directory in case it was downloaded, have that
    % take
    tst = fullfile(".", "v1.1.2", "vbr_init.m");
    if isfile(tst)
        addpath(tst)
    else
        path_to_top_level_vbr = getenv("vbrdir");
        if isfile(fullfile(path_to_top_level_vbr, "vbr_init.m"))
            addpath(path_to_top_level_vbr)
        else
            error("Could not find a VBRc installation.")
        end
    end
    vbr_init
    addpath("vbrc_helper")
    
    % read in structures
    disp(" ")
    disp("loading data from scipy output")
    SV_input_file = [vbr_filename, "_SVs.mat"];
    SV = load_process_scipy_io_mat(SV_input_file, 1);    

    domain_info = load_process_scipy_io_mat([vbr_filename, "_domain.mat"], 0);
    extra_params = load_process_scipy_io_mat([vbr_filename, "_extra_params.mat"], 0);   
    
    disp(" ")
    disp("calculating additional state variables")
    
    SV = calculate_pressure_density(SV, domain_info, extra_params);
    SV.Tsolidus_K = hirschmann_solidus(SV.P_GPa);
    % freqeuency and other params
    if extra_params.use_log_freq_range == 1
        min_freq_n = log10(extra_params.frequency_min);
        max_freq_n = log10(extra_params.frequency_max);
        freq = logspace(min_freq_n, max_freq_n, extra_params.n_freqs);
    else
        min_freq = extra_params.frequency_min;
        max_freq = extra_params.frequency_max;
        freq = linspace(min_freq, max_freq, extra_params.n_freqs);
    end    
    SV.f = freq;
    SV.dg_um = extra_params.grain_size_m * 1e6 * ones(size(SV.T_K)); 
    SV.sig_MPa = extra_params.sig_MPa * ones(size(SV.T_K));
    

    disp(" ")
    disp("initializing the VBRc settings")
        
    if isfield(extra_params, "anh_poro")
        if extra_params.anh_poro == 0
            VBR.in.elastic.methods_list={'anharmonic';};
        else
            VBR.in.elastic.methods_list={'anharmonic'; 'anh_poro'};
        end
    else 
        VBR.in.elastic.methods_list={'anharmonic'; 'anh_poro'};
    end 
    
    VBR.in.viscous.methods_list={'HK2003'};

    fullmlist = {'andrade_psp';'xfit_mxw';'eburgers_psp'; 'xfit_premelt'};
    imethds = numel(fullmlist);
    mlist = {};
    n_method = 1;
    has_premelt = 0;
    for imeth = 1:imethds
        if isfield(extra_params, fullmlist{imeth})
            if strcmp(fullmlist{imeth}, 'xfit_premelt')
                has_premelt = 1;
            end
            mlist{n_method} = fullmlist{imeth};
            n_method = n_method + 1;
        end
    end
    if n_method == 1
        mlist = fullmlist;
    end

    VBR.in.anelastic.methods_list=mlist;
    if has_premelt
        % use new pre-melt scaling here if xfit_premelt is in the methods
       VBR.in.anelastic.xfit_premelt.include_direct_melt_effect = 1;
    end

    VBR.in.anelastic.methods_list=mlist;
    VBR.in.elastic.anharmonic=Params_Elastic('anharmonic'); % unrelaxed elasticity
    VBR.in.elastic.anharmonic.Gu_0_ol = 78; %75.5; % olivine reference shear modulus [GPa]
    VBR.in.GlobalSettings.melt_enhancement=0;
    VBR.in.SV=SV;

    if isfield(extra_params, "separate_phases") && extra_params.separate_phases == 1
        disp("Running as separate phases")
        % in this case, the degree of melting is used for calculating the
        % pressure, but then we do 2 separate calculations

        % VBR, VBR2 will be the separate phases at same elevated P,T
        VBR.in.SV.chi = ones(size(VBR.in.SV.T_K));  % one is all olivine
        disp('    ol-only calculation')
        [VBR] = VBR_spine(VBR);
        save_vbr_oct_mat(vbr_filename_out, VBR);

        disp('    secondary-only calculation')
        VBR2 = VBR;
        VBR2.in.SV.chi = zeros(size(VBR.in.SV.T_K)); % zero is all crust
        [VBR2] = VBR_spine(VBR2);
        vbr_file2 = [strsplit(vbr_filename_out,'.'){1}, '_secondary.mat'];
        save_vbr_oct_mat(vbr_file2, VBR2);
    else
        disp("Calling the VBRc as single phase")
        [VBR] = VBR_spine(VBR) ;
        save_vbr_oct_mat(vbr_filename_out, VBR);
    end


    
end


function SV = calculate_pressure_density(SV, domain_info, extra_params)
    
    % maybe add to extra params
    P_0_Pa = 101325 + 1000 * 9.8 * 1000; % 1 km ocean overlying
    dTdz_ad = 0.3 ; % adiabatic gradient, K/km
         
    % construct z
    z_km = linspace(domain_info.z_info(1), domain_info.z_info(2), domain_info.z_info(3));
    z_km = z_km'; 
    
    % calculate absolute from potential temperatures
    Tfac = (z_km + extra_params.z_moho_km) * dTdz_ad;    
    for ix = 1:domain_info.x_info(3)
        SV.T_K(:, ix) = SV.T_K(:, ix) ;%+ Tfac;  
    end
    
    % calculate pressure at top of model domain
    n_crust = 10; % crustal points (cause we integrate below, so having a few points at least is good)
    T_crust_K = linspace(300, min(SV.T_K(:)), 10);
    z_crust_km = linspace(0, extra_params.z_moho_km, n_crust);
    Rho_o_crust = extra_params.rho_crust * ones(size(T_crust_K));

    FracFo = 0.9; 
    
    % calculate pressure, density in the model 
    % (correcting for non-adiabatic thermal expansion, adiabatic compaction)
    SV.P_GPa = zeros(size(SV.T_K));
    SV.rho = zeros(size(SV.T_K));
    Rho_o_mantle = extra_params.rho_s * ones(size(z_km));
    
    if isfield(extra_params, "use_enrichment")
        SV.chi = ones(size(SV.T_K)) - SV.litho_enrichment;
    else 
        SV.chi = ones(size(SV.T_K));        
    end 
    
    
    
    z_ix_m = [z_crust_km'; z_km+ z_crust_km(end)] * 1000;
    T_crust_K = T_crust_K';    
    for ix = 1:domain_info.x_info(3)
    
        Rho_o_mantle_ix = Rho_o_mantle .* SV.chi(:,ix) + (1-SV.chi(:,ix)) * extra_params.rho_crust;
        Rho_o = [Rho_o_crust'; Rho_o_mantle_ix];
        T_K = SV.T_K(:, ix); - Tfac;
        T_ix = [T_crust_K; T_K];
        rho_ix = Density_Thermal_Expansion(Rho_o, T_ix, FracFo);        
        [rho_ix, P_ix_Pa] = Density_Adiabatic_Compression(rho_ix, z_ix_m, P_0_Pa); 
        SV.P_GPa(:, ix) = P_ix_Pa(n_crust+1:end)/1e9;
        SV.rho(:, ix) = rho_ix(n_crust+1:end);
    end
    
    SV.z_km = z_km;

end


function T_solidus_K = hirschmann_solidus(P_GPa)
    A_1 = 1085.7; % deg C
    A_2 = 132.9; % degC/GPa
    A_3 = -3.2; % deg C/GPa^2
    
    T_solidus_K = A_1 + A_2 * P_GPa + A_3 * P_GPa.^2 + 273; 
 
    % P_L = delta_rho g x = 500*9.81*x
end


function save_vbr_oct_mat(fname, VBR)
    isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;
    if isOctave
        save(fname, 'VBR', '-mat-binary')
    else
        save(fname, 'VBR')
    end
end
