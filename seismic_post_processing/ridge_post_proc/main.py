import os
import h5py
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import unyt
import yt
from pyVBRc.materials.materials import (
    AlignedInclusions,
    IsotropicMedium,
    load_isotropic_medium,
)
from pyVBRc.vbrc_structure import VBRCstruct
from scipy.io import savemat
from dask import compute, delayed
import matplotlib

matplotlib.rc('font', size=12)

_default_buff_size = (500, 250)


def add_derived_fields(U: int, K: int, phi_table_file: str, ds=None):
    # adds some derived fields to a yt dataset
    df = pd.read_csv(phi_table_file)

    def get_ref_phi(U, K):
        dfU = df[df.U_0 == U]
        phi = dfU.phi_0[dfU.K_0_neg_exp == K].tolist()
        if len(phi) == 1:
            return phi[0]
        else:
            raise RuntimeWarning("Did not find a single phi_0 for this U, K")

    def _dimensional_T(field, data):
        return data[("connect0", "TVP_solid::Temperature")] * unyt.unyt_quantity(
            1623, "K"
        )

    if ds is not None:
        add_field_func = getattr(ds, "add_field")
    else:
        add_field_func = getattr(yt, "add_field")

    add_field_func(
        name=("connect0", "temperature"),
        function=_dimensional_T,
        sampling_type="local",
        take_log=False,
        units="K",
        force_override=True,
    )

    def _porosity(field, data):
        phi0 = get_ref_phi(U, K)
        return data[("connect0", "Fluid::Porosity")] * phi0

    add_field_func(
        name=("connect0", "porosity"),
        function=_porosity,
        take_log=False,
        sampling_type="local",
        units="",
        force_override=True,
    )


def get_a_slice(ds, base_field, ftype="connect0"):
    field = (ftype, base_field)
    slc = yt.SlicePlot(ds, "z", field, origin="native")
    slc.set_log(field, False)
    return slc


def _get_pixelized_array(
        ds, base_field, buff_size, ftype="connect0", wid=None, height=None
):
    # field=(ftype, base_field)
    # slc = yt.SlicePlot(ds, "z",field, origin="native", buff_size=buff_size)
    # slc.set_log(field, False)
    if wid is None:
        wid = ds.domain_width[0].to("code_length").d
    if height is None:
        height = ds.domain_width[1].to("code_length").d

    slc = ds.slice("z", 0.5, center=ds.domain_center)
    frb = slc.to_frb(wid, buff_size, height=height)
    field = (ftype, base_field)
    return frb[field].to_ndarray()


def _get_pixelized_x_z(ds, nx, nz):
    wids = ds.domain_width.to("km")
    x = np.linspace(-wids[0] / 2.0, wids[0] / 2, nx)
    z = abs(np.linspace(0.0, wids[1], nz) - wids[1])
    return x, z


def prep_for_VBRc(
        T,
        phi,
        x,
        z,
        litho_enrichment=None,
        anelastic_methods=None,
        separate_phases=None,
        base_name=None,
):
    # writes out data to intermediate files that Octave will read back in for
    # use with the VBRc

    if litho_enrichment is None:
        m_struct_fields = np.core.records.fromarrays([T, phi], names=["T_K", "phi"])
    else:
        m_struct_fields = np.core.records.fromarrays(
            [T, phi, litho_enrichment], names=["T_K", "phi", "litho_enrichment"]
        )
    m_struct_domain_info = np.core.records.fromarrays(
        [[x.min(), x.max(), len(x)], [z.min(), z.max(), len(z)]],
        names=["x_info", "z_info"],
    )

    if base_name is None:
        base_name = "input"

    savemat(
        f"{base_name}_SVs.mat", {"input_SVs": m_struct_fields}
    )  # SVs = "state variables"
    savemat(f"{base_name}_domain.mat", {"input_domain": m_struct_domain_info})

    ep = {
        "delta_rho": 500,  # fluid-solid diff
        "rho_s": 3300,  # solid density
        "rho_crust": 2800,  # crustal density
        "z_moho_km": 5,  # crustal thickness -- is this a model output?
        "frequency_min": 1 / 100,  # minimum frequency in Hz
        "frequency_max": 1 / 10,  # maximum frequency in Hz
        "n_freqs": 2,  # number of frequency points
        "use_log_freq_range": 1,  # if 1, range will use log10 scale between min/max
        "grain_size_m": 0.01,  # grain size for calculation in m
        "sig_MPa": 0.1,  # deviatoric stress to use in MPa
    }

    if anelastic_methods is not None:
        for ane_meth in anelastic_methods:
            ep[ane_meth] = 1

    if separate_phases is not None:
        ep["separate_phases"] = 1

    if litho_enrichment is not None:
        ep["use_enrichment"] = 1
    param_file = f"{base_name}_extra_params.mat"
    _write_extra_params(ep, filename=param_file)


def _write_extra_params(extra_params, filename="extra_params.mat"):
    arrays = [[ep, ] for ep in extra_params.values()]
    extra_params = np.core.records.fromarrays(arrays, names=list(extra_params.keys()))
    savemat(filename, {"extra_params": extra_params})


# def get_lith_enrichment(degF, T, Tsolidus, z, zLAB_max_km=40):
#     litho_enrichment = np.tile(
#         degF[-1, :],
#         (T.shape[0], 1),
#     )
#     litho_enrichment = litho_enrichment - degF
#     litho_enrichment[T > Tsolidus] = 0.0
#
#     if zLAB_max_km is not None:
#         litho_enrichment[z > zLAB_max_km, :] = 0.0
#
#     return litho_enrichment


def _extract_lab_info(ds_vbrc, x_locs, method="eburgers_psp"):
    min_depths = []
    min_vals = []
    max_Qinv_depth = []
    max_Qinv_vals = []
    ave_dVsdz = []
    phi_cutoff = 1e-4
    ave_length = 10
    for x_loc in x_locs:
        raysample = ds_vbrc.ray((x_loc, 0, 0.5), (x_loc, 100, 0.5))
        Vs = raysample[f"Vs_{method}"].to("km/s")
        phi = raysample[f"porosity"]
        Qinv = raysample[f"Qinv_{method}"]
        y = raysample["y"].to("km")

        y_min = np.min(y[phi > phi_cutoff])

        # yc = (y[1:] + y[:-1]) / 2
        # dVsdz = (Vs[1:] - Vs[:-1]) / (y[1:] - y[:-1])
        ave_mask_above = (y.d <= y_min.d) & (y.d >= y_min.d - ave_length)
        ave_mask_V = (y.d <= y_min.d + ave_length) & (y.d >= y_min.d)

        V_astheno = np.mean(Vs[ave_mask_V])
        V_lith = np.mean(Vs[ave_mask_above])

        dV = (V_astheno - V_lith) / V_lith * 100

        min_depths.append(y_min)
        min_vals.append(V_astheno)
        max_Qinv_depth.append(y_min)
        max_Qinv_vals.append(np.mean(Qinv[ave_mask_V]))
        ave_dVsdz.append(dV)

    ave_dVsdz = np.array(ave_dVsdz)
    min_depths = np.array(min_depths)
    min_vals = np.array(min_vals)
    max_Qinv_depth = np.array(max_Qinv_depth)
    max_Qinv_vals = np.array(max_Qinv_vals)
    return min_depths, min_vals, max_Qinv_depth, max_Qinv_vals, ave_dVsdz


# available timesteps. keys are (U, K) pairs.
_timesteps = {
    (2, 7): "001001",
    (2, 9): "001001",
    (4, 7): "000301",
    (4, 9): "000251",
    (8, 7): "001001",
    (8, 9): "001001",
}


def _get_filename(U0, K0, data_dir):
    U = int(U0)
    K = int(K0)

    timestep = _timesteps[(U, K)]
    fname = os.path.join(f"U{U}K{K}", f"myridgemodel{timestep}.pvtu")

    fname = os.path.join(data_dir, fname)
    if not os.path.isfile(fname):
        raise FileNotFoundError(f"Could not find {fname}.")
    return fname


def _load_vbr_run_as_ds(fname, x: np.ndarray, z: np.ndarray):
    vbrc_results = VBRCstruct(fname)

    def promote_array(x):
        return np.expand_dims(x.transpose(), axis=-1)

    ifreq = 0
    vbrc_data = {}
    for method in ("xfit_mxw", "xfit_premelt", "eburgers_psp", "andrade_psp"):
        method_results = getattr(vbrc_results.output.anelastic, method)
        vbrc_data["Vs_" + method] = promote_array(method_results.V[:, :, ifreq])
        vbrc_data["Qinv_" + method] = promote_array(method_results.Qinv[:, :, ifreq])

    vbrc_data['porosity'] = promote_array(vbrc_results.input.SV.phi)
    bbox = np.array(
        [
            [x.min(), x.max()],
            [z.min(), z.max()],
            [0, 1],
        ]
    )
    shp = vbrc_data["Vs_xfit_mxw"].shape

    return yt.load_uniform_grid(vbrc_data, shp, bbox=bbox, length_unit="km")


class _RunInfoData:
    _meta_fields = ("U", "K", "fname", "buff_size")
    _vars2d = ("phi", "degF", "refert")

    # simple class for pickling metadata about a run
    def __init__(
            self,
            x=None,
            z=None,
            refert=None,
            phi=None,
            degF=None,
            fname=None,
            length_unit=None,
            buff_size=None,
            U=None,
            K=None,
            vbrc_output_files=None,
    ):
        self.x = x
        self.z = z
        self.refert = refert
        self.phi = phi
        self.degF = degF
        self.fname = fname
        self.length_unit = length_unit
        if isinstance(buff_size, np.ndarray):
            buff_size = tuple(buff_size.tolist())
        self.buff_size = buff_size
        self.U = U
        self.K = K
        self.vbrc_output_files = vbrc_output_files

    def save(self, fname: str):
        if fname.endswith(".hdf5") is False:
            fname += ".hdf5"

        with h5py.File(fname, "w") as ds:
            grp = ds.create_group('dims')
            grp.create_dataset('x', data=self.x)
            grp.create_dataset('z', data=self.z)

            grp2d = ds.create_group('data2d')
            for nm in self._vars2d:
                vals = getattr(self, nm)
                grp2d.create_dataset(nm, data=vals)

            for nm in self._meta_fields:
                val = getattr(self, nm)
                ds.attrs[nm] = val
            ds.attrs["length_unit_name"] = self.length_unit[1]
            ds.attrs["length_unit_val"] = self.length_unit[0]

            grp_files = ds.create_group('vbrc_output_files')
            for ky, val in self.vbrc_output_files.items():
                grp_files.attrs[ky] = val


def _load_run_info_data(fname):
    with h5py.File(fname, 'r') as ds:
        rid_inputs = {
            "x": ds['dims']['x'][:],
            "z": ds['dims']['z'][:],
        }
        for v in _RunInfoData._vars2d:
            rid_inputs[v] = ds['data2d'][v][:]

        for attr in ds.attrs.keys():
            if attr.startswith('length') is False:
                rid_inputs[attr] = ds.attrs[attr]

        vbrc_files = {}
        for attr in ds['vbrc_output_files'].attrs.keys():
            vbrc_files[attr] = ds['vbrc_output_files'].attrs[attr]

        rid_inputs['vbrc_output_files'] = vbrc_files
        length_unit = (ds.attrs['length_unit_val'], ds.attrs['length_unit_name'])
        rid_inputs['length_unit'] = length_unit
    return _RunInfoData(**rid_inputs)


def _load_run_build_array(U, K, fname, data_dir, buff_size, length_unit):
    # load in the base yt dataset
    ds = yt.load(
        fname, detect_null_elements=True, units_override={"length_unit": length_unit}
    )
    phi_file = os.path.join(data_dir, "phi_0_redim.csv")
    if not os.path.isfile(phi_file):
        raise FileNotFoundError(f"Could not find phi table, {phi_file}")
    add_derived_fields(U, K, phi_file, ds=ds)

    # get all the pixelized arrays
    T = _get_pixelized_array(ds, "temperature", buff_size=buff_size)
    phi = _get_pixelized_array(ds, "porosity", buff_size=buff_size)
    phi[phi < 0] = 0
    # some small neg numbers where T<Tsol, just zero them out.
    degF = _get_pixelized_array(ds, "DegreeMelting::DegreeMelt", buff_size=buff_size)
    x, z = _get_pixelized_x_z(ds, buff_size[0], buff_size[1])

    # calculate refert factor
    ref_F = degF[:, int(buff_size[0] / 2)]
    ref_F = np.tile(ref_F, (degF.shape[1], 1)).T
    ref_F.shape
    refert = ref_F - degF
    refert[refert < 0] = 0.0
    return ds, T, phi, degF, x, z, refert


def process_single_run(
        U,
        K,
        data_dir,
        buff_size,
        output_dir,
):
    """ """

    yt.set_log_level(50)
    length_unit = (100, "km")
    fname = _get_filename(U, K, data_dir)
    ds, T, phi, degF, x, z, refert = _load_run_build_array(U, K, fname,
                                                           data_dir, buff_size, length_unit)

    # baseline VBRc calculation prep
    baseline_f = os.path.join(output_dir, f"U{U}K{K}_baseline")
    separate_f = os.path.join(output_dir, f"U{U}K{K}_separate")
    prep_for_VBRc(T, phi, x, z, base_name=baseline_f)

    # refertilized calculation prep
    prep_for_VBRc(
        T,
        phi,
        x,
        z,
        litho_enrichment=refert,
        separate_phases=True,
        base_name=separate_f,
    )

    # run both
    oct_cmnd = f"run_VBRc('{baseline_f}'); clear all; run_VBRc('{separate_f}');"
    the_command = 'octave --eval "' + oct_cmnd + '"'
    os.system(the_command)

    # save relevant metadata for easily loading data back in
    vbrc_files = {
        "baseline": f"U{U}K{K}_baseline_VBRc_output.mat",
        "separate": f"U{U}K{K}_separate_VBRc_output.mat",
        "separate_secondary": f"U{U}K{K}_separate_VBRc_output_secondary.mat",
    }
    rid = _RunInfoData(
        x=x,
        z=z,
        refert=refert,
        phi=phi,
        degF=degF,
        fname=fname,
        length_unit=length_unit,
        buff_size=buff_size,
        U=U,
        K=K,
        vbrc_output_files=vbrc_files,
    )

    fname = os.path.join(output_dir, f"U{U}K{K}_metadata.hdf5")
    rid.save(fname)


def process_all_runs(data_dir=None, buff_size=None, output_dir=None):
    """ """

    yt.set_log_level(50)
    if data_dir is None:
        data_dir = "."

    if buff_size is None:
        buff_size = _default_buff_size

    if output_dir is None:
        output_dir = "."

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    vbrc_runs = []
    for UK in _timesteps.keys():
        vbrc_runs.append(
            delayed(process_single_run)(UK[0], UK[1], data_dir, buff_size, output_dir)
        )

    compute(*vbrc_runs)


def _get_run_data(U, K, output_dir):
    fn = os.path.join(output_dir, f"U{U}K{K}_metadata.hdf5")
    return _load_run_info_data(fn)


_clr_opts = {
    "U": {
        2: '#F05039',  # (1.0, 0.4, 0.4, 1),
        4: '#1F449C',  # (0.4, 1.0, 0.4, 1),
        8: '#A8B6CC',  # (0.4, 0.4, 1.0, 1),
    },  # line colors
    "K": {7: '-', 9: '--'},  # linestyle
}


def baseline_plots(output_dir, anelastic_method="eburgers_psp"):
    f, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4.5))
    x_locs = np.linspace(0, 99.9, 50)

    fld = f"Vs_{anelastic_method}"
    for UK in _timesteps.keys():
        U = UK[0]
        K = UK[1]
        rd = _get_run_data(U, K, output_dir)

        fn = os.path.join(output_dir, rd.vbrc_output_files["baseline"])
        ds_vbrc = _load_vbr_run_as_ds(fn, rd.x, rd.z)

        slc = yt.SlicePlot(ds_vbrc, "z", ("stream", fld), origin="native")
        slc.set_log(("stream", fld), False)
        slc.flip_vertical()
        slc.save(os.path.join(output_dir, f"U{U}K{K}_{anelastic_method}"))

        Vs_z, Vs, Q_z, Qinv, dV = _extract_lab_info(
            ds_vbrc, x_locs, method=anelastic_method
        )

        axs[1].plot(
            x_locs,
            Vs,
            label=f"U{U}K{K}",
            linestyle=_clr_opts["K"][K],
            color=_clr_opts["U"][U],
        )
        axs[0].plot(
            x_locs,
            Vs_z,
            label=f"U{U}K{K}",
            linestyle=_clr_opts["K"][K],
            color=_clr_opts["U"][U],
        )
        axs[2].plot(
            x_locs,
            dV,
            label=f"U{U}K{K}",
            linestyle=_clr_opts["K"][K],
            color=_clr_opts["U"][U],
        )

    axs[1].set_xlabel("Distance from ridge axis [km]")
    axs[1].set_ylabel("V$_s$(z$_{LAB}$) [km/s]")
    # axs[1].set_ylim([3.2, 4.4])
    axs[0].legend()
    axs[0].set_xlabel("Distance from ridge axis [km]")
    axs[0].set_ylabel("z$_{LAB}$ [km]")
    axs[2].set_xlabel("Distance from ridge axis [km]")
    axs[2].set_ylabel("Vs reduction at LAB [%]")
    figname = os.path.join(output_dir, f"summary_fig_Vs_vs_x_{anelastic_method}.png")
    print(f"Saving {figname}")
    f.set_tight_layout('tight')
    f.savefig(figname)


def separate_phases_plots(output_dir, anelastic_method="eburgers_psp"):
    f, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))
    shear_mod_loc = ("anelastic", anelastic_method, "M")
    for UK in _timesteps.keys():
        U = UK[0]
        K = UK[1]
        rd = _get_run_data(U, K, output_dir)

        main_phase_file = os.path.join(output_dir, rd.vbrc_output_files["separate"])
        sec_phase_file = os.path.join(
            output_dir, rd.vbrc_output_files["separate_secondary"]
        )

        vbrc_main = VBRCstruct(main_phase_file)
        vbrc_secondary = VBRCstruct(sec_phase_file)

        main_phase = load_isotropic_medium(vbrc_main, shear_mod_loc, ifreq=0)
        sec_phase = load_isotropic_medium(vbrc_secondary, shear_mod_loc, ifreq=0)

        # voigt average of final index
        frac_main = np.flipud(1 - rd.refert[:, -1])
        frac_sec = np.flipud(rd.refert[:, -1])
        dens = (
                main_phase.density[:, -1] * frac_main + sec_phase.density[:, -1] * frac_sec
        )
        G = (
                main_phase.shear_modulus[:, -1] * frac_main
                + sec_phase.shear_modulus[:, -1] * frac_sec
        )
        M_Pwave = (
                main_phase.pwave_effective_modulus[:, -1] * frac_main
                + sec_phase.pwave_effective_modulus[:, -1] * frac_sec
        )
        # Vs = np.sqrt(G / dens)
        Vp = np.sqrt(M_Pwave / dens)

        # build an LAB mask
        phi = vbrc_main.input.SV.phi[:, -1]
        phi_s = phi.copy()
        for _ in range(2):
            phi_s[1:] = (phi_s[1:] + phi_s[:-1]) / 2

        LAB_mask = (phi_s < 1e-5) * (rd.z > 40)

        aspect_ratio = 0.01
        ai = AlignedInclusions(aspect_ratio)
        v_p_max_vals = []
        v_p_min_vals = []
        for iz in range(frac_main.size):
            nu = np.atleast_1d(main_phase.poisson_ratio[iz, -1])
            G = np.atleast_1d(main_phase.shear_modulus[iz, -1])
            dens = np.atleast_1d(main_phase.density[iz, -1])
            primary_phase = IsotropicMedium(nu, G, "shear", density=dens)
            nu = np.atleast_1d(sec_phase.poisson_ratio[iz, -1])
            G = np.atleast_1d(sec_phase.shear_modulus[iz, -1])
            dens = np.atleast_1d(sec_phase.density[iz, -1])
            inclusions = IsotropicMedium(nu, G, "shear", density=dens)
            ai.set_material(primary_phase, inclusions, frac_sec[iz] * LAB_mask[iz])
            theta = np.linspace(0, np.pi, 50)
            azi = theta * 180 / np.pi - 90.0
            v_p, v_sv, v_sh = ai.velocities(theta)

            v_p_max_vals.append(v_p.max())
            v_p_min_vals.append(v_p.min())
        v_p_max_vals = np.array(v_p_max_vals)
        v_p_min_vals = np.array(v_p_min_vals)

        lw = _clr_opts["K"][K]
        clr = _clr_opts["U"][U]

        axs[0].plot(Vp / 1e3, rd.z, color=clr, linestyle=lw, label=f"U{U}K{K}", linewidth=1)
        axs[1].plot(
            100 * (v_p_max_vals - v_p_min_vals) / v_p_max_vals, rd.z, color=clr, linestyle=lw
        )

    axs[1].yaxis.set_ticklabels([])
    axs[0].set_xlabel("Isotropic Vp [km/s]")
    axs[0].legend()
    axs[0].set_ylabel("z [km]")
    axs[1].set_xlabel("Anisotropic Vp: percent anisotropy")

    for i in range(2):
        axs[i].set_ylim([70, 100])

    figname = os.path.join(
        output_dir, f"summary_fig_vp_anisotropy_{anelastic_method}.png"
    )
    print(f"Saving {figname}")
    f.savefig(figname)


def plot_all_porosity(data_dir, output_dir):
    """ plot 2d porosity fields for all data """
    for UK in _timesteps.keys():
        U = UK[0]
        K = UK[1]

        fname = _get_filename(U, K, data_dir)

        length_unit = (100, "km")

        # load in the base yt dataset
        ds = yt.load(
            fname,
            detect_null_elements=True,
            units_override={"length_unit": length_unit},
        )
        phi_file = os.path.join(data_dir, "phi_0_redim.csv")
        if not os.path.isfile(phi_file):
            raise FileNotFoundError(f"Could not find phi table, {phi_file}")
        add_derived_fields(U, K, phi_file, ds=ds)

        slc = yt.SlicePlot(ds, "z", ("connect0", "porosity"), origin="native")
        slc.set_log(("connect0", "porosity"), False)
        slc.save(os.path.join(output_dir,f"U{U}K{K}_phi"))


def _get_effective_sill_thickness(refert, phi, z):
    refert_z = refert[:, -1]
    phi_z = phi[:, -1]
    phi_z[phi_z < 1e-10] = 0.0
    for _ in range(3):
        phi_z[1:-1] = (phi_z[0:-2] + phi_z[1:-1] + phi_z[2:]) / 3
    lab_mask = phi[:, -1] < 1e-5 * (z < 50.)
    z_sill = np.abs(np.trapz(refert_z * lab_mask, z))
    lithosphere_fert = refert_z * lab_mask
    return z_sill, lithosphere_fert


def integrate_refert_all_runs(data_dir, output_dir, buff_size=None, ):
    if buff_size is None:
        buff_size = _default_buff_size

    thicknesses = []
    u_vals = []
    k_vals = []
    for U, K in _timesteps.keys():
        length_unit = (100, "km")
        fname = _get_filename(U, K, data_dir)
        ds, T, phi, degF, x, z, refert = _load_run_build_array(U, K, fname, data_dir, buff_size, length_unit)

        z_sill, _ = _get_effective_sill_thickness(refert, phi, z)
        thicknesses.append(z_sill.d)
        u_vals.append(U)
        k_vals.append(K)

    thicknesses = np.array(thicknesses)
    k_vals = np.array(k_vals)
    u_vals = np.array(u_vals)
    f, ax = plt.subplots(1)
    ax.plot(u_vals[k_vals == 7], thicknesses[k_vals == 7], "k", linestyle='none', marker='.', label='K7')
    ax.plot(u_vals[k_vals == 9], thicknesses[k_vals == 9], "k", linestyle='none', marker='x', label='K9')
    ax.set_xlabel('U')
    ax.set_ylabel('integrated thickness [km]')
    plt.legend()
    figname = os.path.join(output_dir, f"summary_fig_effective_sill.png")
    f.savefig(figname)
    plt.show()

# def redistribute_sill(U, K, data_dir, buff_size=None):
#
#     if buff_size is None:
#         buff_size = _default_buff_size
#
#     length_unit = (100, "km")
#     fname = _get_filename(U, K, data_dir)
#     ds, T, phi, degF, x, z, refert = _load_run_build_array(U, K, fname, data_dir, buff_size, length_unit)
#
#     z_sill, lithosphere_fert = get_effective_sill_thickness(refert, phi, z)
#
#     # distribute that z_sill proportionally by fertilization peaks
#     # print(z_sill)
#
#     peak_locs, _ = find_peaks(lithosphere_fert)
#     peak_vals = lithosphere_fert[peak_locs]
#     peak_z_locs = z[peak_locs]
#     peak_power = peak_vals / peak_vals.sum()
#
#     sill_thicknesses = peak_power * z_sill
#
#     z_lith = np.linspace(0, np.max(z), 5000)
#     in_sill = np.zeros(z_lith.shape)
#     for peak_z, peak_power in zip(peak_z_locs, peak_power):
#         z_sill_peak = z_sill * peak_power
#         z_center = peak_z
#         z_shallow = z_center - z_sill_peak / 2
#         z_deep = z_center + z_sill_peak / 2
#         in_sill[(z_lith <= z_deep) & (z_lith >= z_shallow)] = 1.0
#
#     f, axs = plt.subplots(nrows=1, ncols=2)
#     axs[0].plot(lithosphere_fert, z)
#     axs[0].plot(phi_z, z)
#     axs[0].set_ylim([z.min(), z.max()])
#     axs[0].invert_yaxis()
#
#     axs[1].plot(in_sill, z_lith)
#     axs[1].set_ylim([z.min(), z.max()])
#     axs[1].invert_yaxis()
#     plt.show()
#
#     return lithosphere_fert, z
