from ridge_post_proc import baseline_plots, baseline_freq_dep, separate_phases_plots, integrate_refert_all_runs, plot_all_porosity

output_dir = 'output'
data_dir = 'data'

plot_all_porosity(data_dir, output_dir)
nfreq = 10 # number of frequencies that were run for VBRc calculation
for ane_meth in ('eburgers_psp', 'xfit_premelt'):
    for ifreq in range(nfreq):
        baseline_plots(output_dir, anelastic_method=ane_meth, ifreq=ifreq)
        baseline_plots(output_dir, anelastic_method=ane_meth, ifreq=ifreq)
    baseline_freq_dep(output_dir, anelastic_method=ane_meth)

separate_phases_plots(output_dir, anelastic_method='eburgers_psp')
integrate_refert_all_runs(data_dir, output_dir)
