from ridge_post_proc import baseline_plots, separate_phases_plots, integrate_refert_all_runs, plot_all_porosity

output_dir = 'output'
data_dir = 'data'

plot_all_porosity(data_dir, output_dir)
for ifreq in range(4):
    baseline_plots(output_dir, anelastic_method='eburgers_psp', ifreq=ifreq)
    baseline_plots(output_dir, anelastic_method='xfit_premelt', ifreq=ifreq)

separate_phases_plots(output_dir, anelastic_method='eburgers_psp')
integrate_refert_all_runs(data_dir, output_dir)
