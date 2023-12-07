from ridge_post_proc import baseline_plots, separate_phases_plots, integrate_refert_all_runs, plot_all_porosity

output_dir = 'output'
data_dir = 'data'

plot_all_porosity(data_dir, output_dir)
baseline_plots(output_dir, anelastic_method='eburgers_psp')
baseline_plots(output_dir, anelastic_method='xfit_premelt')

separate_phases_plots(output_dir, anelastic_method='eburgers_psp')
integrate_refert_all_runs(data_dir, output_dir)
