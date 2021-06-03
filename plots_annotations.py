from run_annotations import save_bad_resp_and_no_stim_annotations

# Plot bad respiration annotation example --> example subject 01
fig_plot_EDA_EEG_bad_resp = save_bad_resp_and_no_stim_annotations(
                                    subject_number=['01'],
                                    plot_EDA_EEG_bad_no_stim= False,
                                    plot_EDA_EEG_bad_resp = True
                                    )
# Save figure ouput
fig_plot_EDA_EEG_bad_resp.savefig('outputs/figures/figure_respiration_artifact')

# Plot annotations regarding non stimuli presentation --> example subject 01
fig_plot_EDA_EEG_bad_no_stim = save_bad_resp_and_no_stim_annotations(
                                    subject_number=['01'],
                                    plot_EDA_EEG_bad_no_stim= True,
                                    plot_EDA_EEG_bad_resp = False
                                    )
# Save figure ouput
fig_plot_EDA_EEG_bad_no_stim.savefig('outputs/figures/figure_no_stim_annotation')

