from vib_analyzer import VibrationAnalyzer

# Define a callback function to save the plots to a file
def save_plot(plt):
    plt.savefig("vibration_plot.png")
analyzer = VibrationAnalyzer("vibration_data.csv", axis="x")
analyzer.preprocess()
analyzer.plot_time_domain()
analyzer.plot_freq_domain()

