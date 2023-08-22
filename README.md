# Project: Utilizing Eye-Tracking for Empathy-Focused Hiring in Human Resources

The aim of this Python project is to predict individuals' levels of empathy using data collected from eye-tracking. This undertaking utilizes machine learning techniques, including **RandomForestRegressor**, in conjunction with **GroupKFold validation**, to create a predictive model.

### Getting Started

This manual offers you a detailed, sequential guide to set up the project on your local computer, streamlining the process of development and testing activities.

### Prerequisites

To operate this project, it's essential to have Python 3.x installed, along with the following required libraries:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- pickle

###### You can obtain these libraries by running the following command:

pip install pandas numpy matplotlib seaborn scikit-learn pickle

### Project Structure

The project is organized in the following structure:

    lib.py: Contains utility functions for data preprocessing, feature extraction, and model evaluation.
    empathy-Dataset-analysis.ipynb: Encompasses the machine learning pipeline, including exploration and illustrative examples.

Usage
To start the project, run the empathy.ipynb file. This will perform data preprocessing, extract relevant features, train the RandomForestRegressor model, and evaluate its performance using cross-validation.

Make sure to include all necessary details to help users understand the project and its operational aspects effectively.

### Data Acknowledgement

The dataset utilized in this study consists of the following data. To utilize the dataset, simply download it and adjust the file paths to match your setup.

I would like to express my gratitude to the individuals and teams responsible for making the EyeT4Empathy dataset openly available. You can access the dataset through the link provided below:

[EyeT4Empathy Dataset](https://doi.org/10.1038/s41597-022-01862-w)

Please provide attribution to the dataset using the following reference:

P. Lencastre, S. Bhurtel, A. Yazidi, S. Denysov, P. G. Lind, et al. EyeT4Empathy: Dataset of foraging for visual information, gaze typing and empathy assessment. Scientific Data, 9(1):1–8, 2022

<pre>
```bibtex
@article{Lencastre2022,
  author = {Lencastre, Pedro and Bhurtel, Sanchita and Yazidi, Anis and et al.},
  title = {EyeT4Empathy: Dataset of foraging for visual information, gaze typing and empathy assessment},
  journal = {Sci Data},
  volume = {9},
  pages = {752},
  year = {2022},
  doi = {10.1038/s41597-022-01862-w}
}
```
</pre>
