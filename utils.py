
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import glob


from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_score, GroupKFold, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error, explained_variance_score

import pickle

warnings.filterwarnings("ignore")


def process_input(input_data):
    # Exclude the first column
    processed_data = input_data.iloc[:, 1:]

    # Columns to exclude if not relevant for Empathy analysis
    excluded_cols = ['X Coordinate', 'Y Coordinate', 'Normalized X Fixation', 'Normalized Y Fixation',
                     'Event Type', 'Event Value', 'System Timestamp', 'Export Date', 'Recording Date',
                     'Recording UTC Date', 'Start Time', 'Timeline', 'Fixation Filter',
                     'Software Version', 'Resolution Height', 'Resolution Width', 'Monitor Latency',
                     'Media Width', 'Media Height', 'Media X Coordinate', 'Media Y Coordinate', 'Original Width',
                     'Recording UTC Start Time', 'Original Height', 'Sensor Information']

    # Forward-fill values for pupil diameter and fixation point columns
    columns_to_forward_fill = [
        'Left Eye Pupil Diameter', 'Right Eye Pupil Diameter', 'Fixation Point X', 'Fixation Point Y']
    processed_data[columns_to_forward_fill] = processed_data[columns_to_forward_fill].ffill()

    # Convert specified columns to numeric format
    numeric_columns = ['Left Gaze X Coordinate', 'Left Gaze Y Coordinate', 'Left Gaze Z Coordinate',
                       'Right Gaze X Coordinate', 'Right Gaze Y Coordinate', 'Right Gaze Z Coordinate',
                       'Left Eye X Coordinate', 'Left Eye Y Coordinate', 'Left Eye Z Coordinate',
                       'Right Eye X Coordinate', 'Right Eye Y Coordinate', 'Right Eye Z Coordinate',
                       'Left Gaze Point X Coordinate', 'Left Gaze Point Y Coordinate', 'Right Gaze Point X Coordinate', 'Right Gaze Point Y Coordinate',
                       'Gaze X Coordinate (Normalized)', 'Gaze Y Coordinate (Normalized)',
                       'Left Gaze X Coordinate (Normalized)', 'Left Gaze Y Coordinate (Normalized)', 'Right Gaze X Coordinate (Normalized)', 'Right Gaze Y Coordinate (Normalized)',
                       'Left Eye Pupil Diameter', 'Right Eye Pupil Diameter']

    # Convert string values to numeric, handling commas
    for col_name in numeric_columns:
        processed_data[col_name] = pd.to_numeric(
            processed_data[col_name].str.replace(',', '.'), errors='coerce')

    return processed_data


def summarized_eye_tracking_data(input_data, group_name):
    # Filter out valid eye data based on validity
    valid_eye_data = input_data[(input_data['Validity left'] == 'Valid') &
                                (input_data['Validity right'] == 'Valid')]

    # Calculate total fixation count
    total_fixations = input_data[input_data['Eye movement type']
                                 == 'Fixation'].shape[0]

    # Compute average duration of fixations
    avg_fixation_duration = input_data[input_data['Eye movement type']
                                       == 'Fixation']['Event duration'].mean()

    # Compute statistics for pupil diameter, gaze point X, gaze point Y, fixation point X, and fixation point Y
    pupil_diameter_stats = input_data[['Pupil diameter left', 'Pupil diameter right']].mean(
        axis=1).agg(['mean', 'median', 'std']).rename(lambda x: f'Pupil Diameter {x.capitalize()}')
    gaze_x_stats = input_data['Gaze point X'].agg(
        ['mean', 'median', 'std']).rename(lambda x: f'Gaze Point X {x.capitalize()}')
    gaze_y_stats = input_data['Gaze point Y'].agg(
        ['mean', 'median', 'std']).rename(lambda x: f'Gaze Point Y {x.capitalize()}')
    fixation_x_stats = input_data['Fixation point X'].agg(
        ['mean', 'median', 'std']).rename(lambda x: f'Fixation Point X {x.capitalize()}')
    fixation_y_stats = input_data['Fixation point Y'].agg(
        ['mean', 'median', 'std']).rename(lambda x: f'Fixation Point Y {x.capitalize()}')

    # Create a summary dictionary containing relevant data
    merged_info = {
        'Participant': input_data['Participant name'].iloc[0],
        'Group': group_name,
        'Recording': input_data['Recording name'].iloc[0],
        'Total Fixations': total_fixations,
        'Average Fixation Duration': avg_fixation_duration
    }
    merged_info.update(pupil_diameter_stats)
    merged_info.update(gaze_x_stats)
    merged_info.update(gaze_y_stats)
    merged_info.update(fixation_x_stats)
    merged_info.update(fixation_y_stats)

    # Create a summary DataFrame
    merged_dataframe = pd.DataFrame(merged_info, index=[0])

    return merged_dataframe


def display_diameter_certainty_visuals(data_records):
    # Extract unique participant IDs
    unique_participants = data_records['ID'].unique()

    # Iterate through each unique individual
    for person in unique_participants:
        # Select data specific to the current individual
        individual_data = data_records[data_records['ID'] == person]

        # Prepare a subset of the data for analysis
        subset_data = individual_data.reset_index().rename(
            columns={'index': 'event_num'}).head(6)

        # Group the subset data by event number
        grouped_data = subset_data.groupby('event_num').agg({
            'Avg_Pupil_Diameter': 'mean',
            'Median_Pupil_Diameter': 'mean',
            'Pupil_Diameter_StdDev': 'mean'
        }).reset_index()

        # Create a new figure and axis
        fig, ax = plt.subplots()

        # Plot the mean with error bars and the median
        ax.errorbar(grouped_data['event_num'], grouped_data['Avg_Pupil_Diameter'],
                    grouped_data['Pupil_Diameter_StdDev'], linestyle='-', marker='o', capsize=5,
                    ecolor="green", elinewidth=0.5, label='Average')
        ax.plot(grouped_data['event_num'], grouped_data['Median_Pupil_Diameter'],
                linestyle='-', marker='s', label='Median')

        # Set labels and title
        ax.set_xlabel('Event Number')
        ax.set_ylabel('Average Pupil Diameter (mm)')
        ax.set_title(
            f'Mean and Median Pupil Diameter Certainty for Participant {person}')

        # Add legend and display the plot
        ax.legend()
        plt.show()


def visualize_comparison_scores(data_table):
    # Extract original and estimated empathy scores from the data table
    orig_scores = data_table['Original Score'].tolist()
    estimated_scores = data_table['Estimated Score'].tolist()

    # Create a scatter plot to compare original vs. estimated scores
    plt.scatter(orig_scores, estimated_scores,
                color='blue', label='Estimated')
    plt.xlabel('Original Scores')
    plt.ylabel('Estimated Scores')
    plt.title('Comparison of Original and Estimated Scores')

    # Add a diagonal line representing perfect predictions
    min_score = min(min(orig_scores), min(estimated_scores))
    max_score = max(max(orig_scores), max(estimated_scores))
    plt.plot([min_score, max_score], [min_score, max_score],
             color='red', label='Ideal Prediction')

    # Display the legend and the plot
    plt.legend()
    plt.show()

    return


def train_evaluate_model(input_data, study_group):
    # Prepare the feature matrix X and target vector y
    features_X = input_data.drop(
        columns=['Total_Score_extended', 'Group_Name', 'Session_Name'])
    target_y = input_data['Total_Score_extended']

    # Initialize a DataFrame to store evaluation outcomes
    evaluation_results = pd.DataFrame(
        columns=['Participant', 'Original_Score', 'Predicted_Score'])

    # Encode the 'Participant' column using LabelEncoder
    label_encoder = LabelEncoder()
    features_X['Participant'] = label_encoder.fit_transform(
        features_X['Participant'])
    participant_groups = input_data['Participant']

    # Set the number of splits for GroupKFold
    num_folds = 30  # Number of participants
    group_kfold = GroupKFold(n_splits=num_folds)

    # Initialize lists to store evaluation metrics
    mean_squared_err_list = []
    r2_scores_list = []
    root_mean_squared_err_list = []
    median_abs_err_list = []
    all_actual_scores = []  # Initialize list to store all actual scores
    all_predicted_scores = []  # Initialize list to store all predicted scores

    # Loop over each fold in GroupKFold
    for fold, (train_indices, test_indices) in enumerate(group_kfold.split(features_X, target_y, groups=participant_groups)):

        X_train, X_test = features_X.iloc[train_indices], features_X.iloc[test_indices]
        y_train, y_test = target_y.iloc[train_indices], target_y.iloc[test_indices]

        ###################################
        # Initialize and train the model (RandomForestRegressor)
        model = RandomForestRegressor()

        model.fit(X_train, y_train)

        # Make predictions using the trained model
        y_predicted = model.predict(X_test)
        ######################################

        print(f"Fold {fold + 1}:")

        for idx, (original, predicted) in enumerate(zip(y_test, y_predicted)):
            participant = input_data.iloc[test_indices[idx]]['Participant']
            print(
                f"  Participant: {participant}, Original Score: {original}, Predicted Score: {predicted:.2f}")
            evaluation_results = evaluation_results.append(
                {'Participant': participant, 'Original_Score': original, 'Predicted_Score': predicted}, ignore_index=True)

        mse = mean_squared_error(y_test, y_predicted)
        root_mse = np.sqrt(mean_squared_error(y_test, y_predicted))
        r2 = r2_score(y_test, y_predicted)
        median_abs_error = median_absolute_error(y_test, y_predicted)
        explained_variance = explained_variance_score(y_test, y_predicted)

        mean_squared_err_list.append(mse)
        r2_scores_list.append(r2)
        root_mean_squared_err_list.append(root_mse)
        median_abs_err_list.append(median_abs_error)
        all_actual_scores.extend(y_test)
        all_predicted_scores.extend(y_predicted)

    # Calculate average evaluation metrics
    avg_r2_score = np.mean(r2_scores_list)
    avg_root_mse = np.mean(root_mean_squared_err_list)
    avg_median_abs_err = np.mean(median_abs_err_list)
    avg_mean_squared_err = np.mean(mean_squared_err_list)

    print(f"Average Root Mean Squared Error: {avg_root_mse}")
    print(f"Average Median Absolute Error: {avg_median_abs_err}")
    print(f"Average Mean Squared Error: {avg_mean_squared_err}")

    return evaluation_results


def display_corr_heatmap(data_table, target_col, top_count=15):

    # Compute the correlation matrix
    corr_matrix = data_table.corr()

    # Select the top_n columns with the highest correlation to the target column
    top_correlated = corr_matrix.nlargest(
        top_count, target_col)[target_col].index

    # Compute the correlation matrix for the selected columns
    selected_corr_matrix = data_table[top_correlated].corr()

    # Create a heatmap using seaborn
    plt.figure(figsize=(10, 10))
    heatmap_ax = sns.heatmap(selected_corr_matrix, annot=True, cmap='coolwarm')

    # Set the title of the heatmap
    heatmap_ax.set_title(f'Correlation Heatmap for {target_col}')

    # Display the heatmap
    plt.show()

    return


def display_score_visualization(data_table):
    # Calculate the mean of original and estimated empathy scores grouped by participant
    averaged_scores = data_table.groupby('Participant').agg(
        {'Original_Score': 'first', 'Estimated_Score': 'mean'})

    # Set options to display all rows and columns
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    # Print the grouped DataFrame
    print("Grouped DataFrame:")
    print(averaged_scores)

    # Reshape the data for visualization
    reshaped_data = averaged_scores.reset_index().melt(id_vars=['Participant'], value_vars=[
        'Original_Score', 'Estimated_Score'], var_name='Score_Type', value_name='Score')

    # Display the scores for the first 7 participants
    first_7_participants = reshaped_data['Participant'].unique()[:7]
    filtered_data = reshaped_data[reshaped_data['Participant'].isin(
        first_7_participants)]

    # Create a bar plot for visualization
    plt.figure(figsize=(10, 5))
    sns.barplot(data=filtered_data, x='Participant',
                y='Score', hue='Score_Type')

    plt.title(
        'Bar Plot of Actual and Estimated Empathy Scores for the Initial Participants')
    plt.xlabel('Participant')
    plt.ylabel('Empathy Score')

    plt.show()

    return
