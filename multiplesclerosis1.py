#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 23:31:18 2023

@author: chadikattouah
"""

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import random
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

MSdata = pd.read_csv('conversion_predictors_of_clinically_isolated_syndrome_to_multiple_sclerosis.csv')
MSdata.info()
MSdata.describe()
MSdata.isnull().sum()
MSdata.dropna(subset=['Schooling', 'Initial_Symptom'], inplace=True)
MSdata.isnull().sum()
# Replace null values with '0' in 'Initial_EDSS' and 'Final_EDSS' columns
MSdata['Initial_EDSS'].fillna(0, inplace=True)
MSdata['Final_EDSS'].fillna(0, inplace=True)
MSdata.isnull().sum()

# Map the labels for 'group' and 'gender' columns
group_labels = {1: 'CDMS', 2: 'No CDMS'}
gender_labels = {1: 'Male', 2: 'Female'}

# Create a cross-tabulation of 'group' and 'gender'
cross_tab = pd.crosstab(MSdata['group'].map(group_labels), MSdata['Gender'].map(gender_labels))

# Set seaborn style
sns.set(style='whitegrid')

# Create a bar plot to visualize the cross-tabulation
fig, ax = plt.subplots()

# Customize the plot appearance
ax = sns.barplot(data=cross_tab, x=cross_tab.index, y='Male', color='blue', alpha=0.9)
ax = sns.barplot(data=cross_tab, x=cross_tab.index, y='Female', color='pink', alpha=0.9, bottom=cross_tab['Male'])

# Set plot title and labels
ax.set_xlabel('Group', fontsize=12)
ax.set_ylabel('Count', fontsize=12)

# Streamlit Heading
st.header('Group Diagnosis Across Gender')

# Show the filtered plot
st.pyplot(fig)

# Add radio buttons for gender selection
selected_gender = st.radio('Select Gender', list(gender_labels.values()))

# Filter the cross-tabulation based on the selected gender
filtered_cross_tab = cross_tab[selected_gender]

# Show the filtered cross-tabulation as a table
st.dataframe(filtered_cross_tab)
st.header('Age and Schooling variations against people with and without Mutlipe Sclerosis')
# Filter the data for labels '1' and '2' in the 'group' column
filtered_data = MSdata[MSdata['group'].isin([1, 2])]

# Get unique labels in the 'group' column
group_labels = filtered_data['group'].unique()

# Set seaborn style
sns.set(style='whitegrid')

# Allow user to select the label
selected_label = st.selectbox('Select Group Label:', group_labels)

# Filter the data based on the selected label
filtered_data = MSdata[MSdata['group'] == selected_label]

# Create subplots for each variable
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Set advanced color palettes
colors = ['#9c27b0', '#e91e63']

# Plot 'Age' with advanced color
sns.boxplot(data=filtered_data, x='Age', ax=axes[0], color=colors[0])


# Plot 'Schooling' with advanced color
sns.boxplot(data=filtered_data, x='Schooling', ax=axes[1], color=colors[1])


# Set overall plot title
fig.suptitle(f"Boxplot Visualization of 'Age' and 'Schooling' for Label '{selected_label}'", fontsize=16)

# Adjust layout spacing
plt.tight_layout()

# Show the plots
st.pyplot(fig)

# Create Streamlit app
st.header('Number of Patients Across Initial Symptoms in CDMS or non-CDMS')

# Filter data based on user selection
filter_value = st.selectbox("Select filter value:", options=MSdata['group'].unique())
filtered_data = MSdata[MSdata['group'] == filter_value]

# Count unique values in 'Initial_Symptom' column
symptom_counts = filtered_data['Initial_Symptom'].value_counts()

# Generate random colors for each unique value
colors = [f'#{random.randint(0, 16777215):06x}' for _ in range(len(symptom_counts))]

# Create a DataFrame to store unique values and their counts
unique_values_df = pd.DataFrame({'Initial_Symptom': symptom_counts.index, 'Count': symptom_counts.values})

# Set the figure size
fig, ax = plt.subplots(figsize=(8, 6))

# Create the bar chart
bars = ax.bar(range(len(unique_values_df)), unique_values_df['Count'], color=colors)

# Set the x-axis labels to the initial symptoms as integers
ax.set_xticks(range(len(unique_values_df)))
ax.set_xticklabels(unique_values_df['Initial_Symptom'].astype(int), rotation=90, ha='center')

# Add count labels inside each bar
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height, height,
            ha='center', va='bottom', fontweight='bold', fontsize=8)

# Remove y-axis labels
ax.set_yticklabels([])

# Set the chart title and axis labels
ax.set_xlabel('Initial Symptoms')

# Define the simplified legend text
legend_text = '''
<div style="display: flex; flex-wrap: wrap; justify-content: center;">
    <div style="width: 200px; text-align: center; padding: 5px;">
        <span style="padding: 5px;">1</span>: Visual
    </div>
    <div style="width: 200px; text-align: center; padding: 5px;">
        <span style="padding: 5px;">2</span>: Sensory
    </div>
    <div style="width: 200px; text-align: center; padding: 5px;">
        <span style="padding: 5px;">3</span>: Motor
    </div>
    <div style="width: 200px; text-align: center; padding: 5px;">
        <span style="padding: 5px;">4</span>: Other
    </div>
    <div style="width: 200px; text-align: center; padding: 5px;">
        <span style="padding: 5px;">5</span>: Visual and Sensory
    </div>
    <div style="width: 200px; text-align: center; padding: 5px;">
        <span style="padding: 5px;">6</span>: Visual and Motor
    </div>
    <div style="width: 200px; text-align: center; padding: 5px;">
        <span style="padding: 5px;">7</span>: Visual and Others
    </div>
    <div style="width: 200px; text-align: center; padding: 5px;">
        <span style="padding: 5px;">8</span>: Sensory and Motor
    </div>
    <div style="width: 200px; text-align: center; padding: 5px;">
        <span style="padding: 5px;">9</span>: Sensory and Other
    </div>
    <div style="width: 200px; text-align: center; padding: 5px;">
        <span style="padding: 5px;">10</span>: Motor and Other
    </div>
    <div style="width: 200px; text-align: center; padding: 5px;">
        <span style="padding: 5px;">11</span>: Visual, Sensory, and Motor
    </div>
    <div style="width: 200px; text-align: center; padding: 5px;">
        <span style="padding: 5px;">12</span>: Visual, Sensory, and Other
    </div>
    <div style="width: 200px; text-align: center; padding: 5px;">
        <span style="padding: 5px;">13</span>: Visual, Motor, and Other
    </div>
    <div style="width: 200px; text-align: center; padding: 5px;">
        <span style="padding: 5px;">14</span>: Sensory, Motor, and Other
    </div>
    <div style="width: 200px; text-align: center; padding: 5px;">
        <span style="padding: 5px;">15</span>: Visual, Sensory, Motor and other
    </div>
</div>
'''

# Display the simplified legend
st.markdown(f'<div style="text-align: center;">Legend:</div>', unsafe_allow_html=True)
st.markdown(legend_text, unsafe_allow_html=True)

# Display the bar chart in Streamlit
st.pyplot(fig)

st.header("MRI Analysis With Machine Learning")
# Filter data based on user input
def filter_data(data, periventricular_mri, cortical_mri):
    return data[(data['Periventricular_MRI'] == periventricular_mri) & (data['Cortical_MRI'] == cortical_mri)]

# Perform analysis
def perform_analysis(data):
    X = MSdata[['Periventricular_MRI', 'Cortical_MRI']]
    y = MSdata['group']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Display classification report
    st.subheader("Classification Report")
    classification_rep = classification_report(y_test, predictions)
    st.text_area("Report", classification_rep)

    # Feature importance
    st.subheader("Feature Importance")
    # Add colors to the bar chart
    feature_imp = pd.Series(model.feature_importances_, index=X.columns)
    st.bar_chart(feature_imp)

# Show the sidebar for user input
st.subheader("MRI Input")
periventricular_mri = st.selectbox("Periventricular MRI", [0, 1])
cortical_mri = st.selectbox("Cortical MRI", [0, 1])

# Filter data based on user input
filtered_data = filter_data(MSdata, periventricular_mri, cortical_mri)

# Perform analysis
perform_analysis(filtered_data)

# Filter data based on user input
def filter_data(data, infratentorial_mri, spinal_cord_mri):
    return data[(data['Infratentorial_MRI'] == infratentorial_mri) & (data['Spinal_Cord_MRI'] == spinal_cord_mri)]

# Perform analysis
def perform_analysis(data):
    X = MSdata[['Infratentorial_MRI', 'Spinal_Cord_MRI']]
    y = MSdata['group']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Display classification report
    st.subheader("Classification Report")
    classification_rep = classification_report(y_test, predictions)
    st.text_area("Report", classification_rep)

    # Feature importance
    st.subheader("Feature Importance")
    # Add colors to the bar chart
    feature_imp = pd.Series(model.feature_importances_, index=X.columns)
    st.bar_chart(feature_imp)

# Show the sidebar for user input
st.subheader("MRI Input")
infratentorial_mri = st.selectbox("Infratentorial MRI", [0, 1])
spinal_cord_mri = st.selectbox("Spinal Cord MRI", [0, 1])

# Filter data based on user input
filtered_data = filter_data(MSdata, infratentorial_mri, spinal_cord_mri)

# Perform analysis
perform_analysis(filtered_data)

st.header('Mono or Polysymptomatic')

# Selection
selected_group = st.selectbox('Select Group', group_labels)

# Filter the data based on the selected group
filtered_df = MSdata[MSdata['group'] == selected_group]

# Calculate the percentage of Mono_or_Polysymptomatic labels
percentage = filtered_df['Mono_or_Polysymptomatic'].value_counts(normalize=True) * 100

# Map the label values to their names
label_names = {
    1: 'Monosymptomatic',
    2: 'Polysymptomatic',
    3: 'Unknown'
}

# Convert the label values in the filtered DataFrame to their corresponding names
filtered_df['Mono_or_Polysymptomatic'] = filtered_df['Mono_or_Polysymptomatic'].map(label_names)

# Calculate the percentage of Mono_or_Polysymptomatic labels with updated names
percentage = filtered_df['Mono_or_Polysymptomatic'].value_counts(normalize=True) * 100

# Create a pie chart with custom colors and updated labels
colors = ['#FFA500', '#800080', '#008000']
fig, ax = plt.subplots()
patches, texts, autotexts = ax.pie(percentage, labels=percentage.index, autopct='%1.1f%%', startangle=90, colors=colors)

# Set the font weight to bold for the pie chart labels
for text in texts:
    text.set_weight('bold')

# Set the font weight to bold for the percentage values inside the pie chart
for autotext in autotexts:
    autotext.set_weight('bold')

ax.axis('equal')

# Display the pie chart using Streamlit
st.pyplot(fig)

st.header('History of Varicella or chicken-pox')

# Selection
selected_group1 = st.selectbox('Select Group1', group_labels)

# Filter the data based on the selected group
filtered_df = MSdata[MSdata['group'] == selected_group1]

# Calculate the percentage of varicella labels
percentage = filtered_df['Varicella'].value_counts(normalize=True) * 100

# Map the label values to their names
label_names = {
    1: 'Yes',
    2: 'No',
    3: 'Unknown'
}

# Convert the label values in the filtered DataFrame to their corresponding names
filtered_df['Varicella'] = filtered_df['Varicella'].map(label_names)

# Calculate the percentage of varicella labels with updated names
percentage = filtered_df['Varicella'].value_counts(normalize=True) * 100

# Create a pie chart with custom colors and updated labels
colors = ['#FF00FF', '#FFFF00', '#00FFFF']  
fig, ax = plt.subplots()
patches, texts, autotexts = ax.pie(percentage, labels=percentage.index, autopct='%1.1f%%', startangle=90, colors=colors)

# Set the font weight to bold for the pie chart labels
for text in texts:
    text.set_weight('bold')

# Set the font weight to bold for the percentage values inside the pie chart
for autotext in autotexts:
    autotext.set_weight('bold')

ax.axis('equal')

# Display the pie chart using Streamlit
st.pyplot(fig)

st.header('History of Breastfeeding')

# Selection
selected_group2 = st.selectbox('Select Group2', group_labels)

# Filter the data based on the selected group
filtered_df = MSdata[MSdata['group'] == selected_group2]

# Calculate the percentage of Breastfeeding labels
percentage = filtered_df['Breastfeeding'].value_counts(normalize=True) * 100

# Map the label values to their names
label_names = {
    1: 'Yes',
    2: 'No',
    3: 'Unknown'
}

# Convert the label values in the filtered DataFrame to their corresponding names
filtered_df['Breastfeeding'] = filtered_df['Breastfeeding'].map(label_names)

# Calculate the percentage of Breastfeeding labels with updated names
percentage = filtered_df['Breastfeeding'].value_counts(normalize=True) * 100

# Create a pie chart with custom colors and updated labels
colors = ['#FF0000', '#00FF00', '#000FFF']
fig, ax = plt.subplots()
patches, texts, autotexts = ax.pie(percentage, labels=percentage.index, autopct='%1.1f%%', startangle=90, colors=colors)

# Set the font weight to bold for the pie chart labels
for text in texts:
    text.set_weight('bold')

# Set the font weight to bold for the percentage values inside the pie chart
for autotext in autotexts:
    autotext.set_weight('bold')

ax.axis('equal')

# Display the pie chart using Streamlit
st.pyplot(fig)