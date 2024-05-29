# Epileptic-Seizure-Recognition
This dataset is a modified version of a well-known dataset used for detecting epileptic seizures.

The original dataset had recordings from 500 people, each recorded for about 23.6 seconds. These recordings were split into 4097 data points. Each data point represents the brain's electrical activity at a specific time.

We rearranged the data into 23-second chunks, each containing 178 data points, representing 1 second of brain activity. So now, we have 11500 pieces of information, each with 178 data points and a label.

The label (y) indicates what was happening during the recording:

5 means the person's eyes were open.
4 means the person's eyes were closed.
3 means the recording was from a healthy brain area, with the location of a tumor identified.
2 means the recording was from the area where the tumor was located.
1 means there was seizure activity.
Most researchers focus on binary classification: identifying seizures (class 1) versus non-seizure activity (classes 2, 3, 4, and 5).

This dataset is sourced from the UCI Machine Learning Repository
