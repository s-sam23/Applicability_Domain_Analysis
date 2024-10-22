from ad_calc import CombinedApplicabilityDomain  # Import the combined AD class
import streamlit as st
import pandas as pd
from rdkit import Chem
from sklearn.preprocessing import StandardScaler

# Set wide layout for the dashboard
st.set_page_config(layout="wide")

# App title
st.title("Applicability Domain Analysis")

# Sidebar for uploading datasets
st.sidebar.title("Upload Datasets")
train_file = st.sidebar.file_uploader("Upload Training Data", key="train_file")
test_file = st.sidebar.file_uploader("Upload Test Data", key="test_file")

# Input column names for SMILES and target column
cols = st.text_input("Enter the name of the SMILES column and actual label column (separate them with a comma)")

# Initialize the AD checker if training data has been uploaded
if train_file and cols:
    cols = cols.split(",")
    train_data = pd.read_csv(train_file) if train_file.name.endswith('.csv') else pd.read_excel(train_file)[cols]

    # Rename columns for easier access
    train_data.rename(columns={cols[0]: 'SMILES', cols[1]: 'ACT'}, inplace=True)

    # Cache the AD checker and training descriptors if not already cached
    if 'combined_ad_checker' not in st.session_state:
        training_smiles = train_data['SMILES'].tolist()
        binary_target = train_data['ACT'].tolist()

        # Initialize and cache the AD checker (training descriptors are calculated here)
        st.session_state.combined_ad_checker = CombinedApplicabilityDomain(training_smiles, target=binary_target)
        st.success("Training data descriptors have been calculated and cached.")

    # Show the training data
    st.write("Training Data:")
    st.dataframe(train_data[['SMILES', 'ACT']])

    # Check if test data has been uploaded
    if test_file:
        test_data = pd.read_csv(test_file) if test_file.name.endswith('.csv') else pd.read_excel(test_file)[cols]
        test_data.rename(columns={cols[0]: 'SMILES', cols[1]: 'ACT'}, inplace=True)

        # Display the test data
        st.write("Testing Data:")
        st.dataframe(test_data[['SMILES', 'ACT']])

        # UI for choosing AD method
        choice = st.selectbox("Choose Your AD Method", options=[
            '', 'Descriptors_Range_Based_Method', 'Convex_Hull_Based_Method', 'Leverage_Based_Method'
        ])

        # Get the cached AD checker instance
        combined_ad_checker = st.session_state.combined_ad_checker

        # Ensure that a method is selected before applying the AD checks
        if choice:
            if choice == "Descriptors_Range_Based_Method":
                # Range-based AD check
                within_ad_range = combined_ad_checker.is_within_range_domain(test_data['SMILES'])
                within_ad_range_np = within_ad_range.get()  # Convert CuPy array to NumPy

                st.write("Compounds outside the range-based AD:")
                st.dataframe(test_data[~within_ad_range_np])

            elif choice == "Convex_Hull_Based_Method":
                # Convex Hull-based AD check
                within_ad_hull = combined_ad_checker.is_within_convex_hull_domain(test_data['SMILES'])
                # within_ad_hull = within_ad_hull.get()
                st.write("Compounds outside the Convex Hull-based AD:")
                st.dataframe(test_data[~within_ad_hull])

            elif choice == "Leverage_Based_Method":
                # Leverage-based AD check
                within_ad_leverage = combined_ad_checker.is_within_leverage_domain(test_data['SMILES'])
                within_ad_leverage = within_ad_leverage.get()
                st.write("Compounds outside the leverage-based AD:")
                st.dataframe(test_data[~within_ad_leverage])

        else:
            st.warning("Please choose an applicability domain method to proceed.")

# If no training data has been uploaded, display a warning message
else:
    if not train_file:
        st.warning("Please upload training data to proceed.")
    if not cols:
        st.warning("Please enter the SMILES and target column names.")

