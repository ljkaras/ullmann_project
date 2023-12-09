# IMPORTS: 

import sys

import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem import Draw

from IPython.display import display

import pickle
from sklearn.neighbors import NearestNeighbors

import warnings

# SET MATPLOTLIB STANDARDS:

matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 10

# HELPER FUNCTIONS:

# Function to load pickle files
def load_pickle(file_path):
    """
    Loads DataFrames, dictionaries, and trained models from a .pkl file.

    Parameters:
    - file_path (str): The file path of the .pkl file.

    Returns:
    - Any: The loaded file object.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        with open(file_path, 'rb') as file:
            return pickle.load(file)
# Function to convert a given SMILES string to isomeric SMILES
def convert_to_isomeric_smiles(smiles):
    """
    Uses RDKit, a cheminformatics toolkit, to convert the entered SMILES string into isomeric SMILES.

    Parameters:
    - smiles (str): The SMILES string to convert.

    Returns:
    - str: The isomeric SMILES representation of the input molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    isomeric_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    return isomeric_smiles
# Function to get the substrates smiles
def get_input(smiles_type, substrate_dataframe, property_substrate):
    """
    Obtains user input for a SMILES string, converts it to isomeric SMILES,
    and retrieves a relevant property value from the specified DataFrame.

    Parameters:
    - smiles_type (str): Type of substrate (e.g., "primary amine", "aryl-bromide").
    - substrate_dataframe (pd.DataFrame): DataFrame containing substrate information.
    - property_substrate (str): Name of the property to retrieve from the DataFrame.

    Returns:
    - tuple: A tuple containing the entered isomeric SMILES and the corresponding property value.
    """

    while True:
        # Prompt the user to enter a SMILES string
        input_string = input(f"Enter SMILES string of a {smiles_type} in the library: ")
        
        # Convert the entered SMILES to isomeric SMILES
        input_smiles = convert_to_isomeric_smiles(input_string)
        
        # Initialize pred to an empty list
        pred = []
        
        try:
            # Try to find the relevant value for the entered isomeric SMILES in the DataFrame
            pred = substrate_dataframe.loc[substrate_dataframe['smiles'] == input_smiles, property_substrate].values[0]
            print(f"The value for the entered compound: {pred}")
            break  # Exit the loop if a valid SMILES is entered
        except IndexError:
            # Handle the case where the entered SMILES is not found in the library
            print(f"{smiles_type} not found in the library.")
            retry = input("Do you want to try again? [yes/no]: ")
            if retry.lower() != 'yes' and retry.lower() != 'y':
                print("Exiting the program.")
                sys.exit()
            else:
                continue  # Ask for input again if the user wants to retry
        
    # Check if pred is a valid numeric value
    if not isinstance(pred, (int, float)):
        print(f"No {smiles_type} entered. Exiting the program.")
        sys.exit()
    
    return input_smiles, pred
# Function to plot the smiles structure
def display_structure(smiles_type, smiles, substrate_dataframe):
    """
    Displays the molecular structure of the entered smiles strings.

    Parameters:
    - smiles_type (str): Type of substrate (e.g., "primary amine", "aryl-bromide").
    - smiles (str): Isomeric SMILES representation of the entered compound.
    - substrate_dataframe (pd.DataFrame): DataFrame containing substrate information.

    Returns:
    - plt: RDKit plot of the entered smiles
    """

    mol = Chem.MolFromSmiles(smiles)
    plt_mol = Chem.Draw.MolToImage(mol, size=(150, 150), sanitize=False)

    # Display the structure of the entered compound
    print(f"{smiles_type.capitalize()} structure:")
    Draw.ShowMol(mol)
    # <PILIMAGE>.show()
    # display(plt_mol)
# Function to normalize the amine_pred and br_pred values
def normalize_and_stack(amine_pred, br_pred, loaded_x_values, loaded_y_values):
    """
    Normalize and stack the target coordinates to create input data for machine learning models.

    Parameters:
    - amine_pred (float): N𝛿– value for the entered primary amine
    - br_pred (float): Steric value for the entered aryl-bromide
    - loaded_x_values (tuple): Tuple containing min and max values for x-axis normalization.
    - loaded_y_values (tuple): Tuple containing min and max values for y-axis normalization.

    Returns:
    - x_target_norm (float): Normalized value of the N𝛿– value for the entered primary amine
    - y_target_norm (float): Normalized value of the Steric value for the entered aryl-bromide
    - input_data (numpy.ndarray): Stacked normalized coordinates as input data.
    """
    # Convert predictions to NumPy arrays
    X = np.array([amine_pred])
    Y = np.array([br_pred])

    # Normalize the target coordinates
    x_target_norm = (X - loaded_x_values[0]) / (loaded_x_values[1] - loaded_x_values[0])
    y_target_norm = (Y - loaded_y_values[0]) / (loaded_y_values[1] - loaded_y_values[0])

    # Stack the normalized coordinates to create input data
    input_data = np.column_stack((x_target_norm, y_target_norm))

    return x_target_norm, y_target_norm, input_data
# Function to get the confidence value for the entered coupling partners
def get_confidence_prediction(x_target_norm, y_target_norm, confidence_model):
    """
    Get confidence predictions for the entered coupling partners using the interpolation model.

    Parameters:
    - x_target_norm (float):  Normalized value of the N𝛿– value for the entered primary amine
    - y_target_norm (float): Normalized value of the Steric value for the entered aryl-bromide
    - confidence_model (callable): Model for predicting confidence.

    Returns:
    - confidence_pred (numpy.ndarray): Prediction (below or above 20%) and the confidence value
    - message (str): Message describing the prediction results.
    """
    # Make predictions with the confidence model
    confidence_pred = confidence_model(x_target_norm, y_target_norm)

    # Display prediction results
    if confidence_pred[0] < 0:
        message = f'This combination of aryl-bromide and primary amine is predicted to yield <20% with {-confidence_pred[0]:.1f}% confidence.'
    else: 
        message = f'This combination of aryl-bromide and primary amine is predicted to yield >20% with {confidence_pred[0]:.1f}% confidence.'

    print(message)
    return confidence_pred, message
# Function to find the nearest neighbors 
def get_nearest_neighbors(input_data, knn_model, training_dict):
    """
    Get information about the nearest neighbors using the k-nearest neighbors model.

    Parameters:
    - input_data (numpy.ndarray): Input data for which nearest neighbors are sought.
    - knn_model (object): k-nearest neighbors model.
    - training_dict (dict): Dictionary containing information about products in the training set.

    Returns:
    - P1 (str): Product ID for the first nearest neighbor.
    - P1_info (dict): Information about the first nearest neighbor.
    - P2 (str): Product ID for the second nearest neighbor.
    - P2_info (dict): Information about the second nearest neighbor.
    """
    # Find the nearest neighbors and their distances
    distances, indices = knn_model.kneighbors(input_data)

    # Define product IDs for the first and second nearest neighbors
    P1 = f'P{100 + indices[0][0]}'
    P1_info = training_dict[P1]
    P2 = f'P{100 + indices[0][1]}'
    P2_info = training_dict[P2]

    return P1, P1_info, P2, P2_info
# Function to plot the confidence contour map
def plot_confidence_contour(x_grid, y_grid, input_data, confidence_pred):
    """
    Plot the confidence contour and the nearest neighbors.

    Parameters:
    - x_grid (numpy.ndarray): X-coordinates for the meshgrid.
    - y_grid (numpy.ndarray): Y-coordinates for the meshgrid.
    - confidence_model (function): Model for predicting confidence values.
    - input_data (numpy.ndarray): Input data points.
    - confidence_pred (numpy.ndarray): Confidence predictions.
    - P1_info (dict): Information about the first nearest neighbor.
    - P2_info (dict): Information about the second nearest neighbor.
    """
    # Calculate confidence values on the meshgrid
    confidence_interp = confidence_model(x_grid, y_grid)
    # Clip confidence values to a reasonable range
    confidence_interp_norm = np.clip(confidence_interp, -100, 100)
    # Extract normalized coordinates of the nearest neighbors
    X_NN = np.array([P1_info['Normalized_N𝛿–'], P2_info['Normalized_N𝛿–']])
    Y_NN = np.array([P1_info['Normalized_Steric'], P2_info['Normalized_Steric']])
    neighbors_data = np.column_stack((X_NN, Y_NN))

    # Plot the confidence contour and data points
    plt.figure(figsize=(3, 3))
    contour = plt.contourf(x_grid, y_grid, confidence_interp_norm, cmap='RdBu', levels=100)
    scatter = plt.scatter(input_data[:, 0], input_data[:, 1], c=confidence_pred, cmap='RdBu', marker='X', s=75, edgecolors='black', linewidth=1, vmin=-1, vmax=1)
    scatter = plt.scatter(neighbors_data[:, 0], neighbors_data[:, 1], marker='o', s=10, color='black', linewidth=1)
    plt.xlabel('Amine [Normalized N$^\delta$$^–$]')
    plt.ylabel('Aryl-Bromide [Normalized Steric]')
    plt.show()
# Function to display the best ligands information
def display_ligands_info(P_info, ligands):
    """
    Display information about the top ligands of a given nearest neighbor.

    Parameters:
    - P_info (dict): Information about the product, including top ligands.
    - ligands (dict): Dictionary mapping ligand names to PubChem links.
    """
    # Define top 3 ligands as L1, L2, and L3
    smiles_list = [ligands_df.loc[ligands_df['Compound_Name'] == P_info[f'Top_{i}_Ligand'], 'smiles'].values[0] for i in range(1, 4)]
    name_list = [P_info[f'Top_{i}_Ligand'] for i in range(1, 4)]
    yield_list = [P_info[f'Top_{i}_Yield'] for i in range(1, 4)]

    # Create a list of Mol objects from SMILES
    best_ligands = [Chem.MolFromSmiles(smi) for smi in smiles_list]

    # Create images of ligands and display them
    plt_ligands = [Chem.Draw.MolToImage(mol, size=(300, 150), sanitize=False) for mol in best_ligands]
    fig, axs = plt.subplots(1, 3, figsize=(8, 4))
    for i, ax in enumerate(axs):
        ax.imshow(plt_ligands[i])
        ax.text(0.5, -0.1, f"{name_list[i]}\nYield: {yield_list[i]}%", ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.axis("off")

    plt.show()

    print("Here are pubchem links for the three ligands:")
    for i in range(1, 4):
        print(f"{P_info[f'Top_{i}_Ligand']}: {ligands[P_info[f'Top_{i}_Ligand']]}")

# LOADING PKL FILES:

# Load min and max values for normalization
loaded_x_values, loaded_y_values = load_pickle('min_max_values.pkl')
# Load the confidence model
confidence_model = load_pickle('confidence_model.pkl')
# Load the k-nearest neighbors model
knn_model = load_pickle('knn_model.pkl')
# Load the best ligands data
best_ligands = load_pickle('best_ligands.pkl')
# Load grid data (x and y values)
x_grid, y_grid = load_pickle('grid_data.pkl')
# Load a dictionary containing training data
training_dict = load_pickle('training_dict.pkl')
# Read substrates and ligands DataFrames from pickle
br_df = pd.read_pickle('br_df.pkl')
amine_df = pd.read_pickle('amine_df.pkl')
ligands_df = pd.read_pickle('ligands_df.pkl')
# Load pubchem links for ligands
ligands = load_pickle('pubchem_lig.pkl')
    
# INPUT SMILES FOR PRIMARY AMINE AND ARYL-BROMIDE:

# Primary amine input loop
amine_smiles, amine_pred = get_input("primary amine", amine_df, 'N𝛿–')

# Display and retrieve information for primary amine
display_structure("primary amine", amine_smiles, amine_df,)

# Aryl-bromide input loop
br_smiles, br_pred = get_input("aryl-bromide", br_df, 'Steric')

# Display and retrieve information for aryl-bromide
display_structure("aryl-bromide", br_smiles, br_df)

# PREDICT THE YIELD OUTCOME, CONFIDENCE LEVEL, AND TOP LIGANDS

x_target_norm, y_target_norm, input_data = normalize_and_stack(amine_pred, br_pred, loaded_x_values, loaded_y_values)
confidence_pred, message = get_confidence_prediction(x_target_norm, y_target_norm, confidence_model)
P1, P1_info, P2, P2_info = get_nearest_neighbors(input_data, knn_model, training_dict)
plot_confidence_contour(x_grid, y_grid, input_data, confidence_pred)
display_ligands_info(P1_info, ligands)
display_ligands_info(P2_info, ligands)