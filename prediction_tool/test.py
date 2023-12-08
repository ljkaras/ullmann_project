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

    # Retrieve InChIKey and structure for the entered compound
    mol = Chem.MolFromSmiles(smiles)
    plt_mol = Chem.Draw.MolToImage(mol, size=(150, 150), sanitize=False)

    # Display the structure of the entered compound
    print(f"{smiles_type.capitalize()} structure:")
    Draw.ShowMol(mol)
    # <PILIMAGE>.show()
    # display(plt_mol)

# STILL NEED TO COMMENT: 
def convert_to_normalized_coordinates(X, Y, loaded_x_values, loaded_y_values):
    x_target_norm = (X - loaded_x_values[0]) / (loaded_x_values[1] - loaded_x_values[0])
    y_target_norm = (Y - loaded_y_values[0]) / (loaded_y_values[1] - loaded_y_values[0])
    return np.column_stack((x_target_norm, y_target_norm))

def display_prediction_results(confidence_pred):
    if confidence_pred[0] < 0:
        print(f'This combination is predicted to yield <20% with {-confidence_pred[0]:.1f}% confidence.')
    else: 
        print(f'This combination is predicted to yield >20% with {confidence_pred[0]:.1f}% confidence.')

def display_confidence_contour(input_data, confidence_model, x_grid, y_grid):
    confidence_interp = confidence_model(x_grid, y_grid)
    confidence_interp_norm = np.clip(confidence_interp, -100, 100)
    plt.figure(figsize=(3, 3))
    contour = plt.contourf(x_grid, y_grid, confidence_interp_norm, cmap='RdBu', levels=100)
    scatter = plt.scatter(input_data[:, 0], input_data[:, 1], c=confidence_pred, cmap='RdBu', marker='X', s=75, edgecolors='black', linewidth=1, vmin=-1, vmax=1)
    scatter = plt.scatter(neighbors_data[:, 0], neighbors_data[:, 1], marker='o', s=10, color='black', linewidth=1)
    plt.xlabel('Amine [Normalized N$^\delta$$^–$]')
    plt.ylabel('Aryl-Bromide [Normalized Steric]')
    plt.show()

def display_nearest_neighbors_info(P_info, ligands_df, ligands):
    smiles_list = [ligands_df.loc[ligands_df['Compound_Name'] == P_info[f'Top_{i}_Ligand'], 'smiles'].values[0] for i in range(1, 4)]
    name_list = [P_info[f'Top_{i}_Ligand'] for i in range(1, 4)]
    yield_list = [P_info[f'Top_{i}_Yield'] for i in range(1, 4)]
    best_ligands = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    plt_ligands = [Chem.Draw.MolToImage(mol, size=(300, 150), sanitize=False) for mol in best_ligands]
    fig, axs = plt.subplots(1, 3, figsize=(8, 4))
    for i, ax in enumerate(axs):
        ax.imshow(plt_ligands[i])
        ax.text(0.5, -0.1, f"{name_list[i]}\nYield: {yield_list[i]}%", ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.axis("off")
    plt.show()
    print(f"Here are pubchem links for the three ligands:"
        f"\n{name_list[0]}: {ligands[name_list[0]]}"
        f"\n{name_list[1]}: {ligands[name_list[1]]}"
        f"\n{name_list[2]}: {ligands[name_list[2]]}")

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

# # Primary amine input loop
# amine_smiles, amine_pred = get_input("primary amine", amine_df, 'N𝛿–')

# # Display and retrieve information for primary amine
# display_structure("primary amine", amine_smiles, amine_df,)

# # Aryl-bromide input loop
# br_smiles, br_pred = get_input("aryl-bromide", br_df, 'Steric')

# # Display and retrieve information for aryl-bromide
# display_structure("aryl-bromide", br_smiles, br_df)

# PREDICTION
amine_pred = -0.79202
br_pred = 30.5476

input_data = convert_to_normalized_coordinates(amine_pred, br_pred, loaded_x_values, loaded_y_values)
display_prediction_results(confidence_pred)
display_confidence_contour(input_data, confidence_model, x_grid, y_grid)
display_nearest_neighbors_info(P1_info, ligands_df, ligands)
display_nearest_neighbors_info(P2_info, ligands_df, ligands)