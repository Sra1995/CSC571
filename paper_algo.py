"""
CSC571 Data Mining
Project 1
Programmers: Jeray Neely-Speaks, Sajjad Alsaffar, Abigail Garrido
Professor: Dr. Jeonghwa Lee
File Created: 4/8/2024
File Updated: 4/13/2024
"""

import numpy as np                                                                                          # Import numpy for numerical computations
from scipy.linalg import svd, pinv                                                                          # Import SVD and Moore-Penrose pseudoinverse functions from scipy

                                                                                                            # Function to generate 10% missing values
def generate_missing(sequence):
                                                                                                            # Calculate the number of missing values to be generated
    num_missing = round(len(sequence) * 0.1)                                                                # Generate 10% missing values
    
                                                                                                            # Randomly select positions in the sequence
    missing_positions = np.random.choice(len(sequence), num_missing, replace=False)
    
                                                                                                            # Replace the selected positions with None
    for pos in missing_positions:                                                                           # For each missing position selected in the sequence
        sequence[pos] = None                                                                                # Replace the position with None
    
    return sequence                                                                                         # Return the sequence with  x% missing values

                                                                                                            # Dynamic Local Least Squares Imputation Function
def DLLSimpute(G):
                                                                                                            # While there are still missing values in G
    while None in G:
                                                                                                            # Step 1: Sort each row by the number of missing values
        G.sort(key=lambda x: x is None)                                                                     # lmabda function to sort the rows by the number of missing values
        
                                                                                                            # Step 2: Find the first missing position in the first row
        missing_pos = G.index(None)                                                                         # Find the first missing position in the first row
        
                                                                                                            # Step 3: Starting from the missing value position (i,j), the column j is scanned.
                                                                                                            # Increasing i, if the position (i,j) is a missing value, remove the whole row.
        G = [x for i, x in enumerate(G) if i != missing_pos]                                                # Remove the row with the missing value
        
                                                                                    # Step 4: Separate the rest of the matrix into left and right matrices, selecting the largest matrix between these two.
        left_matrix = G[:missing_pos]                                               # Select the left matrix
        right_matrix = G[missing_pos+1:]                                            # Select the right matrix
        if len(left_matrix) > len(right_matrix):                                    # If the left matrix is larger than the right matrix
            G = left_matrix                                                         # Set G to the left matrix
        else:                                                                       # If the right matrix is larger than the left matrix
            G = right_matrix                                                        # Set G to the right matrix
    
                                                                                    # Return the imputed matrix
    return G

                                                                                    # Singular Value Decomposition (SVD) Function
def svd_impute(G):
                                                                                    # Perform SVD
    U, s, VT = svd(G, full_matrices=False)                                          # Perform SVD on the matrix G and return the U, s, and VT matrices
    
                                                                                    # Compute the pseudoinverse of G
    G_pinv = pinv(G)
    
                                                                                    # Return the imputed matrix
    return G_pinv

# Usage
# 'sequence'
sequence = 'ATGCCCCAACTAAATACCGCCGTATGACCCACCATAATTACCCCCATACTCCTGACACTATTTCTCGTCACCCAACTAAAAATATTAAATTCAAATTACCATCTACCCCCCTCACCAAAACCCATAAAAATAAAAAACTACAATAAACCCTGAGAACCAAAATGAACGAAAATCTATTCGCTTCATTCGCTGCCCCCACAATCCTAG'

                                                                                    # Create a dictionary to convert the sequence to numerical data
seq_dict = {'A': 1, 'T': 2, 'G': 3, 'C': 4, None: None}

                                                                                    # Convert the sequence to numerical data
sequence_num = [seq_dict[i] for i in sequence]                                      # for each nucleotide in the sequence, convert it to numerical data
seq_copy = sequence_num                                                             # Copy the sequence_num to seq_copy to be used for comparison
                                                                                    # Generate 10% missing values
sequence_num = generate_missing(sequence_num)

                                                                                    # Perform the DLLSimpute, SVD, computation reduction, and Moore-Penrose pseudoinverse calculations
G_imputed = DLLSimpute(sequence_num)
G_imputed_svd = svd_impute(np.array(G_imputed).reshape(-1,1))


                                                                                    # Compare seq_copy with G_imputed and print %
match_count = sum([1 for i, j in zip(seq_copy, G_imputed) if i == j])               # Count the number of matching values between seq_copy and G_imputed
percentage = (match_count / len(seq_copy)) * 100                                    # Calculate the percentage of matching values between seq_copy and G_imputed
print(f"Matching percentage: {percentage}%")                                        # print the percentage of matching values between seq_copy and G_imputed

                                                                                    # Convert seq_copy to numpy array to perform NRMSE calculation
seq_copy = np.array(seq_copy)

                                                                                    # Ensure seq_copy and G_imputed have the same length
if len(seq_copy) > len(G_imputed):
    seq_copy = seq_copy[:len(G_imputed)]
else:
    G_imputed = G_imputed[:len(seq_copy)]


                                                                                    # calculate NRMSE with out the part to set the diff to 1 if its zero
nrmse_diff = (seq_copy - G_imputed)**2
nrmse = np.sqrt(np.mean(nrmse_diff)) / np.std(seq_copy)
print(f"NRMSE: {nrmse}")

                                                                                    # Calculate NRMSE with the part to handle cases where the difference between original and imputed values is zero
nrmse_diff = (seq_copy - G_imputed)**2

if np.all(nrmse_diff == 0):
    nrmse = 0
else:
    nrmse = np.sqrt(np.mean(nrmse_diff)) / np.std(G_imputed)

print(f"NRMSE: {nrmse:.8f}")
