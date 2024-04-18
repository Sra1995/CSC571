import time

"""
CSC571 Data Mining
Project 1
Programmers: Jeray Neely-Speaks, Sajjad Alsaffar, Abigail Garrido
Professor: Dr. Jeonghwa Lee
File Created: 4/8/2024
File Updated: 4/13/2024
"""


import numpy as np                                                                                              # Import numpy for numerical computations
from sklearn.metrics import mean_squared_error                                                                  # Import mean_squared_error function

def generate_missing_sequence(sequence, missing_percentage):                                                    # Function to generate missing values in a sequence
    num_missing = int(len(sequence) * missing_percentage)                                                       # Calculate the number of missing values to be generated based on user input
    missing_indices = np.random.choice(len(sequence), num_missing, replace=False)                               # Generate random indices for missing values
    missing_sequence = list(sequence)                                                                           # Convert the sequence to a list
    for idx in missing_indices:                                                                                 # For each missing index
        missing_sequence[idx] = 'N'                                                                             # Replace the value at the index with None
    return ''.join(missing_sequence)                                                                            # Return the sequence with missing values as a string

def KNNimpute_single_sequence(sequence, k):                                                                     # Function to impute missing values using K-Nearest Neighbors
    sequence_array = list(sequence)                                                                             # Convert the sequence to a list
    
    for i, nucleotide in enumerate(sequence_array):                                                             # For each nucleotide in the sequence
        if nucleotide == 'N':                                                                                   # If the value is missing
            local_neighborhood = sequence_array[max(0, i-k):min(len(sequence_array), i+k+1)]                    # Get the local neighborhood of size k centered around the missing value
            local_neighborhood = [x for x in local_neighborhood if x != 'N']                                    # Remove missing values from the local neighborhood
            
            if len(local_neighborhood) == 0:                                                                    # If no data points available
                sequence_array[i] = 'A'                                                                         # Impute with a default nucleotide (e.g., 'A') or whatever
            else:
                                                                                                                # Impute with the most common nucleotide in the local neighborhood
                sequence_array[i] = max(set(local_neighborhood), key=local_neighborhood.count)
    
    return ''.join(sequence_array)                                                                              # Return the imputed sequence as a string 

def LLSimpute_single_sequence(sequence, neighborhood_size):                                                     # Function to impute missing values using Local Least Squares
    sequence_array = list(sequence)                                                                             # Convert the sequence to a list
    
    for i, nucleotide in enumerate(sequence_array):                                                             # For each nucleotide in the sequence
        if nucleotide == 'N':                                                                                   # If the value is missing
            local_neighborhood = sequence_array[max(0, i-neighborhood_size):min(len(sequence_array), i+neighborhood_size+1)]            # Get the local neighborhood of size neighborhood_size centered around the missing value
            local_neighborhood = [x for x in local_neighborhood if x != 'N']                                    # Remove missing values from the local neighborhood
            
            if len(local_neighborhood) == 0:                                                                    # If no data points available
                sequence_array[i] = 'A'                                                                         # Impute with a default nucleotide (e.g., 'A') or whatever
            else:
                                                                                                                # Impute with the most common nucleotide in the local neighborhood
                sequence_array[i] = max(set(local_neighborhood), key=local_neighborhood.count)
    
    return ''.join(sequence_array)                                                                              # Return the imputed sequence as a string

def DLLSimpute_single_sequence(sequence, max_neighborhood_size):                                                # Function to impute missing values using Dynamic Local Least Squares
    sequence_array = list(sequence)                                                                             # Convert the sequence to a list
    
    for i, nucleotide in enumerate(sequence_array):                                                             # For each nucleotide in the sequence 
        if nucleotide == 'N':                                                                                   # If the value is missing
            neighborhood_size = 1                                                                               # Start with a small neighborhood size
            
            while neighborhood_size <= max_neighborhood_size:                                                   # While the neighborhood size is less than or equal to the maximum neighborhood size
                                                                                                                # Expand the local neighborhood
                local_neighborhood = sequence_array[max(0, i-neighborhood_size):min(len(sequence_array), i+neighborhood_size+1)]        # Get the local neighborhood of size neighborhood_size centered around the missing value
                local_neighborhood = [x for x in local_neighborhood if x != 'N']                                # Remove missing values from the local neighborhood
                
                if len(local_neighborhood) == 0:                                                                # If no data points available in the local neighborhood
                    neighborhood_size += 1                                                                      # Increase the neighborhood size
                else:
                                                                                                                # Impute with the most common nucleotide in the local neighborhood
                    sequence_array[i] = max(set(local_neighborhood), key=local_neighborhood.count)
                    break                                                                                       # Exit the loop once the missing value is imputed
            
            if neighborhood_size > max_neighborhood_size:                                                       # If no suitable neighborhood found
                sequence_array[i] = 'A'                                                                         # Impute with a default nucleotide (e.g., 'A')
    
    return ''.join(sequence_array)                                                                              # Return the imputed sequence as a string

def calculate_nrmse(original_sequence, imputed_sequence):                                                       # Function to calculate the Normalized Root Mean Squared Error (NRMSE)
    return np.sqrt(mean_squared_error([ord(x) for x in original_sequence], [ord(x) for x in imputed_sequence])) / (np.max([ord(x) for x in original_sequence]) - np.min([ord(x) for x in original_sequence]))

def print_difference_table(original_sequence, imputed_sequence):                                                # Function to print the difference table between the original and imputed sequences
    print("Position | Original | Imputed | Difference")
    print("-------------------------------------------")
    for i, (original_nucleotide, imputed_nucleotide) in enumerate(zip(original_sequence, imputed_sequence)):    # For each nucleotide in the original and imputed sequences
        if original_nucleotide != imputed_nucleotide:                                                           # If the nucleotides are different
            print(f"{i+1:8} | {original_nucleotide:8} | {imputed_nucleotide:7} | {'Different':10}")             # Print the position, original nucleotide, imputed nucleotide, and 'Different'
        else:                                                                                                   # If the nucleotides are the same
            print(f"{i+1:8} | {original_nucleotide:8} | {imputed_nucleotide:7} | {'Same':10}")                  # Print the position, original nucleotide, imputed nucleotide, and 'Same'

                                                                                                                # Sample DNA sequences selected from chimpanzee.txt
sequences = [
    "ATGCCCCAACTAAATACCGCCGTATGACCCACCATAATTACCCCCATACTCCTGACACTATTTCTCGTCACCCAACTAAAAATATTAAATTCAAATTACCATCTACCCCCCTCACCAAAACCCATAAAAATAAAAAACTACAATAAACCCTGAGAACCAAAATGAACGAAAATCTATTCGCTTCATTCGCTGCCCCCACAATCCTAG",
    "ATGGCCTCGCGCTGGTGGCGGTGGCGACGCGGCTGCTCCTGGAGGCCGGCGGCGCGGAGCTCCGGGCCCGGCTCCCCAGGCCGTGCGGGACCGTCGGGGCCGAGCGCCGCTGCCGACGTCCGCGCGCAGGTTCATAGGCGGAAGGGACTTGACTTGTCTCAGATACCCTATATTAATCTTGTGAAGCATTTAACATCTGCCTGTCCAAATGTATGTCGTATATCACGGTTTCATCACACAACCCCAGACAGTAAAACACACAGTGGTGAAAAATACACCGATCCTTTCAAACTCGGTTGGAGAGACTTGAAAGGTCTGTATGAGGACATTAGAAAGGAACTGCTTATATCAACATCAGAACTTAAGGAAATGTCTGAGTACTACTTTGATGGGAAAGGGAAAGCCTTTCGACCAATTATTGTGGCGCTAATGGCCCGAGCATGCAATATTCATCATAACAACTCCCGACATGTGCAAGCCAGCCAGCGCACCATAGCCTTAATTGCAGAAATGATCCACACTGCTAGTCTGGTTCACGATGACGTTATTGACGATGCAAGTTCTCGAAGAGGAAAACACACAGTTAATAAGATCTGGGGTGAAAAGAAGGCTGTTCTTGCTGGAGATTTAATTCTTTCTGCAGCATCTATAGCTCTGGCACGAATTGGAAATACAACTGTTATATCTATTTTAACCCAAGTTATTGAAGATTTGGTGCGTGGTGAATTTCTTCAGCTCGGGTCAAAAGAAAATGAGAATGAAAGATTTGCACACTACCTTGAGAAGACGTTCAAGAAGACCGCCAGCCTGATAGCCAACAGTTGTAAAGCAGTCTCTGTTCTAGGATGTCCCGACCCAGTGGTGCATGAGATCGCCTATCAGTACGGAAAAAATGTAGGAATAGCTTTTCAGCTTATAGATGATGTATTGGACTTCACCTCATGTTCTGACCAGATGGGCAAACCAACATCAGCTGATCTGAAGCTCGGGTTAGCCACTGGTCCTGTCCTGTTTGCCTGTCAGCAGTTCCCAGAAATGAATGCTATGATCATGCGACGGTTCAGTTTGCCGGGAGATGTAGACAGAGCTCGACAGTATGTATTACAGAGTGATGGTGTGCAACAAACAACCTACCTCGCCCAGCAGTACTGCCATGAAGCAATAAGAGAGATCAGTAAACTTCGACCATCCCCAGAAAGAGATGCCCTCATTCAGCTTTCAGAAATTGTACTCACAAGAGATAAATGA"
]
missing_percentage = 0.2                                                                                        # select missing values

for sequence in sequences:
                                                                                                                # Generate missing sequence
    missing_sequence = generate_missing_sequence(sequence, missing_percentage)

                                                                                                                # KNN imputation
    knn_imputed_sequence = KNNimpute_single_sequence(missing_sequence, k=5)

                                                                                                                # LLSimpute
    lls_imputed_sequence = LLSimpute_single_sequence(missing_sequence, neighborhood_size=5)

                                                                                                                # DLLSimpute
    dll_imputed_sequence = DLLSimpute_single_sequence(missing_sequence, max_neighborhood_size=5)

                                                                                                                # Calculate NRMSE
    knn_nrmse = calculate_nrmse(sequence, knn_imputed_sequence)
    lls_nrmse = calculate_nrmse(sequence, lls_imputed_sequence)
    dll_nrmse = calculate_nrmse(sequence, dll_imputed_sequence)

                                                                                                                # Open output file
    with open('output.txt', 'w') as f:
                                                                                                                # Write NRMSE to file
        f.write(f"\nNRMSE for sequence:\n{sequence}\n")
        f.write(f"KNN Imputation: {knn_nrmse}\n")
        f.write(f"LLSimpute: {lls_nrmse}\n")
        f.write(f"DLLSimpute: {dll_nrmse}\n")

                                                                                                                # Write difference table - KNN to file
        f.write("\nDifference Table for KNN Imputation:\n")
        f.write("Position | Original | Imputed | Difference\n")
        f.write("-------------------------------------------\n")
        diff_count = 0
        same_count = 0
        start_time = time.time_ns()
        for i, (original_nucleotide, imputed_nucleotide) in enumerate(zip(sequence, knn_imputed_sequence)):
            if original_nucleotide != imputed_nucleotide:
                f.write(f"{i+1}\t{original_nucleotide}\t{imputed_nucleotide}\tDifferent\n")
                diff_count += 1
            else:
                f.write(f"{i+1}\t{original_nucleotide}\t{imputed_nucleotide}\tSame\n")
                same_count += 1
        end_time = time.time_ns()
        cpu_time = (end_time - start_time) / 1000
        f.write(f"\nTotal: {len(sequence)}\n")
        f.write(f"Different: {diff_count}\n")
        f.write(f"Same: {same_count}\n")
        f.write(f"Percentage Different: {diff_count / len(sequence) * 100:.2f}%\n")
        f.write(f"Percentage Same: {same_count / len(sequence) * 100:.2f}%\n")
        f.write(f"CPU Time: {cpu_time} microseconds\n")

                                                                                                                # Write difference table - LLS to file
        f.write("\nDifference Table for LLSimpute:\n")
        f.write("Position | Original | Imputed | Difference\n")
        f.write("-------------------------------------------\n")
        diff_count = 0
        same_count = 0
        start_time = time.time_ns()
        for i, (original_nucleotide, imputed_nucleotide) in enumerate(zip(sequence, lls_imputed_sequence)):
            if original_nucleotide != imputed_nucleotide:
                f.write(f"{i+1}\t{original_nucleotide}\t{imputed_nucleotide}\tDifferent\n")
                diff_count += 1
            else:
                f.write(f"{i+1}\t{original_nucleotide}\t{imputed_nucleotide}\tSame\n")
                same_count += 1
        end_time = time.time_ns()
        cpu_time = (end_time - start_time) / 1000
        f.write(f"\nTotal: {len(sequence)}\n")
        f.write(f"Different: {diff_count}\n")
        f.write(f"Same: {same_count}\n")
        f.write(f"Percentage Different: {diff_count / len(sequence) * 100:.2f}%\n")
        f.write(f"Percentage Same: {same_count / len(sequence) * 100:.2f}%\n")
        f.write(f"CPU Time: {cpu_time} microseconds\n")

                                                                                                                # Write difference table - DLLS to file
        f.write("\nDifference Table for DLLSimpute:\n")
        f.write("Position | Original | Imputed | Difference\n")
        f.write("-------------------------------------------\n")
        diff_count = 0
        same_count = 0
        start_time = time.time_ns()
        for i, (original_nucleotide, imputed_nucleotide) in enumerate(zip(sequence, dll_imputed_sequence)):
            if original_nucleotide != imputed_nucleotide:
                f.write(f"{i+1}\t{original_nucleotide}\t{imputed_nucleotide}\tDifferent\n")
                diff_count += 1
            else:
                f.write(f"{i+1}\t{original_nucleotide}\t{imputed_nucleotide}\tSame\n")
                same_count += 1
        end_time = time.time_ns()
        cpu_time = (end_time - start_time) / 1000
        f.write(f"\nTotal: {len(sequence)}\n")
        f.write(f"Different: {diff_count}\n")
        f.write(f"Same: {same_count}\n")
        f.write(f"Percentage Different: {diff_count / len(sequence) * 100:.2f}%\n")
        f.write(f"Percentage Same: {same_count / len(sequence) * 100:.2f}%\n")
        f.write(f"CPU Time: {cpu_time} microseconds\n")
