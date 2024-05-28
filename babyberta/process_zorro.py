import os

def process_files_in_directory(directory_path):
    file_results = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            total_pairs = len(lines) #// 2
            count = 0
            
            for i in range(0, len(lines), 2):
                first_row_decimal = float(lines[i].strip().split()[-1])
                # second_row_decimal = float(lines[i + 1].strip().split()[-1])
                
                # if first_row_decimal > second_row_decimal:
                #     count += 1

                # count += abs(first_row_decimal - second_row_decimal)

                count += first_row_decimal
            
            fraction = count / total_pairs
            file_results.append(('\'' + filename + '\'', fraction))
    
    # Sort the results based on filenames
    file_results.sort(key=lambda x: x[0])
    
    # Extract sorted filenames and fractions
    filenames_sorted = [result[0] for result in file_results]
    fractions_sorted = [result[1] for result in file_results]
    
    filenames_str = ",".join(filenames_sorted)
    fractions_str = ",".join(map(str, fractions_sorted))
    
    return filenames_str, fractions_str


# Example usage:
for i in range(1):
    directory_path = f"4/run_{i}/babyberta"
    filenames_str, fractions_str = process_files_in_directory(directory_path)
    if i == 0:
        print(filenames_str)
    print(fractions_str)
