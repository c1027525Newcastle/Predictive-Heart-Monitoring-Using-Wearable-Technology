import os

def correct_json(input_filename, output_filename):
    print(f"Correcting JSON file: {input_filename} to {output_filename}")

    # Create a temporary file to write the corrected JSON
    temp_filename = output_filename + ".tmp"

    with open(input_filename, 'r') as infile, open(temp_filename, 'w') as outfile:
        first_line = infile.readline().strip()

        # Handle the first line and check for missing opening bracket
        if not first_line.startswith('['):
            outfile.write('[\n')  # Write the first line after adding the opening bracket
        
        # Write the first line and handle its comma
        if first_line.endswith('}') and not first_line.endswith('},'):
            outfile.write(first_line + ',\n')  # Add a comma if it's not the last item
        else:
            outfile.write(first_line + '\n')

        # Read and process the rest of the file line by line
        previous_line = ''
        for line in infile:
            line = line.strip()

            # Write the previous line (except the first one)
            if previous_line:
                outfile.write(previous_line + '\n')

            # Prepare the current line as the next previous_line
            if line.endswith('}') and not line.endswith('},'):
                previous_line = line + ','  # Add a comma
            else:
                previous_line = line

        # Write the last processed line without adding a comma
        if previous_line.endswith(','):
            previous_line = previous_line[:-1]  # Remove the trailing comma
        outfile.write(previous_line + '\n')

        # Handle the last line and check for missing closing bracket
        if not previous_line.endswith(']'):
            outfile.write(']\n')

    os.rename(temp_filename, output_filename)
    print(f"JSON file has been corrected and saved as {output_filename}")

# Usage
# input_filename = ''
# output_filename = ''
# correct_json(input_filename, output_filename)
