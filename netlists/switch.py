import sys

def replace_q_with_z(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()

    with open(file_name, 'w') as file:
        for line in lines:
            if '.INP' in line:
                modified_line = line.replace('.Q', '.Z')
                file.write(modified_line)
            else:
                file.write(line)

# Example usage
replace_q_with_z(sys.argv[1])
print(f'Replaced ".Q" with ".Z" in lines containing ".INP" in the file {sys.argv[1]}.')
