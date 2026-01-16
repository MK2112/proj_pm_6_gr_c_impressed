import os

def search_string_in_files(search_string, directory='.'):
    found_files = []
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    if search_string in f.read():
                        found_files.append(filepath)
            except (UnicodeDecodeError, PermissionError, IsADirectoryError, OSError):
                # Skip files that can't be read as text (e.g. binary files, permissions)
                pass
    
    return found_files

# Example usage
if __name__ == '__main__':
    search_term = input("Enter the string to search for: ")
    results = search_string_in_files(search_term)
    
    if results:
        print(f"\nFound in {len(results)} file(s):")
        for file in results:
            print(file)
    else:
        print(f"\nString '{search_term}' not found in any file.")
