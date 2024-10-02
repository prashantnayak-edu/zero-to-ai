import os
import subprocess
import shutil
import json

def convert_notebooks_to_markdown():
    # Create book/notes directory if it doesn't exist
    os.makedirs('book/notes', exist_ok=True)
    
    # Convert all .ipynb files in notes to markdown
    for filename in os.listdir('notes'):
        if filename.endswith('.ipynb'):
            input_path = os.path.join('notes', filename)
            output_filename = filename.replace('.ipynb', '.md')
            output_path = os.path.join('book', 'notes', output_filename)
            
            # Check if file exists and is not empty
            if os.path.exists(input_path) and os.path.getsize(input_path) > 0:
                try:
                    # Check if the file is a valid JSON
                    with open(input_path, 'r', encoding='utf-8') as f:
                        json.load(f)
                    
                    # If JSON is valid, proceed with conversion
                    result = subprocess.run([
                        'jupyter', 'nbconvert', 
                        '--to', 'markdown', 
                        input_path,
                        '--output-dir', os.path.join('book', 'notes'),
                        '--output', output_filename
                    ], check=True, capture_output=True, text=True)
                    
                    print(f"Converted {filename} to markdown")
                    
                    # Log the output of the subprocess for debugging
                    print(f"Conversion output: {result.stdout}")
                    if result.stderr:
                        print(f"Conversion errors: {result.stderr}")
                        
                except (json.JSONDecodeError, subprocess.CalledProcessError) as e:
                    print(f"Error processing {filename}: {str(e)}")
                    create_empty_markdown(output_path, filename)
            else:
                create_empty_markdown(output_path, filename)

def create_empty_markdown(output_path, filename):
    # Create an empty markdown file with a placeholder message
    with open(output_path, 'w') as f:
        f.write(f"# {filename.replace('.ipynb', '')}\n\nThis section is under development.")
    print(f"Created empty markdown file for {filename}")

def copy_readme():
    # Copy the original README.md to book/index.md
    try:
        shutil.copy('README.md', 'book/index.md')
        print("Copied README.md to book/index.md")
    except FileNotFoundError:
        print("README.md not found. Skipping index creation.")

if __name__ == "__main__":
    # Create book directory if it doesn't exist
    os.makedirs('book', exist_ok=True)
    
    convert_notebooks_to_markdown()
    copy_readme()
    
    print("Book build complete!")