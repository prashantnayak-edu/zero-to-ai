import os
import subprocess
import shutil
import json
import re

def convert_notebooks_to_html():
    # Create docs/notes directory if it doesn't exist
    os.makedirs('docs/notes', exist_ok=True)
    
    # Convert all .ipynb files in notes to HTML
    for filename in os.listdir('notes'):
        if filename.endswith('.ipynb'):
            input_path = os.path.join('notes', filename)
            output_filename = filename.replace('.ipynb', '.html')
            output_path = os.path.join('docs', 'notes', output_filename)
            
            # Check if file exists and is not empty
            if os.path.exists(input_path) and os.path.getsize(input_path) > 0:
                try:
                    # Proceed with conversion
                    result = subprocess.run([
                        'jupyter', 'nbconvert', 
                        '--to', 'html', 
                        input_path,
                        '--output-dir', os.path.join('docs', 'notes'),
                        '--output', output_filename
                    ], check=True, capture_output=True, text=True)
                    
                    print(f"Converted {filename} to HTML")
                    
                    # Log the output of the subprocess for debugging
                    print(f"Conversion output: {result.stdout}")
                    if result.stderr:
                        print(f"Conversion errors: {result.stderr}")
                        
                except subprocess.CalledProcessError as e:
                    print(f"Error processing {filename}: {str(e)}")
                    create_empty_html(output_path, filename)
            else:
                create_empty_html(output_path, filename)

def create_empty_html(output_path, filename):
    # Create an empty HTML file with a placeholder message
    with open(output_path, 'w') as f:
        f.write(f"<h1>{filename.replace('.ipynb', '')}</h1><p>This section is under development.</p>")
    print(f"Created empty HTML file for {filename}")

def copy_readme():
    # Copy the original README.md to docs/index.md
    try:
        with open('README.md', 'r', encoding='utf-8') as readme_file:
            content = readme_file.read()
        
        # Replace .ipynb extensions with .html
        modified_content = re.sub(r'(\w+)\.ipynb', r'\1.html', content)
        
        with open('docs/index.md', 'w', encoding='utf-8') as index_file:
            index_file.write(modified_content)
        
        print("Copied and modified README.md to docs/index.md")
    except FileNotFoundError:
        print("README.md not found. Skipping index creation.")

if __name__ == "__main__":
    # Create docs directory if it doesn't exist
    os.makedirs('docs', exist_ok=True)
    
    convert_notebooks_to_html()
    copy_readme()
    
    print("Docs build complete!")