import os
import subprocess
import shutil
import json
import re

def convert_notebooks_to_markdown():
    # Create docs/notes directory if it doesn't exist
    os.makedirs('docs/notes', exist_ok=True)
    
    # Convert all .ipynb files in notes to markdown
    for filename in os.listdir('notes'):
        if filename.endswith('.ipynb'):
            input_path = os.path.join('notes', filename)
            output_filename = filename.replace('.ipynb', '.md')
            output_path = os.path.join('docs', 'notes', output_filename)
            
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
                        '--output-dir', os.path.join('docs', 'notes'),
                        '--output', output_filename
                    ], check=True, capture_output=True, text=True)
                    
                    print(f"Converted {filename} to markdown")
                    
                    # Add MathJax support to the converted markdown file
                    add_mathjax_support(output_path)
                    
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
    # Copy the original README.md to docs/index.md
    try:
        with open('README.md', 'r', encoding='utf-8') as readme_file:
            content = readme_file.read()
        
        # Replace .ipynb extensions with .md
        modified_content = re.sub(r'(\w+)\.ipynb', r'\1.md', content)
        
        with open('docs/index.md', 'w', encoding='utf-8') as index_file:
            index_file.write(modified_content)
        
        print("Copied and modified README.md to docs/index.md")
    except FileNotFoundError:
        print("README.md not found. Skipping index creation.")

def add_mathjax_support(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    mathjax_script = """
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML" async></script>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
        inlineMath: [['$','$'], ['\\(','\\)']],
        processEscapes: true
    }
});
</script>
"""
    
    modified_content = mathjax_script + content
    
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(modified_content)

if __name__ == "__main__":
    # Create docs directory if it doesn't exist
    os.makedirs('docs', exist_ok=True)
    
    convert_notebooks_to_markdown()
    copy_readme()
    
    # Add MathJax support to index.md as well
    add_mathjax_support('docs/index.md')
    
    print("Docs build complete!")