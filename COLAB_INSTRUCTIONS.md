
## Running on Google Colab

To run this project on Google Colab for faster GPU execution:

1. Open [Google Colab](https://colab.research.google.com/).
2. Create a new notebook.
3. In the first cell, run the following code to clone the repository and install dependencies:

```python
import os
repo_name = "Isolating-Race-Vectors-in-Latent-Space"

if os.path.isdir(repo_name):
    %cd {repo_name}
    !git pull
else:
    !git clone https://github.com/Arnavsharma2/{repo_name}.git
    %cd {repo_name}

# Uninstall conflicting Colab packages
!pip uninstall -y opencv-python-headless

# Install requirements (now pins numpy<2)
!pip install -r requirements.txt

# Force reinstall packages to ensure they match numpy<2
# We explicitly include numpy<2 here to prevent pip from upgrading it
!pip install --force-reinstall "pandas" "matplotlib" "scikit-image" "numpy<2"

import sys
import os
from pathlib import Path

# Add project root to path
current_path = Path(os.getcwd())
if str(current_path) not in sys.path:
    sys.path.append(str(current_path))
print(f"Added {current_path} to path")
```

4   git push origin main
   ```

5. After setup, you can open `notebooks/01_getting_started.ipynb` in Colab or copy its contents to your Colab notebook.

## Option 2: Run with Python 3.11 (Recommended if dependencies fail)

If you are still seeing "binary incompatibility" or version errors, the most robust fix is to use Python 3.11 (which has better support for our specific library versions).

Run this in a cell **before** doing anything else:

```python
# 1. Install Python 3.11
!sudo apt-get update -y
!sudo apt-get install python3.11 python3.11-distutils python3.11-dev
!curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
!python3.11 get-pip.py

# 2. Install essential packages for Colab
!python3.11 -m pip install ipykernel

# 3. Register the kernel
!python3.11 -m ipykernel install --user --name=python3.11

print("Python 3.11 installed! Now go to 'Runtime' > 'Change runtime type' and select 'python3.11' from the dropdown (if available) or simply continue using the commands below which use python3.11 explicitly.")
```

**Note:** Changing the actual Colab runtime kernel to custom Python versions can be tricky in the UI. A simpler way is to just run your scripts using `!python3.11 script.py`.

However, for a Notebook, you really want the kernel. If the dropdown doesn't show up, you might need to stick to the "Option 1" fixes I provided (pinning numpy).
```
