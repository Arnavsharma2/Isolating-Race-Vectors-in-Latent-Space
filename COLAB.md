
## Running on Google Colab

To run this project on Google Colab for faster GPU execution:

1. Open [Google Colab](https://colab.research.google.com/).
2. Create a new notebook.
3. In the first cell, run the following code to install Python 3.11 and set up the environment:

```python
# 1. Install Python 3.11
!sudo apt-get update -y
!sudo apt-get install python3.11 python3.11-distutils python3.11-dev
!curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
!python3.11 get-pip.py

# 2. Install packages for Colab
!python3.11 -m pip install ipykernel

# 3. Register kernel
!python3.11 -m ipykernel install --user --name=python3.11

print("Python 3.11 installed!")
```

4. In the second cell, run the following code to clone the repository and install dependencies:

```python
import os
repo_name = "Isolating-Race-Vectors-in-Latent-Space"

if os.path.isdir(repo_name):
    %cd {repo_name}
    !git pull
else:
    !git clone https://github.com/Arnavsharma2/{repo_name}.git
    %cd {repo_name}

# Install requirements 
!pip install -r requirements.txt

import sys
import os
from pathlib import Path

# Add project root to path
current_path = Path(os.getcwd())
if str(current_path) not in sys.path:
    sys.path.append(str(current_path))
print(f"Added {current_path} to path")
```

5. After setup, you can open `notebooks/01_getting_started.ipynb` in Colab or copy its contents to your Colab notebook.
