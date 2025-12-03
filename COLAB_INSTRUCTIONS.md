
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

# Install requirements
!pip install -r requirements.txt

# Force reinstall packages that might be compiled against incompatible NumPy
!pip install --force-reinstall pandas matplotlib scikit-image

import sys
import os
from pathlib import Path

# Add project root to path
current_path = Path(os.getcwd())
if str(current_path) not in sys.path:
    sys.path.append(str(current_path))
print(f"Added {current_path} to path")
```

4. After setup, you can open `notebooks/01_getting_started.ipynb` in Colab or copy its contents to your Colab notebook.
