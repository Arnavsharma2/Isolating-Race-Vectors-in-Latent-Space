
## Running on Google Colab

To run this project on Google Colab for faster GPU execution:

1. Open [Google Colab](https://colab.research.google.com/).
2. Create a new notebook.
3. In the first cell, run the following code to clone the repository and install dependencies:

```python
!git clone https://github.com/Arnavsharma2/Isolating-Race-Vectors-in-Latent-Space.git
%cd Isolating-Race-Vectors-in-Latent-Space
!pip install -r requirements.txt

import sys
sys.path.append('.')
```

4. **Important**: Since you have local changes (the `src` folder), you must push them to GitHub first!
   ```bash
   git add src requirements.txt
   git commit -m "Add source code and requirements"
   git push origin main
   ```

5. After setup, you can open `notebooks/01_getting_started.ipynb` in Colab or copy its contents to your Colab notebook.
