# Tutorial_Differential_Evolution
This repository contains the code, data and images used in the [Genetic Algorithm to Optimize Machine Learning Hyper-parameters](https://medium.com/p/genetic-algorithm-to-optimize-machine-learning-hyperparameters-72bd6e2596fc?source=email-1e4964370d8--writer.postDistributed&sk=1492c86ef94ddeefc4d903d4f891ad08) article published in Towards Data Science

---
## Contents
- **generate_data.py**: it generates and plots x<s>1</s>,x<s>2</s>,f(x<s>1</s>,x<s>2</s>) data
- **hyperparams_grid_search.py**: it calculates and plots RMSE for a grid of alpha,gamma values
- **results_grid.dat**: contains the alpha,gamma,RMSE values from hyperparams_grid_search.py
- **hyperparams_diff_evol.py**: it uses differential evolution to converge the alpha and gamma that minimize RMSE
- **hyperparams_grid_and_diff_evol.py**: it then plots hyperparameter grid, as well as configurations visited with differential evolution algorithm and optimized values
- **figures**: folder with all figures used in article

---

## Prerequisites
The necessary packages (with the tested versions with Python 3.8.5) are specified in the file requirements.txt. These packages can be installed with pip:

```
pip3 install -r requirements.txt
```

---

## License and copyright

&copy; Marcos del Cueto Cordones

Licensed under the [MIT License](LICENSE.md).
