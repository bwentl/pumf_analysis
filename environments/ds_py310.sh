conda create -y -n ds_py310 -c conda-forge --override-channels python=3.10 "numpy>=1.16.1" "pandas>=1.1.0" "geopandas>=0.10.2" "psutil>=4.1" "pyarrow>=2.0" "numba>0.51.2" "pyyaml>=5.1" "requests>=2.7" pytest pytest-cov coveralls pycodestyle pytest-regressions jupyter jupyterlab matplotlib descartes pandasql scipy seaborn pyodbc sqlalchemy openpyxl xlrd xlsxwriter sympy nose scikit-learn scikit-learn-intelex autopep8 pip ipykernel pyreadstat
conda activate ds_py310
conda env export -n ds_py310 -f environments/ds_py310_win10.yml --no-builds
conda deactivate
# re create the same environment
conda env create -n ds_py310 -f environments/ds_py310_win10.yml
