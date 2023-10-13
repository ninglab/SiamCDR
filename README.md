# SiamCDR
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Project description...

## Table of Contents:
- [Setup](#setup)
- [Evaluate model](#eval)
- [Use model](#use)
- [Train model](#train)
- [Citing](#cite)
- [Credits and contact information](#contact)
- [Copyright and license notice](#license)

## Setup: <a name="setup"></a>
To being using the model, 
please create a new conda environment with the necessary dependencies.
This can be done by executing the following command:
```bash
conda env create -f environment.yml
conda activate SiamCDR
```

To enable use of this enviornment in the Jupyter notebooks we provide, please execute the following:
```bash
python -m ipykernel install --user --name=siam-cdr
```


## Evaluate model: <a name="eval"></a>
The processed data used to train and evaluate our model is available in `data/processed` directory.
The code used to implement the experiements published in our manuscrpt is available in [`notebooks/experiments`](notebooks/experiments).

## Use model: <a name="use"></a>



## Train model: <a name="train"></a>



## Citing: <a name="cite"></a>
Our manuscript is currently under review at _Nature Communications_.

<!---
If you find this method useful in your research, please cite it using the following BibTex entry:
```
@article{lawrence_ad1_2023,
  title = {},
  journal = {},
  month = {},
  year = {2023},
  doi = {},
  author = {Lawrence, P. and Burns B. and Ning, X.},
}
```
-->

If you use any part of this library in your research, please cite it using the following BibTex entry:
```
@online{codeSiamCDR2023,
  title = {SiamCDR library for enhancing learned drug and cell line representations for cancer drug response via contrastive learning},
  author = {Lawrence, P. and Ning, X.},
  url = {https://github.com/ninglab/SiamCDR},
  year = {2023},
}
```

## Credits and contact information: <a name="contact"></a>
This implementation of `SiamCDR` was written by Patrick J. Lawrence with contributiuons by Xia Ning, PhD.
If you have any questions or encounter a problem, 
please contact <a href='mailto:patrick.skillman-lawrence@osumc.edu'>Patrick J. Lawrence</a>

## Copyright and license notice: <a name="license"></a>
Copyright 2023, The Ohio State University

We gratefully acknowledge support by National Science Foundation grant no. IIS-2133650.

Licensed under the [GNU GENERAL PUBLIC LICENSE, version 3.0](LICENSE). You may not use this library except in compliance with the License.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

