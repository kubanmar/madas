# similarity-data-framework

A framework to compare different similarity measures of materials. <br>
Features: <br>
  * Automatic download of materials from the NOMAD Encyclopedia
  * ASE Database to locally store materials data
  * Fingerprints:
    * DOS --> Calculated from DFT DOS values
    * SYM --> All symmetry operations of a material
    * SOAP --> A SOAP implementation using DScribe (https://singroup.github.io/dscribe/)
    * IAD --> InterAtomic Distances

More information can be found in the Wiki: https://git.physik.hu-berlin.de/kuban/similarity-data-framework/wikis/home <br>

For installation:
  * clone the repository:
    * ``git clone git@git.physik.hu-berlin.de:kuban/similarity-data-framework.git``
  * install it locally with pip:
    * ``cd similarity_data_framework``
    * ``pip install -e .``
  * you may import the module in python with:
    * ``import simdatframe``

