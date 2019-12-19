# Pytorch trRosetta

This package is a fork of ***trRosetta***, the current state of the art protein structure prediction protocol developed in: [Improved protein structure prediction using predicted inter-residue orientations](https://www.biorxiv.org/content/10.1101/846279v1). 

It was rewritten in Pytorch in order to make the code more extendable and for eventually providing embeddings from the pre-trained models.

The link to the original repository is [here](https://github.com/gjoni/trRosetta)

## Requirements

- pytorch
- numpy
- fire

`> pip install -r requirements`

## Download

### download repository
```
> git clone https://github.com/lucidrains/trRosetta
> cd trRosetta
```

### unzip the compressed model file
```
> tar xvf models.tar.gz
```

## Usage

After unzipping all the current model files, simply run the predict command with the first argument pointing at the amino acid sequence file, the compressed numpy array containing the ensemble predicted inter-residue distances and angles will be saved to the directory of the input file.

```
python predict.py ./T1001.a3m
```


## Links

- [structure modeling scripts](http://yanglab.nankai.edu.cn/trRosetta/download/) (require [PyRosetta](http://www.pyrosetta.org/))
- [***trRosetta*** server](http://yanglab.nankai.edu.cn/trRosetta/)


## References

```
@article {Yang846279,
  author = {Yang, Jianyi and Anishchenko, Ivan and Park, Hahnbeom and Peng, Zhenling and Ovchinnikov, Sergey and Baker, David},
  title = {Improved protein structure prediction using predicted inter-residue orientations},
  elocation-id = {846279},
  year = {2019},
  doi = {10.1101/846279},
  publisher = {Cold Spring Harbor Laboratory},
  URL = {https://www.biorxiv.org/content/early/2019/11/18/846279},
  eprint = {https://www.biorxiv.org/content/early/2019/11/18/846279.full.pdf},
  journal = {bioRxiv}
}
```