# Scripts for Processing Data

These scripts can modify data.root into data files that is readable for human and TridentNet.

Fast test:
```
# compile generate_csv.cpp file
make  

# modify data.root into data/*.csv which is readable by human
./generate_csv

# modify data/*.csv into data/*.pt which can be recognized by TridentNet
python generate_pt.py 
```
Details for these scripts is intorduced below.

# Generate csv
This script load PmtHit and SipmHit from data.root, apply photon efficiency with 0.3, calculate some physics values which you are interested in and save them into the directory ``` ./data```.

For a fast test, you can only modify the "Modify this region to control input and output" region inside ```genData()``` to control io, and then run the script to get your own result. You can check the details of output files with ```pandas.read_csv(filename)``` command inside python.


# Generate pt
```.pt``` is often used as the suffix of a pytorch readable data file. So this script is used to generate ```.pt``` files from ```.csv``` files.

The created files can be read with the following python commands:
```
import torch
data, index = torch.load('data/xyz_0.pt')
print(data.fstNode) # show the location of first hit
```

The created ```xyz*.pt``` files will be further used by TridentNet.

