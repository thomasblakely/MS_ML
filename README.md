Hi guys

1. After cloning the project, please use setup.sh to create the right folder. If the csv file is in the current directory, it will copy into the data folder. And this shell file will create the right structure for you.    
```bash
chmod +x setup.sh
./setup.sh
```
2. Run the basic_analysis and advanced, it should create some files in the output and data folder.
```bash
python src/basic_analysis.py
python src/advanced.py

or

python3 src/basic_analysis.py
python3 src/advanced.py
```
3. The gragh.py will generate the graph of the dataset, the first function will create the a linear graph based on all the patient. The second sunction will generate one patient graph according to your choice.
```bash
python src2/gragh.py

or

python3 src2/gragh.py
```

4. I updated all the unsupersived learning things under the /unsupersived folder.
