# LipSpeak
### UC Berkeley - MIDS Capstone Fall 2020
#### Using latest advances in data science, we strive to improve quality of life for people who lost their ability to speak by helping them communicate effectively

**Team**: Lina Gurevich, Erik Hou, Avinash Chandrasekaran, Daisy Ya

[[Final Project Website]](https://groups.ischool.berkeley.edu/LIPSPEAK/)


## Contents
* [1. Preparation](https://github.com/avinashsc/Lipspeak/#1-preparation)
* [2. Running a Demo](https://github.com/avinashsc/Lipspeak/#2-demo)
* [3. Mobile Setup](https://github.com/avinashsc/Lipspeak/#2-mobile)
* [4. Citation](https://github.com/avinashsc/Lipspeak/#4-citation)


### 1. Preparation

Install python dependencies by creating a new virtual environment and then running 

```
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Running a Demo

To verify that everything works:

* Run a simple demo 
``` bash
bash misc/download_models.py
./download_models.sh
python app.py
```

* In a different shell, send demo video and sample phrasebook to the server
```bash
python
>>> import requests, json, os
>>> data = {'queries': ['call an ambulance', 'difficulty breathing']}
>>> files = {"file": (os.path.basename('./demo.mp4'),
             open('./demo.mp4','rb'),'application/octet-stream'),"phrasebook": (None, json.dumps(data))}
>>> resp = requests.post("http://url-where-server-is-running.com:5000/predict",files=files) 
```

* Expected output...

### 3. Mobile Setup
For details regarding our mobile setup, please refer to (https://github.com/gurlina/LipSpeakApp)

### 4. Citation
If you use this code, please cite the following:
```
@misc{momeni2020seeing,
    title={Seeing wake words: Audio-visual Keyword Spotting},
    author={Liliane Momeni and Triantafyllos Afouras and Themos Stafylakis and Samuel Albanie and Andrew Zisserman},
    year={2020},
    eprint={2009.01225},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
