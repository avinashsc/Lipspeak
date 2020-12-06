# LipSpeak
## UC Berkeley - MIDS Capstone Fall 2020
**Mission**: Using latest advances in data science, we strive to improve quality of life for people who lost their ability to speak by helping them communicate effectively

**Team**: Lina Gurevich, Erik Hou, Avinash Chandrasekaran, Daisy Ya

[[Final Project Website]](https://groups.ischool.berkeley.edu/LIPSPEAK/)


## Contents
* [1. Preparation](https://github.com/avinashsc/Lipspeak/#1-preparation)
* [2. Running a Demo](https://github.com/avinashsc/Lipspeak/#2-running-a-demo)
* [3. Description](https://github.com/avinashsc/Lipspeak/#3-description))
* [4. Mobile Setup](https://github.com/avinashsc/Lipspeak/#4-mobile-setup)
* [5. Limitations](https://github.com/avinashsc/Lipspeak/#5-limitations)
* [6. Citation](https://github.com/avinashsc/Lipspeak/#6-citation)


### 1. Preparation

Install python dependencies by creating a new virtual environment and then running 

```
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Running a Demo

In a linux environment, we can verify the end to end flows through the following commands:

* Run a simple demo 
``` bash
git clone https://github.com/avinashsc/Lipspeak.git
bash misc/download_models.py
./download_models.sh
python app.py
```

The above set of commands clones our repository, downloads the required pre-trained models and starts the backend flask server.

* In a different shell, send demo video and sample phrasebook to the server as:
```bash
python
>>> import requests, json, os
>>> data = {'queries': ['call an ambulance', 'difficulty breathing']}
>>> files = {"file": (os.path.basename('./demo.mp4'),
             open('./demo.mp4','rb'),'application/octet-stream'),"phrasebook": (None, json.dumps(data))}
>>> resp = requests.post("http://url-where-server-is-running.com:5000/predict",files=files) 
```

The above set of commands, pass a demo video and user defined phrasebook to the server that has been setup earlier.

* Expected output: Prediction is "difficulty breathing"

The demo video corresponds to mouthing the words "difficulty breathing". The model predicts this correctly. If tried out in our app, the app would
sound "I have difficulty breathing" phrase

### 3. Description
* The models used in our project have been trained and evaluated on [LRW and LRS datasets](http://www.robots.ox.ac.uk/~vgg/data/lip_reading/). The pre-trained deep lip reading model can be located at ```models/lrs2_lip_model``` and the keyword-spotting model can be located at ```misc/pretrained_models```
* When mouthing a video through the app, the appropriate features required by the lip reading model are pre-computed and saved in ```data/lipspeak```.  ```config.py``` specifies the necessary configuration setup for the lip reading model.
* KWSNet keyword spotting model configuration is available in ```configs/demo/eval.json```
* ```app.py``` is the python script that initializes all models & starts the flask server for backend inference task. It exposes a REST API to the mobile app, and expects the video and a user defined phrasebook as inputs. When inputs are obtained, we first extract the visual features from the video, run the keyword spotting model and compute probabilities to identify what the mouthed phrase was. This is then reported back to the app

### 4. Mobile Setup
For details regarding our mobile setup, please refer to [LipSpeak App Project](https://github.com/gurlina/LipSpeakApp)

### 5. Limitations
We would like to emphasise that this research represents a working progress towards, and as such, has a few limitations that we are aware of.

* Homophemes - for example, the words "may", "pay", "bay" cannot be distinguished without audio as the visemes "m", "p", "b" visually look the same.
* Accents, speed of speech and mumbling which modify lip movements.
* Variable imaging conditions such as lighting, motion and resolution which modiy the appearance of the lips.
* Shorter keywords which are harder to visually spot.

### 6. Citation
If you use this code, please cite the following:
```
@misc{momeni2020seeing,
    title={Seeing wake words: Audio-visual Keyword Spotting},
    author={Liliane Momeni and Triantafyllos Afouras and Themos Stafylakis
            and Samuel Albanie and Andrew Zisserman},
    year={2020},
    eprint={2009.01225},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
