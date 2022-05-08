# Web app demo for Pytorch Retinaface

This app uses the pytorch retinaface library to detect faces from either a video or an image. 
I forked [ternaus retinaface_demo](https://github.com/ternaus/retinaface_demo) repo and added the video functionality to the web application.

## Install dependencies
```
pip install -r requirements.txt
```
## Run App from terminal
1. clone this repo 
```
git clone https://github.com/phemiloyeAI/face-detection.git
```
2. change the directory to the path containing the streamlit app
```
cd ./app/app.py
```
3.  Run app.py script
```
streamlit run app.py
```
## Deploy App 
If you wish to deploy this app yourself, here is how to:

* [sign up](https://share.streamlit.io/signup) to open a streamlit cloud account.
* Connect your streamlit account to your github account containing this repo you cloned.
* On your streamlit cloud account, create an app. You can call it any name.
* Click on "Paste GitHub URL".
* Paste the link to your github repo in the bar.
* And finally click on "Deploy", right below the bar.


**The code for the network:** https://github.com/ternaus/retinaface \

**The Web app:** [https://share.streamlit.io/phemiloyeai/face-detection/app/app.py](https://share.streamlit.io/phemiloyeai/face-detection/app/app.py)
