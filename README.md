## Running:

- use Python version 3.12
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python continuous.py
```

- After that simply speak in front of you mac and it should transcribe with fairly high accuracy

Current known limitations:
- I am only chunking the audio every 5 seconds without overlap
    - this means that some meaning can be lost if the last word is spread across 2 chunks

Ideas for future iterations:
- adding in an overlap feature so the model considers multiple different chunks for 1 audio segment