import pickle
import re
from pathlib import Path
import string

__version__ =  '0.1.0'

BASE_DIR = Path(__file__).resolve(strict=True).parent

model = pickle.load(open(f"{BASE_DIR}/trained_pipeline-{__version__}.pkl", "rb"))

classes = ['Arabic', 'Danish', 'Dutch', 'English', 'French', 'German',
       'Greek', 'Hindi', 'Italian', 'Kannada', 'Malayalam', 'Portugeese',
       'Russian', 'Spanish', 'Sweedish', 'Tamil', 'Turkish']


def predict_pipeline(text):
    
    text = str(text).lower()
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub(r'\[\d+\]', '', text)
    
    pred = model.predict([text])

    return classes[pred[0]]