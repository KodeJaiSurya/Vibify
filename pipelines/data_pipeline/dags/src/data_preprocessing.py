import pandas as pd
import numpy as np
import gdown

def download_emotion_data(file_id):
    url = f"https://drive.google.com/uc?id={file_id}"
    file_path = "dags/data/raw/dataset_fer2013.csv"
    gdown.download(url, file_path, quiet=False)
    print("FER2013 dataset downloaded.")
    return file_path

def load_emotion_data(file_path):
    df = pd.read_csv(file_path)
    print("FER2013 dataset loaded.")
    return df

def filter_emotions(data, selected_emotions=[0,3,4,6]):
    """Filter the dataset to include only the selected emotions"""
    filtered_data = data[data['emotion'].isin(selected_emotions)]
    return filtered_data

def map_emotions(data, emotion_map={0:0,3:1,4:2,6:3}):
    """Map the original emotions to a new set of labels"""
    data['emotion']=data['emotion'].map(emotion_map)
    return data

def preprocess_pixels(data, width=48,height=48):
    """Convert the pixel strings to numpy arrays and normalize the pixel values"""
    pixels = data['pixels'].tolist()
    X=[]
 
    #convert each pixel sequence to an array
    for pixel_sequence in pixels:
        pixel_array = np.array(pixel_sequence.split(' '),dtype=float).reshape(width,height,1)
        X.append(pixel_array)
 
    #convert to a numpy array and normalize
    X=np.array(X)
    X=X/255.0
    return X
 
def preprocess_labels(data):
    """Extarct emotio labels from the data"""
    return np.array(data['emotion'].tolist())
 
def aggregate_filtered_data(X,y):
    try:
        np.save("X.npy",X)
        np.save("y.npy",y)
        print("X and Y saved!")
        return True
    except Exception as e:
        print(f"Error : {e}")