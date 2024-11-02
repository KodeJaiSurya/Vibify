import pandas as pd



def load_data():
    '''Load data from GCP'''
    path ='data/fer2013.csv'
    data = pd.readcsv(path)
    return data

def filter_emotions(data, selected_emotions=[0,3,4,6]):
    """Filter the dataset to include only the selected emotions"""
    filtered_data = data[data['emotion'].isin(selected_emotions)]
    return filtered_data

def map_emotions(data, emotion_map={0:0,3:1,4:2,6:3}):
    """Map the original emotions to a new set of labels"""
    data['emotions']=data['emotions'].map(emotion_map)
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

def preprocess_data(data):
    data = filter_emotions(data)
    data = map_emotions(data)
    X = preprocess_pixels(data)
    y = preprocess_labels(data)

    final_data = pd.concat([X,y] , axis=1)
    final_data.to_csv('data/processed_data.csv')
    print("Processing completed!")

    return X,y



    
    
