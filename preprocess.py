import numpy as np
import joblib

dictionary=joblib.load(r'data/dictionary.pkl')
fets=['facility_type','state_factor','building_class']



def __boBinary(x,y):
    addTo=y-len(bin(x)[2:])
    return '0'*addTo+bin(x)[2:]

def process(values):
    temp=[]
    for fet in fets:
        if fet=='facility_type':
            bnNo=__boBinary(dictionary[fet][values[0]],6)
            for i in bnNo:
                temp.append(np.int8(i))
        elif fet=='state_factor':
            bnNo=__boBinary(dictionary[fet][values[1]],3)
            for i in bnNo:
                temp.append(np.int8(i))
        elif fet=='building_class':
            bnNo=__boBinary(dictionary[fet][values[2]],2)
            for i in bnNo:
                temp.append(np.int8(i))
        
    return temp

def get_prediction(data,model):
    pred=model.predict(data)
    return pred

