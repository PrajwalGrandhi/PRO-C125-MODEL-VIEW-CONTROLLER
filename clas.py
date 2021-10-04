import cv2
import numpy as np
from numpy.random.mtrand import multinomial
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd
from sklearn.metrics import accuracy_score


X=np.load('image.npz')['arr_0']
y=pd.read_csv('labels.csv')["labels"]
print(pd.Series(y).value_counts())
classes=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
nclasses=len(classes)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=500,train_size=4000,random_state=9)
X_train_scaled=X_train/255.0
X_test_scaled=X_test/255.0

clas=LogisticRegression(solver='saga',multi_class='multinomial').fit(X_train_scaled,y_train)

y_pred=clas.predict(X_test_scaled)
accur=accuracy_score(y_test,y_pred)
print("The accuracy is: ",accur)

def imagePredict(image):
    im_pil=Image.open(image)
    image_bw=im_pil.convert('L')
    image_bw_resized=image_bw.resize((22,30),Image.ANTIALIAS)

    px_filter=20

    min_px=np.percentile(image_bw_resized,px_filter)

    image_bw_resized_inverted_scaled=np.clip(image_bw_resized-min_px,0,255)

    max_px=np.max(image_bw_resized)

    image_bw_resized_inverted_scaled=np.asarray(image_bw_resized_inverted_scaled)/max_px

    test_sample=np.array(image_bw_resized_inverted_scaled).reshape(1,660)
    test_pred=clas.predict(test_sample)

    return test_pred[0]

