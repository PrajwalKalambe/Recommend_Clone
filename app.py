
from flask import Flask,request,render_template
from sklearn.metrics.pairwise import cosine_similarity
import os
from google.cloud import vision
from collections import Counter
import pandas as pd
import requests

os.environ['GOOGLE_APPLICATION_CREDENTIALS']='D:\\Python\\GoogleVision_API\\API_key.json'

app= Flask(__name__)


def is_url_image(image_url):
        image_formats = ("image/png", "image/jpeg", "image/jpg")
        response = requests.head(image_url)
        return response.headers["content-type"] in image_formats

def detect_feature(uri):
    Output_list=[]
    client = vision.ImageAnnotatorClient()
    image = vision.Image()
    image.source.image_uri = uri
    objects = client.object_localization(image=image).localized_object_annotations
    obj_list=[]
    for object_ in objects:
        obj_list.append(object_.name)
    obj_list=set(obj_list)
    Output_list.append(list(obj_list))
    
    return Output_list      



output_arr=[]
@app.route('/visionA',methods=['GET','POST'])
def visionA():
    
    output_arr=[]
    projectpath=''
    if(request.method=='POST'):
        projectpath = request.form.get('URL_link')
        if(projectpath==''):
            err_msg='NO URL DETECTED !!! PLEASE ENTEER AGAIN !!'
            return render_template('error404.html',err_msg=err_msg)
        if(is_url_image(projectpath)):
            
            a=detect_feature(projectpath) 
            df = pd.read_csv('D:\\Python\\Recommend_Clone\\final_result.csv')
            URL_link = df['URL']
            new_df=df
            
          
            if not (df['OBJECT'].str.contains((a[0])[0])).any():
                err_msg='NO IMAGE FOUND IN THE DATABASE'
                return render_template('error404.html',err_msg=err_msg)
                
            for i in range(len(URL_link)):
                b=str(df.iloc[i]['OBJECT'])
                a_vals = Counter(str(a))
                b_vals = Counter(b)
            

                # convert to word-vectors
                words  = list(a_vals.keys() | b_vals.keys())
                a_vect = [a_vals.get(word, 0) for word in words]     
                b_vect = [b_vals.get(word, 0) for word in words]        
                sim = cosine_similarity([a_vect], [b_vect])
                new_df.loc[i, "SIM"] = sim[0][0]

            new_df.sort_values(["SIM"],axis=0,ascending=False,inplace=True)
           
            output_arr=[]
            for i in range(5):
                output_arr.append(new_df.iloc[i]['URL'])
            
            
        else:
            err_msg='PLEASE ENTER A VALID URL'
            return render_template('error404.html',err_msg=err_msg)
            
        
    return render_template('index.html',output_arr=output_arr)
    

if __name__ == '__main__':
    app.run()
