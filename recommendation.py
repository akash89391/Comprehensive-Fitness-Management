from .models import Profile
from django.shortcuts import render, redirect
from django.contrib import messages

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Load the dataset once to avoid reloading it for each request
df = pd.read_csv(r'D:\\final year project\\food recapp\\minor\\website\\dataset.csv')
df.reset_index(drop=True, inplace=True)  # Reset index to align with model indices

def Recommend(request):
    if request.user.is_authenticated:
        class Recommender:
            def __init__(self):
                # Load the dataset and reset the index to align with model indices
                self.df = pd.read_csv(r'D:\\final year project\\food recapp\\minor\\website\\dataset.csv')
                self.df.reset_index(drop=True, inplace=True)
            
            def get_features(self):
                # Create dummies for categorical features
                nutrient_dummies = self.df['Nutrient'].str.get_dummies()
                disease_dummies = self.df['Disease'].str.get_dummies(sep=' ')
                diet_dummies = self.df['Diet'].str.get_dummies(sep=' ')
                
                # Combine all features into one dataframe
                feature_df = pd.concat([nutrient_dummies, disease_dummies, diet_dummies], axis=1)
                return feature_df
            
            def k_neighbor(self, inputs):
                feature_df = self.get_features()
                
                # Initialize NearestNeighbors model with 40 neighbors
                model = NearestNeighbors(n_neighbors=40, algorithm='ball_tree')
                model.fit(feature_df)
                
                # Create an empty dataframe to store the results
                df_results = pd.DataFrame(columns=list(self.df.columns))
                
                # Get distances and indices for k nearest neighbors
                distances, indices = model.kneighbors(inputs)
                
                for i in list(indices[0]):  # indices[0] because kneighbors returns a 2D array
                    if i < len(self.df):
                        # Only add to results if the index is valid
                        df_results = pd.concat([df_results, self.df.loc[[i]]])
                    else:
                        print(f"Index {i} out of bounds.")

                # Filter and clean the result dataframe
                df_results = df_results.filter(
                    ['Meal_Id', 'Name', 'catagory', 'Nutrient', 'Veg_Non', 'Price', 'Review', 'Diet', 'Disease', 'description']
                )
                df_results = df_results.drop_duplicates(subset=['Name'])  # Remove duplicates
                df_results = df_results.reset_index(drop=True)  # Reset the index
                return df_results
        
        # Instantiate the Recommender class
        ob = Recommender()
        data = ob.get_features()
        
        # Prepare the input vector for recommendation
        total_features = data.columns
        d = {feature: 0 for feature in total_features}
        
        # Get user profile from the database
        p = Profile.objects.get(number=request.user.username)
        diet = list(p.diet.split('++'))
        disease = list(p.disease.split('++'))
        nutrient = list(p.nutrient.split('++'))
        
        Recommend_input = diet + disease + nutrient
        image = p.image.url
        
        # Set the corresponding feature values to 1 based on user profile
        for feature in Recommend_input:
            if feature in d:
                d[feature] = 1
        
        # Convert the feature dictionary into a list of values
        final_input = list(d.values())
        
        # Pass the input to the model and get recommendations
        results = ob.k_neighbor([final_input])
        
        # Convert the results into a dictionary for easy access
        data = dict(results)
        
        ids = list(data['Meal_Id'])
        names = list(data['Name'])
        category = list(data['catagory'])
        veg_non = list(data['Veg_Non'])
        review = list(data['Review'])
        nutrient = list(data['Nutrient'])  
        price = list(data['Price'])
        description = list(data['description'])
        
        # Zip the data to pass it to the template
        data = zip(names, ids, names, category, category, veg_non, review, nutrient, price, description)
        
        return render(request, "website/recommend.html", {'data': data, 'image': image})
    
    else:
        messages.error(request, 'You must be logged in for meal recommendations.')
        return redirect('Home')






# from .models import Profile
# from django.shortcuts import render,redirect
# from django.contrib import messages

# import pandas as pd
# import numpy as np
# import sklearn
# from sklearn.neighbors import NearestNeighbors

# df = pd.read_csv(r'D:\\final year project\\food recapp\\minor\\website\\dataset.csv')

# def Recommend(request):
    
#     if request.user.is_authenticated:
#         class Recommender:
            
#             def __init__(self):
#                 self.df = pd.read_csv(r'D:\\final year project\\food recapp\\minor\\website\\dataset.csv')
            
#             def get_features(self):
#                 #getting dummies of dataset
#                 nutrient_dummies = self.df.Nutrient.str.get_dummies()
#                 disease_dummies = self.df.Disease.str.get_dummies(sep=' ')
#                 diet_dummies = self.df.Diet.str.get_dummies(sep=' ')
#                 feature_df = pd.concat([nutrient_dummies,disease_dummies,diet_dummies],axis=1)
             
#                 return feature_df
            
#             def k_neighbor(self,inputs):
                
#                 feature_df = self.get_features()
               
#                 #initializing model with k=20 neighbors
#                 model = NearestNeighbors(n_neighbors=40,algorithm='ball_tree')
                
#                 # fitting model with dataset features
#                 model.fit(feature_df)
                
#                 df_results = pd.DataFrame(columns=list(self.df.columns))
                
                
#                 # getting distance and indices for k nearest neighbor
#                 distances , indices = model.kneighbors(inputs)
                
                
#                 for i in list(indices):
#                     #df_results = df_results.concat(self.df.loc[i])
#                     df_results = pd.concat([df_results, self.df.loc[[i]]])
                
#                 df_results = df_results.filter(['Meal_Id','Name','catagory','Nutrient','Veg_Non','Price','Review','Diet','Disease','description'])
#                 df_results = df_results.drop_duplicates(subset=['Name'])
#                 df_results = df_results.reset_index(drop=True)
#                 return df_results
        
#         ob = Recommender()
#         data = ob.get_features()
        
#         total_features = data.columns
#         d = dict()
#         for i in total_features:
#             d[i]= 0
       
        
#         p=Profile.objects.get(number=request.user.username) #extract values from database where Table name is Profie
#         diet=list(p.diet.split('++'))
#         disease=list(p.disease.split('++'))
#         nutrient=list(p.nutrient.split('++'))
        
#         Recommend_input=diet+disease+nutrient
        
#         image=p.image.url
       
        
#         for i in Recommend_input: 
#             d[i] = 1
#         final_input = list(d.values())
        
#         results = ob.k_neighbor([final_input]) # pass 2d array []
       
#         data=dict(results)
        
#         ids=list(data['Meal_Id'])
#         n=list(data['Name'])
#         c=list(data['catagory'])
#         vn=list(data['Veg_Non'])
#         r=list(data['Review'])
#         nt=list(data['Nutrient'])  
#         p=list(data['Price'])
#         d=list(data['description'])
#         i=range(len(n))   
#         sc=c
        
#         data=zip(n,ids,n,c,sc,vn,r,nt,p,d)
        
#         return render(request,"website/recommend.html",{'data':data,'image':image})
    
#     else:
#         messages.error(request,'You must be logged in for meal recommendations..')
#         return redirect('Home')
        