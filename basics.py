import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress tensorflow warnings https://stackoverflow.com/a/40871012
from deepface import DeepFace
import subprocess
import numpy as np
from decimal import Decimal # for proper rounding
import random
import time
import pandas as pd
from datetime import datetime
import struct
import tensorflow as tf
import accuracy as ac
import quantisations as qt

def run_sfe(x, y, y_0, y_1):
    # write the original 2 vectors to a file (second vector used only for verification)
    with open(f"{EXECUTABLE_PATH}/{INPUT_FILE_NAME}", 'w') as f:
        for x_i, y_i in zip(x, y):
            f.write(f"{x_i} {y_i}\n")
            
    # write the shares into separate files
    with open(f"{EXECUTABLE_PATH}/share0.txt", 'w') as f:
        for i in y_0:
            f.write(f"{i}\n")
    with open(f"{EXECUTABLE_PATH}/share1.txt", 'w') as f:
        for i in y_1:
            f.write(f"{i}\n")
            
    # execute the ABY cos sim computation
    output = subprocess.run(CMD_SCENARIO, shell=True, capture_output=True, text=True, cwd=EXECUTABLE_PATH)
    assert (output.returncode == 0) # make sure the process executed successfully
    
    return output

def get_embedding(imagepath):
    return DeepFace.represent(img_path = imagepath, model_name="SFace", enforce_detection=True)[0]["embedding"]

def get_embedding_facenet(imagepath):
    return DeepFace.represent(img_path = imagepath, model_name="Facenet", enforce_detection=True)[0]["embedding"]

def get_cos_dist_numpy(x, y):
    return 1 - np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))
    
def get_cos_dist_nom(x, y):
    return 1 - np.dot(x, y)
        
def get_two_random_embeddings(same_person):
    """Get two random embeddings of either the same person or two different people out of all the images available"""
    people = [p for p in os.listdir('lfw') if os.path.isdir(os.path.join('lfw', p))] # list of all people that have images
    people_with_multiple_images = [p for p in people if len([img for img in os.listdir(os.path.join("lfw", p)) if img != '.DS_Store']) > 3]  # list of people with more than one image in folder
    embedding1, embedding2 = None, None # face embeddings
    while embedding1 is None or embedding2 is None: # try until the chosen images have detectable faces
        try:
            if same_person:
                # same person should have more than one image (we might still end up choosing the same image of that person with prob 1/n, but that's ok)
                person1 = random.choice(people_with_multiple_images)
                person2 = person1
            else:
                # two persons chosen should be different
                person1 = random.choice(people)
                person2 = random.choice([p for p in people if p != person1])
            # get two random images
            img1 = f"lfw/{person1}/{random.choice(os.listdir(f'lfw/{person1}'))}"
            img2 = f"lfw/{person2}/{random.choice(os.listdir(f'lfw/{person2}'))}"
            # try to extract embeddings from both images
            embedding1 = get_embedding(img1)
            embedding2 = get_embedding(img2)
        except Exception as e:
            # failed to detect faces in images, try again
            # print(e)
            pass
    return img1,img2

def write_two_random_vecs(as_int=False):
    x, y = get_two_random_embeddings(False)
    if as_int:
        x = x.astype(int)
        y = y.astype(int)
    with open(f"{EXECUTABLE_PATH}/{INPUT_FILE_NAME}", 'w') as f:
        for x_i, y_i in zip(x, y):
            f.write(f"{x_i} {y_i}\n")

def get_two_random_images(same_person):
    """Get two random embeddings of either the same person or two different people out of all the images available"""
    people = [p for p in os.listdir('lfw') if os.path.isdir(os.path.join('lfw', p))] # list of all people that have images
    people_with_multiple_images = [p for p in people if len([img for img in os.listdir(os.path.join("lfw", p)) if img != '.DS_Store']) > 3]  # list of people with more than one image in folder
    img1, img2 = None, None # face embeddings
    while img1 is None or img2 is None: # try until the chosen images have detectable faces
        try:
            if same_person:
                # same person should have more than one image (we might still end up choosing the same image of that person with prob 1/n, but that's ok)
                person1 = random.choice(people_with_multiple_images)
                person2 = person1
            else:
                # two persons chosen should be different
                person1 = random.choice(people_with_multiple_images)
                person2 = random.choice([p for p in people_with_multiple_images if p != person1])
            # get two random images
            img1 = f"lfw/{person1}/{random.choice(os.listdir(f'lfw/{person1}'))}"
            img2 = f"lfw/{person2}/{random.choice(os.listdir(f'lfw/{person2}'))}"
        except Exception as e:
            # failed to detect faces in images, try again
            # print(e)
            pass

    return img1,img2


# Below are the functions for euclidean distance we created and adpapted versions of get_two_random_embeddings and get_embedding for facenet.

def euclidean_distance(x, y):
    """
    Compute the euclidean distance between two vectors using numpy
    """
    return np.linalg.norm(np.array(x) - np.array(y))

def get_two_random_embeddings_facenet(same_person):
    """Get two random embeddings of either the same person or two different people out of all the images available"""
    people = [p for p in os.listdir('lfw') if os.path.isdir(os.path.join('lfw', p))] # list of all people that have images
    people_with_multiple_images = [p for p in people if len([img for img in os.listdir(os.path.join("lfw", p)) if img != '.DS_Store']) > 3]  # list of people with more than one image in folder
    img1, img2 = None, None # face embeddings
    embedding1, embedding2 = None, None # face embeddings
    while embedding1 is None or embedding2 is None: # try until the chosen images have detectable faces
        try:
            if same_person:
                # same person should have more than one image (we might still end up choosing the same image of that person with prob 1/n, but that's ok)
                person1 = random.choice(people_with_multiple_images)
                person2 = person1
            else:
                # two persons chosen should be different
                person1 = random.choice(people_with_multiple_images)
                person2 = random.choice([p for p in people_with_multiple_images if p != person1])
            # get two random images
            img1 = f"lfw/{person1}/{random.choice(os.listdir(f'lfw/{person1}'))}"
            img2 = f"lfw/{person2}/{random.choice(os.listdir(f'lfw/{person2}'))}"
            # try to extract embeddings from both images
            embedding1 = get_embedding_facenet(img1)
            embedding2 = get_embedding_facenet(img2)
        except Exception as e:
            # failed to detect faces in images, try again
            # print(e)
            pass
    return img1,img2
