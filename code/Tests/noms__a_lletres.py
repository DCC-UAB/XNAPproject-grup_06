# -*- coding: utf-8 -*-
"""
Created on Wed May 29 16:08:39 2024

@author: alexg
"""

import torch

import cv2
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import pandas as pd


def segment_letters(image_dir, output_dir,csv,csv_out):
    # Crear el directorio de salida si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)   #crea un directori per guardar les imatges
    df = pd.read_csv(csv)  #Carrega el csv
    df.dropna(inplace=True)
    
    list_letters = [] #Crea una llista per guardar les lletres
    num_img = -1
    for element in os.listdir(wdir): #Recorre tots els noms
        image_path = str(wdir+'/'+element)  
        # Leer la imagen y convertirla a escala de grises
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Aplicar umbral para binarizar la imagen
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Encontrar los contornos
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Ordenar los contornos de izquierda a derecha
        contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
        
        image = element  #assignem el nom de la imatge
        filtro = df[df['FILENAME'] == image]  #afegim un filtre
        if not filtro.empty:
            nom = filtro['IDENTITY'].values[0]  #Extreiem el nom
         
        # print(nom)
            if len(contours) == len(nom):#si la quantitat de lletres correspon a la paraula
                # print(nom)
                # Iterar sobre los contornos y guardar cada letra como una imagen separada
                for i,lletra in enumerate(nom):
                    list_letters.append(nom[i])
                for i, contour in enumerate(contours):
                    # Obtener el bounding box del contorno
                    x, y, w, h = cv2.boundingRect(contour)
            
                    # Extraer la letra
                    letter = img[y:y+h, x:x+w]
                    num_img += 1
                    # Guardar la imagen de la letra
                    letter_path = os.path.join(output_dir, f'letter_{num_img}.png')
                    cv2.imwrite(letter_path, letter)
                
        
    list_id = [i for i in range(0, num_img + 1)]
        
    datos = {
        'id': list_id,
        'nombre': list_letters,
    }

    df = pd.DataFrame(datos)
    # Escribir el DataFrame a un archivo CSV
    nombre_archivo = csv_out
    df.to_csv(nombre_archivo, index=False)
    print(num_img)
    return(0)



wdir = 'train_v2/train'  #Directorio de las imagenes a clasificar
dir_out = "imatges_proc" #Directorio donde se clasificarán las imagenes
csv_in = 'written_name_train_v2.csv' #CSV con los nombres
csv_out = 'written_letters_train_v2.csv' #CSV donde se guardarán las letras
segment_letters(wdir,dir_out,csv_in,csv_out)

