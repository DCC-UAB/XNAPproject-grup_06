# -*- coding: utf-8 -*-
"""
Created on Wed May 29 16:08:39 2024

@author: alexg
"""

import cv2
import os
import pandas as pd
from PIL import Image


def delete_image_rows(input_csv, output_csv, images_to_delete):
    """
    Función para eliminar de un csv las filas de las cuales no tenemos imagenes

    Parameters
    ----------
    input_csv : CSV de las letras con su correspondiente nombre
    output_csv : CSV de salida 
    images_to_delete : Lista con los nombres de las imagenes que se deben eliminar

    Returns
    -------
    None.

    """
    # Leer el archivo CSV
    df = pd.read_csv(input_csv)

    # Eliminar las filas donde el nombre de la imagen está en la lista de imágenes a eliminar
    df = df[~df.iloc[:, 0].isin(images_to_delete)]

    # Guardar el DataFrame resultante en un nuevo archivo CSV
    df.to_csv(output_csv, index=False)


def process_images(directory, output_directory,csv_in,csv_out):
    """
    Función para clasificar el conjunto de letras en imagenes estandarizadas a blanco y negro y a un 
    tamaño de 10 x 14 eliminando las mayores a esta resolución y rellenando con bordes blanco las inferiores a este

    Parameters
    ----------
    directory :Directorio de las imagenes a clasificar
    output_directory : Directorio donde se clasificarán las imagenes
    csv_in : CSV con los nombres
    csv_out : CSV donde se guardarán las letras

    Returns
    -------
    None.

    """
    #Crea un directorio si no existe
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)  
    imatges_a_eliminar = []
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            try:
                filepath = os.path.join(directory, filename)
                with Image.open(filepath) as img:
                    # Convertir la imagen a escala de grises
                    img = img.convert('L')
                    # Convertir la imagen a blanco y negro (0 o 1)
                    img = img.point(lambda x: 0 if x < 128 else 1, mode='1')

                    # Obtener las dimensiones de la imagen
                    width, height = img.size

                    # Descartar imágenes con tamaño superior a 10x14
                    if width > 10 or height > 14:
                        imatges_a_eliminar.append(filename)
                        #print(f"Discarding {filename} due to size {width}x{height}")
                        
                        
                        continue

                    # Crear una nueva imagen de 10x14 con fondo blanco
                    new_img = Image.new('1', (10, 14), 1)  # 1 para color blanco

                    # Calcular la posición para centrar la imagen en el fondo blanco
                    x_offset = (10 - width) // 2
                    y_offset = (14 - height) // 2

                    # Pegar la imagen en la nueva imagen de fondo blanco
                    new_img.paste(img, (x_offset, y_offset))

                    # Guardar la imagen procesada en el directorio de salida
                    output_path = os.path.join(output_directory, filename)
                    new_img.save(output_path)
                    
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    #Elimina del csv las imagenes con tamaño superior a 10 x 14
    delete_image_rows(csv_in, csv_out, imatges_a_eliminar)

def segment_letters(image_dir, output_dir,csv,csv_out):
    """
    Codigo que dado un directorio con imagenes de palabras las recorta extrayendo las letras y guardando a que letra corresponde
    a cada imagen en un csv. Descarta aquellas palabras de las cuales se extraen un numero de imagenes diferentes al numero
    de palabras 
    
    Parameters
    ----------
    image_dir : Directorio de las imagenes a clasificar
    output_dir : Directorio de las imagenes clasificadas 
    csv : CSV de las letras
    csv_out : CSV de las letras sin las no clasificadas

    Returns
    -------
    None.

    """

    # Crear el directorio de salida si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)   #crea un directori per guardar les imatges
    df = pd.read_csv(csv)  #Carrega el csv
    df.dropna(inplace=True)
    list_id = []
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
                    if num_img%1000 == 0:
                        print(num_img)
                    # Guardar la imagen de la letra
                    letter_path = os.path.join(output_dir, f'letter_{num_img}.png')
                    string = str("letter_"+str(num_img)+".png")
                    list_id.append(string)
                    cv2.imwrite(letter_path, letter)
                
        
        
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
dir_out = "imatges_proc/" #Directorio donde se clasificarán las imagenes
csv_in = 'written_name_train_v2.csv' #CSV con los nombres
csv_out = 'written_letters_train_v2.csv' #CSV donde se guardarán las letras
segment_letters(wdir,dir_out,csv_in,csv_out)

directory = "imatges_proc/"  # Directorio de las imagenes a clasificar
output_directory = "imatges_proc1/"  # Directorio de las imagenes clasificadas 
csv_inp = "written_letters_train_v2.csv" #CSV de las letras
csv_outp = "written_letters_train_v3.csv" #CSV de las letras sin las no clasificadas
process_images(directory, output_directory,csv_inp,csv_outp)

