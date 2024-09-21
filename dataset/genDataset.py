import cv2
import os
import pandas as pd
import random

def processImagesAndGenerateLabels(sourceFolder, outputFolder):
    os.makedirs(outputFolder, exist_ok=True)

    csvData = []

    # Traitement des images
    for filename in os.listdir(sourceFolder):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            imgPath = os.path.join(sourceFolder, filename)
            img = cv2.imread(imgPath)
            outPath = os.path.join(outputFolder, filename)
            cv2.imwrite(outPath, cv2.resize( img, (448,448)))
            csvData.append([filename, filename, 0])

            for i in range(1, 4):
                # Transformation modifiée
                angleOriginal = random.uniform(-0.5, 0.5)
                modifiedImg = rotateWithVariation(img, angle=angleOriginal)
                modifiedFilename = filename.split('.')[0] + f"_original_{i}." + filename.split('.')[1]
                modifiedPath = os.path.join(outputFolder, modifiedFilename)

                cv2.imwrite(modifiedPath, cv2.resize(modifiedImg, (448,448)))
                csvData.append([filename, modifiedFilename, 0])  # Paire originale-modifiée (label 0)

                # Transformation pivotée
                angleRotated = random.uniform(179.5, 180.5)
                rotatedImg = rotateWithVariation(img, angle=angleRotated)
                rotatedFilename = filename.split('.')[0] + f"_rotated_{i}." + filename.split('.')[1]
                rotatedPath = os.path.join(outputFolder, rotatedFilename)

                cv2.imwrite(rotatedPath, cv2.resize( rotatedImg, (448,448)))
                csvData.append([filename, rotatedFilename, 1])

    # Enregistrer les données dans un fichier CSV
    df = pd.DataFrame(csvData, columns=['filename1', 'filename2', 'label'])
    df.to_csv(os.path.join(outputFolder, 'labels.csv'), index=False)






def rotateWithVariation(image, angle):
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    rotationMatrix = cv2.getRotationMatrix2D(center, angle, 1)
    rotatedImg = cv2.warpAffine(image, rotationMatrix, (width, height))
    return rotatedImg

def processImages(sourceFolder, outputFolder):
    os.makedirs(outputFolder, exist_ok=True)

    for filename in os.listdir(sourceFolder):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            imgPath = os.path.join(sourceFolder, filename)
            img = cv2.imread(imgPath)
            outPath = os.path.join(outputFolder, filename)
            cv2.imwrite(outPath, cv2.resize(img, (448, 448)))

            for i in range(1, 4):
                # Transformation modifiée
                angleOriginal = random.uniform(-0.5, 0.5)
                modifiedImg = rotateWithVariation(img, angle=angleOriginal)
                modifiedFilename = filename.split('.')[0] + f"_original_{i}." + filename.split('.')[1]
                modifiedPath = os.path.join(outputFolder, modifiedFilename)
                cv2.imwrite(modifiedPath, cv2.resize(modifiedImg, (448, 448)))

                # Transformation pivotée
                angleRotated = random.uniform(179.5, 180.5)
                rotatedImg = rotateWithVariation(img, angle=angleRotated)
                rotatedFilename = filename.split('.')[0] + f"_rotated_{i}." + filename.split('.')[1]
                rotatedPath = os.path.join(outputFolder, rotatedFilename)
                cv2.imwrite(rotatedPath, cv2.resize(rotatedImg, (448, 448)))

def generateCSV(outputFolder):
    csvData = []
    filenames = os.listdir(outputFolder)

    for filename in filenames:
        base_name = filename.split('_')[0]

        for other_filename in filenames:
            other_base_name = other_filename.split('_')[0]

            # S'assurer que les comparaisons sont faites uniquement avec les images de la même base
            if base_name == other_base_name and filename != other_filename:
                label = 0

                # Cas où aucune des images n'a de suffixe (les deux sont des images de base)
                if "_" not in filename and "_" not in other_filename:
                    label = 0
                # Cas où l'image de base est comparée à ses variantes
                elif "_" not in filename:
                    if "original" in other_filename:
                        label = 0
                    elif "rotated" in other_filename:
                        label = 1
                # Cas où les variantes sont comparées entre elles
                else:
                    if "original" in filename and "original" in other_filename:
                        label = 0
                    elif "rotated" in filename and "rotated" in other_filename:
                        label = 0
                    elif "original" in filename and "rotated" in other_filename:
                        label = 1
                    elif "rotated" in filename and "original" in other_filename:
                        label = 1

                csvData.append([filename, other_filename, label])

    df = pd.DataFrame(csvData, columns=['filename1', 'filename2', 'label'])
    df.to_csv(os.path.join(outputFolder, 'labels.csv'), index=False)
