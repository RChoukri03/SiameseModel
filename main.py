import argparse
import logging
import traceback
from dataset.dataset import createDatasets
from dataset.genDataset import processImagesAndGenerateLabels, generateCSV, processImages
from model.model import RotDecNetwork, ContrastiveLoss
from train import trainModel
from model.callbacks import EarlyStopping, TensorBoardLogger
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from utils.clearml import ClearMLManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
fraction = 0.95

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    logging.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logging.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        logging.warning("CUDA is not available. Using CPU.")
    try:
        parser = argparse.ArgumentParser(description='Entraînement du modèle RotDec Network')
        parser.add_argument('--csvFile', type=str, default='data/dataset/labels.csv', help='Chemin vers le fichier CSV')
        parser.add_argument('--sourceFolder', type=str, default='data/croppedkits', help='Répertoire source des images à traiter')
        parser.add_argument('--datasetFolder', type=str, default='data/dataset', help='Répertoire de sortie du dataset')
        parser.add_argument('--splitRatio', type=float, default=0.70, help='Ratio de split train/validation')
        parser.add_argument('--batchSize', type=int, default=96, help='Taille du batch pour l\'entraînement et la validation')
        parser.add_argument('--numEpochs', type=int, default=100, help='Nombre d\'époques pour l\'entraînement')
        parser.add_argument('--baseModel', type=str, default='efficientnet_b4', help='modèle de base : resnet50, resnet101, resnet152, vgg13, vgg16, vgg19, mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small')
        parser.add_argument('--forceGenerateDataset', action='store_true', help='Forcer la génération du dataset même si le dossier de sortie existe')

        args = parser.parse_args()

        ClearMLManager(projectName='Kit_Rot_Detection', taskName='Train')

        # Vérifiez si le répertoire de sortie (datasetFolder) existe
        if not os.path.exists(args.datasetFolder) or args.forceGenerateDataset:
            # Si le répertoire de sortie n'existe pas ou si l'option --forceGenerateDataset est activée, créez-le et générez le dataset
            os.makedirs(args.datasetFolder, exist_ok=True)
            logging.info(f"Dataset folder: {args.datasetFolder} not found, generate dataset ...")
            processImages(args.sourceFolder, args.datasetFolder)
            generateCSV(args.datasetFolder)
            

        else:
            # Si le répertoire de sortie existe et que l'option --forceGenerateDataset n'est pas activée, utilisez-le comme répertoire de sortie du dataset
            logging.info(f"Using existing Dataset folder: {args.datasetFolder}")


        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        trainDataset, valDataset = createDatasets(args.csvFile, args.datasetFolder, args.splitRatio)  # Utilisez datasetFolder comme répertoire du dataset
        trainLoader = DataLoader(trainDataset, batch_size=args.batchSize, shuffle=True,num_workers=4, pin_memory=True,prefetch_factor=2)
        valLoader = DataLoader(valDataset, batch_size=args.batchSize, shuffle=False,num_workers=4,pin_memory=True ,prefetch_factor=2)
        ##trainLoader = DataLoader(trainDataset, batch_size=args.batchSize, shuffle=True, num_workers=0) #, collate_fn=apply_augmentations, prefetch_factor=2)
        #valLoader = DataLoader(valDataset, batch_size=args.batchSize, shuffle=False, num_workers=0) #, collate_fn=apply_augmentations, prefetch_factor=2)
        model = RotDecNetwork(base_model_name=args.baseModel).to(device)
        # criterion = torch.nn.BCEWithLogitsLoss()
        criterion = ContrastiveLoss(margin=3.0)
        optimizer = optim.Adam(model.parameters(), lr=0.006)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.45)

        callbacks = {
            "earlyStopping": EarlyStopping(patience=200, verbose=True),
            "tensorboard": TensorBoardLogger(log_dir='runs/')
        }

        trainModel(model, trainLoader, valLoader, criterion, optimizer, scheduler, numEpochs=args.numEpochs, callbacks=callbacks, device=device)

    except Exception as e:
        logging.error(f"Main execution error: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()