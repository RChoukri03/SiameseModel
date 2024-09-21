# train.py
import traceback
import torch
import logging
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from utils.clearml import ClearMLManager
from time import time
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
clearMLManager = ClearMLManager(projectName='Kit_Rot_Detection', taskName='Train')



def trainModel(model, trainLoader, valLoader, criterion, optimizer, scheduler, numEpochs, callbacks, device):
    model.to(device)
    logging.info(f"Model is using device: {device}")
    if device.type == 'cuda':
        logging.info(f"Current CUDA Memory Usage: {torch.cuda.memory_allocated(device=device)} bytes")
    bestAcc = 0.0

    for epoch in range(numEpochs):
        model.train()
        runningLoss = 0.0
        progress_bar = tqdm(enumerate(trainLoader), total=len(trainLoader), desc=f"Epoch {epoch + 1}/{numEpochs}")
        for i, (inputs, labels) in progress_bar:
            #start_data_load = time()
            inputs = [input.to(device, dtype=torch.float) for input in inputs]
            labels = labels.to(device, dtype=torch.float)
            #end_data_load = time()
            optimizer.zero_grad()
            try:
                outputs1, outputs2 = model(*inputs)
                loss = criterion(outputs1, outputs2, labels)
                loss.backward()
                optimizer.step()
                runningLoss += loss.item()
                #logging.info(f"Epoch {epoch+1}, Batch {i+1}, Data Load Time: {end_data_load - start_data_load:.4f}, Inference Time: {end_inference - start_inference:.4f}")
                progress_bar.set_postfix({"Train Loss": f"{runningLoss/(i+1):.4f}"})
            except Exception as e:
                logging.error(f"Training error: {e}")
                traceback.print_exc()

        valLoss, valAcc = validateModel(model, valLoader, criterion, device)
        scheduler.step()

        if valAcc > bestAcc:
            bestAcc = valAcc
            torch.save(model.state_dict(), 'weights/best_model.pth')
            clearMLManager.reportValue('BestValAccuracy', bestAcc)
            #clearMLManager.uploadArtifacts(name='bestRotDetKitModel', object='weights/best_model.pth')
        logging.info(f'Epoch {epoch+1}/{numEpochs}, Train Loss: {runningLoss/(i+1):.4f}, Val Loss: {valLoss:.4f}, Val Acc: {valAcc:.4f}')
        callbacks['tensorboard'].logTraining(runningLoss, epoch)
        callbacks['tensorboard'].logValidation(valLoss, valAcc, epoch)
        callbacks['earlyStopping'](valLoss, model)

        if callbacks['earlyStopping'].early_stop:
            logging.info("Early stopping triggered.")
            break
    torch.save(model.state_dict(), 'weights/final_model.pth')
    clearMLManager.uploadArtifacts(name='bestRotDetKitModel', object='weights/best_model.pth')
    clearMLManager.uploadArtifacts(name='finalRotDetKitModel', object='weights/final_model.pth')

def validateModel(model, valLoader, criterion, device):
    model.eval()
    runningLoss = 0.0
    distances = []  # Pour stocker les distances euclidiennes
    labels_list = []  # Pour stocker les labels

    val_progress_bar = tqdm(valLoader, total=len(valLoader), desc="Validation Progress")

    with torch.no_grad():
        for inputs, labels in val_progress_bar:
            inputs = [input.to(device, dtype=torch.float) for input in inputs]
            labels = labels.to(device, dtype=torch.float)

            try:
                output1, output2 = model(*inputs)
                loss = criterion(output1, output2, labels)
                runningLoss += loss.item()

                # Calculer et stocker la distance euclidienne
                euclidean_distance = F.pairwise_distance(output1, output2)
                distances.extend(euclidean_distance.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())

                # Update the progress bar
                val_progress_bar.set_postfix({"Validation Loss": f"{runningLoss/len(distances):.4f}"})
            except Exception as e:
                logging.error(f"Validation error: {e}")
                traceback.print_exc()

    threshold = 1.0  # Ce seuil doit être déterminé en fonction de vos données et expériences
    
    predictions = [1 if dist > threshold else 0 for dist in distances]
    valAccuracy = accuracy_score(labels_list, predictions)

    logging.info(f"Validation Accuracy: {valAccuracy:.4f}")
    logging.info(f"Total Validation Loss: {runningLoss:.4f}")
    return runningLoss / len(valLoader), valAccuracy


    
def validateModel(model, valLoader, criterion, device):
    model.eval()
    runningLoss = 0.0
    allPreds, allLabels = [], []
    val_progress_bar = tqdm(valLoader, total=len(valLoader), desc="Validation Progress")

    with torch.no_grad():
        for inputs, labels in val_progress_bar:
            inputs = [input.to(device, dtype=torch.float) for input in inputs]
            labels = labels.to(device, dtype=torch.float)

            try:
                outputs = model(*inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
                runningLoss += loss.item()
                preds = torch.sigmoid(outputs).data > 0.5
                allPreds.extend(preds.cpu().numpy())
                allLabels.extend(labels.cpu().numpy())

                # Update the progress bar
                val_progress_bar.set_postfix({"Validation Loss": f"{runningLoss/len(allPreds):.4f}"})
            except Exception as e:
                logging.error(f"Validation error: {e}")
                traceback.print_exc()

    valAccuracy = accuracy_score(allLabels, allPreds)
    logging.info(f"Validation Accuracy: {valAccuracy:.4f}")
    logging.info(f"Total Validation Loss: {runningLoss:.4f}")
    return runningLoss / len(valLoader), valAccuracy
