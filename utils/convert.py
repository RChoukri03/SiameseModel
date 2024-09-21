import torch
import torch.onnx
from model.model import RotDecNetwork  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RotDecNetwork(base_model_name='mobilenet_v2')
weights_path = "weights/rotDetfin.pth"

# Charger le modèle sur CPU si CUDA n'est pas disponible
model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))

model.to(device)
model.eval()
print(device)
# Créer deux entrées dummy de la taille attendue par le modèle
dummy_input1 = torch.randn(1, 3, 224, 224,dtype=torch.float32)
dummy_input2 = torch.randn(1, 3, 224, 224,dtype=torch.float32)
dummy_input1 = dummy_input1.to(device)
dummy_input2 = dummy_input2.to(device)
# Définir le chemin pour enregistrer le modèle ONNX
onnx_path = "weights/rotDetfin.onnx"

# Exporter le modèle
# Notez que nous passons maintenant deux entrées dummy
torch.onnx.export(model, (dummy_input1, dummy_input2), onnx_path, export_params=True,  do_constant_folding=True, input_names=['input1', 'input2'], output_names=['output1', 'output2'])


