import os
import torch

def convert(clf, X_test):
    outputDir = 'weights'
    outputPath = os.path.join(outputDir, 'model.onnx')

    clf.network.eval()

    dummyInput = torch.from_numpy(X_test).type(torch.float32).to(device=clf.device)

    os.makedirs(outputDir, exist_ok=True)

    # Exportation du modèle en ONNX
    torch.onnx.export(
        clf.network,
        dummyInput,
        outputPath,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )

    return outputPath
