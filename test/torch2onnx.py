import json
import torch
from models.model_architectures import *

def Convert_ONNX(model, input_size, output_path): 

    # set the model to inference mode 
    model.eval() 

    # Let's create a dummy input tensor  
    dummy_input = torch.randn(*input_size)

    # Export the model   
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         output_path,       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=10,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['modelInput'],   # the model's input names 
         output_names = ['modelOutput'], # the model's output names 
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                'modelOutput' : {0 : 'batch_size'}}) 
    print(" ") 
    print('Model has been converted to ONNX') 

    onnx_model = torch.onnx.load(output_path)
    try:
        torch.onnx.checker.check_model(onnx_model)
    except torch.onnx.checker.ValidationError as e:
        print("The model is invalid: %s"%e)
    else:
        print("The model is valid!")


if __name__ == "__main__": 

    with open('test/config.json', 'r') as f:
        data = json.load(f)

    # Let's build our model 
    #train(5) 
    #print('Finished Training') 

    # Test which classes performed well 
    #testAccuracy() 

    # Let's load the model we just created and test the accuracy per label 
    model = resnet101(3)
    model.load_state_dict(torch.load(data['models_path']['resnet101']['pytorch'])) 
    Convert_ONNX(model, (1,3,224,224), data['models_path']['resnet101']['onnx'])