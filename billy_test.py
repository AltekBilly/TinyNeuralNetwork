import tensorflow as tf

def load_tflite_model(file_path):
    # Load the TFLite model
    with open(file_path, 'rb') as f:
        model_content = f.read()
    model = tf.lite.Interpreter(model_content=model_content)
    model.allocate_tensors()
    return model

def get_model_details(model):
    # Get model details
    details = {
        "input_details": model.get_input_details(),
        "output_details": model.get_output_details(),
        "tensor_details": model.get_tensor_details()
    }
    return details

# Load the models
root = 'examples/converter/out/'
file_path_1 = root + 'Altek_Landmark-FacialLandmark-Visible-20240716-qat-best.tflite'
file_path_2 = root + 'Altek_Landmark-FacialLandmark-merl_rav-20240718-qat-best.tflite'

model_1 = load_tflite_model(file_path_1)
model_2 = load_tflite_model(file_path_2)

# Get details of both models
model_1_details = get_model_details(model_1)
model_2_details = get_model_details(model_2)

# Compare the number of tensors
print(f"Model 1 has {len(model_1_details['tensor_details'])} tensors.")
print(f"Model 2 has {len(model_2_details['tensor_details'])} tensors.")

# Compare input and output details
print("\nModel 1 Input Details:", model_1_details['input_details'])
print("Model 1 Output Details:", model_1_details['output_details'])

print("\nModel 2 Input Details:", model_2_details['input_details'])
print("Model 2 Output Details:", model_2_details)

# Compare layers
def compare_tensors(tensor_details_1, tensor_details_2):
    for i, (tensor_1, tensor_2) in enumerate(zip(tensor_details_1, tensor_details_2)):
        print(f"\nTensor {i}:")
        print("Model 1:", tensor_1)
        print("Model 2:", tensor_2)

compare_tensors(model_1_details['tensor_details'], model_2_details['tensor_details'])
