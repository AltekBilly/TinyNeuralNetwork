import os
import sys

import torch

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))

sys.path.insert(1, os.path.join(CURRENT_PATH, '../../'))

from examples.models.cifar10.mobilenet import DEFAULT_STATE_DICT, Mobilenet
from tinynn.converter import TFLiteConverter

TYPE = 'int8'
NAME = 'Altek_Landmark-FacialLandmark-merl_rav-20p-20240920-qat-best'

def main_worker():
    # model = Mobilenet()
    # model.load_state_dict(torch.load(DEFAULT_STATE_DICT))
    model = torch.jit.load(os.path.join(CURRENT_PATH, f"{NAME}.pt"))
    model.cpu()
    model.eval()

    dummy_input = torch.rand((1, 3, 256, 256))
    # dummy_input = torch.quantize_per_tensor(dummy_input, scale=1/255, zero_point=0, dtype=torch.quint8) #ANCHOR - quant
    output_path = os.path.join(CURRENT_PATH, 'out', f'{NAME}-{TYPE}.tflite')

    # When converting quantized models, please ensure the quantization backend is set.
    torch.backends.quantized.engine = 'qnnpack'

    # The code section below is used to convert the model to the TFLite format
    # If you want perform dynamic quantization on the float models,
    # you may refer to `dynamic.py`, which is in the same folder.
    # As for static quantization (e.g. quantization-aware training and post-training quantization),
    # please refer to the code examples in the `examples/quantization` folder.
    converter = TFLiteConverter(model, dummy_input, output_path, quantize_target_type=TYPE)
    converter.convert()


if __name__ == '__main__':
    main_worker()
