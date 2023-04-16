from torchlm.models import pipnet
from torchlm.tools import faceboxesv2
# will auto download pretrained weights from latest release if pretrained=True
model = pipnet(backbone="resnet101", pretrained=True, num_nb=10, num_lms=98, net_stride=32,
               input_size=256, meanface_type="wflw", backbone_pretrained=True, map_location="cuda")
model.apply_exporting(
    onnx_path="./pipnet_resnet101.onnx",
    opset=12, simplify=True, output_names=None  # use default output names.
)

model2 = faceboxesv2(device="cuda")
model2.apply_exporting(
    onnx_path="./faceboxesv2.onnx",
    opset=12, simplify=True, output_names=None  # use default output names.
)