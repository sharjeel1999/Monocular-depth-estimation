import torch
import torchvision
import torchvision.models.detection.image_list
import torchvision.models.detection.rpn
import torchvision.models.detection.roi_heads
import torchvision.models.detection.transform
import torchvision.ops


# Load the pre-trained Mask R-CNN model (which contains the backbone)
model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
backbone = model.backbone

# backbone.eval()

try:
    scripted_backbone = torch.jit.script(backbone)
except Exception as e:
    print(f"Scripting failed: {e}")
    print("Attempting tracing instead...")

    dummy_input = torch.randn(1, 3, 224, 224) # Adjust shape if needed
    scripted_backbone = torch.jit.trace(backbone, dummy_input)
    print("Tracing successful.")

output_filename = "resnet50_fpn_backbone.pt"
scripted_backbone.save(output_filename)

print(f"Pre-trained ResNet50-FPN backbone saved to {output_filename}")
