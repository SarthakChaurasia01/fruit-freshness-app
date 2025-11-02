import torch
import timm
import os




def build_effnet_b2(num_classes=6, pretrained_backbone=False):
# create timm model with appropriate head
model = timm.create_model('efficientnet_b2', pretrained=pretrained_backbone, num_classes=num_classes)
return model




def load_model_from_path(path, num_classes=6, device='cpu'):
if not os.path.exists(path):
raise FileNotFoundError(f"Model file not found: {path}")
model = build_effnet_b2(num_classes=num_classes, pretrained_backbone=False)
model.to(device)


state = torch.load(path, map_location=device)
# if checkpoint dict contains state_dict under common keys, attempt to extract
if isinstance(state, dict):
if 'state_dict' in state:
state_dict = state['state_dict']
elif 'model_state_dict' in state:
state_dict = state['model_state_dict']
else:
# assume it is a state_dict
state_dict = state
else:
state_dict = state


# strip DataParallel 'module.' prefix if present
new_state = {}
for k, v in state_dict.items():
new_k = k
if k.startswith('module.'):
new_k = k[len('module.'):]
new_state[new_k] = v


try:
model.load_state_dict(new_state, strict=True)
except Exception as e:
# fallback to non-strict
model.load_state_dict(new_state, strict=False)


model.eval()
return model
