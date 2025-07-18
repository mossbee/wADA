import net
import torch
import os
from face_alignment import align
import numpy as np


adaface_models = {
    'ir_50':"pretrained/adaface_ir50_ms1mv2.ckpt",
}

def load_pretrained_model(architecture='ir_50', device=None):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    
    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    statedict = torch.load(adaface_models[architecture], weights_only=False, map_location=device)['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    model = model.to(device)
    return model

def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    bgr_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
    tensor = torch.tensor(bgr_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return tensor

def get_feature(image_path, model, device=None):
    if device is None:
        device = next(model.parameters()).device
    aligned_rgb_img = align.get_aligned_face(image_path, device=device)
    bgr_tensor_input = to_input(aligned_rgb_img)
    if device is not None:
        bgr_tensor_input = bgr_tensor_input.to(device)
    feature, _ = model(bgr_tensor_input)
    return feature

if __name__ == '__main__':
    # Choose device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = load_pretrained_model('ir_50', device=device)
    feature, norm = model(torch.randn(2,3,112,112).to(device))

    test_image_path = 'face_alignment/test_images'
    features = []
    for fname in sorted(os.listdir(test_image_path)):
        path = os.path.join(test_image_path, fname)
        aligned_rgb_img = align.get_aligned_face(path, device=device)
        bgr_tensor_input = to_input(aligned_rgb_img)
        feature, _ = model(bgr_tensor_input.to(device))
        features.append(feature)

    similarity_scores = torch.cat(features) @ torch.cat(features).T
    print(similarity_scores)


