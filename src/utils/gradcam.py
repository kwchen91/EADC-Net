import torch

class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer = dict(model.named_modules())[target_layer_name]
        self.gradients = None
        self.activations = None
        self._register()

    def _register(self):
        def fwd_hook(module, inp, out):
            self.activations = out.detach()
        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_full_backward_hook(bwd_hook)

    def __call__(self, x):
        self.model.zero_grad()
        out = self.model(x)  
        score = out.sigmoid().mean()
        score.backward(retain_graph=True)
        weights = self.gradients.mean(dim=(2,3), keepdim=True)  
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam / (cam.max()+1e-6)
        return out, cam
