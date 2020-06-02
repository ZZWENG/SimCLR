import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
#from hyptorch.nn import FromPoincare, ToPoincare, HypLinear
import geoopt

c = 1.0
ball = geoopt.PoincareBall(c)

class ToPoincare(torch.nn.Module):
    def __init__(self, in_dim, ball):
        super().__init__()
        self.x = torch.zeros(in_dim)
        self.ball_ = ball
        
    def forward(self, u):
        device = u.device
        u = u.cpu()
        mapped = self.ball_.expmap(self.x, u)
        mapped.to(device)
        return mapped
        

class HResNetSimCLR(nn.Module):
    def __init__(self, base_model, out_dim):
        super(HResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=True),
                            "resnet50": models.resnet50(pretrained=True)}

        resnet = self._get_basemodel(base_model)
        num_ftrs = resnet.fc.in_features

        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.embedding = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            #nn.Linear(256, 256),
            #nn.ReLU(),
            ToPoincare(in_dim=256, ball=ball)
        )

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        try:
            h = self.encoder(x)
        except:
            print(x.shape)
            h = self.encoder(x)
        h = h.squeeze()
        x = self.embedding(h)
        return h, x
