import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from hyptorch.nn import FromPoincare, ToPoincare, HypLinear

c = 1.0
train_c = False
train_x = False


class HResNetSimCLR(nn.Module):
    def __init__(self, base_model, out_dim):
        super(HResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
                            "resnet50": models.resnet50(pretrained=False)}

        resnet = self._get_basemodel(base_model)
        num_ftrs = resnet.fc.in_features

        self.encoder = nn.Sequential(*list(resnet.children())[:-1])

        self.e2p = ToPoincare(train_c=train_c, c=c, train_x=train_x)

        self.p2e = FromPoincare(train_c=train_c, c=c, train_x=train_x)
        self.embedding = nn.Sequential(
            HypLinear(num_ftrs, num_ftrs, c=c),
            self.p2e,
            nn.ReLU(inplace=True),
            self.e2p,
            HypLinear(num_ftrs, out_dim, c=c),
        )

        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

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
