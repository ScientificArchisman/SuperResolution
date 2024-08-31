import torch 
from torch import nn 
from torchvision.models import vgg19
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_activation: bool, kernel_size, stride, padding) -> None:
        super().__init__()    
        self.conv = nn.Conv2d(in_channels, out_channels, bias = True, kernel_size=kernel_size, stride=stride, padding=padding)
        self.activation = nn.LeakyReLU(0.2, inplace=True) if use_activation else nn.Identity()

    def forward(self, x):
        return self.activation(self.conv(x))
    

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, scale_factor: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.activation(self.conv(self.upsample(x)))
    

class DenseResidualBlock(nn.Module):
    def __init__(self, in_channels: int, channels = 32, beta: float = 0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta 
        self.conv = nn.ModuleList()

        for block_no in range(5):
            self.conv.append(ConvBlock(in_channels + channels * block_no, 
                                       channels if block_no < 4 else in_channels,
                                         use_activation=True if block_no < 4 else False,
                                           kernel_size=3, stride=1, padding=1))
            
            
    def forward(self, x):
        new_inputs = x
        for block in self.conv:
            out = block(new_inputs)
            new_inputs = torch.cat([new_inputs, out], dim=1)
        return self.beta * out + x
        
class RRDB(nn.Module):
    def __init__(self, in_channels, residual_beta, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.residual_beta = residual_beta
        self.rrdb = nn.Sequential(*[DenseResidualBlock(in_channels, beta=residual_beta) for _ in range(3)])

    def forward(self, x):
        return self.rrdb(x) * self.residual_beta + x
    
class Generator(nn.Module):
    def __init__(self, in_channels = 3, num_blocks = 23, num_channels = 64, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1, bias = True)
        self.residual_blocks = nn.Sequential(*[RRDB(num_channels, residual_beta=0.2) for _ in range(num_blocks)])
        self.conv2 = nn.Conv2d(in_channels = num_channels, out_channels = num_channels, kernel_size=3, stride=1, padding=1, bias = True)
        self.upsample = nn.Sequential(*[UpsampleBlock(in_channels=num_channels, scale_factor=2) for _ in range(2)])
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=num_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=True))
        
    def forward(self, x):
        x = self.conv1(x)
        residual = x
        x = self.residual_blocks(x)
        x = self.conv2(x)
        x += residual
        x = self.upsample(x)
        x = self.conv3(x)
        return x
    

class Discriminator(nn.Module):
    def __init__(self, in_channels = 3, features = [64, 64, 128, 128, 256, 256, 512, 512], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        blocks = []
        for idx, feature in enumerate(features):
            blocks.append(ConvBlock(in_channels, feature, use_activation=True, kernel_size=3, stride=1 + idx % 2, padding=1))
            in_channels = feature

        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1))
        
    def forward(self, x):
        x = self.blocks(x)
        return self.classifier(x)
    

def initialize_weights(model, scale = 0.1):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)
            m.weight.data *= scale
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data)
            m.weight.data *= scale
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    


class VGGLoss(nn.Module):
    def __init__(self, device = device) -> None:
        super().__init__()
        self.vgg = vgg19(pretrained=True).features[:35].eval().to(device)

        for param in self.vgg.parameters():
            param.requires_grad = False

        self.loss = nn.MSELoss()

    def forward(self, x, y):
        x_features = self.vgg(x)
        y_features = self.vgg(y)
        return self.loss(x_features, y_features)



def test():
    gen = Generator()
    discrim = Discriminator()
    low_res = 24
    x = torch.randn((5, 3, low_res, low_res))
    gen_out = gen(x)
    disc_out = discrim(gen_out)

    print(gen_out.shape)
    print(disc_out.shape)

if __name__ == '__main__':
    test()
