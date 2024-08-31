from torch import nn 
import torch 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GeneratorResidualBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, 
                 padding=1, use_activation: bool = True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True) if use_activation else nn.Identity()
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, 
                                kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        x = self.bn2(self.conv2(self.activation(self.bn1(self.conv1(x)))))
        return x + residual
    
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=0, 
                 scate_factor=4, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scate_factor)
        self.activation = nn.PReLU()
    
    def forward(self, x):
        return self.pixel_shuffle(self.activation(self.conv(x)))
    

class Generator(nn.Module):
    def __init__(self, in_channels: int = 3, num_residual_blocks: int = 16, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=9, stride=1, padding=4, bias=True)
        self.activation = nn.PReLU()
        self.residual_blocks = nn.Sequential(*[GeneratorResidualBlock() for _ in range(num_residual_blocks)])
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(64)
        self.upsample = nn.Sequential(*[UpsampleBlock(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, scate_factor=2), 
                                        UpsampleBlock(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, scate_factor=2)])
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4, bias=True)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x += self.bn(self.conv2(self.residual_blocks(x))) + x
        return self.conv3(self.upsample(x))    



class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=2, 
                 padding=1, use_activation: bool = True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding, bias=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True) if use_activation else nn.Identity()

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))
    
class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 3, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.disc_blocks = nn.Sequential(*[DiscriminatorBlock(in_channels=64, out_channels=64, stride=2), 
                                           DiscriminatorBlock(in_channels=64, out_channels=128, stride=1),
                                           DiscriminatorBlock(in_channels=128, out_channels=128, stride=2),
                                           DiscriminatorBlock(in_channels=128, out_channels=256, stride=1),
                                           DiscriminatorBlock(in_channels=256, out_channels=256, stride=2),
                                           DiscriminatorBlock(in_channels=256, out_channels=512, stride=1),
                                           DiscriminatorBlock(in_channels=512, out_channels=512, stride=2)])
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d((6, 6)), nn.Flatten(), nn.Linear(512*6*6, 1024), 
                                        nn.LeakyReLU(0.2, inplace=True), nn.Linear(1024, 1), nn.Sigmoid())
        
    def forward(self, x):
        return self.classifier(self.disc_blocks(self.activation(self.conv1(x))))

    


if __name__ == "__main__":
    test_tensor = torch.randn(1, 3, 64, 64).to(DEVICE)
    generator = Generator().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)

    gen_output = generator(test_tensor)
    disc_output = discriminator(gen_output)

    print(f"Generator Output Shape: {gen_output.shape}")
    print(f"Discriminator Output Shape: {disc_output.shape}")