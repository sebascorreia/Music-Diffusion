from diffusers import UNet2DModel

def Unet2d():
    return UNet2DModel(
        sample_size=256,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        blocks_per_group=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2d",
            "DownBlock2d",
            "DownBlock2d",
            "DownBlock2d",
            "AttnDownBlock2d",
            "DownBlock2d",
        ),
        up_block_types=(
            "UpBlock2d",
            "AttnUpBlock2d",
            "UpBlock2d",
            "UpBlock2d",
            "UpBlock2d",
            "UpBlock2d",
        ),
    )

