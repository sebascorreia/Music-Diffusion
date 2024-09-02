from diffusers import UNet2DModel
import torch
import torch.nn as nn
def Unet2d(sample_size):
    return UNet2DModel(
        sample_size=sample_size,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )


def CustomUnet2DConditional(sample_size=128, num_classes=10, class_emb_size=4):
    return CustomConditionalModel(sample_size=sample_size, num_classes=num_classes, class_emb_size=class_emb_size).model


class CustomConditionalModel(UNet2DModel):
    def __init__(self, sample_size=128, num_classes=10, class_emb_size=4):
        super().__init__()

        self.class_emb = nn.Embedding(num_classes, class_emb_size)
        self.model = UNet2DModel(
            sample_size=sample_size,
            in_channels=1 + class_emb_size,  # Adding class embedding channels
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

    def forward(self, x, t, class_labels):
        bs, ch, w, h = x.shape

        # Embed class labels and prepare for concatenation
        class_cond = self.class_emb(class_labels)
        class_cond = class_cond.view(bs, class_cond.shape[1], 1, 1).expand(bs, class_cond.shape[1], w, h)

        # Concatenate input with class condition
        net_input = torch.cat((x, class_cond), dim=1)
        output = self.model(net_input, t)

        return output


def CondUnet2d(sample_size=128, num_classes=10):
    return UNet2DModel(
        sample_size=sample_size,
        class_emb_type = "None",
        num_class_embeds = num_classes,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )



