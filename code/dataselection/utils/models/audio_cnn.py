# dataselection/utils/models/audio_cnn.py
import torch
import torch.nn as nn

class AudioCNN(nn.Module):
    """
    CNN 2D type VGG pour log-mels / spectrogrammes.
    Entrées acceptées:
      - [B, 1, F, T]
      - [B, F, T]   (on ajoute la dimension canal)
      - [B, T, F]   (on permute -> [B, F, T])

    Args:
        n_mels (int): dim fréquentielle attendue (indicatif)
        num_classes (int): nb de classes (défaut=2)
        dropout (float): dropout avant la FC
        gn_groups (int): GroupNorm groups
    """
    def __init__(self, n_mels: int = 64, num_classes: int = 2,
                 dropout: float = 0.3, gn_groups: int = 8):
        super().__init__()
        self.n_mels = n_mels
        self.embDim = 128  # <-- AJOUT: dimension d'embedding attendue par la sélection

        def GN(cout: int) -> nn.GroupNorm:
            return nn.GroupNorm(num_groups=min(gn_groups, cout), num_channels=cout)

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1),
                GN(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, kernel_size=3, padding=1),
                GN(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

        self.features = nn.Sequential(
            block(1, 32),   # /2
            block(32, 64),  # /4
            block(64, 128), # /8
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(128, num_classes)

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, F] -> [B, F, T] si besoin
        if x.dim() == 3 and x.shape[-1] < x.shape[-2]:
            x = x.transpose(1, 2)
        # [B, F, T] -> [B, 1, F, T]
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return x

    # <-- MODIF MINIMALE: on ajoute les paramètres last et freeze avec des valeurs par défaut
    def forward(self, x: torch.Tensor, last: bool = False, freeze: bool = False):
        x = self._normalize_input(x)  # [B, 1, F, T]
        if freeze:
            with torch.no_grad():
                x = self.features(x)
                e = x.mean(dim=[2, 3])  # GAP -> [B, 128]
        else:
            x = self.features(x)
            e = x.mean(dim=[2, 3])  # GAP -> [B, 128]
        e = self.dropout(e)
        logits = self.classifier(e)
        return (logits, e) if last else logits

    # <-- AJOUT: méthode requise par la sélection
    def get_embedding_dim(self):
        return self.embDim
