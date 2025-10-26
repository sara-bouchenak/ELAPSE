# dataselection/utils/models/audio_lstm.py
import torch
import torch.nn as nn
from typing import Optional, Literal, Tuple

__all__ = ["AudioLSTM"]

class AudioLSTM(nn.Module):
    """
    BiLSTM pour séquences de features (MFCC, log-mels, etc.).

    Entrées supportées (x):
      - [B, T, C]
      - [B, C, T]  (permuté en [B, T, C])
      - [B, 1, F, T] (squeeze -> [B, F, T] -> [B, T, F])
      - [B, F, T]  (permuté en [B, T, F])
      - [T, C]     (promu en [1, T, C])

    Construction:
      - Si `in_dim` est fourni au __init__, les couches sont construites immédiatement (eager).
      - Sinon, elles sont construites lors du premier forward (lazy) en détectant la dimension C.

    Args:
        in_dim: dimension des features par frame (C). Si None, détectée au 1er forward.
        hidden: taille de l'état caché LSTM.
        layers: nombre de couches LSTM.
        num_classes: nombre de classes en sortie (logits).
        dropout: dropout appliqué après le pooling temporel (pas dans LSTM si layers==1).
        bidirectional: BiLSTM si True.
        pooling: "mean" (moyenne temporelle) ou "last" (dernier pas de temps).
    """

    def __init__(
        self,
        in_dim: Optional[int] = None,
        hidden: int = 128,
        layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        pooling: Literal["mean", "last"] = "mean",
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.num_classes = int(num_classes)
        self.dropout_p = float(dropout)
        self.bidirectional = bool(bidirectional)
        self.pooling = pooling

        # Modules construits à la volée
        self.lstm: Optional[nn.LSTM] = None
        self.fc: Optional[nn.Linear] = None
        self.drop = nn.Dropout(self.dropout_p)

        # Ajout minime : dimension de l'embedding penultimate
        self.embDim: int = self.hidden * (2 if self.bidirectional else 1)

        self._in_dim: Optional[int] = None
        if in_dim is not None:
            self._lazy_build(int(in_dim))

    # ------------ Helpers forme & build ------------

    @staticmethod
    def _promote_batch(x: torch.Tensor) -> torch.Tensor:
        # [T, C] -> [1, T, C]
        if x.dim() == 2:
            x = x.unsqueeze(0)
        return x

    @staticmethod
    def _is_time_first(t: int, c: int) -> bool:
        # Heuristique: le temps est souvent la plus grande des deux dims
        # En cas d'égalité, on suppose déjà [B, T, C].
        return t >= c

    def _to_BTC(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalise en [B, T, C] (float32, contiguous).
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"AudioLSTM attend un torch.Tensor, reçu {type(x)}")

        if x.dim() == 4:
            # ex: [B, 1, F, T] -> [B, F, T]
            if x.shape[1] == 1:
                x = x.squeeze(1)
            else:
                raise ValueError(
                    f"Forme 4D non supportée (attendu [B,1,F,T]), reçu {tuple(x.shape)}"
                )

        x = self._promote_batch(x)  # [T,C] -> [1,T,C]

        if x.dim() != 3:
            raise ValueError(
                f"Entrée inattendue: {tuple(x.shape)}. "
                f"Formes valides: [B,T,C], [B,C,T], [B,1,F,T], [B,F,T], [T,C]."
            )

        B, A, B_ = x.shape  # ambigu: soit [B,T,C] soit [B,C,T]
        if self._is_time_first(A, B_):
            # suppose [B, T, C] déjà
            x = x
        else:
            # suppose [B, C, T] -> [B, T, C]
            x = x.transpose(1, 2)

        # sécurité dtype/contiguïté
        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()
        return x  # [B, T, C]

    def _lazy_build(self, in_dim: int) -> None:
        if in_dim <= 0:
            raise ValueError(f"in_dim doit être > 0, reçu {in_dim}")
        if (self.lstm is None) or (self._in_dim != in_dim):
            self._in_dim = in_dim
            self.lstm = nn.LSTM(
                input_size=in_dim,
                hidden_size=self.hidden,
                num_layers=self.layers,
                batch_first=True,
                bidirectional=self.bidirectional,
                # dropout interne LSTM seulement si layers > 1 (PyTorch requirement)
                dropout=0.0 if self.layers <= 1 else 0.2,
            )
            out_dim = self.hidden * (2 if self.bidirectional else 1)
            self.fc = nn.Linear(out_dim, self.num_classes)
            # Ajout minime : garder embDim cohérent
            self.embDim = out_dim

    # ------------ Forward ------------

    def _temporal_pool(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, H*dir]
        if self.pooling == "mean":
            return x.mean(dim=1)                  # [B, H*dir]
        elif self.pooling == "last":
            return x[:, -1, :]                    # [B, H*dir]
        else:
            raise ValueError(f"Pooling inconnu: {self.pooling}")

    def forward(self, x: torch.Tensor, *, last: bool = False, freeze: bool = False) -> torch.Tensor:
        """
        Args:
            x: Tensor des features, formes listées en docstring.
            last: si True, retourne aussi le vecteur penultimate (avant dropout).
            freeze: si True et last=True, détache le penultimate du graphe.

        Returns:
            - si last=False: logits [B, num_classes]
            - si last=True: (logits [B, num_classes], features [B, embDim])
        """
        x = self._to_BTC(x)  # [B, T, C]
        if self.lstm is None or self.fc is None:
            self._lazy_build(x.shape[2])  # C

        # LSTM
        x, _ = self.lstm(x)               # [B, T, H*dir]

        # Pooling -> penultimate (sans dropout, stable)
        penultimate = self._temporal_pool(x)  # [B, H*dir]

        # Logits comme avant (dropout puis fc)
        logits = self.fc(self.drop(penultimate))

        if last:
            feats = penultimate.detach() if freeze else penultimate
            return logits, feats

        return logits

    def get_embedding_dim(self):
        return self.embDim
