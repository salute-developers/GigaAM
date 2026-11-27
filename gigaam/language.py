"""Language tags and checkpoint-compatible conditioning layers."""

import json
from collections.abc import Mapping

import torch
from torch import nn


class LanguageEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.languages = tuple(cfg["languages"])
        self.slots = {code: i + 1 for i, code in enumerate(self.languages)}
        self.slots["auto"] = 0
        self.default = cfg.get("default_tag") or "auto"
        self.weight = nn.Parameter(torch.empty(cfg["num_slots"], cfg["embedding_dim"]))
        if cfg.get("init") == "zeros":
            nn.init.zeros_(self.weight)
        else:
            nn.init.normal_(self.weight, std=cfg.get("init_std", 0.02))

    def _vector(self, tag):
        if not isinstance(tag, Mapping):
            values = (
                json.loads(tag)
                if isinstance(tag, str) and tag.startswith("[")
                else ([tag] if isinstance(tag, str) else tag)
            )
            tag = {
                code: 1.0
                for value in values
                for code in value.strip().lower().split("-")
            }
        vector = self.weight.new_zeros(self.weight.shape[0])
        for code, value in tag.items():
            code = code.strip().lower()
            vector[self.slots["kk" if code == "kz" else code]] += float(value)
        return vector / vector.sum()

    def encode_tags(self, language, batch_size=1):
        language = self.default if language is None else language
        tags = (
            [language] * batch_size
            if isinstance(language, (str, Mapping))
            else language
        )
        return torch.stack([self._vector(tag) for tag in tags])

    def forward(self, language, batch_size=1):
        return self.forward_for_export(self.encode_tags(language, batch_size))

    def forward_for_export(self, vectors):
        return (vectors.unsqueeze(-1) * self.weight.unsqueeze(0)).sum(1)


class LanguageFiLM(nn.Module):
    def __init__(self, widths, embedding_dim, cfg):
        super().__init__()
        every = cfg.get("every_n", 4)
        placement = cfg.get("placement", "every_n")
        layers = range(len(widths))
        if placement == "custom":
            layers = cfg["layers"]
        elif placement != "all":
            layers = range(every - 1, len(widths), every)
        self.projections = nn.ModuleDict(
            {str(i): nn.Linear(embedding_dim, 2 * widths[i]) for i in layers}
        )

        for projection in self.projections.values():
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(self, hidden, index, embedding):
        if str(index) not in self.projections:
            return hidden
        gamma, beta = (
            self.projections[str(index)](embedding.to(hidden.dtype))
            .unsqueeze(1)
            .chunk(2, -1)
        )
        return hidden * (1 + gamma) + beta


class LanguageLateFusion(nn.Module):
    def __init__(self, width, embedding_dim, cfg):
        super().__init__()
        self.kernel = nn.Sequential(
            nn.Linear(width + embedding_dim, cfg.get("hidden", 2048)),
            nn.ReLU(),
            nn.Dropout(cfg.get("dropout", 0.0)),
            nn.Linear(cfg.get("hidden", 2048), width),
        )

        nn.init.zeros_(self.kernel[-1].weight)
        nn.init.zeros_(self.kernel[-1].bias)

    def forward(self, encoded, embedding):
        with torch.autocast(device_type=encoded.device.type, enabled=False):
            encoded = encoded.float()
            frames = encoded.transpose(1, 2)
            language = embedding.float().unsqueeze(1).expand(-1, frames.shape[1], -1)
            return encoded + self.kernel(torch.cat((frames, language), -1)).transpose(
                1, 2
            )
