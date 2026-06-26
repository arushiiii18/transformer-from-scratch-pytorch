import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformer import make_transformer
import time
import os


# ── Mask utilities ────────────────────────────────────────────────────────────

def make_src_mask(src: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Padding mask for encoder.
    Positions where src == pad_idx are masked out (set to 0).
    Shape: (batch, 1, 1, src_len) — broadcasts over heads and query positions.
    """
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Combined padding + causal mask for decoder.
    1. Padding mask: ignore pad tokens
    2. Causal mask: position i cannot attend to j > i

    Both masks ANDed together.
    Shape: (batch, 1, tgt_len, tgt_len)
    """
    tgt_len = tgt.size(1)
    # Padding mask: (batch, 1, 1, tgt_len)
    pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
    # Causal mask: (1, 1, tgt_len, tgt_len)
    causal_mask = torch.tril(torch.ones(1, 1, tgt_len, tgt_len, device=tgt.device)).bool()
    # AND: a position is valid only if it's not padding AND not future
    return pad_mask & causal_mask


# ── Learning rate schedule ─────────────────────────────────────────────────────

class WarmupScheduler:
    """
    Learning rate schedule from Section 5.3 of the paper.

    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))

    Why warmup?
    At the start, weights are random. A large lr causes chaotic updates.
    Warmup linearly increases lr for warmup_steps, then decays it.
    This stabilizes early training when the model is most sensitive.
    """

    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self._compute_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def _compute_lr(self):
        s = self.step_num
        w = self.warmup_steps
        return (self.d_model ** -0.5) * min(s ** -0.5, s * w ** -1.5)


# ── Label smoothing loss ───────────────────────────────────────────────────────

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing from Section 5.4 of the paper. epsilon=0.1

    Instead of training with hard targets (one-hot),
    we smooth: 0.9 probability on correct token, 0.1 spread over vocab.

    Why? Hard targets are overconfident. The model wastes capacity
    becoming extremely confident on training examples.
    Label smoothing improves generalization and BLEU scores.
    """

    def __init__(self, vocab_size: int, pad_idx: int = 0, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred   : (batch * tgt_len, vocab_size) — logits
        target : (batch * tgt_len,) — token indices
        """
        vocab_size = pred.size(-1)

        # Build smooth distribution
        with torch.no_grad():
            smooth_target = torch.zeros_like(pred)
            smooth_target.fill_(self.smoothing / (vocab_size - 2))  # -2: correct + pad
            smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
            smooth_target[:, self.pad_idx] = 0  # never predict pad

        # Mask padding positions entirely
        pad_mask = (target == self.pad_idx)
        smooth_target[pad_mask] = 0

        log_prob = torch.log_softmax(pred, dim=-1)
        loss = -(smooth_target * log_prob).sum(dim=-1)
        non_pad = (~pad_mask).sum()
        return loss.sum() / non_pad.clamp(min=1)


# ── Training loop ──────────────────────────────────────────────────────────────

def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    total_tokens = 0

    for batch_idx, (src, tgt) in enumerate(dataloader):
        src = src.to(device)
        tgt = tgt.to(device)

        # Teacher forcing: feed tgt[:-1] as input, predict tgt[1:]
        # i.e. given <sos> w1 w2 ... wN, predict w1 w2 ... <eos>
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask = make_src_mask(src)
        tgt_mask = make_tgt_mask(tgt_input)

        # Forward pass
        logits = model(src, tgt_input, src_mask, tgt_mask)

        # Reshape for loss: (batch * tgt_len, vocab_size)
        logits_flat = logits.reshape(-1, logits.size(-1))
        tgt_flat = tgt_output.reshape(-1)

        loss = criterion(logits_flat, tgt_flat)

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping — prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        lr = scheduler.step()

        non_pad_tokens = (tgt_output != 0).sum().item()
        total_loss += loss.item() * non_pad_tokens
        total_tokens += non_pad_tokens

        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx:4d} | Loss {loss.item():.4f} | LR {lr:.6f}")

    return total_loss / total_tokens


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            src_mask = make_src_mask(src)
            tgt_mask = make_tgt_mask(tgt_input)

            logits = model(src, tgt_input, src_mask, tgt_mask)
            logits_flat = logits.reshape(-1, logits.size(-1))
            tgt_flat = tgt_output.reshape(-1)

            loss = criterion(logits_flat, tgt_flat)
            non_pad_tokens = (tgt_output != 0).sum().item()
            total_loss += loss.item() * non_pad_tokens
            total_tokens += non_pad_tokens

    return total_loss / total_tokens


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data import build_dataloaders  # we'll write this next

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters — paper values scaled down for Multi30k
    D_MODEL = 256
    H = 8
    N = 3
    D_FF = 512
    DROPOUT = 0.1
    BATCH_SIZE = 64
    EPOCHS = 15
    WARMUP_STEPS = 4000

    train_loader, val_loader, src_vocab, tgt_vocab = build_dataloaders(BATCH_SIZE)

    model = make_transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=D_MODEL,
        h=H,
        N=N,
        d_ff=D_FF,
        dropout=DROPOUT
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0,  # scheduler controls lr
        betas=(0.9, 0.98),
        eps=1e-9  # paper Section 5.3 exact values
    )
    scheduler = WarmupScheduler(optimizer, D_MODEL, WARMUP_STEPS)
    criterion = LabelSmoothingLoss(len(tgt_vocab), pad_idx=0, smoothing=0.1)

    best_val_loss = float('inf')
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - start

        print(f"Epoch {epoch:2d} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "checkpoints/best_model.pt")
            print(f"  Saved best model.")