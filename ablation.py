import torch
import os
import json
from transformer import make_transformer
from train import train_epoch, evaluate, WarmupScheduler, LabelSmoothingLoss
from data import build_dataloaders


def run_ablation(config: dict, train_loader, val_loader, src_vocab, tgt_vocab, device):
    """Train a single configuration for a few epochs and return val loss."""
    model = make_transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config['d_model'],
        h=config['h'],
        N=config['N'],
        d_ff=config['d_ff'],
        dropout=0.1
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0,
        betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = WarmupScheduler(optimizer, config['d_model'], warmup_steps=4000)
    criterion = LabelSmoothingLoss(len(tgt_vocab), pad_idx=0, smoothing=0.1)

    results = []
    for epoch in range(1, config['epochs'] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"  Epoch {epoch} | Train {train_loss:.4f} | Val {val_loss:.4f}")
        results.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})

    return results


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    train_loader, val_loader, src_vocab, tgt_vocab = build_dataloaders(batch_size=64)

    # ── Ablation configurations ────────────────────────────────────────────────
    # We vary ONE parameter at a time, keep everything else fixed (control)
    # Base: d_model=256, h=8, N=3, d_ff=512, epochs=3

    experiments = [
        # Vary number of attention heads
        {'name': 'heads_1',  'd_model': 256, 'h': 1, 'N': 3, 'd_ff': 512, 'epochs': 3},
        {'name': 'heads_2',  'd_model': 256, 'h': 2, 'N': 3, 'd_ff': 512, 'epochs': 3},
        {'name': 'heads_4',  'd_model': 256, 'h': 4, 'N': 3, 'd_ff': 512, 'epochs': 3},
        {'name': 'heads_8',  'd_model': 256, 'h': 8, 'N': 3, 'd_ff': 512, 'epochs': 3},  # baseline

        # Vary number of layers
        {'name': 'layers_1', 'd_model': 256, 'h': 8, 'N': 1, 'd_ff': 512, 'epochs': 3},
        {'name': 'layers_2', 'd_model': 256, 'h': 8, 'N': 2, 'd_ff': 512, 'epochs': 3},
        {'name': 'layers_3', 'd_model': 256, 'h': 8, 'N': 3, 'd_ff': 512, 'epochs': 3},  # baseline
        {'name': 'layers_6', 'd_model': 256, 'h': 8, 'N': 6, 'd_ff': 512, 'epochs': 3},

        # Vary FFN size
        {'name': 'dff_256',  'd_model': 256, 'h': 8, 'N': 3, 'd_ff': 256,  'epochs': 3},
        {'name': 'dff_512',  'd_model': 256, 'h': 8, 'N': 3, 'd_ff': 512,  'epochs': 3},  # baseline
        {'name': 'dff_1024', 'd_model': 256, 'h': 8, 'N': 3, 'd_ff': 1024, 'epochs': 3},
        {'name': 'dff_2048', 'd_model': 256, 'h': 8, 'N': 3, 'd_ff': 2048, 'epochs': 3},
    ]

    all_results = {}
    os.makedirs("ablation_results", exist_ok=True)

    for config in experiments:
        print(f"\n{'='*50}")
        print(f"Running: {config['name']}")
        print(f"Config: h={config['h']}, N={config['N']}, d_ff={config['d_ff']}")
        print(f"{'='*50}")
        results = run_ablation(config, train_loader, val_loader, src_vocab, tgt_vocab, device)
        all_results[config['name']] = {'config': config, 'results': results}

    # Save results
    with open("ablation_results/results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n\nAblation Summary (Final Val Loss):")
    print(f"{'Config':<12} {'h':>4} {'N':>4} {'d_ff':>6} {'Val Loss':>10}")
    print("-" * 40)
    for name, data in all_results.items():
        c = data['config']
        final_val = data['results'][-1]['val_loss']
        print(f"{name:<12} {c['h']:>4} {c['N']:>4} {c['d_ff']:>6} {final_val:>10.4f}")

    print("\nResults saved to ablation_results/results.json")