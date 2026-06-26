import torch
from transformer import make_transformer
from data import build_dataloaders, tokenize, SOS_IDX, EOS_IDX, PAD_IDX


def greedy_decode(
    model,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    device: torch.device
) -> torch.Tensor:
    """
    Greedy decoding — always pick the highest probability token at each step.
    Not the best translation quality (beam search is better) but simple and
    sufficient to verify the model works correctly.

    Autoregressive: we generate one token at a time, feeding each
    predicted token back as input for the next step.
    """
    model.eval()
    with torch.no_grad():
        encoder_output = model.encode(src, src_mask)

        # Start with <sos>
        tgt = torch.tensor([[SOS_IDX]], dtype=torch.long, device=device)

        for _ in range(max_len):
            tgt_mask = torch.tril(
                torch.ones(1, 1, tgt.size(1), tgt.size(1), device=device)
            ).bool()

            decoder_output = model.decode(tgt, encoder_output, src_mask, tgt_mask)
            # Take last token's logits, pick argmax
            logits = model.projection(decoder_output[:, -1, :])
            next_token = logits.argmax(dim=-1).unsqueeze(0)

            tgt = torch.cat([tgt, next_token], dim=1)

            if next_token.item() == EOS_IDX:
                break

    return tgt.squeeze(0)


def translate(sentence: str, model, src_vocab, tgt_vocab, device, max_len=50) -> str:
    model.eval()
    tokens = tokenize(sentence)
    src_ids = [SOS_IDX] + src_vocab.encode(tokens) + [EOS_IDX]
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_mask = (src != PAD_IDX).unsqueeze(1).unsqueeze(2)

    output_ids = greedy_decode(model, src, src_mask, max_len, device)

    # Convert ids back to tokens, strip special tokens
    translated = []
    for idx in output_ids.tolist():
        if idx in (SOS_IDX, PAD_IDX):
            continue
        if idx == EOS_IDX:
            break
        translated.append(tgt_vocab.idx2token.get(idx, '<unk>'))

    return ' '.join(translated)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Rebuild vocab — must match exactly what was used during training
    _, _, src_vocab, tgt_vocab = build_dataloaders(batch_size=32)

    model = make_transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=256,
        h=8,
        N=3,
        d_ff=512,
        dropout=0.0  # disable dropout at inference
    ).to(device)

    model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=device))
    model.eval()
    print("Model loaded.\n")

    # Test sentences — German to English
    test_sentences = [
        "ein mann spielt gitarre .",
        "eine frau läuft durch den park .",
        "zwei kinder spielen im garten .",
        "ein hund rennt über das feld .",
    ]

    print("Translations (15 epochs, GPU, val loss 2.84):\n")
    for sentence in test_sentences:
        translation = translate(sentence, model, src_vocab, tgt_vocab, device)
        print(f"DE: {sentence}")
        print(f"EN: {translation}\n")