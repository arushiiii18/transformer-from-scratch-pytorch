import torch
import sacrebleu
from tqdm import tqdm
from transformer import make_transformer
from data import build_dataloaders, tokenize, SOS_IDX, EOS_IDX, PAD_IDX
from inference import translate
from inference_beam import translate_beam


def evaluate_bleu(model, val_loader, src_vocab, tgt_vocab, device, beam_size=4):
    """
    Evaluate translation quality on the full validation set.
    Reports corpus BLEU — a single score over all sentences together,
    not averaged per-sentence. Corpus BLEU is standard in MT research.

    Why corpus BLEU not average sentence BLEU?
    Short sentences get disproportionately penalized by sentence BLEU.
    Corpus BLEU pools all n-gram matches before computing the score,
    giving a more stable and meaningful metric.
    """
    model.eval()

    greedy_hypotheses = []
    beam_hypotheses = []
    references = []

    print("Running inference on validation set...")

    # Rebuild raw val sentences for reference
    from data import read_file, tokenize as tok
    val_src = read_file('data_files/val.de')
    val_tgt = read_file('data_files/val.en')

    for src_sentence, tgt_sentence in tqdm(zip(val_src, val_tgt), total=len(val_src)):
        # Greedy translation
        greedy_out = translate(src_sentence, model, src_vocab, tgt_vocab, device)
        greedy_hypotheses.append(greedy_out)

        # Beam search translation
        beam_out = translate_beam(src_sentence, model, src_vocab, tgt_vocab, device, beam_size=beam_size)
        beam_hypotheses.append(beam_out)

        # Reference — lowercase to match model output
        references.append(tgt_sentence.lower().strip())

    # SacreBLEU expects references as list of lists
    refs = [references]

    greedy_bleu = sacrebleu.corpus_bleu(greedy_hypotheses, refs)
    beam_bleu = sacrebleu.corpus_bleu(beam_hypotheses, refs)

    print("\n" + "="*50)
    print("BLEU Evaluation Results")
    print("="*50)
    print(f"Greedy Decoding : {greedy_bleu.score:.2f}")
    print(f"Beam Search k=4 : {beam_bleu.score:.2f}")
    print("="*50)
    print("\nNote on scores:")
    print("  - Multi30k is a clean, constrained dataset (29k pairs)")
    print("  - Whitespace tokenization (BPE would improve further)")
    print("  - Small model (d_model=256, N=3 vs paper's 512, N=6)")
    print("  - Paper reports 27.3 BLEU on WMT (4.5M pairs, harder task)")

    return greedy_bleu.score, beam_bleu.score


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    _, val_loader, src_vocab, tgt_vocab = build_dataloaders(batch_size=32)

    model = make_transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=256, h=8, N=3, d_ff=512, dropout=0.0
    ).to(device)

    model.load_state_dict(torch.load('checkpoints/best_model.pt', map_location=device))
    model.eval()
    print("Model loaded.\n")

    evaluate_bleu(model, val_loader, src_vocab, tgt_vocab, device, beam_size=4)