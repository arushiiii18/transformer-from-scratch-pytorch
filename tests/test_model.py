import torch
import pytest
from transformer import make_transformer


class TestTransformer:

    def setup_method(self):
        self.model = make_transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=128,  # small for test speed
            h=4,
            N=2,
            d_ff=256
        )

    def test_output_shape(self):
        src = torch.randint(0, 1000, (2, 10))
        tgt = torch.randint(0, 1000, (2, 7))
        out = self.model(src, tgt)
        assert out.shape == (2, 7, 1000)

    def test_make_src_mask_blocks_padding(self):
        """make_src_mask must return False for padding positions."""
        from train import make_src_mask
        src = torch.tensor([[1, 2, 3, 0, 0]])
        mask = make_src_mask(src, pad_idx=0)
        assert mask.shape == (1, 1, 1, 5)
        assert mask[0, 0, 0, 0].item() == True   # real token
        assert mask[0, 0, 0, 3].item() == False  # padding
        assert mask[0, 0, 0, 4].item() == False  # padding


    def test_make_tgt_mask_is_causal(self):
        """make_tgt_mask must block future positions."""
        from train import make_tgt_mask
        tgt = torch.tensor([[1, 2, 3, 4]])
        mask = make_tgt_mask(tgt, pad_idx=0)
        assert mask.shape == (1, 1, 4, 4)
        assert mask[0, 0, 0, 1].item() == False  # pos 0 cannot see pos 1
        assert mask[0, 0, 1, 0].item() == True   # pos 1 can see pos 0
        assert mask[0, 0, 3, 0].item() == True   # pos 3 can see pos 0
        assert mask[0, 0, 3, 2].item() == True   # pos 3 can see pos 2

    def test_make_tgt_mask_blocks_padding(self):
        """make_tgt_mask must block padding tokens."""
        from train import make_tgt_mask
        tgt = torch.tensor([[1, 2, 0, 0]])  # last two are padding
        mask = make_tgt_mask(tgt, pad_idx=0)
        assert mask[0, 0, 0, 2].item() == False  # padding blocked
        assert mask[0, 0, 0, 3].item() == False  # padding blocked

    def test_encoder_decoder_output_shapes(self):
        src = torch.randint(0, 1000, (3, 12))
        tgt = torch.randint(0, 1000, (3, 8))
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = torch.tril(torch.ones(1, 1, 8, 8))
        enc_out = self.model.encode(src, src_mask)
        dec_out = self.model.decode(tgt, enc_out, src_mask, tgt_mask)
        assert enc_out.shape == (3, 12, 128)
        assert dec_out.shape == (3, 8, 128)

    def test_parameter_count_reasonable(self):
        total = sum(p.numel() for p in self.model.parameters())
        assert total > 1_000_000, "Model seems too small"