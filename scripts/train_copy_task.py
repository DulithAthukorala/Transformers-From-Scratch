import random
import torch
import torch.nn as nn

from src.model.transformer import Transformer


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def token_accuracy(logits, targets, pad_id=0):
    pred = logits.argmax(dim=-1)  # (B, L)
    mask = (targets != pad_id)
    correct = (pred == targets) & mask
    return correct.sum().item() / mask.sum().item()


def make_fixed_copy_dataset(train_size, seq_len, vocab_size, bos_id=1):
    # src tokens in [2, vocab_size-1]
    src = torch.randint(2, vocab_size, (train_size, seq_len), dtype=torch.long)

    bos = torch.full((train_size, 1), bos_id, dtype=torch.long)
    trg_full = torch.cat([bos, src], dim=1)  # (train_size, seq_len+1)

    trg_in = trg_full[:, :-1]   # (train_size, seq_len)
    trg_out = trg_full[:, 1:]   # (train_size, seq_len)  == src

    return src, trg_in, trg_out


def main():
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # --- Task settings (make it easy to memorize) ---
    vocab_size = 20
    pad_id = 0
    bos_id = 1

    seq_len = 6
    train_size = 512
    batch_size = 256

    # fixed dataset (THIS is what makes it an overfit proof)
    train_src, train_trg_in, train_trg_out = make_fixed_copy_dataset(
        train_size=train_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        bos_id=bos_id,
    )

    # --- Model settings ---
    model = Transformer(
        src_vocab_size=vocab_size,
        trg_vocab_size=vocab_size,
        src_pad_idx=pad_id,
        trg_pad_idx=pad_id,
        embed_size=128,
        num_layers=3,
        heads=4,
        forward_expansion=4,
        dropout=0.0,
        device=device,
        max_length=64,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    steps = 1000
    print_every = 200

    model.train()
    for step in range(1, steps + 1):
        idx = torch.randint(0, train_size, (batch_size,))

        src = train_src[idx].to(device)
        trg_in = train_trg_in[idx].to(device)
        trg_out = train_trg_out[idx].to(device)

        logits = model(src, trg_in)  # (B, L, V)

        loss = criterion(
            logits.reshape(-1, vocab_size),
            trg_out.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % print_every == 0 or step == 1:
            acc = token_accuracy(logits, trg_out, pad_id=pad_id)
            print(f"step {step:4d} | loss {loss.item():.4f} | token_acc {acc*100:.2f}%")

    print("done")


if __name__ == "__main__":
    main()