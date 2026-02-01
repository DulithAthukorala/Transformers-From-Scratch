import os
import random
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.model.transformer import Transformer


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)  # CPU randomness
    torch.cuda.manual_seed_all(seed)  # GPU randomness


def save_checkpoint(path, model, optimizer, step):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optim_state"])
    return ckpt["step"]


@torch.no_grad()  # Disable gradient calculation
def token_accuracy(logits, targets, pad_id=0):
    """
    Compute token-level accuracy, ignoring padding tokens.
    """
    pred = logits.argmax(dim=-1)  # (B, L, V) -> (B, L)
    mask = targets != pad_id  # True = valid token, False = padding
    correct = (pred == targets) & mask
    return correct.sum().item() / mask.sum().item()


def make_fixed_copy_dataset(train_size, seq_len, vocab_size, bos_id=1):
    """
    Create random numbers from 2 to vocab_size-1 as the source sequence
    and BOS + source sequence as the target sequence.
    """
    src = torch.randint(2, vocab_size, size=(train_size, seq_len), dtype=torch.long)
    bos = torch.full((train_size, 1), bos_id, dtype=torch.long)
    trg_full = torch.cat([bos, src], dim=1)  # (train_size, seq_len+1)
    trg_in = trg_full[:, :-1]  # (train_size, seq_len)
    trg_out = trg_full[:, 1:]  # (train_size, seq_len)
    return src, trg_in, trg_out


def main():
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # TensorBoard writer (logs go into runs/copy_task/)
    writer = SummaryWriter("runs/overfit_proof")

    vocab_size = 20
    pad_id = 0
    bos_id = 1

    seq_len = 6
    train_size = 512
    batch_size = 256

    train_src, train_trg_in, train_trg_out = make_fixed_copy_dataset(
        train_size=train_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        bos_id=bos_id,
    )

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
        pre_norm=True,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    start_step = 0
    ckpt_path = "checkpoints/overfit_proof_latest.pt"

    if os.path.exists(ckpt_path):
        start_step = load_checkpoint(ckpt_path, model, optimizer, device)
        print(f"Resumed from step {start_step}")

    steps = 1000
    print_every = 200
    best_acc = 0.0

    model.train()

    # ✅ resume-aware loop
    for step in range(start_step + 1, steps + 1):
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

        acc = token_accuracy(logits, trg_out, pad_id=pad_id)

        # TensorBoard logs (scalar curves)
        writer.add_scalar("train/loss", loss.item(), step)
        writer.add_scalar("train/token_acc", acc, step)

        if step % print_every == 0 or step == 1:
            print(f"step {step:4d} | loss {loss.item():.4f} | token_acc {acc*100:.2f}%")

        if step % 200 == 0:
            save_checkpoint("checkpoints/overfit_proof_latest.pt", model, optimizer, step)

        # ✅ fixed indentation + best checkpoint
        if acc > best_acc:
            best_acc = acc
            save_checkpoint("checkpoints/overfit_proof_best.pt", model, optimizer, step)

    # ✅ save final at the actual last step executed
    save_checkpoint("checkpoints/overfit_proof_final.pt", model, optimizer, step)

    writer.close()
    print("done")


if __name__ == "__main__":
    main()