import os
import sys
import random
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# Add the project root to sys.path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.transformer import Transformer


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


@torch.no_grad()
def token_accuracy(logits, targets, pad_id=0):
    pred = logits.argmax(dim=-1)  # (B, L, V) -> (B, L)
    mask = targets != pad_id
    correct = (pred == targets) & mask
    return correct.sum().item() / mask.sum().item()


def make_fixed_copy_dataset(train_size, seq_len, vocab_size, bos_id=1):
    src = torch.randint(2, vocab_size, size=(train_size, seq_len), dtype=torch.long)
    bos = torch.full((train_size, 1), bos_id, dtype=torch.long)
    trg_full = torch.cat([bos, src], dim=1)
    trg_in = trg_full[:, :-1]
    trg_out = trg_full[:, 1:]
    return src, trg_in, trg_out

def get_or_create_dataset(path, train_size, seq_len, vocab_size, bos_id):
    """
    Truly fixed dataset:
    - if file exists: load it (same tokens every run)
    - else: create once and save
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        data = torch.load(path)
        return data["src"], data["trg_in"], data["trg_out"]

    src, trg_in, trg_out = make_fixed_copy_dataset(train_size, seq_len, vocab_size, bos_id)
    torch.save({"src": src, "trg_in": trg_in, "trg_out": trg_out}, path)
    return src, trg_in, trg_out


def get_or_create_dataset(path, train_size, seq_len, vocab_size, bos_id):
    """
    Truly fixed dataset:
    - if file exists: load it (same data every run)
    - else: create once and save
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        data = torch.load(path)
        return data["src"], data["trg_in"], data["trg_out"]

    src, trg_in, trg_out = make_fixed_copy_dataset(train_size, seq_len, vocab_size, bos_id)
    torch.save({"src": src, "trg_in": trg_in, "trg_out": trg_out}, path)
    return src, trg_in, trg_out


@torch.no_grad()
def greedy_decode(model, src, bos_id, max_len):
    model.eval()
    B = src.shape[0]
    device = src.device

    trg_in = torch.full((B, 1), bos_id, dtype=torch.long, device=device)

    for _ in range(max_len - 1):
        logits = model(src, trg_in)             # (B, cur_len, V)
        next_token = logits[:, -1, :].argmax(-1)  # (B,)
        trg_in = torch.cat([trg_in, next_token.unsqueeze(1)], dim=1)

    return trg_in


@torch.no_grad()
def log_attention_heatmap(writer, model, src, step, tag="attn/encoder_layer0_head0"):
    model.eval()

    try:
        if not hasattr(model, "encoder") or not hasattr(model, "make_src_mask"):
            return False

        encoder = model.encoder
        if not hasattr(encoder, "word_embedding") or not hasattr(encoder, "position_embedding"):
            return False
        if not hasattr(encoder, "layers") or len(encoder.layers) == 0:
            return False

        B, L = src.shape
        device = src.device
        positions = torch.arange(0, L, device=device).unsqueeze(0).expand(B, L)

        x = encoder.word_embedding(src) + encoder.position_embedding(positions)
        if hasattr(encoder, "dropout"):
            x = encoder.dropout(x)

        src_mask = model.make_src_mask(src)
        layer0 = encoder.layers[0]

        out = layer0(x, x, x, src_mask, return_attention=True)
        if isinstance(out, (tuple, list)) and len(out) == 2:
            _, attn = out
            if torch.is_tensor(attn) and attn.dim() == 4:
                attn_img = attn[0, 0].detach().float().clamp(0, 1)  # (L, L)
                writer.add_image(tag, attn_img.unsqueeze(0), step, dataformats="CHW")
                return True
    except Exception:
        pass

    return False


def main():
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    writer = SummaryWriter("runs/overfit_proof")

    # ---- Controls ----
    force_retrain = False  # set True if you want to ignore checkpoint and start fresh
    extra_steps_after_resume = 1000

    # ---- Task config ----
    vocab_size = 20
    pad_id = 0
    bos_id = 1

    seq_len = 6
    train_size = 512
    batch_size = 256

    dataset_path = "data/fixed_copy_task.pt"

    train_src, train_trg_in, train_trg_out = get_or_create_dataset(
        dataset_path,
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
    ckpt_latest = "checkpoints/overfit_proof_latest.pt"

    if (not force_retrain) and os.path.exists(ckpt_latest):
        start_step = load_checkpoint(ckpt_latest, model, optimizer, device)
        print(f"Resumed from step {start_step}")

    # ✅ always trains extra after resume
    steps = max(1000, start_step + 1000)
    print_every = 200
    best_acc = 0.0

    model.train()

    for step in range(start_step + 1, steps + 1):
        idx = torch.randint(0, train_size, (batch_size,))

        src = train_src[idx].to(device)
        trg_in = train_trg_in[idx].to(device)
        trg_out = train_trg_out[idx].to(device)

        logits = model(src, trg_in)  # (B, L, V)

        loss = criterion(logits.reshape(-1, vocab_size), trg_out.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        acc = token_accuracy(logits, trg_out, pad_id=pad_id)

        writer.add_scalar("train/loss", loss.item(), step)
        writer.add_scalar("train/token_acc", acc, step)

        if step % print_every == 0 or step == start_step + 1:
            print(f"step {step:4d} | loss {loss.item():.4f} | token_acc {acc*100:.2f}%")

        if step % 200 == 0:
            save_checkpoint(ckpt_latest, model, optimizer, step)

        if acc > best_acc:
            best_acc = acc
            save_checkpoint("checkpoints/overfit_proof_best.pt", model, optimizer, step)

        if step in {start_step + 1, 200, 600, 1000, steps}:
            ok = log_attention_heatmap(writer, model, src, step)
            if not ok and step == start_step + 1:
                print("note: attention heatmap not logged (model didn't expose attention without refactor)")

    save_checkpoint("checkpoints/overfit_proof_final.pt", model, optimizer, step)

    # ---- Greedy decoding proof ----
    model.eval()
    demo_B = 4
    src_demo = train_src[:demo_B].to(device)
    gen = greedy_decode(model, src_demo, bos_id=bos_id, max_len=seq_len + 1)

    print("\n--- Greedy decode demo ---")
    for i in range(demo_B):
        src_tokens = src_demo[i].tolist()
        pred_tokens = gen[i, 1:].tolist()
        print(f"src : {src_tokens}")
        print(f"pred: {pred_tokens}\n")

    writer.close()
    print("done")


if __name__ == "__main__":
    main()