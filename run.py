import subprocess

models = [
    ("neuralmind/bert-base-portuguese-cased", 32),
    ("neuralmind/bert-large-portuguese-cased", 32),
    ("PORTULAN/albertina-100m-portuguese-ptbr-encoder", 32),
    ("PORTULAN/albertina-900m-portuguese-ptbr-encoder", 32),
    ("PORTULAN/albertina-1b5-portuguese-ptbr-encoder-256", 32),
]

if __name__ == "__main__":
    print("Testing models")
    for model, batch_size in models:
        subprocess.run(
            [
                "torchrun",
                "--standalone",
                "--nproc_per_node=2",
                "dist.py",
                model,
                str(batch_size),
            ]
        )
