import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import html

def clean_text(x: str) -> str:
    x = str(x)

    x = html.unescape(x)
    x = re.sub(r"<[^>]+>", " ", x)

    x = x.replace("\n", " ").replace("\r", " ")
    x = " ".join(x.split()).strip()

    return x

def main(input_csv: str, text_column: str, out_dir: str, test_size: float, seed: int):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    # Auto-build combined_description if requested
    if text_column == "combined_description":
        required = ["about_the_game", "detailed_description"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Cannot build 'combined_description'. Missing columns: {missing}")

        df["about_the_game"] = df["about_the_game"].fillna("")
        df["detailed_description"] = df["detailed_description"].fillna("")
        df["combined_description"] = (
            df["about_the_game"].astype(str) + " " + df["detailed_description"].astype(str)
        ).map(clean_text)
    elif text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found. The columns: {list(df.columns)}")

    texts = df[text_column].dropna().astype(str).map(clean_text)
    texts = texts[texts.str.len() > 20]
    texts = texts.drop_duplicates().reset_index(drop=True)

    train_texts, val_texts = train_test_split(
        texts, test_size=test_size, random_state=seed, shuffle=True
    )

    all_file = out_path / "all_descriptions.txt"
    train_file = out_path / "train.txt"
    val_file = out_path / "validation.txt"

    all_file.write_text("\n".join(texts.tolist()), encoding="utf-8")
    train_file.write_text("\n".join(train_texts.tolist()), encoding="utf-8")
    val_file.write_text("\n".join(val_texts.tolist()), encoding="utf-8")

    print(f"Total: {len(texts)}")
    print(f"Train: {len(train_texts)}")
    print(f"Validation: {len(val_texts)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--text_column", type=str, default="description")
    parser.add_argument("--out_dir", type=str, default="data")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1102)
    args = parser.parse_args()

    main(
        input_csv=args.input_csv,
        text_column=args.text_column,
        out_dir=args.out_dir,
        test_size=args.test_size,
        seed=args.seed,
    )