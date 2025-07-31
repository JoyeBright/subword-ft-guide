import pandas as pd

def prepare_vocab_src_tgt_files(wmt_path, medical_path,
                                 src_out_path, tgt_out_path,
                                 source_col="source", target_col="target",
                                 shuffle=True, seed=42):
    """
    Load WMT and Medical train.csv files, oversample Medical to match WMT size,
    and write source and target texts to separate files for tokenizer training.

    Parameters:
        wmt_path (str): Path to WMT CSV (must have source/target columns).
        medical_path (str): Path to Medical CSV (must have source/target columns).
        src_out_path (str): Output file path for source lines.
        tgt_out_path (str): Output file path for target lines.
        source_col (str): Name of the source language column.
        target_col (str): Name of the target language column.
        shuffle (bool): Whether to shuffle combined data before saving.
        seed (int): Random seed for reproducibility.
    """
    df_wmt = pd.read_csv(wmt_path)
    df_med = pd.read_csv(medical_path)

    n_wmt = len(df_wmt)
    n_med = len(df_med)

    if n_med == 0 or n_wmt == 0:
        raise ValueError(f"❌ Data error: WMT={n_wmt} samples, Medical={n_med} samples.")

    # Oversample medical to match WMT
    factor = (n_wmt // n_med) + 1
    df_med_os = pd.concat([df_med] * factor, ignore_index=True).sample(n=n_wmt, random_state=seed)

    # Merge
    df_combined = pd.concat([df_wmt, df_med_os], ignore_index=True)
    if shuffle:
        df_combined = df_combined.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Save to separate files
    df_combined[source_col].to_csv(src_out_path, index=False, header=False)
    df_combined[target_col].to_csv(tgt_out_path, index=False, header=False)

    print(f"✅ Saved source to {src_out_path}")
    print(f"✅ Saved target to {tgt_out_path}")
    print(f"   Total: {len(df_combined)} sentence pairs (WMT={n_wmt}, Medical={n_wmt} oversampled)")


prepare_vocab_src_tgt_files("data/wmt/test.csv", "data/medical/test.csv", "data/medical_combined/test.src", "data/medical_combined/test.tgt", seed=422)