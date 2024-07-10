import os
from pathlib import Path
from typing import Optional

import pandas as pd

from config import CFG

# TODO: ADD SEED !


def get_data_train_and_data_irm(
    data_train_size: int = 60000,
    min_number_of_photo: int = 20,
    max_number_of_photo: Optional[int] = None,
    train_reset_labels: bool = True,
):
    """
    min_number_of_photo - if persone have less photo, we skip this sample
    max_number_of_photo - take NO MORE than this number, if None take all photo
    """
    df_identity = pd.read_csv(CFG.identity_path, sep=" ", header=None).sort_values(by=0).reset_index(drop=True)
    img_col_name = "path"
    df_identity.columns = [img_col_name, "label"]
    cropped_imgs = os.listdir(CFG.img_folder_dst)
    data = pd.DataFrame({img_col_name: cropped_imgs})
    data["is_query"] = None
    data["is_gallery"] = None
    data = data.join(df_identity.set_index(img_col_name), on=img_col_name, how="left")
    data[img_col_name] = data[img_col_name].map(lambda x: Path(x))
    data_label_count = (
        data.groupby(["label"])
        .agg({"label": "count"})
        .rename(columns={"label": "label_count"})
        .sort_values(by="label_count", ascending=False)
    )
    print(f"Число уникальных людей {len(data_label_count)}. Всего фото {len(data)}")

    train_data_list = []
    train_data_counter = 0
    idx = 0
    data_label_index = data_label_count.index

    while train_data_counter < data_train_size:
        if max_number_of_photo:
            tmp = data[data["label"] == data_label_index[idx]]
            if len(tmp) > max_number_of_photo:
                train_data_list.append(tmp.sample(max_number_of_photo))
            else:
                train_data_list.append(tmp)

        else:
            train_data_list.append(data[data["label"] == data_label_index[idx]])

        train_data_counter += len(train_data_list[-1])
        idx += 1
        error_msg = f"Не хватает данных, уменmшите {min_number_of_photo} или задайте как None"
        assert idx < len(data_label_index), error_msg

    train_data = pd.concat(train_data_list).reset_index(drop=True)

    train_data_labels = train_data["label"].unique()
    if train_reset_labels:
        train_data_map_lables = {l: i for i, l in enumerate(train_data_labels)}
        train_data["label"] = train_data["label"].map(lambda x: train_data_map_lables[x])
    print(f"Датасет для тренировки содержит {len(train_data_labels)} людей")

    data_irm_index = data_label_index[idx:]
    mask = data["label"].isin(set(data_irm_index))
    data_irm = data[mask].reset_index(drop=True)
    print(f"data {len(data)} -> train_data {len(train_data)} data_irm {len(data_irm)}")

    train_data["split"] = "train"
    data_irm["split"] = "valid"

    return train_data, data_irm


def split_by_person(
    data_irm: pd.DataFrame,
    query_size: int = 40,
    distractors_size: int = 160,
    query_sample: int = 3,
    distractors_sample: int = 3,
):
    """
    query_size - amount unique person in query set
    distractors_size - amount unique person in distractors set

    """
    # TODO: add asserts
    data_label_count = (
        data_irm.groupby(["label"])
        .agg({"label": "count"})
        .rename(columns={"label": "label_count"})
        .sort_values(by="label_count", ascending=False)
    )

    query_label_set = data_label_count.head(query_size)
    data_label_count = data_label_count.drop(query_label_set.index)
    distractors_label_set = data_label_count.head(distractors_size)

    query_list = []
    distractors_list = []

    for label in query_label_set.index:
        label_data = data_irm[data_irm["label"] == label]
        query_list.append(label_data.sample(query_sample))

    for label in distractors_label_set.index:
        label_data = data_irm[data_irm["label"] == label]
        distractors_list.append(label_data.sample(distractors_sample))

    query_df = pd.concat(query_list).reset_index(drop=True)
    distractors_df = pd.concat(distractors_list).reset_index(drop=True)
    # проверка что датасеты не пересекаются
    assert bool(set(query_df["label"]) & set(distractors_df["label"])) is False

    print(f"query_df {len(query_df)} distractors_df {len(distractors_df)}")

    return query_df, distractors_df


def split_dataset_by_photo(df, label_col, num_val_samples_per_class):
    validation_data = []
    train_data = []
    test_data = []

    for label in df[label_col].unique():
        label_data = df[df[label_col] == label]

        val_samples = label_data.sample(num_val_samples_per_class)
        validation_data.append(val_samples)
        label_data = label_data.drop(val_samples.index)

        test_samples = label_data.sample(num_val_samples_per_class)
        test_data.append(test_samples)
        label_data = label_data.drop(test_samples.index)

        train_data.append(label_data)

    train_df = pd.concat(train_data).reset_index(drop=True)
    validation_df = pd.concat(validation_data).reset_index(drop=True)
    test_df = pd.concat(test_data).reset_index(drop=True)
    print(f"df({len(df)}) -> train({len(train_df)}) val({len(validation_df)}) test({len(test_df)})")

    return train_df, validation_df, test_df


def get_valid_dataset(
    data_irm: pd.DataFrame,
    valid_size: int = 2000,
    photos_in_one_sample: int = 6,
    is_query_amount: int = 2,
):
    data_label_count = (
        data_irm.groupby(["label"])
        .agg({"label": "count"})
        .rename(columns={"label": "label_count"})
        .sort_values(by="label_count", ascending=False)
    )

    result = []
    is_gallery_amount = photos_in_one_sample - is_query_amount
    counter = 0
    for label in data_label_count.index:
        label_data = data_irm[data_irm["label"] == label]
        sample = label_data.sample(photos_in_one_sample)
        sample["is_query"] = [True] * is_query_amount + [False] * is_gallery_amount
        sample["is_gallery"] = [False] * is_query_amount + [True] * is_gallery_amount
        result.append(sample)
        counter += len(sample)
        if counter >= valid_size:
            return pd.concat(result).reset_index(drop=True)
