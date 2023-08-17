import os
import zipfile
import pickle
from glob import glob
from pathlib import Path

import pandas as pd
import gradio as gr

from indexrl.training import (
    DynamicBuffer,
    create_model,
    save_model,
    explore,
    train_iter,
)
from indexrl.environment import IndexRLEnv
from indexrl.utils import get_n_channels, state_to_expression

data_dir = "data/"
os.makedirs(data_dir, exist_ok=True)

meta_data_file = os.path.join(data_dir, "metadata.csv")
if not os.path.exists(meta_data_file):
    with open(meta_data_file, "w") as fp:
        fp.write("Name,Channels,Path\n")


def save_dataset(name, zip):
    with zipfile.ZipFile(zip.name, "r") as zip_ref:
        data_path = os.path.join(data_dir, name)
        zip_ref.extractall(data_path)

    img_path = glob(os.path.join(data_path, "images", "*.npy"))[0]
    n_channels = get_n_channels(img_path)

    with open(meta_data_file, "a") as fp:
        fp.write(f"{name},{n_channels},{data_path}\n")
    meta_data_df = pd.read_csv(meta_data_file)
    return meta_data_df, gr.Dropdown.update(choices=meta_data_df["Name"].to_list())


def find_expression(dataset_name: str):
    meta_data_df = pd.read_csv(meta_data_file, index_col="Name")
    n_channels = meta_data_df["Channels"][dataset_name]
    data_dir = meta_data_df["Path"][dataset_name]

    image_dir = os.path.join(data_dir, "images")
    mask_dir = os.path.join(data_dir, "masks")

    cache_dir = os.path.join(data_dir, "cache")
    logs_dir = os.path.join(data_dir, "logs")
    models_dir = os.path.join(data_dir, "models")
    for dir_name in (cache_dir, logs_dir, models_dir):
        Path(dir_name).mkdir(parents=True, exist_ok=True)

    action_list = (
        list("()+-*/=") + ["sq", "sqrt"] + [f"c{c}" for c in range(n_channels)]
    )
    env = IndexRLEnv(action_list, 12)
    agent, optimizer = create_model(len(action_list))
    seen_path = os.path.join(cache_dir, "seen.pkl") if cache_dir else ""
    env.save_seen(seen_path)
    data_buffer = DynamicBuffer()

    i = 0
    while True:
        i += 1
        print(f"----------------\nIteration {i}")
        print("Collecting data...")
        data = explore(
            env.copy(),
            agent,
            image_dir,
            mask_dir,
            1,
            logs_dir,
            seen_path,
            n_iters=1000,
        )
        print(
            f"Data collection done. Collected {len(data)} examples. Buffer size = {len(data_buffer)}."
        )

        data_buffer.add_data(data)
        print(f"Buffer size new = {len(data_buffer)}.")

        agent, optimizer, loss = train_iter(agent, optimizer, data_buffer)
        print("Loss:", loss)

        i_str = str(i).rjust(3, "0")
        if models_dir:
            save_model(agent, f"{models_dir}/model_{i_str}_loss-{loss}.pt")
        if cache_dir:
            with open(f"{cache_dir}/data_buffer_{i_str}.pkl", "wb") as fp:
                pickle.dump(data_buffer, fp)

        with open(os.path.join(logs_dir, "tree_1.txt"), "r", encoding="utf-8") as fp:
            tree = fp.read()

        top_5 = data_buffer.get_top_n(5)
        top_5_str = "\n".join(
            map(
                lambda x: " ".join(state_to_expression(x[0], action_list))
                + " "
                + str(x[1]),
                top_5,
            )
        )

        yield tree, top_5_str


with gr.Blocks(title="IndexRL") as demo:
    gr.Markdown("# IndexRL")
    meta_data_df = pd.read_csv(meta_data_file)

    with gr.Tab("Find Expressions"):
        select_dataset = gr.Dropdown(
            label="Select Dataset",
            choices=meta_data_df["Name"].to_list(),
        )
        find_exp_btn = gr.Button("Find Expressions")
        stop_btn = gr.Button("Stop")
        best_exps = gr.Textbox(label="Best Expressions", interactive=False)
        out_exp_tree = gr.Textbox(label="Latest Expression Tree", interactive=False)

    with gr.Tab("Datasets"):
        dataset_upload = gr.File(label="Upload Data ZIP file")
        dataset_name = gr.Textbox(label="Dataset Name")
        dataset_upload_btn = gr.Button("Upload")

        dataset_table = gr.Dataframe(meta_data_df, label="Dataset Table")

    find_exp_event = find_exp_btn.click(
        find_expression, inputs=[select_dataset], outputs=[out_exp_tree, best_exps]
    )
    stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[find_exp_event])

    dataset_upload.upload(
        lambda x: ".".join(os.path.basename(x.orig_name).split(".")[:-1]),
        inputs=dataset_upload,
        outputs=dataset_name,
    )
    dataset_upload_btn.click(
        save_dataset,
        inputs=[dataset_name, dataset_upload],
        outputs=[dataset_table, select_dataset],
    )

demo.queue(concurrency_count=10).launch(debug=True)
