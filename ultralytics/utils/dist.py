# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import shutil
import socket
import sys
import tempfile

from . import USER_CONFIG_DIR
from .torch_utils import TORCH_1_9


def find_free_network_port() -> int:
    """
    Find a free port on localhost.

    It is useful in single-node training when we don't want to connect to a real main node but have to set the
    `MASTER_PORT` environment variable.

    Returns:
        (int): The available network port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]  # port


def generate_ddp_file(trainer):
    """Generates a DDP file and returns its file name."""
    module, name = f"{trainer.__class__.__module__}.{trainer.__class__.__name__}".rsplit(".", 1)
    overrides_dict = vars(trainer.args)
    model_url = getattr(trainer.hub_session, "model_url", trainer.args.model)
    # Build content via a list and join for improved performance with moderate string size
    content_lines = [
        "# Ultralytics Multi-GPU training temp file (should be automatically deleted after use)\n",
        f"overrides = {overrides_dict}\n",
        "\n",
        'if __name__ == "__main__":\n',
        f"    from {module} import {name}\n",
        "    from ultralytics.utils import DEFAULT_CFG_DICT\n",
        "\n",
        "    cfg = DEFAULT_CFG_DICT.copy()\n",
        "    cfg.update(save_dir='')   # handle the extra key 'save_dir'\n",
        f"    trainer = {name}(cfg=cfg, overrides=overrides)\n",
        f'    trainer.args.model = "{model_url}"\n',
        "    results = trainer.train()\n",
    ]
    content = "".join(content_lines)

    ddp_dir = _ensure_ddp_dir()
    # Use mode="w" (not w+) to avoid unnecessary read pointer efforts, as file is not read after write
    with tempfile.NamedTemporaryFile(
        prefix="_temp_",
        suffix=f"{id(trainer)}.py",
        mode="w",
        encoding="utf-8",
        dir=ddp_dir,
        delete=False,
    ) as file:
        file.write(content)
        return file.name


def generate_ddp_command(world_size, trainer):
    """
    Generate command for distributed training.

    Args:
        world_size (int): Number of processes to spawn for distributed training.
        trainer (object): The trainer object containing configuration for distributed training.

    Returns:
        cmd (List[str]): The command to execute for distributed training.
        file (str): Path to the temporary file created for DDP training.
    """

    if not trainer.resume:
        shutil.rmtree(trainer.save_dir)  # remove the save_dir
    file = generate_ddp_file(trainer)
    dist_cmd = "torch.distributed.run" if TORCH_1_9 else "torch.distributed.launch"
    port = find_free_network_port()
    cmd = [sys.executable, "-m", dist_cmd, "--nproc_per_node", f"{world_size}", "--master_port", f"{port}", file]
    return cmd, file


def ddp_cleanup(trainer, file):
    """Delete temp file if created."""
    if f"{id(trainer)}.py" in file:  # if temp_file suffix in file
        os.remove(file)


def _ensure_ddp_dir():
    # Helper to minimize repeated DDP dir existence checks
    ddp_path = USER_CONFIG_DIR / "DDP"
    # Use exist_ok=False for performance, ignore FileExistsError if necessary
    try:
        os.mkdir(ddp_path)
    except FileExistsError:
        pass
    return ddp_path
