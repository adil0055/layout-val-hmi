# SAM 3 server for layoutval

Runs Meta's SAM 3 on the computer with the GPU and answers layoutval over HTTP:
a photograph and a text prompt in, the masks SAM 3 finds out. layoutval uses
the mask to find the display — any shape, round ones included — in place of
tapped corners or the chessboard. Only `sam3_server.py` goes on that computer.

## Install (GPU computer)

Needs Python 3.12+, an NVIDIA GPU with CUDA 12.6+, PyTorch 2.7+.

```bash
conda create -n sam3 python=3.12 -y && conda activate sam3
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/facebookresearch/sam3.git
pip install -e ./sam3
pip install pillow numpy
```

The checkpoint is gated: open https://huggingface.co/facebook/sam3, press
**Request access**, then on this computer

```bash
hf auth login          # paste a Hugging Face read token
```

(or download `sam3.pt` once and pass `--checkpoint /path/to/sam3.pt`).

## Run

```bash
python sam3_server.py --port 8765 --token choose-a-secret
```

The first start downloads the checkpoint. Check it from the bench computer:

```bash
curl http://<gpu-computer>:8765/health
```

Open the port in the GPU computer's firewall if needed (Windows: allow
inbound TCP 8765; Ubuntu: `sudo ufw allow 8765/tcp`).

## Use it from layoutval (bench computer)

```bash
layoutval go --sam3-url http://<gpu-computer>:8765 --sam3-token choose-a-secret
```

On the phone or webcam page pick **SAM 3**, optionally type what to look for
(empty means `display screen`; for example `round instrument cluster` or
`car dashboard screen`), and shoot the Corners step as usual: the dots arrive
already placed, round or square, and can be dragged.

Options: `--device cpu` works without a GPU but takes tens of seconds a
photograph; `--fp32` turns off bfloat16 on older cards; `SAM3_TOKEN` can hold
the token instead of `--token`.

## Licence

SAM 3's code and weights are under Meta's SAM License: royalty-free, and the
grant does not exclude commercial use, but it has conditions (no use prohibited
by trade controls, keep the licence with any copy you pass on, acknowledge SAM
in publications). It is not an OSI open-source licence, so check it for your
use. It stays on this computer: layoutval only talks to it over HTTP and does
not include or import it. `sam3_server.py` itself needs only Pillow and NumPy
beside SAM 3.
