# Master thesis (xlapsa00)

This is the accompanying python script for master thesis at Brno university of technology, Faculty of information technologies that includes a utility for running specific models with different settings. 

## Setup

This project requires Python 3.8. After installing Python, you can install the project's dependencies using `pip`. 

Here are the step by step instructions to setup the project:

1. Clone the repository:
   ```bash
   git clone https://github.com/TomasLapsansky/Master-thesis.git
   cd Master-thesis
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```

3. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

After setting up the project, you can run the script with different command-line arguments.

Here's the list of the arguments:

- `-m, --training_model`: Model name (case sensitive). This is a required argument.
- `-d, --dataset`: Dataset name (case sensitive).
- `-e, --eval`: Set if execute evaluation.
- `-r, --dropout`: Set dropout rate. Default is 0.5.
- `-t, --trained`: Use pre-trained model. Default is False.
- `-f, --frozen`: Freeze layers of base model until a specified layer.
- `--type`: Set type of efficient net.
- `--lr, --learning_rate`: Set learning rate for model. Default is 0.0001.
- `-c, --checkpoint`: Path to loaded checkpoint.
- `-p, --print`: Path to image.

Here is an example of how to run the script:

```bash
python main.py -m efficientdet --type L -d celeb-df -t
```

## License

This project is licensed under the terms of the MIT license.
