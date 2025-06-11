# Configurable GAN Trainer

A Python application for training Generative Adversarial Networks (GANs) with configurable architectures and training parameters, featuring a Tkinter-based graphical user interface.

This project allows users to:

- Define Generator and Discriminator architectures via a JSON configuration file.
- Set training parameters (learning rates, loss functions, epochs, batch sizes) through the UI.
- Load and preview network architectures.
- Start, pause, resume, and switch active training between the Generator and Discriminator.
- Monitor training progress with a simple log.
- Visualize training losses.
- Generate sample images from the current Generator.

## Project Structure

GAN_Project_Root/
├── GAN/ # Main Python package for the GAN application
│ ├── init.py
│ ├── main.py # Tkinter GUI and main application logic
│ ├── controller.py # Orchestrates model building and training
│ ├── model_builder.py # Dynamically builds PyTorch models from config
│ ├── train_manager.py # Handles the GAN training loops and logic
│ ├── data_loader.py # Loads data (e.g., MNIST)
│ └── config.json # Example JSON configuration for network architecture & training
├── .venv/ # Virtual environment (managed by uv or venv)
├── data/ # Default directory for datasets (e.g., MNIST)
├── pyproject.toml # Project metadata and build configuration
├── requirements.txt # Project dependencies
├── README.md # This file
└── LICENSE # Project license

## Prerequisites

- Python 3.8+
- `uv` (or `pip` and `venv`) for environment and package management

## Setup and Installation

1.  **Clone the repository (if applicable):**

    ```bash
    # git clone <repository-url>
    # cd <repository-name>
    ```

2.  **Create and activate a virtual environment:**
    Using `uv`:

    ```bash
    uv venv .venv
    source .venv/bin/activate
    ```

    Or using standard `venv`:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    Using `uv`:
    ```bash
    uv pip install -r requirements.txt
    ```
    Or using `pip`:
    ```bash
    pip install -r requirements.txt
    ```
    If you also want to install the project itself in editable mode (useful for development):
    ```bash
    uv pip install -e . # or pip install -e .
    ```

## Usage

1.  **Ensure you have a `config.json` file** (an example is provided in `GAN/config.json`) defining the GAN architecture and default training settings.
2.  **Run the application:**

    ```bash
    python -m GAN.main
    ```

    This will launch the Tkinter GUI.

3.  **Using the GUI:**
    - **Network Preview Tab:** Load your `config.json` to see a textual representation of the Generator and Discriminator architectures.
    - **Training Parameters Tab:**
      - Adjust learning rates, loss functions, epochs, and batch sizes for both Generator and Discriminator.
      - Select the data folder (defaults to `./data` for MNIST).
      - Choose which network (Generator or Discriminator) should be active initially.
      - Use the control buttons (Start, Pause, Resume, Switch, Stop) to manage the training process.
      - Training logs will appear in the text area.
    - **Summary & Generation Tab:**
      - View a summary of the loaded configuration.
      - Plot the training losses (Generator vs. Discriminator).
      - Generate a sample image using the current state of the Generator.

## Configuration (`config.json`)

The `config.json` file defines:

- `generator`: Input dimension, layer configurations (type, units/filters, activation, etc.), output size.
- `discriminator`: Input shape, layer configurations, output size.
- `training`: Default training parameters for generator and discriminator (loss function, learning rate, epochs, batch size), data folder path, and initial active network.

See `GAN/config.json` for a detailed example.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
