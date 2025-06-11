import gradio as gr
import os
import json
import time
import threading # For UI actions like plotting that might block Gradio's main processing
from PIL import Image # Gradio uses PIL Images
import torch # For generating image from tensor

# Assuming your other GAN modules are in the same directory or a package
from .controller import GANController 
from .model_builder import detect_gpu 
# data_loader and train_manager are used by controller

# --- Global State Management ---
# This is a simple way to hold the controller; for more complex apps, consider classes or Gradio's State
APP_STATE = {
    "gan_controller": None,
    "loaded_config_data": None,
    "training_active_flag": False, # UI's perception of training state
    "log_messages": [],
    "last_generated_image": None,
    "current_g_lr": "0.0002", # To hold UI values
    "current_d_lr": "0.0002",
    "current_g_epochs": "100",
    "current_d_epochs": "100",
    "current_g_batch": "64",
    "current_d_batch": "64",
    "current_data_folder": os.path.abspath("./data"),
    "current_initial_network": "generator"
}

MAX_LOG_LINES = 200 # To prevent log box from growing indefinitely

# --- Helper Functions for UI Callbacks ---

def add_log(message):
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    full_message = f"{timestamp} - {message}"
    APP_STATE["log_messages"].append(full_message)
    if len(APP_STATE["log_messages"]) > MAX_LOG_LINES:
        APP_STATE["log_messages"] = APP_STATE["log_messages"][-MAX_LOG_LINES:]
    print(full_message) # Also print to console for debugging
    return "\n".join(APP_STATE["log_messages"])

def ui_training_callback(message):
    """This function will be called by the Trainer threads with updates."""
    # Gradio UI updates need to happen in functions returned by Gradio event handlers.
    # So, this callback will just add to the log queue.
    # A separate Gradio function will be used to periodically refresh the log display.
    add_log(message)
    # We can't directly return gr.Textbox.update here as this is called from a bg thread.

def load_configuration_file(file_obj):
    if APP_STATE["training_active_flag"]:
        add_log("ERROR: Cannot load new configuration while training is active. Stop first.")
        return APP_STATE["loaded_config_data"], "\n".join(APP_STATE["log_messages"]), gr.Button.update(interactive=False) # Keep start disabled

    if file_obj is None:
        # Try loading default config.json from GAN package directory
        _config_dir = os.path.dirname(os.path.abspath(__file__))
        default_config_path = os.path.join(_config_dir, "config.json")
        if os.path.exists(default_config_path):
            add_log(f"No file uploaded. Attempting to load default: {default_config_path}")
            try:
                with open(default_config_path, "r") as f:
                    APP_STATE["loaded_config_data"] = json.load(f)
                add_log("Default configuration loaded successfully.")
            except Exception as e:
                add_log(f"Error loading default config: {e}")
                APP_STATE["loaded_config_data"] = None
        else:
            add_log("No file uploaded and no default config.json found in app directory.")
            APP_STATE["loaded_config_data"] = None
    else:
        try:
            APP_STATE["loaded_config_data"] = json.load(file_obj)
            add_log(f"Configuration loaded successfully from: {file_obj.name}")
        except Exception as e:
            add_log(f"Error loading or parsing configuration file: {e}")
            APP_STATE["loaded_config_data"] = None

    # Populate UI vars from loaded config (if successful)
    if APP_STATE["loaded_config_data"] and "training" in APP_STATE["loaded_config_data"]:
        training_conf = APP_STATE["loaded_config_data"]["training"]
        gen_train = training_conf.get("generator", {})
        disc_train = training_conf.get("discriminator", {})
        APP_STATE["current_g_lr"] = str(gen_train.get("learning_rate", APP_STATE["current_g_lr"]))
        APP_STATE["current_d_lr"] = str(disc_train.get("learning_rate", APP_STATE["current_d_lr"]))
        APP_STATE["current_g_epochs"] = str(gen_train.get("epochs", APP_STATE["current_g_epochs"]))
        APP_STATE["current_d_epochs"] = str(disc_train.get("epochs", APP_STATE["current_d_epochs"]))
        APP_STATE["current_g_batch"] = str(gen_train.get("batch_size", APP_STATE["current_g_batch"]))
        APP_STATE["current_d_batch"] = str(disc_train.get("batch_size", APP_STATE["current_d_batch"]))
        APP_STATE["current_data_folder"] = training_conf.get("data_folder", APP_STATE["current_data_folder"])
        APP_STATE["current_initial_network"] = training_conf.get("initial_network", APP_STATE["current_initial_network"])
        add_log("Training parameters populated from config.")

    preview_g = get_arch_preview("generator")
    preview_d = get_arch_preview("discriminator")
    summary = get_config_summary()
    
    start_btn_interactive = APP_STATE["loaded_config_data"] is not None and not APP_STATE["training_active_flag"]

    return (
        gr.Textbox.update(value=preview_g),
        gr.Textbox.update(value=preview_d),
        gr.Textbox.update(value=summary),
        # Update training param UI elements
        gr.Textbox.update(value=APP_STATE["current_g_lr"]),
        gr.Textbox.update(value=APP_STATE["current_d_lr"]),
        gr.Textbox.update(value=APP_STATE["current_g_epochs"]),
        gr.Textbox.update(value=APP_STATE["current_d_epochs"]),
        gr.Textbox.update(value=APP_STATE["current_g_batch"]),
        gr.Textbox.update(value=APP_STATE["current_d_batch"]),
        gr.Textbox.update(value=APP_STATE["current_data_folder"]),
        gr.Radio.update(value=APP_STATE["current_initial_network"]),
        # Log update
        gr.Textbox.update(value="\n".join(APP_STATE["log_messages"])),
        # Button states
        gr.Button.update(interactive=start_btn_interactive), # Start
        gr.Button.update(interactive=False), # Pause
        gr.Button.update(interactive=False), # Resume
        gr.Button.update(interactive=False), # Switch
        gr.Button.update(interactive=False)  # Stop
    )


def get_arch_preview(network_name):
    if not APP_STATE["loaded_config_data"]:
        return "No configuration loaded."
    section = APP_STATE["loaded_config_data"].get(network_name, {})
    if not section:
        return f"{network_name.capitalize()} section not found in config."
    text = f"{network_name.capitalize()} Architecture:\n"
    text += f"  Input: {section.get('input_dim' if network_name == 'generator' else 'input_shape', 'N/A')}\n"
    text += f"  Output Size (Info): {section.get('output_size', 'N/A')}\n"
    text += f"  Global Activation: {section.get('global_activation', 'N/A')}\n"
    text += "  Layers:\n"
    for i, layer_conf in enumerate(section.get("layers", [])):
        layer_desc = f"    {i+1}. Type: {layer_conf.get('type', '?')}"
        details = [f"{k}: {v}" for k, v in layer_conf.items() if k.lower() not in ['type']]
        if details: layer_desc += f" ({', '.join(details)})"
        text += layer_desc + "\n"
    return text

def get_config_summary():
    if not APP_STATE["loaded_config_data"]:
        return "No configuration loaded."
    summary = "=== Loaded Configuration Summary ===\n\n"
    summary += "-- Generator Arch --\n" + json.dumps(APP_STATE["loaded_config_data"].get("generator", {}), indent=2) + "\n\n"
    summary += "-- Discriminator Arch --\n" + json.dumps(APP_STATE["loaded_config_data"].get("discriminator", {}), indent=2) + "\n\n"
    summary += "-- Training Defaults (from file) --\n" + json.dumps(APP_STATE["loaded_config_data"].get("training", {}), indent=2) + "\n"
    return summary

def get_current_ui_training_config_for_controller(g_lr, d_lr, g_epochs, d_epochs, g_batch, d_batch, data_folder_ui, initial_net_ui):
    """Reads training parameters from UI fields and returns a config dict for the controller."""
    # Store current UI values in APP_STATE as well, so they persist if user changes tabs
    APP_STATE["current_g_lr"] = g_lr
    APP_STATE["current_d_lr"] = d_lr
    APP_STATE["current_g_epochs"] = g_epochs
    APP_STATE["current_d_epochs"] = d_epochs
    APP_STATE["current_g_batch"] = g_batch
    APP_STATE["current_d_batch"] = d_batch
    APP_STATE["current_data_folder"] = data_folder_ui
    APP_STATE["current_initial_network"] = initial_net_ui
    try:
        config = {
            "generator": {
                "loss_function": APP_STATE["loaded_config_data"]["training"]["generator"].get("loss_function", "BCEWithLogitsLoss"), # Get from loaded config
                "learning_rate": float(g_lr),
                "epochs": int(g_epochs),
                "batch_size": int(g_batch)
            },
            "discriminator": {
                "loss_function": APP_STATE["loaded_config_data"]["training"]["discriminator"].get("loss_function", "BCEWithLogitsLoss"), # Get from loaded config
                "learning_rate": float(d_lr),
                "epochs": int(d_epochs),
                "batch_size": int(d_batch)
            },
            "data_folder": data_folder_ui,
            "initial_network": initial_net_ui.lower()
        }
        return config
    except ValueError as e:
        add_log(f"ERROR: Invalid training parameter value: {e}. Check numeric fields.")
        return None
    except KeyError: # If loaded_config_data or training sections are missing
        add_log("ERROR: Loaded configuration is missing 'training' section or loss functions.")
        return None


def update_button_states_gradio():
    # This function determines what the interactive state of buttons should be.
    # It returns a list of gr.Button.update objects.
    # Order: Start, Pause, Resume, Switch, Stop
    if not APP_STATE["training_active_flag"]: # Not started or has been stopped
        # Start enabled only if config is loaded
        start_interactive = APP_STATE["loaded_config_data"] is not None
        return [
            gr.Button.update(interactive=start_interactive),
            gr.Button.update(interactive=False),
            gr.Button.update(interactive=False),
            gr.Button.update(interactive=False),
            gr.Button.update(interactive=False)
        ]
    else: # Training is active (could be running or paused)
        is_paused_for_active_net = True # Assume paused if controller/trainer not fully ready
        if APP_STATE["gan_controller"] and APP_STATE["gan_controller"].trainer:
            trainer = APP_STATE["gan_controller"].trainer
            active_net = trainer.current_network
            if active_net == "generator":
                is_paused_for_active_net = not trainer.pause_event_gen.is_set()
            elif active_net == "discriminator":
                is_paused_for_active_net = not trainer.pause_event_disc.is_set()
        
        return [
            gr.Button.update(interactive=False), # Start disabled
            gr.Button.update(interactive=not is_paused_for_active_net), # Pause enabled if running
            gr.Button.update(interactive=is_paused_for_active_net),  # Resume enabled if paused
            gr.Button.update(interactive=True),  # Switch always enabled when active
            gr.Button.update(interactive=True)   # Stop always enabled when active
        ]

def start_training_action(g_lr, d_lr, g_epochs, d_epochs, g_batch, d_batch, data_folder_ui, initial_net_ui):
    if APP_STATE["training_active_flag"]:
        add_log("INFO: Training is already active.")
        btns = update_button_states_gradio()
        return "\n".join(APP_STATE["log_messages"]), btns[0], btns[1], btns[2], btns[3], btns[4]

    if not APP_STATE["loaded_config_data"]:
        add_log("ERROR: No network configuration loaded. Please load a JSON config file first.")
        btns = update_button_states_gradio()
        return "\n".join(APP_STATE["log_messages"]), btns[0], btns[1], btns[2], btns[3], btns[4]

    current_ui_train_config = get_current_ui_training_config_for_controller(
        g_lr, d_lr, g_epochs, d_epochs, g_batch, d_batch, data_folder_ui, initial_net_ui
    )
    if not current_ui_train_config:
        btns = update_button_states_gradio()
        return "\n".join(APP_STATE["log_messages"]), btns[0], btns[1], btns[2], btns[3], btns[4]

    add_log("Initializing GAN Controller...")
    try:
        # Clean up old controller if exists (e.g., after a stop)
        if APP_STATE["gan_controller"]:
            if (APP_STATE["gan_controller"].generator_thread and APP_STATE["gan_controller"].generator_thread.is_alive()) or \
               (APP_STATE["gan_controller"].discriminator_thread and APP_STATE["gan_controller"].discriminator_thread.is_alive()):
                add_log("Waiting for previous threads to fully terminate before restarting...")
                APP_STATE["gan_controller"].signal_stop_training_threads()
                APP_STATE["gan_controller"].join_training_threads(timeout=1) # Quick join
            del APP_STATE["gan_controller"]
            APP_STATE["gan_controller"] = None

        APP_STATE["gan_controller"] = GANController(
            APP_STATE["loaded_config_data"].get("generator", {}),
            APP_STATE["loaded_config_data"].get("discriminator", {}),
            current_ui_train_config
        )
        add_log("Controller initialized. Starting training threads...")
        APP_STATE["gan_controller"].start_persistent_training(ui_training_callback)
        APP_STATE["training_active_flag"] = True
        add_log("Training threads initiated.")
    except Exception as e:
        APP_STATE["training_active_flag"] = False
        APP_STATE["gan_controller"] = None
        add_log(f"FATAL ERROR starting training: {e} {traceback.format_exc()}")
    
    btns = update_button_states_gradio()
    return "\n".join(APP_STATE["log_messages"]), btns[0], btns[1], btns[2], btns[3], btns[4]


def pause_action():
    if not APP_STATE["training_active_flag"] or not APP_STATE["gan_controller"]:
        add_log("Cannot pause: No active training.")
    else:
        active_net = APP_STATE["gan_controller"].trainer.current_network
        APP_STATE["gan_controller"].pause_training()
        add_log(f"Paused {active_net} training.")
    btns = update_button_states_gradio()
    return "\n".join(APP_STATE["log_messages"]), btns[0], btns[1], btns[2], btns[3], btns[4]

def resume_action(g_lr, d_lr, g_epochs, d_epochs, g_batch, d_batch, data_folder_ui, initial_net_ui): # Pass current UI params
    if not APP_STATE["training_active_flag"] or not APP_STATE["gan_controller"]:
        add_log("Cannot resume: No active training.")
    else:
        # Update controller's params from UI before resuming
        current_ui_train_config = get_current_ui_training_config_for_controller(
            g_lr, d_lr, g_epochs, d_epochs, g_batch, d_batch, data_folder_ui, initial_net_ui
        )
        if not current_ui_train_config: # Error already logged
            btns = update_button_states_gradio()
            return "\n".join(APP_STATE["log_messages"]), btns[0], btns[1], btns[2], btns[3], btns[4]
        
        APP_STATE["gan_controller"].update_runtime_train_params_from_ui(current_ui_train_config)
        
        active_net = APP_STATE["gan_controller"].trainer.current_network
        APP_STATE["gan_controller"].resume_training()
        add_log(f"Resumed {active_net} training with current UI parameters.")
    btns = update_button_states_gradio()
    return "\n".join(APP_STATE["log_messages"]), btns[0], btns[1], btns[2], btns[3], btns[4]

def switch_action():
    if not APP_STATE["training_active_flag"] or not APP_STATE["gan_controller"]:
        add_log("Cannot switch: No active training.")
    else:
        old_active = APP_STATE["gan_controller"].trainer.current_network
        APP_STATE["gan_controller"].switch_network()
        new_active = APP_STATE["gan_controller"].trainer.current_network
        add_log(f"Switched. Was: {old_active}, Now active: {new_active}.")
    btns = update_button_states_gradio()
    return "\n".join(APP_STATE["log_messages"]), btns[0], btns[1], btns[2], btns[3], btns[4]

def stop_action():
    if not APP_STATE["training_active_flag"] or not APP_STATE["gan_controller"]:
        add_log("Stop: No training was active.")
    else:
        add_log("Signaling training threads to stop...")
        APP_STATE["gan_controller"].signal_stop_training_threads()
        # The threads are daemons. For a cleaner stop, we might want to join,
        # but that could block the Gradio UI.
        # For Gradio, usually better to signal and let them die or clean up on app exit.
        # We can update the flag and buttons assuming they will stop.
        APP_STATE["training_active_flag"] = False
        add_log("Stop signal sent. Threads should terminate. Controller remains for generation until new start.")
        # Controller might need a full join in a cleanup function if app is closed.
    btns = update_button_states_gradio()
    return "\n".join(APP_STATE["log_messages"]), btns[0], btns[1], btns[2], btns[3], btns[4]


def generate_image_action():
    if not APP_STATE["gan_controller"] or not APP_STATE["gan_controller"].generator:
        add_log("ERROR: Generator not initialized. Load config and ensure controller is set up.")
        return APP_STATE["last_generated_image"], "\n".join(APP_STATE["log_messages"]) # Return previous image
    
    add_log("Generating image...")
    pil_img = None
    try:
        # This part should be quick enough for Gradio, but if G is huge, could be slow.
        # For very slow generation, a separate thread and polling mechanism might be needed for Gradio too.
        APP_STATE["gan_controller"].generator.eval()
        noise = torch.randn(1, APP_STATE["gan_controller"].latent_dim, device=APP_STATE["gan_controller"].device)
        with torch.no_grad():
            generated_output = APP_STATE["gan_controller"].generator(noise)
        
        img_tensor_raw = generated_output[0].cpu()
        img_tensor_norm = (img_tensor_raw + 1) / 2.0
        img_tensor_norm = img_tensor_norm.clamp(0, 1)
        
        from torchvision import transforms # Ensure this import
        pil_img = transforms.ToPILImage()(img_tensor_norm)
        pil_img = pil_img.resize((256, 256), Image.Resampling.NEAREST) # Larger for Gradio display
        APP_STATE["last_generated_image"] = pil_img
        add_log("Image generated.")
    except Exception as e:
        add_log(f"ERROR generating image: {e} {traceback.format_exc()}")
    
    return APP_STATE["last_generated_image"], "\n".join(APP_STATE["log_messages"])

def plot_losses_action():
    if not APP_STATE["gan_controller"] or not APP_STATE["gan_controller"].trainer:
        add_log("No training data to plot losses.")
        return None, "\n".join(APP_STATE["log_messages"])
    
    trainer = APP_STATE["gan_controller"].trainer
    if not trainer.gen_losses and not trainer.disc_losses:
        add_log("No loss data recorded yet.")
        return None, "\n".join(APP_STATE["log_messages"])

    import matplotlib.pyplot as plt # Import here for the function scope
    fig = plt.figure(figsize=(8, 4)) # Smaller for Gradio display
    if trainer.gen_losses:
        plt.plot(trainer.gen_losses, label="Generator Loss")
    if trainer.disc_losses:
        plt.plot(trainer.disc_losses, label="Discriminator Loss")
    plt.xlabel("Epoch (respective)")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()
    plt.grid(True)
    # fig.canvas.draw() # Not needed, Gradio handles rendering plot from figure object
    add_log("Loss plot generated.")
    return fig, "\n".join(APP_STATE["log_messages"])

def refresh_logs(): # Function to be called by gr. κάθε for periodic refresh
    return "\n".join(APP_STATE["log_messages"])

# --- Gradio UI Definition ---
with gr.Blocks(theme=gr.themes.Soft(), title="Interactive GAN Trainer") as demo:
    gr.Markdown("# Interactive GAN Training Framework")
    gr.Markdown(f"GPU Status: {detect_gpu()}")

    with gr.Tabs():
        with gr.TabItem("1. Configuration & Network Preview"):
            with gr.Row():
                config_file_upload = gr.File(label="Upload Configuration JSON", file_types=[".json"])
            with gr.Row():
                preview_g_textbox = gr.Textbox(label="Generator Architecture", lines=15, interactive=False)
                preview_d_textbox = gr.Textbox(label="Discriminator Architecture", lines=15, interactive=False)
        
        with gr.TabItem("2. Training Setup & Controls"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Generator Parameters")
                    g_lr_input = gr.Textbox(label="G Learning Rate", value=APP_STATE["current_g_lr"])
                    g_epochs_input = gr.Textbox(label="G Max Epochs", value=APP_STATE["current_g_epochs"])
                    g_batch_input = gr.Textbox(label="G Batch Size", value=APP_STATE["current_g_batch"])
                with gr.Column(scale=1):
                    gr.Markdown("### Discriminator Parameters")
                    d_lr_input = gr.Textbox(label="D Learning Rate", value=APP_STATE["current_d_lr"])
                    d_epochs_input = gr.Textbox(label="D Max Epochs", value=APP_STATE["current_d_epochs"])
                    d_batch_input = gr.Textbox(label="D Batch Size", value=APP_STATE["current_d_batch"])
            
            gr.Markdown("### Common Settings")
            data_folder_input = gr.Textbox(label="Data Folder Path", value=APP_STATE["current_data_folder"])
            initial_network_radio = gr.Radio(["generator", "discriminator"], label="Initial Active Network", value=APP_STATE["current_initial_network"])

            gr.Markdown("### Training Controls")
            with gr.Row():
                start_button = gr.Button("Start Training", variant="primary", interactive=False) # Starts disabled
                pause_button = gr.Button("Pause Active", interactive=False)
                resume_button = gr.Button("Resume Active", interactive=False)
                switch_button = gr.Button("Switch Active Net", interactive=False)
                stop_button = gr.Button("Stop All Training", variant="stop", interactive=False)
        
        with gr.TabItem("3. Output & Summary"):
            with gr.Row():
                summary_textbox = gr.Textbox(label="Full Configuration Summary", lines=20, interactive=False)
            with gr.Row():
                generate_image_button = gr.Button("Generate & Show Image")
                plot_losses_button = gr.Button("Plot Training Losses")
            with gr.Row():
                generated_image_display = gr.Image(label="Generated Image", type="pil", width=256, height=256)
                loss_plot_display = gr.Plot(label="Loss Curves")

    log_textbox = gr.Textbox(label="Training Log / Status", lines=15, interactive=False, autoscroll=True)

    # --- Event Handlers ---
    # Config loading and initial UI population
    config_file_upload.upload(
        fn=load_configuration_file,
        inputs=[config_file_upload],
        outputs=[
            preview_g_textbox, preview_d_textbox, summary_textbox,
            g_lr_input, d_lr_input, g_epochs_input, d_epochs_input, g_batch_input, d_batch_input,
            data_folder_input, initial_network_radio,
            log_textbox,
            start_button, pause_button, resume_button, switch_button, stop_button
        ]
    )
    
    # Collect all training param inputs for actions that need them
    training_param_inputs = [
        g_lr_input, d_lr_input, g_epochs_input, d_epochs_input,
        g_batch_input, d_batch_input, data_folder_input, initial_network_radio
    ]
    # Define outputs for button state updates
    button_state_outputs = [start_button, pause_button, resume_button, switch_button, stop_button]

    start_button.click(
        fn=start_training_action,
        inputs=training_param_inputs,
        outputs=[log_textbox] + button_state_outputs
    )
    pause_button.click(
        fn=pause_action,
        inputs=None, # Pause doesn't need current UI params, uses controller state
        outputs=[log_textbox] + button_state_outputs
    )
    resume_button.click(
        fn=resume_action,
        inputs=training_param_inputs, # Resume needs current UI params for potential LR updates
        outputs=[log_textbox] + button_state_outputs
    )
    switch_button.click(
        fn=switch_action,
        inputs=None,
        outputs=[log_textbox] + button_state_outputs
    )
    stop_button.click(
        fn=stop_action,
        inputs=None,
        outputs=[log_textbox] + button_state_outputs
    )

    generate_image_button.click(
        fn=generate_image_action,
        inputs=None,
        outputs=[generated_image_display, log_textbox]
    )
    plot_losses_button.click(
        fn=plot_losses_action,
        inputs=None,
        outputs=[loss_plot_display, log_textbox]
    )

    # Periodically refresh logs (e.g., every 2 seconds)
    # This is one way to get updates from background threads.
    demo.load(None, None, None, every=2, fn=lambda: gr.Textbox.update(value="\n".join(APP_STATE["log_messages"])), outputs=[log_textbox])


# --- Application Cleanup ---
def cleanup_on_exit():
    print("Gradio app attempting to shut down...")
    if APP_STATE["gan_controller"]:
        print("Signaling training threads to stop...")
        APP_STATE["gan_controller"].signal_stop_training_threads()
        print("Waiting for training threads to join (max 5s)...")
        APP_STATE["gan_controller"].join_training_threads(timeout=5)
        print("Threads joined or timed out.")
    print("Cleanup complete. Exiting.")

# Gradio doesn't have a direct on_closing hook like Tkinter.
# Cleanup for threads is best handled if the script is Ctrl+C'd or if Gradio offers
# a way to run code before full exit. Daemon threads will die with main process.
# For more robust cleanup, you might need to wrap the demo.launch() in a try/finally block
# or use atexit module, though atexit and threads can be tricky.

if __name__ == "__main__":
    import traceback # For more detailed error logging
    add_log("Gradio Application Starting...")
    # To make the default config load on startup if no file is immediately uploaded by user:
    # This requires the load_configuration_file to be callable without a file_obj
    # and to be able to update the necessary output components.
    # We can trigger it once with None.
    # initial_outputs = load_configuration_file(None) # Call it once to load default
    # This is tricky because Gradio expects outputs to match.
    # Better to let user click "Load" or upload a file for initial setup.

    try:
        demo.launch()
    except KeyboardInterrupt:
        print("Keyboard interrupt received...")
    finally:
        cleanup_on_exit()