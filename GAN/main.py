# main.py
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from .controller import GANController # Relative import
from .model_builder import detect_gpu # Relative import
import json
# import threading # Controller handles threads
from PIL import Image, ImageTk
import torch
import os # For default data path
from torchvision import transforms



class GanConfigurator:
    def __init__(self, root_window): # Renamed root to root_window for clarity
        self.root = root_window
        self.root.title("GAN Training Application")
        self.root.geometry("1000x750") # Slightly taller for stats
        
        self.data_folder_var = tk.StringVar(value=os.path.abspath("./data")) # Default to absolute path
        self.loaded_config_data = None # Stores entire loaded JSON (arch + default training)
        
        self.gan_controller = None # Initialized on start_training
        
        # UI callback for training messages
        self.training_ui_callback = lambda message: self.root.after(0, self.update_training_stats_text, message)
        
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.preview_frame = ttk.Frame(self.notebook)
        self.train_params_frame = ttk.Frame(self.notebook) # Renamed for clarity
        self.summary_frame = ttk.Frame(self.notebook)
        
        self.notebook.add(self.preview_frame, text="Network Preview")
        self.notebook.add(self.train_params_frame, text="Training Parameters")
        self.notebook.add(self.summary_frame, text="Summary & Generation")
        
        self.build_preview_tab()
        self.build_training_params_tab()
        self.build_summary_tab()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing) # Handle graceful exit

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to stop training and quit?"):
            if self.gan_controller:
                print("Attempting to stop training on close...")
                self.gan_controller.stop_training() # This will join threads
            self.root.destroy()

    def build_preview_tab(self):
        frame = self.preview_frame
        ttk.Label(frame, text="Network Architecture Preview (from loaded JSON)", 
                  font=("Arial", 14)).pack(pady=10)
        
        ttk.Button(frame, text="Load Configuration File (*.json)", command=self.load_json_configuration).pack(pady=10)
        
        preview_container = ttk.Frame(frame)
        preview_container.pack(fill="both", expand=True, padx=10, pady=5)
        preview_container.columnconfigure(0, weight=1)
        preview_container.columnconfigure(1, weight=1)

        gen_lf = ttk.LabelFrame(preview_container, text="Generator Architecture")
        gen_lf.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.gen_preview_text = tk.Text(gen_lf, height=15, width=60, state="disabled", wrap="word")
        self.gen_preview_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        disc_lf = ttk.LabelFrame(preview_container, text="Discriminator Architecture")
        disc_lf.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.disc_preview_text = tk.Text(disc_lf, height=15, width=60, state="disabled", wrap="word")
        self.disc_preview_text.pack(fill="both", expand=True, padx=5, pady=5)
    
    def load_json_configuration(self):
        file_path = filedialog.askopenfilename(title="Select Configuration File", 
                                               filetypes=[("JSON files", "*.json")])
        if not file_path: return
        try:
            with open(file_path, "r") as f:
                self.loaded_config_data = json.load(f)
            messagebox.showinfo("Success", "Configuration loaded successfully.")
            self.populate_preview_from_config()
            self.populate_training_params_from_config() # Populate UI fields
            self.update_summary_text_from_config() # Also update summary tab
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load or parse configuration: {e}")
            self.loaded_config_data = None

    def populate_preview_from_config(self):
        if not self.loaded_config_data: return
        
        def format_layers(config_section_name):
            section = self.loaded_config_data.get(config_section_name, {})
            text = f"{config_section_name.capitalize()}:\n"
            text += f"  Input: {section.get('input_dim' if config_section_name == 'generator' else 'input_shape', 'N/A')}\n"
            text += f"  Output Size (Informational): {section.get('output_size', 'N/A')}\n"
            text += f"  Global Activation: {section.get('global_activation', 'N/A')}\n"
            text += "  Layers:\n"
            for layer in section.get("layers", []):
                layer_desc = f"    - Type: {layer.get('type', '?')}"
                details = []
                for k, v in layer.items():
                    if k.lower() != 'type': details.append(f"{k}: {v}")
                if details: layer_desc += f" ({', '.join(details)})"
                text += layer_desc + "\n"
            return text

        self.gen_preview_text.config(state="normal")
        self.gen_preview_text.delete("1.0", tk.END)
        self.gen_preview_text.insert(tk.END, format_layers("generator"))
        self.gen_preview_text.config(state="disabled")
        
        self.disc_preview_text.config(state="normal")
        self.disc_preview_text.delete("1.0", tk.END)
        self.disc_preview_text.insert(tk.END, format_layers("discriminator"))
        self.disc_preview_text.config(state="disabled")

    def build_training_params_tab(self):
        frame = self.train_params_frame
        ttk.Label(frame, text="Live Training Parameters", font=("Arial", 14)).pack(pady=10)
        
        params_container = ttk.Frame(frame)
        params_container.pack(pady=10, fill="x", padx=20)
        
        # Helper to create entry fields
        def create_param_entry(parent, label_text, default_value_var):
            row_frame = ttk.Frame(parent)
            row_frame.pack(fill="x", pady=2)
            ttk.Label(row_frame, text=label_text, width=15).pack(side="left")
            entry = ttk.Entry(row_frame, textvariable=default_value_var, width=15)
            entry.pack(side="left", padx=5)
            return entry

        # Generator Params UI
        gen_lf = ttk.LabelFrame(params_container, text="Generator Training")
        gen_lf.pack(side="left", fill="y", expand=True, padx=10)
        self.gen_loss_var = tk.StringVar(value="BCEWithLogitsLoss")
        self.gen_lr_var = tk.StringVar(value="0.0002")
        self.gen_epochs_var = tk.StringVar(value="100") # No. of full passes for G
        self.gen_batch_size_var = tk.StringVar(value="64")

        ttk.Label(gen_lf, text="Loss Function:").pack(pady=(5,0))
        self.gen_loss_combo = ttk.Combobox(gen_lf, textvariable=self.gen_loss_var, 
                                           values=["BCEWithLogitsLoss", "MSELoss", "BCELoss"], width=18)
        self.gen_loss_combo.pack(pady=(0,5))
        create_param_entry(gen_lf, "Learning Rate:", self.gen_lr_var)
        create_param_entry(gen_lf, "Max Epochs:", self.gen_epochs_var)
        create_param_entry(gen_lf, "Batch Size:", self.gen_batch_size_var)

        # Discriminator Params UI
        disc_lf = ttk.LabelFrame(params_container, text="Discriminator Training")
        disc_lf.pack(side="left", fill="y", expand=True, padx=10)
        self.disc_loss_var = tk.StringVar(value="BCEWithLogitsLoss")
        self.disc_lr_var = tk.StringVar(value="0.0002")
        self.disc_epochs_var = tk.StringVar(value="100") # No. of full dataset passes for D
        self.disc_batch_size_var = tk.StringVar(value="64")

        ttk.Label(disc_lf, text="Loss Function:").pack(pady=(5,0))
        self.disc_loss_combo = ttk.Combobox(disc_lf, textvariable=self.disc_loss_var, 
                                            values=["BCEWithLogitsLoss", "MSELoss", "BCELoss"], width=18)
        self.disc_loss_combo.pack(pady=(0,5))
        create_param_entry(disc_lf, "Learning Rate:", self.disc_lr_var)
        create_param_entry(disc_lf, "Max Epochs:", self.disc_epochs_var)
        create_param_entry(disc_lf, "Batch Size:", self.disc_batch_size_var)

        # Common Training Params
        common_lf = ttk.LabelFrame(frame, text="Common Settings")
        common_lf.pack(pady=10, fill="x", padx=20)

        data_frame = ttk.Frame(common_lf)
        data_frame.pack(fill="x", pady=5)
        ttk.Label(data_frame, text="Data Folder (for MNIST):", width=20).pack(side="left")
        self.data_folder_entry = ttk.Entry(data_frame, textvariable=self.data_folder_var, width=50)
        self.data_folder_entry.pack(side="left", padx=5, expand=True, fill="x")
        ttk.Button(data_frame, text="Browse", command=self.select_data_folder).pack(side="left")
        
        initial_net_frame = ttk.Frame(common_lf)
        initial_net_frame.pack(fill="x", pady=5)
        ttk.Label(initial_net_frame, text="Initial Active Network:", width=20).pack(side="left")
        self.initial_train_choice_var = tk.StringVar(value="generator")
        ttk.Radiobutton(initial_net_frame, text="Generator", variable=self.initial_train_choice_var, value="generator").pack(side="left", padx=5)
        ttk.Radiobutton(initial_net_frame, text="Discriminator", variable=self.initial_train_choice_var, value="discriminator").pack(side="left", padx=5)

        # Control Buttons
        control_lf = ttk.LabelFrame(frame, text="Training Controls")
        control_lf.pack(pady=10, fill="x", padx=20)
        btn_frame = ttk.Frame(control_lf)
        btn_frame.pack(pady=5)
        self.start_btn = ttk.Button(btn_frame, text="Start Training", command=self.start_gan_training, width=15)
        self.start_btn.pack(side="left", padx=5)
        self.pause_btn = ttk.Button(btn_frame, text="Pause Active", command=self.pause_active_training, width=15, state="disabled")
        self.pause_btn.pack(side="left", padx=5)
        self.resume_btn = ttk.Button(btn_frame, text="Resume Active", command=self.resume_active_training, width=15, state="disabled")
        self.resume_btn.pack(side="left", padx=5)
        self.switch_btn = ttk.Button(btn_frame, text="Switch Active Net", command=self.switch_active_network, width=15, state="disabled")
        self.switch_btn.pack(side="left", padx=5)
        self.stop_btn = ttk.Button(btn_frame, text="Stop All", command=self.stop_all_training, width=15, state="disabled")
        self.stop_btn.pack(side="left", padx=5)
        
        # Training Stats Area
        stats_lf = ttk.LabelFrame(frame, text="Training Log / Status")
        stats_lf.pack(pady=10, fill="both", expand=True, padx=20)
        self.training_stats_text = tk.Text(stats_lf, height=10, width=80, wrap="word")
        self.training_stats_text.pack(fill="both", expand=True, padx=5, pady=5)

    def populate_training_params_from_config(self):
        if not self.loaded_config_data or "training" not in self.loaded_config_data:
            # Keep defaults if no training section in config
            return

        training_conf = self.loaded_config_data["training"]
        gen_train = training_conf.get("generator", {})
        disc_train = training_conf.get("discriminator", {})

        self.gen_loss_var.set(gen_train.get("loss_function", "BCEWithLogitsLoss"))
        self.gen_lr_var.set(str(gen_train.get("learning_rate", "0.0002")))
        self.gen_epochs_var.set(str(gen_train.get("epochs", "100")))
        self.gen_batch_size_var.set(str(gen_train.get("batch_size", "64")))

        self.disc_loss_var.set(disc_train.get("loss_function", "BCEWithLogitsLoss"))
        self.disc_lr_var.set(str(disc_train.get("learning_rate", "0.0002")))
        self.disc_epochs_var.set(str(disc_train.get("epochs", "100")))
        self.disc_batch_size_var.set(str(disc_train.get("batch_size", "64")))
        
        self.data_folder_var.set(training_conf.get("data_folder", os.path.abspath("./data")))
        self.initial_train_choice_var.set(training_conf.get("initial_network", "generator"))

    def get_current_ui_training_config(self):
        """Reads training parameters from UI fields and returns a config dict."""
        try:
            config = {
                "generator": {
                    "loss_function": self.gen_loss_var.get(),
                    "learning_rate": float(self.gen_lr_var.get()),
                    "epochs": int(self.gen_epochs_var.get()),
                    "batch_size": int(self.gen_batch_size_var.get())
                },
                "discriminator": {
                    "loss_function": self.disc_loss_var.get(),
                    "learning_rate": float(self.disc_lr_var.get()),
                    "epochs": int(self.disc_epochs_var.get()),
                    "batch_size": int(self.disc_batch_size_var.get())
                },
                "data_folder": self.data_folder_var.get(),
                "initial_network": self.initial_train_choice_var.get().lower()
            }
            return config
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid training parameter value: {e}")
            return None

    def select_data_folder(self):
        folder = filedialog.askdirectory(title="Select Data Folder (e.g., for MNIST)")
        if folder:
            self.data_folder_var.set(os.path.abspath(folder))
    
    def update_button_states(self, training_started=False):
        if training_started:
            self.start_btn.config(state="disabled")
            self.pause_btn.config(state="normal")
            self.resume_btn.config(state="normal")
            self.switch_btn.config(state="normal")
            self.stop_btn.config(state="normal")
        else: # Training stopped or not started
            self.start_btn.config(state="normal")
            self.pause_btn.config(state="disabled")
            self.resume_btn.config(state="disabled")
            self.switch_btn.config(state="disabled")
            self.stop_btn.config(state="disabled")


    def start_gan_training(self):
        if not self.loaded_config_data:
            messagebox.showerror("Error", "Please load a network configuration file first.")
            return
        
        current_ui_train_config = self.get_current_ui_training_config()
        if not current_ui_train_config: return # Error handled in getter

        try:
            # Pass only arch configs to GANController constructor
            self.gan_controller = GANController(
                self.loaded_config_data.get("generator", {}),
                self.loaded_config_data.get("discriminator", {}),
                current_ui_train_config # Pass the full UI training config for initial setup
            )
            self.gan_controller.start_persistent_training(self.training_ui_callback)
            self.update_training_stats_text("Training initialized. Check console for details.")
            self.update_button_states(training_started=True)
        except Exception as e:
            messagebox.showerror("Training Error", f"Failed to start training: {e}")
            print(f"Full traceback for training start error: {e}", exc_info=True) # For dev
            self.gan_controller = None # Ensure controller is None if setup fails
            self.update_button_states(training_started=False)

    def pause_active_training(self):
        if self.gan_controller:
            self.gan_controller.pause_training()
            self.update_training_stats_text(f"Paused {self.gan_controller.trainer.current_network} training.")

    def resume_active_training(self):
        if self.gan_controller:
            # Update controller's runtime params from UI before resuming
            current_ui_train_config = self.get_current_ui_training_config()
            if not current_ui_train_config: return
            self.gan_controller.update_runtime_train_params_from_ui(current_ui_train_config)
            
            self.gan_controller.resume_training()
            self.update_training_stats_text(f"Resumed {self.gan_controller.trainer.current_network} training with current UI parameters.")
    
    def switch_active_network(self):
        if self.gan_controller:
            self.gan_controller.switch_network()
            self.update_training_stats_text(f"Switched. Active network: {self.gan_controller.trainer.current_network}.")
    
    def stop_all_training(self):
        if self.gan_controller:
            self.gan_controller.stop_training() # This will join threads
            self.update_training_stats_text("All training stopped by user.")
            self.update_button_states(training_started=False)
            self.gan_controller = None # Clear controller instance
    
    def update_training_stats_text(self, message):
        if self.training_stats_text.winfo_exists(): # Check if widget still exists
            self.training_stats_text.insert(tk.END, str(message) + "\n")
            self.training_stats_text.see(tk.END) # Auto-scroll
    
    def build_summary_tab(self):
        frame = self.summary_frame
        ttk.Label(frame, text="Configuration Summary & Image Generation", font=("Arial", 14)).pack(pady=10)
        
        summary_controls = ttk.Frame(frame)
        summary_controls.pack(pady=5)
        ttk.Button(summary_controls, text="Refresh Summary from Loaded Config", command=self.update_summary_text_from_config).pack(side="left", padx=5)
        ttk.Button(summary_controls, text="Plot Losses", command=self.plot_training_losses).pack(side="left", padx=5)
        
        self.summary_text_widget = tk.Text(frame, height=15, width=100, wrap="word", state="disabled")
        self.summary_text_widget.pack(pady=10, fill="x", padx=20)
        
        generation_lf = ttk.LabelFrame(frame, text="Generate Image (from current Generator)")
        generation_lf.pack(pady=10, fill="x", padx=20)
        ttk.Button(generation_lf, text="Generate & Show Image", command=self.generate_and_show_image).pack(pady=5)
        self.generated_image_label = ttk.Label(generation_lf, text="No image generated yet.")
        self.generated_image_label.pack(pady=5)
        self.generated_image_pil = None # To keep a reference to PhotoImage

    def update_summary_text_from_config(self):
        if not self.loaded_config_data:
            summary = "No configuration loaded."
        else:
            summary = "=== Loaded Configuration Summary ===\n\n"
            summary += "--- Generator Architecture ---\n"
            summary += json.dumps(self.loaded_config_data.get("generator", {}), indent=2) + "\n\n"
            summary += "--- Discriminator Architecture ---\n"
            summary += json.dumps(self.loaded_config_data.get("discriminator", {}), indent=2) + "\n\n"
            summary += "--- Default Training Parameters (from file) ---\n"
            summary += json.dumps(self.loaded_config_data.get("training", {}), indent=2) + "\n"
        
        self.summary_text_widget.config(state="normal")
        self.summary_text_widget.delete("1.0", tk.END)
        self.summary_text_widget.insert(tk.END, summary)
        self.summary_text_widget.config(state="disabled")
    
    def plot_training_losses(self):
        if self.gan_controller and self.gan_controller.get_trainer_instance():
            self.gan_controller.get_trainer_instance().plot_losses()
        else:
            messagebox.showinfo("Info", "No training data available to plot losses. Start training first.")
    
    def generate_and_show_image(self):
        if not self.gan_controller or not self.gan_controller.generator:
            messagebox.showerror("Error", "Generator not initialized. Load config and start training first.")
            return
        try:
            self.gan_controller.generator.eval() # Set to evaluation mode
            noise = torch.randn(1, self.gan_controller.latent_dim, device=self.gan_controller.device)
            with torch.no_grad():
                generated_output = self.gan_controller.generator(noise)
            
            # Assuming output is [1, C, H, W] and C=1 for MNIST, range [-1, 1] due to Tanh
            img_tensor = generated_output[0].cpu().squeeze(0) # Remove batch and channel for 1-channel image
            img_tensor = (img_tensor + 1) / 2.0 # Denormalize from [-1,1] to [0,1]
            img_tensor = img_tensor.clamp(0, 1) # Ensure values are in [0,1]
            
            # Convert to PIL Image
            pil_img = transforms.ToPILImage()(img_tensor.unsqueeze(0)) # Add channel dim back for ToPILImage if squeezed
            
            # Resize for display if too small/large (optional)
            pil_img = pil_img.resize((128, 128), Image.Resampling.NEAREST) 

            self.generated_image_pil = ImageTk.PhotoImage(pil_img)
            self.generated_image_label.config(image=self.generated_image_pil, text="")
            # self.generated_image_label.image = self.generated_image_pil # Keep reference (done by PhotoImage itself)
        except Exception as e:
            messagebox.showerror("Image Generation Error", f"Failed to generate image: {e}")
            print(f"Full traceback for image generation error: {e}", exc_info=True)


if __name__ == "__main__":
    main_root = tk.Tk()
    app = GanConfigurator(main_root)
    main_root.mainloop()