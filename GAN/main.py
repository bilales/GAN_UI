# GAN/main.py
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from .controller import GANController # Relative import
from .model_builder import detect_gpu # Relative import
import json
import threading
from PIL import Image, ImageTk # Ensure Pillow is installed
import torch
import os
import time # For timestamps in logger
from torchvision import transforms # For image generation

class GanConfigurator:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("Interactive GAN Training Framework")
        self.root.geometry("950x800") # Adjusted size

        self.data_folder_var = tk.StringVar(value=os.path.abspath("./data"))
        self.loaded_config_data = None
        self.gan_controller = None
        self.training_active = False # Manages if training process is ongoing
        self.generated_image_pil_ref = None # For PhotoImage persistence

        self.training_ui_callback = lambda message: self.root.after(0, self.update_training_stats_text, message)

        # --- Main Layout ---
        # Top frame for title and load button
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill="x", padx=10, pady=(10,0))
        ttk.Label(header_frame, text="GAN Configurator & Trainer", font=("Arial", 16, "bold")).pack(side="left")
        self.load_config_button = ttk.Button(header_frame, text="Load Config JSON", command=self.load_json_configuration)
        self.load_config_button.pack(side="right")


        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self.preview_frame = ttk.Frame(self.notebook)
        self.train_params_frame = ttk.Frame(self.notebook)
        self.summary_frame = ttk.Frame(self.notebook)

        self.notebook.add(self.preview_frame, text="1. Network Architectures")
        self.notebook.add(self.train_params_frame, text="2. Training Setup & Controls")
        self.notebook.add(self.summary_frame, text="3. Output & Summary")

        self._build_preview_tab()
        self._build_training_params_tab()
        self._build_summary_tab()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.update_training_stats_text(f"GPU Status: {detect_gpu()}")
        self.update_training_stats_text("Welcome! Please load a configuration file to begin.")
        self.update_button_states() # Initialize button states correctly

    def on_closing(self):
        if self.training_active: # If training threads might be running
            if not messagebox.askokcancel("Quit", "Training might be active. Stop threads and quit?"):
                return # User cancelled quit
        
        if self.gan_controller:
            self.update_training_stats_text("Application closing: Signaling training threads to stop...")
            self.root.update_idletasks() # Process UI message
            self.gan_controller.signal_stop_training_threads()
            self.update_training_stats_text("Waiting for threads to join (max 5s each)...")
            self.root.update_idletasks()
            self.gan_controller.join_training_threads(timeout=5)
            self.update_training_stats_text("Threads joined or timed out. Exiting.")
        self.root.destroy()

    def _build_preview_tab(self):
        frame = self.preview_frame
        # Title for this tab, load button is global now
        ttk.Label(frame, text="Preview of Loaded Network Architectures", font=("Arial", 12, "italic")).pack(pady=(5,10))

        preview_container = ttk.Frame(frame)
        preview_container.pack(fill="both", expand=True, padx=5, pady=5)
        preview_container.columnconfigure(0, weight=1)
        preview_container.columnconfigure(1, weight=1)

        gen_lf = ttk.LabelFrame(preview_container, text="Generator Architecture")
        gen_lf.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.gen_preview_text = tk.Text(gen_lf, height=20, width=45, state="disabled", wrap="word", relief="sunken", borderwidth=1)
        gen_scroll = ttk.Scrollbar(gen_lf, command=self.gen_preview_text.yview)
        self.gen_preview_text['yscrollcommand'] = gen_scroll.set
        gen_scroll.pack(side="right", fill="y")
        self.gen_preview_text.pack(side="left", fill="both", expand=True, padx=(5,0), pady=5)
        
        disc_lf = ttk.LabelFrame(preview_container, text="Discriminator Architecture")
        disc_lf.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.disc_preview_text = tk.Text(disc_lf, height=20, width=45, state="disabled", wrap="word", relief="sunken", borderwidth=1)
        disc_scroll = ttk.Scrollbar(disc_lf, command=self.disc_preview_text.yview)
        self.disc_preview_text['yscrollcommand'] = disc_scroll.set
        disc_scroll.pack(side="right", fill="y")
        self.disc_preview_text.pack(side="left", fill="both", expand=True, padx=(5,0), pady=5)

    def load_json_configuration(self):
        if self.training_active:
            messagebox.showwarning("Config Load Denied", "Cannot load new configuration while training is active. Please stop the current training session first.")
            return

        _config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__))) # Directory of main.py
        default_config_path = os.path.join(_config_dir, "config.json")

        file_path = filedialog.askopenfilename(
            title="Select GAN Configuration File",
            initialdir=_config_dir, # Start in the directory containing this script
            filetypes=[("JSON files", "*.json")])
        
        if not file_path: # User cancelled
            if os.path.exists(default_config_path):
                if messagebox.askyesno("Load Default?", f"No file selected. Load default '{os.path.basename(default_config_path)}' from the application directory?"):
                    file_path = default_config_path
                else:
                    self.update_training_stats_text("Configuration loading cancelled by user.")
                    return
            else:
                self.update_training_stats_text("Configuration loading cancelled. No default found.")
                return

        try:
            with open(file_path, "r") as f:
                self.loaded_config_data = json.load(f)
            print(f"[DEBUG] Config loaded successfully. self.loaded_config_data is set: {self.loaded_config_data is not None}")
            self.update_training_stats_text(f"Configuration loaded from: {os.path.basename(file_path)}")
            self.populate_preview_from_config()
            self.populate_training_params_from_config()
            self.update_summary_text_from_config()
        except Exception as e:
            messagebox.showerror("Configuration Error", f"Failed to load or parse configuration file '{os.path.basename(file_path)}':\n{e}")
            self.loaded_config_data = None # Ensure it's None on failure
            print(f"[DEBUG] Config load FAILED. self.loaded_config_data is None.")
            self.update_training_stats_text(f"ERROR loading config: {e}")
        finally:
            print(f"[DEBUG] load_json_configuration finally block. training_active: {self.training_active}, loaded_config_data: {self.loaded_config_data is not None}")
            self.update_button_states() # Crucial for enabling Start button

    def populate_preview_from_config(self):
        if not self.loaded_config_data:
            no_config_msg = "No configuration loaded. Please use 'Load Config JSON' button."
            for txt_widget in [self.gen_preview_text, self.disc_preview_text]:
                txt_widget.config(state="normal")
                txt_widget.delete("1.0", tk.END)
                txt_widget.insert(tk.END, no_config_msg)
                txt_widget.config(state="disabled")
            return
        
        def format_layers_preview(config_section_name):
            section = self.loaded_config_data.get(config_section_name, {})
            text = f"{config_section_name.capitalize()} Configuration:\n"
            text += f"  Input: {section.get('input_dim' if config_section_name == 'generator' else 'input_shape', 'N/A')}\n"
            text += f"  Output Size (Informational): {section.get('output_size', 'N/A')}\n"
            text += f"  Global Activation: {section.get('global_activation', 'N/A')}\n"
            text += "  Layers:\n"
            for i, layer_conf in enumerate(section.get("layers", [])):
                layer_desc = f"    {i+1}. Type: {layer_conf.get('type', '?')}"
                details = [f"{k}: {v}" for k, v in layer_conf.items() if k.lower() not in ['type']]
                if details: layer_desc += f" ({', '.join(details)})"
                text += layer_desc + "\n"
            return text

        for txt_widget, section_name in [(self.gen_preview_text, "generator"), (self.disc_preview_text, "discriminator")]:
            txt_widget.config(state="normal")
            txt_widget.delete("1.0", tk.END)
            txt_widget.insert(tk.END, format_layers_preview(section_name))
            txt_widget.config(state="disabled")

    def _build_training_params_tab(self):
        frame = self.train_params_frame
        # Using pack for the overall sections in this tab for simplicity, as requested
        
        # --- Parameters Section ---
        params_outer_lf = ttk.LabelFrame(frame, text="Training Parameters")
        params_outer_lf.pack(pady=10, padx=10, fill="x")

        # Sub-frames for G and D to sit side-by-side if window is wide enough
        params_container_frame = ttk.Frame(params_outer_lf)
        params_container_frame.pack(fill="x", expand=True, pady=5, padx=5)

        gen_lf = ttk.LabelFrame(params_container_frame, text="Generator")
        gen_lf.pack(side="left", padx=5, fill="y", expand=True)
        self.gen_loss_var = tk.StringVar(value="BCEWithLogitsLoss")
        self.gen_lr_var = tk.StringVar(value="0.0002")
        self.gen_epochs_var = tk.StringVar(value="100")
        self.gen_batch_size_var = tk.StringVar(value="64")
        self._create_param_entry_row(gen_lf, "Loss Function:", self.gen_loss_var, is_combo=True, combo_values=["BCEWithLogitsLoss", "MSELoss", "BCELoss"])
        self._create_param_entry_row(gen_lf, "Learning Rate:", self.gen_lr_var)
        self._create_param_entry_row(gen_lf, "Max Epochs:", self.gen_epochs_var)
        self._create_param_entry_row(gen_lf, "Batch Size (G):", self.gen_batch_size_var)

        disc_lf = ttk.LabelFrame(params_container_frame, text="Discriminator")
        disc_lf.pack(side="left", padx=5, fill="y", expand=True)
        self.disc_loss_var = tk.StringVar(value="BCEWithLogitsLoss")
        self.disc_lr_var = tk.StringVar(value="0.0002")
        self.disc_epochs_var = tk.StringVar(value="100")
        self.disc_batch_size_var = tk.StringVar(value="64")
        self._create_param_entry_row(disc_lf, "Loss Function:", self.disc_loss_var, is_combo=True, combo_values=["BCEWithLogitsLoss", "MSELoss", "BCELoss"])
        self._create_param_entry_row(disc_lf, "Learning Rate:", self.disc_lr_var)
        self._create_param_entry_row(disc_lf, "Max Epochs:", self.disc_epochs_var)
        self._create_param_entry_row(disc_lf, "Batch Size (D):", self.disc_batch_size_var)
        
        # --- Common Settings Section ---
        common_lf = ttk.LabelFrame(frame, text="Common Settings")
        common_lf.pack(pady=5, padx=10, fill="x")

        data_frame = ttk.Frame(common_lf)
        data_frame.pack(fill="x", pady=3, padx=5)
        ttk.Label(data_frame, text="Data Folder:", width=15).pack(side="left")
        self.data_folder_entry = ttk.Entry(data_frame, textvariable=self.data_folder_var, width=40)
        self.data_folder_entry.pack(side="left", padx=5, expand=True, fill="x")
        ttk.Button(data_frame, text="Browse...", command=self.select_data_folder, width=10).pack(side="left")
        
        initial_net_frame = ttk.Frame(common_lf)
        initial_net_frame.pack(fill="x", pady=3, padx=5)
        ttk.Label(initial_net_frame, text="Initial Active Net:", width=15).pack(side="left")
        self.initial_train_choice_var = tk.StringVar(value="generator")
        ttk.Radiobutton(initial_net_frame, text="Generator", variable=self.initial_train_choice_var, value="generator").pack(side="left", padx=10)
        ttk.Radiobutton(initial_net_frame, text="Discriminator", variable=self.initial_train_choice_var, value="discriminator").pack(side="left", padx=10)

        # --- Controls Section ---
        control_lf = ttk.LabelFrame(frame, text="Training Controls")
        control_lf.pack(pady=10, padx=10, fill="x")
        
        # Using a frame for buttons to allow them to be centered or arranged more easily
        btn_container = ttk.Frame(control_lf)
        btn_container.pack(pady=5)

        btn_width = 17 
        self.start_btn = ttk.Button(btn_container, text="Start Training", command=self.start_gan_training, width=btn_width)
        self.start_btn.grid(row=0, column=0, padx=3, pady=3)
        self.pause_btn = ttk.Button(btn_container, text="Pause Active", command=self.pause_active_training, width=btn_width)
        self.pause_btn.grid(row=0, column=1, padx=3, pady=3)
        self.resume_btn = ttk.Button(btn_container, text="Resume Active", command=self.resume_active_training, width=btn_width)
        self.resume_btn.grid(row=0, column=2, padx=3, pady=3)
        self.switch_btn = ttk.Button(btn_container, text="Switch Active Net", command=self.switch_active_network, width=btn_width)
        self.switch_btn.grid(row=1, column=0, padx=3, pady=3) # Second row
        self.stop_btn = ttk.Button(btn_container, text="Stop All Training", command=self.stop_all_training, width=btn_width)
        self.stop_btn.grid(row=1, column=1, padx=3, pady=3, columnspan=2)
        
        # --- Log Section ---
        stats_lf = ttk.LabelFrame(frame, text="Training Log / Status")
        stats_lf.pack(pady=5, padx=10, fill="both", expand=True)
        self.training_stats_text = tk.Text(stats_lf, height=10, wrap="word", relief="sunken", borderwidth=1)
        stats_scroll = ttk.Scrollbar(stats_lf, command=self.training_stats_text.yview)
        self.training_stats_text['yscrollcommand'] = stats_scroll.set
        stats_scroll.pack(side="right", fill="y")
        self.training_stats_text.pack(side="left",fill="both", expand=True, padx=(5,0), pady=5)

    def _create_param_entry_row(self, parent, label_text, string_var, is_combo=False, combo_values=None, entry_width=18):
        row_frame = ttk.Frame(parent)
        row_frame.pack(fill="x", padx=5, pady=3)
        ttk.Label(row_frame, text=label_text, width=15, anchor="w").pack(side="left")
        if is_combo:
            widget = ttk.Combobox(row_frame, textvariable=string_var, values=combo_values or [], width=entry_width - 2, state="readonly")
        else:
            widget = ttk.Entry(row_frame, textvariable=string_var, width=entry_width)
        widget.pack(side="left", padx=5, fill="x", expand=True)


    def populate_training_params_from_config(self):
        if not self.loaded_config_data or "training" not in self.loaded_config_data:
            self.update_training_stats_text("No 'training' section in config, using current UI values or defaults.")
            return # Keep existing UI values if no training section

        training_conf = self.loaded_config_data["training"]
        def safe_get(source_dict, key, default_var):
            return source_dict.get(key, default_var.get()) # Fallback to current UI value if key missing

        gen_train = training_conf.get("generator", {})
        self.gen_loss_var.set(safe_get(gen_train, "loss_function", self.gen_loss_var))
        self.gen_lr_var.set(str(safe_get(gen_train, "learning_rate", self.gen_lr_var)))
        self.gen_epochs_var.set(str(safe_get(gen_train, "epochs", self.gen_epochs_var)))
        self.gen_batch_size_var.set(str(safe_get(gen_train, "batch_size", self.gen_batch_size_var)))

        disc_train = training_conf.get("discriminator", {})
        self.disc_loss_var.set(safe_get(disc_train, "loss_function", self.disc_loss_var))
        self.disc_lr_var.set(str(safe_get(disc_train, "learning_rate", self.disc_lr_var)))
        self.disc_epochs_var.set(str(safe_get(disc_train, "epochs", self.disc_epochs_var)))
        self.disc_batch_size_var.set(str(safe_get(disc_train, "batch_size", self.disc_batch_size_var)))
        
        self.data_folder_var.set(training_conf.get("data_folder", self.data_folder_var.get()))
        self.initial_train_choice_var.set(training_conf.get("initial_network", self.initial_train_choice_var.get()))
        self.update_training_stats_text("Training parameters populated/updated from loaded config.")


    def get_current_ui_training_config(self):
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
            print(f"[DEBUG] Current UI Training Config: {config}") # DEBUG
            return config
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid training parameter value: {e}.\nPlease ensure numeric fields (LR, Epochs, Batch Size) contain valid numbers.")
            return None


    def select_data_folder(self):
        current_path = self.data_folder_var.get()
        initial_dir = os.path.dirname(current_path) if os.path.isdir(current_path) else os.path.expanduser("~")
        folder = filedialog.askdirectory(title="Select Data Folder (e.g., for MNIST dataset)", initialdir=initial_dir)
        if folder:
            self.data_folder_var.set(os.path.abspath(folder))
            self.update_training_stats_text(f"Data folder set to: {folder}")


    def update_button_states(self):
        print(f"[DEBUG] update_button_states: training_active={self.training_active}, loaded_config={self.loaded_config_data is not None}")
        if not hasattr(self, 'start_btn'): return # Widgets not created yet

        # Determine if the start button should be enabled
        can_start = bool(self.loaded_config_data) and not self.training_active
        self.start_btn.config(state="normal" if can_start else "disabled")

        if not self.training_active: # Not started or has been stopped
            self.pause_btn.config(state="disabled")
            self.resume_btn.config(state="disabled")
            self.switch_btn.config(state="disabled")
            self.stop_btn.config(state="disabled")
        else: # Training is active (could be running or paused)
            self.stop_btn.config(state="normal")
            self.switch_btn.config(state="normal")
            
            is_paused_for_active_net = True # Default to assuming paused if controller/trainer not fully ready
            if self.gan_controller and self.gan_controller.trainer:
                active_net = self.gan_controller.trainer.current_network
                if active_net == "generator":
                    # Check if pause_event_gen attribute exists before trying to access is_set()
                    if hasattr(self.gan_controller.trainer, 'pause_event_gen'):
                        is_paused_for_active_net = not self.gan_controller.trainer.pause_event_gen.is_set()
                elif active_net == "discriminator":
                    if hasattr(self.gan_controller.trainer, 'pause_event_disc'):
                        is_paused_for_active_net = not self.gan_controller.trainer.pause_event_disc.is_set()
            
            self.pause_btn.config(state="normal" if not is_paused_for_active_net else "disabled")
            self.resume_btn.config(state="normal" if is_paused_for_active_net else "disabled")
        self.root.update_idletasks() # Force UI refresh


    def start_gan_training(self):
        if self.training_active:
            messagebox.showinfo("Training Info", "A training session is already active. Please stop it before starting a new one.")
            return

        if not self.loaded_config_data:
            messagebox.showerror("Configuration Error", "No network configuration loaded. Please load a JSON config file first.")
            self.update_button_states() # Ensure start button is disabled
            return
        
        current_ui_train_config = self.get_current_ui_training_config()
        if not current_ui_train_config: # Error message shown by getter
            self.update_button_states() # Ensure start button might re-enable if it was a temp input error
            return

        # Disable start button immediately to prevent multiple clicks
        self.start_btn.config(state="disabled")
        self.root.update_idletasks() # Make UI show disabled button
        self.update_training_stats_text("Initializing GAN Controller...")

        try:
            # If restarting after a stop, ensure old controller/threads are cleaned up
            if self.gan_controller:
                if (self.gan_controller.generator_thread and self.gan_controller.generator_thread.is_alive()) or \
                   (self.gan_controller.discriminator_thread and self.gan_controller.discriminator_thread.is_alive()):
                    self.update_training_stats_text("Waiting for previous threads to fully terminate before restarting...")
                    self.gan_controller.signal_stop_training_threads()
                    self.gan_controller.join_training_threads(timeout=2) # Quick join
                del self.gan_controller
            
            self.gan_controller = GANController(
                self.loaded_config_data.get("generator", {}),
                self.loaded_config_data.get("discriminator", {}),
                current_ui_train_config
            )
            self.update_training_stats_text("Controller initialized. Starting training threads...")
            self.root.update_idletasks()

            self.gan_controller.start_persistent_training(self.training_ui_callback)
            self.training_active = True # Set flag AFTER successful start
            self.update_training_stats_text("Training threads initiated. Check console for detailed logs.")
        except Exception as e:
            self.training_active = False # Ensure flag is reset on error
            self.gan_controller = None 
            messagebox.showerror("Training Start Error", f"Failed to start training session: {e}")
            self.update_training_stats_text(f"FATAL ERROR starting training: {e}")
            print(f"Full traceback for training start error:", exc_info=True)
        finally:
            # This will correctly set button states based on self.training_active
            self.update_button_states()


    def _set_controls_during_action(self, disable=True):
        """Temporarily disable/re-enable buttons during a quick action."""
        state = "disabled" if disable else "normal"
        # Only affect buttons relevant if training is active
        if self.training_active:
            self.pause_btn.config(state=state)
            self.resume_btn.config(state=state)
            self.switch_btn.config(state=state)
            # Stop button remains enabled as it's the escape hatch
        if not disable: # When re-enabling, call full update
            self.update_button_states()


    def pause_active_training(self):
        if not self.training_active or not self.gan_controller:
            self.update_training_stats_text("Cannot pause: No active training session.")
            return
        
        self._set_controls_during_action(True)
        active_net_before_pause = self.gan_controller.trainer.current_network
        self.gan_controller.pause_training() # Pauses the currently active network
        self.update_training_stats_text(f"Paused {active_net_before_pause} training.")
        self._set_controls_during_action(False)


    def resume_active_training(self):
        if not self.training_active or not self.gan_controller:
            self.update_training_stats_text("Cannot resume: No active training session.")
            return

        self._set_controls_during_action(True)
        current_ui_train_config = self.get_current_ui_training_config()
        if not current_ui_train_config: 
            self._set_controls_during_action(False)
            return
        self.gan_controller.update_runtime_train_params_from_ui(current_ui_train_config)
            
        active_net_to_resume = self.gan_controller.trainer.current_network
        self.gan_controller.resume_training()
        self.update_training_stats_text(f"Resumed {active_net_to_resume} training.")
        self._set_controls_during_action(False)

    
    def switch_active_network(self):
        if not self.training_active or not self.gan_controller:
            self.update_training_stats_text("Cannot switch: No active training session.")
            return
        
        self._set_controls_during_action(True)
        old_active = self.gan_controller.trainer.current_network
        self.gan_controller.switch_network()
        new_active = self.gan_controller.trainer.current_network
        self.update_training_stats_text(f"Switched. Was: {old_active}, Now active: {new_active}.")
        self._set_controls_during_action(False)

    
    def stop_all_training(self):
        if not self.training_active or not self.gan_controller:
            self.update_training_stats_text("Stop command: No training was active.")
            if not self.training_active: self.update_button_states() # Ensure correct state
            return

        self.update_training_stats_text("Signaling training threads to stop...")
        self.stop_btn.config(text="Stopping...", state="disabled")
        # Disable other controls immediately
        self.pause_btn.config(state="disabled")
        self.resume_btn.config(state="disabled")
        self.switch_btn.config(state="disabled")
        self.root.update_idletasks()

        self.gan_controller.signal_stop_training_threads()
        
        # Polling to update UI when threads actually stop
        def _check_threads_stopped():
            if self.gan_controller and \
               ((self.gan_controller.generator_thread is None or not self.gan_controller.generator_thread.is_alive()) and \
                (self.gan_controller.discriminator_thread is None or not self.gan_controller.discriminator_thread.is_alive())):
                
                self.update_training_stats_text("Training threads have terminated.")
                self.training_active = False
                # Attempt a quick join, mainly to clean up thread objects if controller handles it
                if hasattr(self.gan_controller, 'join_training_threads'):
                     self.gan_controller.join_training_threads(timeout=0.5) # Non-blocking effective
                # Consider self.gan_controller = None here if you want a full reset for next "start"
                # For now, keep controller so user can still generate images or plot last losses
                self.update_button_states()
                self.stop_btn.config(text="Stop All Training") # Reset button text
            elif self.training_active: # If stop was initiated but threads still cleaning up
                self.root.after(500, _check_threads_stopped) # Poll again
            # If !self.training_active and threads are gone, this won't reschedule.
        
        self.root.after(100, _check_threads_stopped) # Start polling
        # Update flag immediately so UI reflects "stopping" intent
        # self.training_active = False # Moved into _check_threads_stopped for more accuracy
        # self.update_button_states() # update_button_states will be called by _check_threads_stopped

    def update_training_stats_text(self, message):
        if hasattr(self, 'training_stats_text') and self.training_stats_text.winfo_exists(): 
            current_time = time.strftime("%H:%M:%S", time.localtime())
            self.training_stats_text.insert(tk.END, f"{current_time} - {str(message)}\n")
            self.training_stats_text.see(tk.END) 

    def _build_summary_tab(self):
        frame = self.summary_frame
        ttk.Label(frame, text="Output & Configuration Summary", font=("Arial", 12, "italic")).pack(pady=(5,10))
        
        controls_frame = ttk.Frame(frame)
        controls_frame.pack(pady=5)
        ttk.Button(controls_frame, text="Refresh Summary Text", command=self.update_summary_text_from_config).pack(side="left", padx=5)
        self.plot_losses_button = ttk.Button(controls_frame, text="Plot Training Losses", command=self.plot_training_losses)
        self.plot_losses_button.pack(side="left", padx=5)
        
        self.summary_text_widget = tk.Text(frame, height=12, width=80, wrap="word", state="disabled", relief="sunken", borderwidth=1)
        summary_scroll = ttk.Scrollbar(frame, command=self.summary_text_widget.yview)
        self.summary_text_widget['yscrollcommand'] = summary_scroll.set
        summary_scroll.pack(side="right", fill="y", padx=(0,5), pady=5)
        self.summary_text_widget.pack(pady=5, padx=(5,0), fill="both", expand=True)
        
        generation_lf = ttk.LabelFrame(frame, text="Generated Image Preview")
        generation_lf.pack(pady=10, padx=10, fill="x")
        self.generate_image_button = ttk.Button(generation_lf, text="Generate & Show New Image", command=self.generate_and_show_image)
        self.generate_image_button.pack(pady=5)
        
        # Image display area - kept your fix
        self.image_display_frame = ttk.Frame(generation_lf, width=130, height=130, relief="sunken", borderwidth=1)
        self.image_display_frame.pack(pady=5, anchor="center") # Centering the frame itself
        self.image_display_frame.pack_propagate(False) 
        self.generated_image_label = ttk.Label(self.image_display_frame, text="No image yet", compound="image", anchor="center")
        self.generated_image_label.pack(fill="both", expand=True)

    def update_summary_text_from_config(self):
        if not self.loaded_config_data:
            summary = "No configuration loaded. Please use 'Load Config JSON' button (top of window)."
        else:
            summary = "=== Loaded Configuration Summary (from JSON) ===\n\n"
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
        if not self.gan_controller or not self.gan_controller.get_trainer_instance():
            messagebox.showinfo("Plot Info", "No training session active or no data to plot. Start training first.")
            return
        
        trainer = self.gan_controller.get_trainer_instance()
        if not trainer.gen_losses and not trainer.disc_losses:
            messagebox.showinfo("Plot Info", "No loss data recorded yet.")
            return

        if hasattr(self, 'plot_losses_button') : self.plot_losses_button.config(state="disabled")
        thread = threading.Thread(target=self._plot_losses_thread_task, daemon=True)
        thread.start()

    def _plot_losses_thread_task(self):
        try:
            trainer_instance = self.gan_controller.get_trainer_instance()
            if trainer_instance: trainer_instance.plot_losses() 
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Plotting Error", f"Failed to display plot: {e}"))
            print(f"Plotting error details:", exc_info=True)
        finally:
            if hasattr(self, 'plot_losses_button') and self.plot_losses_button.winfo_exists():
                 self.root.after(0, lambda: self.plot_losses_button.config(state="normal"))
    
    def generate_and_show_image(self): 
        if not self.gan_controller or not self.gan_controller.generator:
            messagebox.showerror("Generation Error", "Generator model is not initialized. Please load a configuration and ensure the controller is set up.")
            return
        
        if hasattr(self, 'generate_image_button'): self.generate_image_button.config(state="disabled")
        thread = threading.Thread(target=self._generate_image_thread_task, daemon=True)
        thread.start()

    def _generate_image_thread_task(self): 
        try:
            self.gan_controller.generator.eval() 
            noise = torch.randn(1, self.gan_controller.latent_dim, device=self.gan_controller.device)
            with torch.no_grad(): generated_output = self.gan_controller.generator(noise)
            
            img_tensor_raw = generated_output[0].cpu() # Expected shape [C, H, W]
            
            img_tensor_norm = (img_tensor_raw + 1) / 2.0 
            img_tensor_norm = img_tensor_norm.clamp(0, 1) 
            
            pil_img = transforms.ToPILImage()(img_tensor_norm)
            pil_img_resized = pil_img.resize((128, 128), Image.Resampling.NEAREST) 

            self.root.after(0, self._update_generated_image_ui, pil_img_resized)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Image Generation Error", f"An error occurred: {e}"))
            print(f"Image generation error details:", exc_info=True) 
        finally:
            if hasattr(self, 'generate_image_button') and self.generate_image_button.winfo_exists():
                self.root.after(0, lambda: self.generate_image_button.config(state="normal"))

    def _update_generated_image_ui(self, pil_img):
        if hasattr(self, 'generated_image_label') and self.generated_image_label.winfo_exists(): 
            self.generated_image_pil_ref = ImageTk.PhotoImage(pil_img) 
            self.generated_image_label.config(image=self.generated_image_pil_ref, text="") # Clear text when image is shown


if __name__ == "__main__":
    main_root = tk.Tk()
    app = GanConfigurator(main_root)
    # Initial button state is set at the end of __init__ by calling update_button_states
    main_root.mainloop()