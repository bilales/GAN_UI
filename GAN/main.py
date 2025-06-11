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
# Matplotlib imports will be handled inside the thread for plotting

class GanConfigurator:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("Interactive GAN Training Framework (Tkinter)")
        self.root.geometry("950x850") # Adjusted for auto-switch panel

        # State Variables
        self.data_folder_var = tk.StringVar(value=os.path.abspath("./data"))
        self.loaded_config_data = None
        self.gan_controller = None
        self.training_active = False # Manages if training process is ongoing
        self.generated_image_pil_ref = None # For PhotoImage persistence

        # Auto-Switching UI Variables
        self.auto_switch_var = tk.BooleanVar(value=False)
        self.g_train_epochs_auto_var = tk.StringVar(value="1") # For auto-switch ratio
        self.d_train_epochs_auto_var = tk.StringVar(value="1") # For auto-switch ratio

        self.training_ui_callback = lambda message: self.root.after(0, self.update_training_stats_text, message)

        # --- Main Layout ---
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill="x", padx=10, pady=(10,0))
        ttk.Label(header_frame, text="GAN Configurator & Trainer", font=("Arial", 16, "bold")).pack(side="left")

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
        if self.training_active: 
            if not messagebox.askokcancel("Quit", "Training might be active. Stop threads and quit?"):
                return 
        
        if self.gan_controller:
            self.update_training_stats_text("Application closing: Signaling training threads to stop...")
            self.root.update_idletasks() 
            self.gan_controller.signal_stop_training_threads()
            self.update_training_stats_text("Waiting for threads to join (max 5s each)...")
            self.root.update_idletasks()
            self.gan_controller.join_training_threads(timeout=5)
            self.update_training_stats_text("Threads joined or timed out. Exiting.")
        self.root.destroy()

    def _build_preview_tab(self):
        frame = self.preview_frame
        
        top_section_frame = ttk.Frame(frame)
        top_section_frame.pack(fill="x", pady=10, padx=10)

        ttk.Label(top_section_frame, text="Preview of Loaded Network Architectures", font=("Arial", 12, "italic")).pack(side="left")
        self.load_config_button = ttk.Button(top_section_frame, text="Load Config JSON", command=self.load_json_configuration)
        self.load_config_button.pack(side="right")

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
            messagebox.showwarning("Config Load Denied", "Cannot load config while training. Stop first.")
            return
        _config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__))) # GAN package directory
        default_config_path = os.path.join(_config_dir, "config.json")

        file_path = filedialog.askopenfilename(
            title="Select GAN Configuration File", 
            initialdir=_config_dir, 
            filetypes=[("JSON files", "*.json")])
        
        if not file_path: 
            if os.path.exists(default_config_path):
                if messagebox.askyesno("Load Default?", f"No file selected. Load default '{os.path.basename(default_config_path)}' from the application directory?"):
                    file_path = default_config_path
                else:
                    self.update_training_stats_text("Configuration loading cancelled by user.")
                    return
            else:
                self.update_training_stats_text("Configuration loading cancelled. No default config.json found in app directory.")
                return

        try:
            with open(file_path, "r") as f:
                self.loaded_config_data = json.load(f)
            print(f"[DEBUG] Config loaded successfully. self.loaded_config_data is set: {self.loaded_config_data is not None}")
            self.update_training_stats_text(f"Configuration loaded from: {os.path.basename(file_path)}")
            self.populate_preview_from_config()
            self.populate_training_params_from_config() # This will also call _on_auto_switch_toggle
            self.update_summary_text_from_config()
        except Exception as e:
            messagebox.showerror("Configuration Error", f"Failed to load or parse configuration file '{os.path.basename(file_path)}':\n{e}")
            self.loaded_config_data = None 
            print(f"[DEBUG] Config load FAILED. self.loaded_config_data is None.")
            self.update_training_stats_text(f"ERROR loading config: {e}")
        finally:
            # This call is crucial to enable the start button if config load was successful
            print(f"[DEBUG] load_json_configuration finally block. training_active: {self.training_active}, loaded_config_data: {self.loaded_config_data is not None}")
            self.update_button_states() 

    def populate_preview_from_config(self):
        if not self.loaded_config_data:
            no_config_msg = "No configuration loaded. Please use 'Load Config JSON' button."
            for txt_widget in [self.gen_preview_text, self.disc_preview_text]:
                txt_widget.config(state="normal"); txt_widget.delete("1.0", tk.END); txt_widget.insert(tk.END, no_config_msg); txt_widget.config(state="disabled")
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
            txt_widget.config(state="normal"); txt_widget.delete("1.0", tk.END); 
            txt_widget.insert(tk.END, format_layers_preview(section_name)); txt_widget.config(state="disabled")

    def _build_training_params_tab(self):
        frame = self.train_params_frame
        
        params_outer_lf = ttk.LabelFrame(frame, text="Training Parameters")
        params_outer_lf.pack(pady=5, padx=10, fill="x") 

        params_container_frame = ttk.Frame(params_outer_lf)
        params_container_frame.pack(fill="x", expand=True, pady=2, padx=5)

        gen_lf = ttk.LabelFrame(params_container_frame, text="Generator")
        gen_lf.pack(side="left", padx=5, fill="y", expand=True)
        self.gen_loss_var = tk.StringVar(value="BCEWithLogitsLoss")
        self.gen_lr_var = tk.StringVar(value="0.0002")
        self.gen_epochs_var = tk.StringVar(value="100") 
        self.gen_batch_size_var = tk.StringVar(value="64")
        self._create_param_entry_row(gen_lf, "Loss Function:", self.gen_loss_var, is_combo=True, combo_values=["BCEWithLogitsLoss", "MSELoss", "BCELoss"])
        self._create_param_entry_row(gen_lf, "Learning Rate:", self.gen_lr_var)
        self._create_param_entry_row(gen_lf, "Max Manual Epochs:", self.gen_epochs_var) 
        self._create_param_entry_row(gen_lf, "Batch Size (G):", self.gen_batch_size_var)

        disc_lf = ttk.LabelFrame(params_container_frame, text="Discriminator")
        disc_lf.pack(side="left", padx=5, fill="y", expand=True)
        self.disc_loss_var = tk.StringVar(value="BCEWithLogitsLoss")
        self.disc_lr_var = tk.StringVar(value="0.0002")
        self.disc_epochs_var = tk.StringVar(value="100")
        self.disc_batch_size_var = tk.StringVar(value="64")
        self._create_param_entry_row(disc_lf, "Loss Function:", self.disc_loss_var, is_combo=True, combo_values=["BCEWithLogitsLoss", "MSELoss", "BCELoss"])
        self._create_param_entry_row(disc_lf, "Learning Rate:", self.disc_lr_var)
        self._create_param_entry_row(disc_lf, "Max Manual Epochs:", self.disc_epochs_var)
        self._create_param_entry_row(disc_lf, "Batch Size (D):", self.disc_batch_size_var)
        
        common_lf = ttk.LabelFrame(frame, text="Common Settings")
        common_lf.pack(pady=5, padx=10, fill="x")
        data_frame = ttk.Frame(common_lf); data_frame.pack(fill="x", pady=2, padx=5)
        ttk.Label(data_frame, text="Data Folder:", width=15).pack(side="left")
        self.data_folder_entry = ttk.Entry(data_frame, textvariable=self.data_folder_var, width=40)
        self.data_folder_entry.pack(side="left", padx=5, expand=True, fill="x")
        ttk.Button(data_frame, text="Browse...", command=self.select_data_folder, width=10).pack(side="left")
        initial_net_frame = ttk.Frame(common_lf); initial_net_frame.pack(fill="x", pady=2, padx=5)
        ttk.Label(initial_net_frame, text="Initial Active Net:", width=15).pack(side="left")
        self.initial_train_choice_var = tk.StringVar(value="generator")
        ttk.Radiobutton(initial_net_frame, text="Generator", variable=self.initial_train_choice_var, value="generator").pack(side="left", padx=10)
        ttk.Radiobutton(initial_net_frame, text="Discriminator", variable=self.initial_train_choice_var, value="discriminator").pack(side="left", padx=10)

        auto_switch_lf = ttk.LabelFrame(frame, text="Automatic G/D Switching Cycle")
        auto_switch_lf.pack(pady=5, padx=10, fill="x")
        auto_switch_check_frame = ttk.Frame(auto_switch_lf); auto_switch_check_frame.pack(fill="x", padx=5, pady=2)
        self.auto_switch_checkbox = ttk.Checkbutton(auto_switch_check_frame, text="Enable Auto-Switching", variable=self.auto_switch_var, command=self._on_auto_switch_toggle)
        self.auto_switch_checkbox.pack(side="left")
        ratio_frame = ttk.Frame(auto_switch_lf); ratio_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(ratio_frame, text="Train G for:").pack(side="left", padx=(0,2))
        self.g_epochs_auto_entry = ttk.Entry(ratio_frame, textvariable=self.g_train_epochs_auto_var, width=5, state="disabled")
        self.g_epochs_auto_entry.pack(side="left")
        ttk.Label(ratio_frame, text="epoch(s), then D for:").pack(side="left", padx=(5,2))
        self.d_epochs_auto_entry = ttk.Entry(ratio_frame, textvariable=self.d_train_epochs_auto_var, width=5, state="disabled")
        self.d_epochs_auto_entry.pack(side="left")
        ttk.Label(ratio_frame, text="epoch(s). Repeats.").pack(side="left", padx=(5,0))

        control_lf = ttk.LabelFrame(frame, text="Training Controls")
        control_lf.pack(pady=5, padx=10, fill="x") 
        btn_container = ttk.Frame(control_lf); btn_container.pack(pady=5) 
        btn_width = 16 
        self.start_btn = ttk.Button(btn_container, text="Start Training", command=self.start_gan_training, width=btn_width)
        self.start_btn.grid(row=0, column=0, padx=3, pady=3, sticky="ew")
        self.pause_btn = ttk.Button(btn_container, text="Pause Active", command=self.pause_active_training, width=btn_width)
        self.pause_btn.grid(row=0, column=1, padx=3, pady=3, sticky="ew")
        self.resume_btn = ttk.Button(btn_container, text="Resume Active", command=self.resume_active_training, width=btn_width)
        self.resume_btn.grid(row=0, column=2, padx=3, pady=3, sticky="ew")
        self.switch_btn = ttk.Button(btn_container, text="Switch Active Net", command=self.switch_active_network, width=btn_width)
        self.switch_btn.grid(row=1, column=0, padx=3, pady=3, sticky="ew")
        self.stop_btn = ttk.Button(btn_container, text="Stop All Training", command=self.stop_all_training, width=btn_width)
        self.stop_btn.grid(row=1, column=1, padx=3, pady=3, columnspan=2, sticky="ew")
        for i in range(3): btn_container.columnconfigure(i, weight=1) # Make button columns expand
        
        stats_lf = ttk.LabelFrame(frame, text="Training Log / Status")
        stats_lf.pack(pady=5, padx=10, fill="both", expand=True)
        self.training_stats_text = tk.Text(stats_lf, height=8, wrap="word", relief="sunken", borderwidth=1)
        stats_scroll = ttk.Scrollbar(stats_lf, command=self.training_stats_text.yview)
        self.training_stats_text['yscrollcommand'] = stats_scroll.set
        stats_scroll.pack(side="right", fill="y")
        self.training_stats_text.pack(side="left",fill="both", expand=True, padx=(5,0), pady=5)

    def _on_auto_switch_toggle(self):
        entry_state = "normal" if self.auto_switch_var.get() else "disabled"
        # Ensure these widgets exist before trying to config them (might be called early)
        if hasattr(self, 'g_epochs_auto_entry'): self.g_epochs_auto_entry.config(state=entry_state)
        if hasattr(self, 'd_epochs_auto_entry'): self.d_epochs_auto_entry.config(state=entry_state)
        
        if self.auto_switch_var.get():
            self.update_training_stats_text("Automatic switching ENABLED.")
        else:
            self.update_training_stats_text("Automatic switching DISABLED.")
        self.update_button_states() # Update manual switch button state based on this toggle

    def _create_param_entry_row(self, parent, label_text, string_var, is_combo=False, combo_values=None, entry_width=18):
        row_frame = ttk.Frame(parent); row_frame.pack(fill="x", padx=5, pady=3) 
        ttk.Label(row_frame, text=label_text, width=15, anchor="w").pack(side="left")
        widget = ttk.Combobox(row_frame, textvariable=string_var, values=combo_values or [], width=entry_width - 2, state="readonly") if is_combo else ttk.Entry(row_frame, textvariable=string_var, width=entry_width)
        widget.pack(side="left", padx=5, fill="x", expand=True)

    def populate_training_params_from_config(self):
        if not self.loaded_config_data or "training" not in self.loaded_config_data:
            self.update_training_stats_text("No 'training' section in config, UI defaults preserved.")
            return # Keep existing UI values if no training section
        training_conf = self.loaded_config_data["training"]; s_get = lambda d, k, v_obj: d.get(k,v_obj.get()) # Helper
        g_p,d_p = training_conf.get("generator",{}), training_conf.get("discriminator",{})
        
        # Populate G & D params
        self.gen_loss_var.set(s_get(g_p,"loss_function",self.gen_loss_var))
        self.gen_lr_var.set(str(s_get(g_p,"learning_rate",self.gen_lr_var)))
        self.gen_epochs_var.set(str(s_get(g_p,"epochs",self.gen_epochs_var))) # Max manual epochs
        self.gen_batch_size_var.set(str(s_get(g_p,"batch_size",self.gen_batch_size_var)))

        self.disc_loss_var.set(s_get(d_p,"loss_function",self.disc_loss_var))
        self.disc_lr_var.set(str(s_get(d_p,"learning_rate",self.disc_lr_var)))
        self.disc_epochs_var.set(str(s_get(d_p,"epochs",self.disc_epochs_var))) # Max manual epochs
        self.disc_batch_size_var.set(str(s_get(d_p,"batch_size",self.disc_batch_size_var)))
        
        self.data_folder_var.set(s_get(training_conf,"data_folder",self.data_folder_var))
        self.initial_train_choice_var.set(s_get(training_conf,"initial_network",self.initial_train_choice_var))
        
        # Populate Auto-Switch params from config if they exist
        auto_s_conf = training_conf.get("automatic_switching", {}) # Default to empty dict
        self.auto_switch_var.set(auto_s_conf.get("enabled", False)) # Default to False if not in config
        self.g_train_epochs_auto_var.set(str(auto_s_conf.get("g_epochs", "1")))
        self.d_train_epochs_auto_var.set(str(auto_s_conf.get("d_epochs", "1")))
        self._on_auto_switch_toggle() # Update entry states based on loaded auto_switch_var
        self.update_training_stats_text("Training parameters populated from config.")

    def get_current_ui_training_config(self):
        try:
            config = {
                "generator": { 
                    "loss_function": self.gen_loss_var.get(), 
                    "learning_rate": float(self.gen_lr_var.get()), 
                    "epochs": int(self.gen_epochs_var.get()), # Max manual epochs
                    "batch_size": int(self.gen_batch_size_var.get()) 
                },
                "discriminator": { 
                    "loss_function": self.disc_loss_var.get(), 
                    "learning_rate": float(self.disc_lr_var.get()), 
                    "epochs": int(self.disc_epochs_var.get()), # Max manual epochs
                    "batch_size": int(self.disc_batch_size_var.get()) 
                },
                "data_folder": self.data_folder_var.get(),
                "initial_network": self.initial_train_choice_var.get().lower()
            }
            # Add automatic_switching configuration
            if self.auto_switch_var.get():
                g_auto_epochs = int(self.g_train_epochs_auto_var.get())
                d_auto_epochs = int(self.d_train_epochs_auto_var.get())
                if g_auto_epochs <= 0 or d_auto_epochs <= 0: 
                    raise ValueError("Auto-switch epoch counts must be positive integers.")
                config["automatic_switching"] = {
                    "enabled": True, 
                    "g_epochs": g_auto_epochs, 
                    "d_epochs": d_auto_epochs
                }
            else:
                config["automatic_switching"] = {"enabled": False}

            print(f"[DEBUG] Current UI Training Config: {json.dumps(config, indent=2)}") 
            return config
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid training parameter value: {e}.\nPlease ensure numeric fields (LR, Epochs, Batch Size) contain valid numbers.")
            return None

    def select_data_folder(self):
        current_path = self.data_folder_var.get(); 
        initial_dir = os.path.dirname(current_path) if os.path.isdir(current_path) and os.path.exists(current_path) else os.path.expanduser("~")
        if not os.path.exists(initial_dir): initial_dir = os.path.expanduser("~") # Fallback if dir doesn't exist

        folder = filedialog.askdirectory(title="Select Data Folder (e.g., for MNIST dataset)", initialdir=initial_dir)
        if folder:
            self.data_folder_var.set(os.path.abspath(folder))
            self.update_training_stats_text(f"Data folder set to: {folder}")

    def update_button_states(self): 
        print(f"[DEBUG] update_button_states: training_active={self.training_active}, loaded_config={self.loaded_config_data is not None}, auto_switch={self.auto_switch_var.get()}")
        if not hasattr(self, 'start_btn'): return # Widgets might not be created yet

        # Load Config button state
        if hasattr(self, 'load_config_button'): 
            self.load_config_button.config(state="normal" if not self.training_active else "disabled")
        
        # Auto-switch checkbox and entries state
        if hasattr(self, 'auto_switch_checkbox'):
            self.auto_switch_checkbox.config(state="normal" if not self.training_active else "disabled")
        if self.training_active: # If training, auto-switch entries are also disabled
             if hasattr(self, 'g_epochs_auto_entry'): self.g_epochs_auto_entry.config(state="disabled")
             if hasattr(self, 'd_epochs_auto_entry'): self.d_epochs_auto_entry.config(state="disabled")
        else: # If not training, state of entries depends on checkbox (handled by _on_auto_switch_toggle)
            if hasattr(self, '_on_auto_switch_toggle'): self._on_auto_switch_toggle()


        # Start button state
        can_start = bool(self.loaded_config_data) and not self.training_active
        self.start_btn.config(state="normal" if can_start else "disabled")

        # Other control buttons state
        if not self.training_active: 
            self.pause_btn.config(state="disabled"); self.resume_btn.config(state="disabled")
            self.switch_btn.config(state="disabled"); self.stop_btn.config(state="disabled")
        else: # Training is active (could be running or paused)
            self.stop_btn.config(state="normal")
            # Manual switch button enabled only if training is active AND auto-switch is OFF
            self.switch_btn.config(state="normal" if not self.auto_switch_var.get() else "disabled")
            
            is_paused_for_active_net = True # Default to assuming paused if controller/trainer not fully ready
            if self.gan_controller and self.gan_controller.trainer:
                active_net = self.gan_controller.trainer.current_network
                pause_event = getattr(self.gan_controller.trainer, f'pause_event_{active_net}', None)
                if pause_event: 
                    is_paused_for_active_net = not pause_event.is_set()
            
            self.pause_btn.config(state="normal" if not is_paused_for_active_net else "disabled")
            self.resume_btn.config(state="normal" if is_paused_for_active_net else "disabled")
        self.root.update_idletasks()


    def start_gan_training(self):
        if self.training_active: 
            messagebox.showinfo("Info", "Training is already active. Please stop the current session before starting a new one.")
            return
        if not self.loaded_config_data: 
            messagebox.showerror("Error", "No network configuration loaded. Please load a JSON config file using the button in the 'Network Architectures' tab first.")
            self.update_button_states() 
            return
        
        current_ui_train_config = self.get_current_ui_training_config() # This now includes auto-switch config
        if not current_ui_train_config: 
            self.update_button_states() 
            return
        
        self.start_btn.config(state="disabled") 
        self.root.update_idletasks() 
        self.update_training_stats_text("Initializing GAN Controller...")
        try:
            if self.gan_controller: 
                if (hasattr(self.gan_controller, 'generator_thread') and self.gan_controller.generator_thread and self.gan_controller.generator_thread.is_alive()) or \
                   (hasattr(self.gan_controller, 'discriminator_thread') and self.gan_controller.discriminator_thread and self.gan_controller.discriminator_thread.is_alive()):
                    self.update_training_stats_text("Ensuring previous threads (if any) are stopped before restart...")
                    self.gan_controller.signal_stop_training_threads()
                    self.gan_controller.join_training_threads(timeout=1) # Quick join
                del self.gan_controller # Allow garbage collection for a full reset
            
            self.gan_controller = GANController(
                self.loaded_config_data.get("generator",{}), 
                self.loaded_config_data.get("discriminator",{}), 
                current_ui_train_config # Pass the full config including auto-switch settings
            )
            self.update_training_stats_text("Controller initialized. Starting training threads...")
            self.root.update_idletasks()

            self.gan_controller.start_persistent_training(self.training_ui_callback)
            self.training_active = True 
            self.update_training_stats_text("Training threads initiated.")
        except Exception as e:
            self.training_active = False 
            self.gan_controller = None 
            messagebox.showerror("Training Start Error", f"Failed to start training session: {e}")
            self.update_training_stats_text(f"FATAL ERROR starting training: {e}")
            print(f"Full traceback for training start error:", exc_info=True)
        finally: 
            self.update_button_states()


    def pause_active_training(self):
        if not self.training_active or not self.gan_controller: 
            self.update_training_stats_text("Cannot pause: No active training.")
            return
        active_net = self.gan_controller.trainer.current_network
        self.gan_controller.pause_training()
        self.update_training_stats_text(f"Paused {active_net} training.")
        self.update_button_states()


    def resume_active_training(self):
        if not self.training_active or not self.gan_controller: 
            self.update_training_stats_text("Cannot resume: No active training.")
            return
        current_ui_train_config = self.get_current_ui_training_config() # Get latest params including auto-switch
        if not current_ui_train_config: 
            self.update_button_states()
            return
        self.gan_controller.update_runtime_train_params_from_ui(current_ui_train_config) # Update controller
            
        active_net_to_resume = self.gan_controller.trainer.current_network
        self.gan_controller.resume_training()
        self.update_training_stats_text(f"Resumed {active_net_to_resume} training.")
        self.update_button_states()
    
    def switch_active_network(self): # Manual Switch
        if not self.training_active or not self.gan_controller: 
            self.update_training_stats_text("Cannot switch: No active training.")
            return
        if self.auto_switch_var.get():
            self.update_training_stats_text("Manual switch disabled: Auto-switching is currently ON.")
            return
        
        old_active = self.gan_controller.trainer.current_network
        self.gan_controller.switch_network(manual_switch=True) # Signal to controller it's a manual switch
        new_active = self.gan_controller.trainer.current_network
        self.update_training_stats_text(f"MANUAL Switch. Was: {old_active}, Now active: {new_active}.")
        self.update_button_states()
    
    def stop_all_training(self):
        if not self.training_active or not self.gan_controller:
            self.update_training_stats_text("Stop command: No training was active.")
            if not self.training_active: self.update_button_states() 
            return

        self.update_training_stats_text("Signaling training threads to stop...")
        self.stop_btn.config(text="Stopping...", state="disabled")
        # Disable other controls immediately
        for btn_attr in ['pause_btn', 'resume_btn', 'switch_btn', 'start_btn', 'load_config_button', 'auto_switch_checkbox']:
            if hasattr(self, btn_attr) and getattr(self, btn_attr).winfo_exists(): 
                getattr(self, btn_attr).config(state="disabled")
        # Also disable auto-switch ratio entries
        if hasattr(self, 'g_epochs_auto_entry'): self.g_epochs_auto_entry.config(state="disabled")
        if hasattr(self, 'd_epochs_auto_entry'): self.d_epochs_auto_entry.config(state="disabled")
        self.root.update_idletasks()

        self.gan_controller.signal_stop_training_threads()
        
        def _check_threads_stopped_after_signal(attempts_left=10):
            controller_exists = bool(self.gan_controller)
            threads_gone = controller_exists and \
               ((getattr(self.gan_controller,'generator_thread',None) is None or not self.gan_controller.generator_thread.is_alive()) and \
                (getattr(self.gan_controller,'discriminator_thread',None) is None or not self.gan_controller.discriminator_thread.is_alive()))

            if threads_gone or not controller_exists: 
                self.update_training_stats_text("Training threads have terminated or controller is cleared.")
                self.training_active = False
                # self.gan_controller = None # Decide if you want to clear controller here or allow post-stop actions
            elif attempts_left > 0 :
                self.update_training_stats_text(f"Waiting for threads to confirm stop ({attempts_left} checks left)..."); 
                self.root.after(500, lambda: _check_threads_stopped_after_signal(attempts_left - 1))
                return # Don't update buttons fully yet, still stopping
            else: 
                self.update_training_stats_text("Threads stop confirmation timeout. Assuming stopped for UI state."); 
                self.training_active = False
            
            # This block runs if threads stopped, no controller, or timeout reached
            self.update_button_states() # This will re-enable start/load config etc.
            if hasattr(self, 'stop_btn') and self.stop_btn.winfo_exists():
                 self.stop_btn.config(text="Stop All Training") # Reset button text (will be re-disabled by update_button_states if training_active is False)
        
        self.root.after(200, lambda: _check_threads_stopped_after_signal())


    def update_training_stats_text(self, message):
        if hasattr(self, 'training_stats_text') and self.training_stats_text.winfo_exists(): 
            current_time = time.strftime("%H:%M:%S", time.localtime()); 
            self.training_stats_text.insert(tk.END, f"{current_time} - {str(message)}\n"); 
            self.training_stats_text.see(tk.END) 

    def _build_summary_tab(self):
        frame = self.summary_frame; ttk.Label(frame, text="Output & Configuration Summary", font=("Arial", 12, "italic")).pack(pady=(5,10))
        c_frame = ttk.Frame(frame); c_frame.pack(pady=5)
        ttk.Button(c_frame, text="Refresh Summary Text", command=self.update_summary_text_from_config).pack(side="left", padx=5)
        self.plot_losses_button = ttk.Button(c_frame, text="Plot Training Losses", command=self.plot_training_losses); self.plot_losses_button.pack(side="left", padx=5)
        self.summary_text_widget = tk.Text(frame, height=12, width=80, wrap="word", state="disabled", relief="sunken", borderwidth=1)
        s_scroll = ttk.Scrollbar(frame, command=self.summary_text_widget.yview); self.summary_text_widget['yscrollcommand'] = s_scroll.set
        s_scroll.pack(side="right", fill="y", padx=(0,5), pady=5); self.summary_text_widget.pack(pady=5, padx=(5,0), fill="both", expand=True)
        gen_lf = ttk.LabelFrame(frame, text="Generated Image Preview"); gen_lf.pack(pady=10, padx=10, fill="x")
        self.generate_image_button = ttk.Button(gen_lf, text="Generate & Show Image", command=self.generate_and_show_image); self.generate_image_button.pack(pady=5)
        self.image_display_frame = ttk.Frame(gen_lf, width=130, height=130, relief="sunken", borderwidth=1); self.image_display_frame.pack(pady=5, anchor="center") 
        self.image_display_frame.pack_propagate(False); self.generated_image_label = ttk.Label(self.image_display_frame, text="No image", compound="image", anchor="center")
        self.generated_image_label.pack(fill="both", expand=True)

    def update_summary_text_from_config(self):
        s = "No config loaded." if not self.loaded_config_data else f"=== Loaded Config ===\n\n--- Generator ---\n{json.dumps(self.loaded_config_data.get('generator',{}),indent=2)}\n\n--- Discriminator ---\n{json.dumps(self.loaded_config_data.get('discriminator',{}),indent=2)}\n\n--- Training Defaults ---\n{json.dumps(self.loaded_config_data.get('training',{}),indent=2)}\n"
        self.summary_text_widget.config(state="normal"); self.summary_text_widget.delete("1.0",tk.END); self.summary_text_widget.insert(tk.END,s); self.summary_text_widget.config(state="disabled")
    
    def plot_training_losses(self):
        if not self.gan_controller or not self.gan_controller.get_trainer_instance(): messagebox.showinfo("Plot Info", "No training session or data."); return
        trainer = self.gan_controller.get_trainer_instance()
        if not trainer.gen_losses and not trainer.disc_losses: messagebox.showinfo("Plot Info", "No loss data recorded."); return
        if hasattr(self, 'plot_losses_button') : self.plot_losses_button.config(state="disabled")
        threading.Thread(target=self._plot_losses_thread_task, daemon=True).start()

    def _plot_losses_thread_task(self):
        plot_filepath = "training_losses.png" 
        try:
            import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt 
            trainer = self.gan_controller.get_trainer_instance()
            if trainer and (trainer.gen_losses or trainer.disc_losses):
                fig = plt.figure(figsize=(10,5))
                if trainer.gen_losses: plt.plot(trainer.gen_losses, label="G Loss")
                if trainer.disc_losses: plt.plot(trainer.disc_losses, label="D Loss")
                plt.xlabel("Epoch (respective)"); plt.ylabel("Loss"); plt.title("Training Losses"); plt.legend(); plt.grid(True)
                plt.savefig(plot_filepath,bbox_inches='tight'); plt.close(fig) 
                self.root.after(0, lambda: messagebox.showinfo("Plot Saved", f"Plot saved to:\n{os.path.abspath(plot_filepath)}"))
                self.root.after(0, lambda: self.update_training_stats_text(f"Loss plot saved: {plot_filepath}"))
            else: self.root.after(0, lambda: messagebox.showinfo("Plot Info", "No loss data to plot."))
        except Exception as e: self.root.after(0, lambda: messagebox.showerror("Plot Error", f"{e}")); print(f"Plot error:", exc_info=True)
        finally:
            if hasattr(self,'plot_losses_button') and self.plot_losses_button.winfo_exists(): self.root.after(0,lambda:self.plot_losses_button.config(state="normal"))
    
    def generate_and_show_image(self): 
        if not self.gan_controller or not self.gan_controller.generator: messagebox.showerror("Gen Error", "Generator not ready."); return
        if hasattr(self, 'generate_image_button'): self.generate_image_button.config(state="disabled")
        threading.Thread(target=self._generate_image_thread_task, daemon=True).start()

    def _generate_image_thread_task(self): 
        try:
            self.gan_controller.generator.eval(); noise = torch.randn(1, self.gan_controller.latent_dim, device=self.gan_controller.device)
            with torch.no_grad(): generated_output = self.gan_controller.generator(noise)
            img_raw = generated_output[0].cpu(); img_norm = ((img_raw + 1) / 2.0).clamp(0,1)
            pil_img = transforms.ToPILImage()(img_norm); pil_resized = pil_img.resize((128,128), Image.Resampling.NEAREST)
            self.root.after(0, self._update_generated_image_ui, pil_resized)
        except Exception as e: self.root.after(0, lambda: messagebox.showerror("Image Gen Error", f"{e}")); print(f"Image gen error:", exc_info=True) 
        finally:
            if hasattr(self,'generate_image_button') and self.generate_image_button.winfo_exists(): self.root.after(0,lambda:self.generate_image_button.config(state="normal"))

    def _update_generated_image_ui(self, pil_img):
        if hasattr(self,'generated_image_label') and self.generated_image_label.winfo_exists(): 
            self.generated_image_pil_ref = ImageTk.PhotoImage(pil_img); self.generated_image_label.config(image=self.generated_image_pil_ref, text="")

if __name__ == "__main__":
    main_root = tk.Tk()
    app = GanConfigurator(main_root)
    main_root.mainloop()