# main.py
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from .controller import GANController 
from .model_builder import detect_gpu 
import json
import threading 
from PIL import Image, ImageTk
import torch
import os 
from torchvision import transforms # Ensure this is imported

class GanConfigurator:
    def __init__(self, root_window): 
        self.root = root_window
        self.root.title("Interactive GAN Training Framework")
        self.root.geometry("1000x800") 
        
        self.data_folder_var = tk.StringVar(value=os.path.abspath("./data")) 
        self.loaded_config_data = None 
        
        self.gan_controller = None 
        self.training_active = False # To manage button states primarily
        
        self.training_ui_callback = lambda message: self.root.after(0, self.update_training_stats_text, message)
        
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.preview_frame = ttk.Frame(self.notebook)
        self.train_params_frame = ttk.Frame(self.notebook) 
        self.summary_frame = ttk.Frame(self.notebook)
        
        self.notebook.add(self.preview_frame, text="1. Network Preview")
        self.notebook.add(self.train_params_frame, text="2. Training Parameters & Controls")
        self.notebook.add(self.summary_frame, text="3. Summary & Generation")
        
        self.build_preview_tab()
        self.build_training_params_tab()
        self.build_summary_tab()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing) 
        self.generated_image_pil_ref = None # For PhotoImage persistence

        self.update_training_stats_text(f"GPU Status: {detect_gpu()}")
        self.update_training_stats_text("Tips: 1. Load Config. 2. Set Params. 3. Start Training.")


    def on_closing(self):
        if self.training_active:
            if not messagebox.askokcancel("Quit", "Training is active. Stop training and quit?"):
                return # User cancelled quit
        
        if self.gan_controller:
            self.update_training_stats_text("Application closing: Signaling threads to stop...")
            self.root.update_idletasks()
            self.gan_controller.signal_stop_training_threads() 
            self.update_training_stats_text("Waiting for threads to join (max 5s each)...")
            self.root.update_idletasks()
            self.gan_controller.join_training_threads(timeout=5) 
            self.update_training_stats_text("Threads joined or timed out.")
        self.root.destroy()

    def build_preview_tab(self):
        frame = self.preview_frame
        top_frame = ttk.Frame(frame)
        top_frame.pack(pady=10, fill="x")
        ttk.Label(top_frame, text="Network Architecture Preview", 
                  font=("Arial", 14, "bold")).pack(side="left", padx=10)
        ttk.Button(top_frame, text="Load Configuration File (*.json)", command=self.load_json_configuration).pack(side="right", padx=10)
        
        preview_container = ttk.Frame(frame)
        preview_container.pack(fill="both", expand=True, padx=10, pady=5)
        preview_container.columnconfigure(0, weight=1)
        preview_container.columnconfigure(1, weight=1)

        gen_lf = ttk.LabelFrame(preview_container, text="Generator Architecture")
        gen_lf.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.gen_preview_text = tk.Text(gen_lf, height=20, width=50, state="disabled", wrap="word", relief="sunken", borderwidth=1)
        gen_scroll = ttk.Scrollbar(gen_lf, command=self.gen_preview_text.yview)
        self.gen_preview_text['yscrollcommand'] = gen_scroll.set
        gen_scroll.pack(side="right", fill="y")
        self.gen_preview_text.pack(side="left", fill="both", expand=True, padx=(5,0), pady=5)
        
        disc_lf = ttk.LabelFrame(preview_container, text="Discriminator Architecture")
        disc_lf.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.disc_preview_text = tk.Text(disc_lf, height=20, width=50, state="disabled", wrap="word", relief="sunken", borderwidth=1)
        disc_scroll = ttk.Scrollbar(disc_lf, command=self.disc_preview_text.yview)
        self.disc_preview_text['yscrollcommand'] = disc_scroll.set
        disc_scroll.pack(side="right", fill="y")
        self.disc_preview_text.pack(side="left", fill="both", expand=True, padx=(5,0), pady=5)
    
    def load_json_configuration(self):
        if self.training_active:
            messagebox.showwarning("Warning", "Cannot load new configuration while training is active. Please stop training first.")
            return

        initial_dir = os.path.dirname(self.data_folder_var.get()) # Start browser near current data folder
        config_path_in_package = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

        file_path = filedialog.askopenfilename(
            title="Select Configuration File", 
            initialdir=os.path.dirname(config_path_in_package), # Start in package dir
            filetypes=[("JSON files", "*.json")])
        
        if not file_path: 
            if os.path.exists(config_path_in_package):
                if messagebox.askyesno("Confirm", "No file selected. Load default 'config.json' from package directory?"):
                    file_path = config_path_in_package
                else:
                    return
            else:
                return

        try:
            with open(file_path, "r") as f:
                self.loaded_config_data = json.load(f)
            self.update_training_stats_text(f"Configuration loaded from: {os.path.basename(file_path)}")
            self.populate_preview_from_config()
            self.populate_training_params_from_config() 
            self.update_summary_text_from_config() 
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
            for i, layer in enumerate(section.get("layers", [])):
                layer_desc = f"    {i+1}. Type: {layer.get('type', '?')}"
                details = []
                for k, v in layer.items():
                    if k.lower() not in ['type']: details.append(f"{k}: {v}")
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
        ttk.Label(frame, text="Live Training Parameters & Controls", font=("Arial", 14, "bold")).pack(pady=10)
        
        params_container = ttk.Frame(frame)
        params_container.pack(pady=5, fill="x", padx=10)
        params_container.columnconfigure(0, weight=1)
        params_container.columnconfigure(1, weight=1)
        
        def create_param_entry(parent, label_text, string_var, width=12):
            row_frame = ttk.Frame(parent)
            row_frame.pack(fill="x", pady=2, padx=5)
            ttk.Label(row_frame, text=label_text, width=15).pack(side="left")
            entry = ttk.Entry(row_frame, textvariable=string_var, width=width)
            entry.pack(side="left", padx=5, fill="x", expand=True)

        # Generator Params UI
        gen_lf = ttk.LabelFrame(params_container, text="Generator Training")
        gen_lf.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.gen_loss_var = tk.StringVar(value="BCEWithLogitsLoss")
        self.gen_lr_var = tk.StringVar(value="0.0002")
        self.gen_epochs_var = tk.StringVar(value="100") 
        self.gen_batch_size_var = tk.StringVar(value="64")

        combo_frame_g = ttk.Frame(gen_lf)
        combo_frame_g.pack(fill="x", pady=2, padx=5)
        ttk.Label(combo_frame_g, text="Loss Function:", width=15).pack(side="left")
        self.gen_loss_combo = ttk.Combobox(combo_frame_g, textvariable=self.gen_loss_var, 
                                           values=["BCEWithLogitsLoss", "MSELoss", "BCELoss"], width=15) # Adjusted width
        self.gen_loss_combo.pack(side="left", padx=5, fill="x", expand=True)
        create_param_entry(gen_lf, "Learning Rate:", self.gen_lr_var)
        create_param_entry(gen_lf, "Max Epochs:", self.gen_epochs_var)
        create_param_entry(gen_lf, "Batch Size:", self.gen_batch_size_var)

        # Discriminator Params UI
        disc_lf = ttk.LabelFrame(params_container, text="Discriminator Training")
        disc_lf.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.disc_loss_var = tk.StringVar(value="BCEWithLogitsLoss")
        self.disc_lr_var = tk.StringVar(value="0.0002")
        self.disc_epochs_var = tk.StringVar(value="100") 
        self.disc_batch_size_var = tk.StringVar(value="64")

        combo_frame_d = ttk.Frame(disc_lf)
        combo_frame_d.pack(fill="x", pady=2, padx=5)
        ttk.Label(combo_frame_d, text="Loss Function:", width=15).pack(side="left")
        self.disc_loss_combo = ttk.Combobox(combo_frame_d, textvariable=self.disc_loss_var, 
                                            values=["BCEWithLogitsLoss", "MSELoss", "BCELoss"], width=15)
        self.disc_loss_combo.pack(side="left", padx=5, fill="x", expand=True)
        create_param_entry(disc_lf, "Learning Rate:", self.disc_lr_var)
        create_param_entry(disc_lf, "Max Epochs:", self.disc_epochs_var)
        create_param_entry(disc_lf, "Batch Size:", self.disc_batch_size_var)

        common_lf = ttk.LabelFrame(frame, text="Common Settings")
        common_lf.pack(pady=10, fill="x", padx=10)

        data_frame = ttk.Frame(common_lf)
        data_frame.pack(fill="x", pady=5, padx=5)
        ttk.Label(data_frame, text="Data Folder:", width=15).pack(side="left")
        self.data_folder_entry = ttk.Entry(data_frame, textvariable=self.data_folder_var, width=40)
        self.data_folder_entry.pack(side="left", padx=5, expand=True, fill="x")
        ttk.Button(data_frame, text="Browse", command=self.select_data_folder, width=8).pack(side="left")
        
        initial_net_frame = ttk.Frame(common_lf)
        initial_net_frame.pack(fill="x", pady=5, padx=5)
        ttk.Label(initial_net_frame, text="Initial Active Net:", width=15).pack(side="left")
        self.initial_train_choice_var = tk.StringVar(value="generator")
        ttk.Radiobutton(initial_net_frame, text="Generator", variable=self.initial_train_choice_var, value="generator").pack(side="left", padx=10)
        ttk.Radiobutton(initial_net_frame, text="Discriminator", variable=self.initial_train_choice_var, value="discriminator").pack(side="left", padx=10)

        control_lf = ttk.LabelFrame(frame, text="Training Controls")
        control_lf.pack(pady=10, fill="x", padx=10)
        btn_frame = ttk.Frame(control_lf)
        btn_frame.pack(pady=5)
        
        btn_width = 17 # Standardize button width
        self.start_btn = ttk.Button(btn_frame, text="Start Training", command=self.start_gan_training, width=btn_width)
        self.start_btn.pack(side="left", padx=3)
        self.pause_btn = ttk.Button(btn_frame, text="Pause Active", command=self.pause_active_training, width=btn_width, state="disabled")
        self.pause_btn.pack(side="left", padx=3)
        self.resume_btn = ttk.Button(btn_frame, text="Resume Active", command=self.resume_active_training, width=btn_width, state="disabled")
        self.resume_btn.pack(side="left", padx=3)
        self.switch_btn = ttk.Button(btn_frame, text="Switch Active Net", command=self.switch_active_network, width=btn_width, state="disabled")
        self.switch_btn.pack(side="left", padx=3)
        self.stop_btn = ttk.Button(btn_frame, text="Stop All Training", command=self.stop_all_training, width=btn_width, state="disabled")
        self.stop_btn.pack(side="left", padx=3)
        
        stats_lf = ttk.LabelFrame(frame, text="Training Log / Status")
        stats_lf.pack(pady=10, fill="both", expand=True, padx=10)
        self.training_stats_text = tk.Text(stats_lf, height=12, width=80, wrap="word", relief="sunken", borderwidth=1)
        stats_scroll = ttk.Scrollbar(stats_lf, command=self.training_stats_text.yview)
        self.training_stats_text['yscrollcommand'] = stats_scroll.set
        stats_scroll.pack(side="right", fill="y")
        self.training_stats_text.pack(side="left",fill="both", expand=True, padx=(5,0), pady=5)

    def populate_training_params_from_config(self):
        if not self.loaded_config_data or "training" not in self.loaded_config_data:
            self.update_training_stats_text("No 'training' section in config, using UI defaults.")
            return

        training_conf = self.loaded_config_data["training"]
        gen_train = training_conf.get("generator", {})
        disc_train = training_conf.get("discriminator", {})

        self.gen_loss_var.set(gen_train.get("loss_function", self.gen_loss_var.get()))
        self.gen_lr_var.set(str(gen_train.get("learning_rate", self.gen_lr_var.get())))
        self.gen_epochs_var.set(str(gen_train.get("epochs", self.gen_epochs_var.get())))
        self.gen_batch_size_var.set(str(gen_train.get("batch_size", self.gen_batch_size_var.get())))

        self.disc_loss_var.set(disc_train.get("loss_function", self.disc_loss_var.get()))
        self.disc_lr_var.set(str(disc_train.get("learning_rate", self.disc_lr_var.get())))
        self.disc_epochs_var.set(str(disc_train.get("epochs", self.disc_epochs_var.get())))
        self.disc_batch_size_var.set(str(disc_train.get("batch_size", self.disc_batch_size_var.get())))
        
        self.data_folder_var.set(training_conf.get("data_folder", self.data_folder_var.get()))
        self.initial_train_choice_var.set(training_conf.get("initial_network", self.initial_train_choice_var.get()))
        self.update_training_stats_text("Training parameters populated from config.")


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
            return config
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid training parameter value: {e}. Please check numeric fields.")
            return None

    def select_data_folder(self):
        folder = filedialog.askdirectory(title="Select Data Folder (e.g., for MNIST)", initialdir=os.path.dirname(self.data_folder_var.get()))
        if folder:
            self.data_folder_var.set(os.path.abspath(folder))
    
    def _set_button_lock(self, lock=True):
        """Helper to disable/enable multiple buttons during an action."""
        state = "disabled" if lock else "normal"
        # Only disable/enable control buttons if training has been started at least once
        if self.training_active or not lock: # If unlocking, always enable relevant ones
            self.pause_btn.config(state=state if lock else "normal") # Resume enables this
            self.resume_btn.config(state=state if lock else "normal") # Pause enables this
            self.switch_btn.config(state=state if lock else "normal")
            # Stop button is more complex, usually enabled when training_active
            # Start button is usually disabled when training_active
        if not lock: # When unlocking, set specific states
            if self.training_active:
                self.start_btn.config(state="disabled")
                self.stop_btn.config(state="normal")
                # Pause/Resume depend on current trainer state if available
                if self.gan_controller and self.gan_controller.trainer:
                    active_net = self.gan_controller.trainer.current_network
                    if active_net == "generator":
                        is_paused = not self.gan_controller.trainer.pause_event_gen.is_set()
                    else:
                        is_paused = not self.gan_controller.trainer.pause_event_disc.is_set()
                    self.pause_btn.config(state="normal" if not is_paused else "disabled")
                    self.resume_btn.config(state="normal" if is_paused else "disabled")

            else: # Not training active
                self.start_btn.config(state="normal")
                self.stop_btn.config(state="disabled")
                self.pause_btn.config(state="disabled")
                self.resume_btn.config(state="disabled")
                self.switch_btn.config(state="disabled")
        else: # When locking
             self.pause_btn.config(state="disabled")
             self.resume_btn.config(state="disabled")
             self.switch_btn.config(state="disabled")


    def update_button_states(self): # Call this after operations
        if not self.training_active: # Not started or stopped
            self.start_btn.config(state="normal" if self.loaded_config_data else "disabled")
            self.pause_btn.config(state="disabled")
            self.resume_btn.config(state="disabled")
            self.switch_btn.config(state="disabled")
            self.stop_btn.config(state="disabled")
        else: # Training is active (running or paused)
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.switch_btn.config(state="normal")
            
            # Determine pause/resume state based on the active network's pause event
            if self.gan_controller and self.gan_controller.trainer:
                active_net = self.gan_controller.trainer.current_network
                is_paused = False
                if active_net == "generator":
                    if hasattr(self.gan_controller.trainer, 'pause_event_gen'): # Check exists
                        is_paused = not self.gan_controller.trainer.pause_event_gen.is_set()
                elif active_net == "discriminator":
                    if hasattr(self.gan_controller.trainer, 'pause_event_disc'): # Check exists
                        is_paused = not self.gan_controller.trainer.pause_event_disc.is_set()
                
                self.pause_btn.config(state="normal" if not is_paused else "disabled")
                self.resume_btn.config(state="normal" if is_paused else "disabled")
            else: # Fallback if trainer not fully ready
                self.pause_btn.config(state="disabled")
                self.resume_btn.config(state="disabled")
        self.root.update_idletasks()


    def start_gan_training(self):
        if self.training_active:
            messagebox.showinfo("Info", "Training is already active. Stop it before starting a new session.")
            return

        if not self.loaded_config_data:
            messagebox.showerror("Error", "Please load a network configuration file first.")
            self.update_button_states()
            return
        
        current_ui_train_config = self.get_current_ui_training_config()
        if not current_ui_train_config: 
            self.update_button_states()
            return 

        self.start_btn.config(state="disabled") # Disable start button immediately
        self.root.update_idletasks()
        self.update_training_stats_text("Initializing GAN Controller...")

        try:
            # Re-initialize controller for a fresh start if called multiple times after stop
            if self.gan_controller: # If a previous instance exists (after a stop)
                del self.gan_controller # Allow garbage collection
            
            self.gan_controller = GANController(
                self.loaded_config_data.get("generator", {}),
                self.loaded_config_data.get("discriminator", {}),
                current_ui_train_config 
            )
            self.update_training_stats_text("Controller initialized. Starting training threads...")
            self.root.update_idletasks()

            self.gan_controller.start_persistent_training(self.training_ui_callback)
            self.training_active = True
            self.update_training_stats_text("Training threads initiated.")
        except Exception as e:
            self.training_active = False
            self.gan_controller = None 
            messagebox.showerror("Training Error", f"Failed to start training: {e}")
            self.update_training_stats_text(f"ERROR starting training: {e}")
            print(f"Full traceback for training start error:", exc_info=True)
        finally:
            self.update_button_states()


    def pause_active_training(self):
        if not self.training_active or not self.gan_controller:
            self.update_training_stats_text("Cannot pause: No active training session.")
            self.update_button_states()
            return
        
        self._set_button_lock(True)
        active_net = self.gan_controller.trainer.current_network # Get before pause
        self.gan_controller.pause_training()
        self.update_training_stats_text(f"Paused {active_net} training.")
        self._set_button_lock(False)
        self.update_button_states()


    def resume_active_training(self):
        if not self.training_active or not self.gan_controller:
            self.update_training_stats_text("Cannot resume: No active training session.")
            self.update_button_states()
            return

        self._set_button_lock(True)
        current_ui_train_config = self.get_current_ui_training_config()
        if not current_ui_train_config: 
            self._set_button_lock(False)
            self.update_button_states()
            return
        self.gan_controller.update_runtime_train_params_from_ui(current_ui_train_config)
            
        active_net = self.gan_controller.trainer.current_network # Get before resume
        self.gan_controller.resume_training()
        self.update_training_stats_text(f"Resumed {active_net} training.")
        self._set_button_lock(False)
        self.update_button_states()

    
    def switch_active_network(self):
        if not self.training_active or not self.gan_controller:
            self.update_training_stats_text("Cannot switch: No active training session.")
            self.update_button_states()
            return
        
        self._set_button_lock(True)
        old_active = self.gan_controller.trainer.current_network
        self.gan_controller.switch_network()
        new_active = self.gan_controller.trainer.current_network
        self.update_training_stats_text(f"Switched. Was: {old_active}, Now active: {new_active}.")
        self._set_button_lock(False)
        self.update_button_states()

    
    def stop_all_training(self):
        if not self.training_active or not self.gan_controller:
            self.update_training_stats_text("Stop command: No training was active.")
            self.update_button_states() # Ensure buttons are in non-training state
            return

        self.update_training_stats_text("Stopping all training...")
        self._set_button_lock(True) # Disable all controls during stop
        self.stop_btn.config(text="Stopping...", state="disabled") # Indicate stopping
        self.root.update_idletasks()

        self.gan_controller.signal_stop_training_threads() 
        # Joining is now handled in on_closing or if start is called again after this.
        # For UI responsiveness, we don't block here.
        # Threads are daemons, will exit with main app if not joined earlier.
        # However, for a clean restart, controller should handle joining or re-init.
        
        # To simulate waiting for threads to acknowledge stop (optional, non-blocking way)
        def check_if_stopped():
            if self.gan_controller and \
               (not self.gan_controller.trainer.running_gen and not self.gan_controller.trainer.running_disc) or \
               ((self.gan_controller.generator_thread is None or not self.gan_controller.generator_thread.is_alive()) and \
                (self.gan_controller.discriminator_thread is None or not self.gan_controller.discriminator_thread.is_alive())):
                self.update_training_stats_text("Training threads appear to have stopped.")
                self.training_active = False
                self.gan_controller.join_training_threads(timeout=1) # Quick join attempt
                self.gan_controller = None # Allow re-creation on next start
                self.update_button_states()
                self.stop_btn.config(text="Stop All Training") # Reset button text
            else:
                self.root.after(500, check_if_stopped) # Check again shortly

        self.root.after(100, check_if_stopped) # Start checking
        # Immediate UI update
        self.training_active = False # Tentatively set, check_if_stopped confirms
        self.update_button_states()
        self.stop_btn.config(text="Stop All Training")


    def update_training_stats_text(self, message):
        if hasattr(self, 'training_stats_text') and self.training_stats_text.winfo_exists(): 
            current_time = time.strftime("%H:%M:%S", time.localtime())
            self.training_stats_text.insert(tk.END, f"{current_time} - {str(message)}\n")
            self.training_stats_text.see(tk.END) 
    
    def build_summary_tab(self):
        frame = self.summary_frame
        ttk.Label(frame, text="Configuration Summary & Image Generation", font=("Arial", 14, "bold")).pack(pady=10)
        
        summary_controls = ttk.Frame(frame)
        summary_controls.pack(pady=5)
        ttk.Button(summary_controls, text="Refresh Summary from Loaded Config", command=self.update_summary_text_from_config).pack(side="left", padx=5)
        self.plot_losses_button = ttk.Button(summary_controls, text="Plot Losses", command=self.plot_training_losses)
        self.plot_losses_button.pack(side="left", padx=5)
        
        self.summary_text_widget = tk.Text(frame, height=15, width=100, wrap="word", state="disabled", relief="sunken", borderwidth=1)
        summary_scroll = ttk.Scrollbar(frame, command=self.summary_text_widget.yview)
        self.summary_text_widget['yscrollcommand'] = summary_scroll.set
        summary_scroll.pack(side="right", fill="y", pady=(0,10), padx=(0,10))
        self.summary_text_widget.pack(pady=(0,10), fill="both", expand=True, padx=(10,0))
        
        generation_lf = ttk.LabelFrame(frame, text="Generate Image (from current Generator)")
        generation_lf.pack(pady=10, fill="x", padx=10)
        self.generate_image_button = ttk.Button(generation_lf, text="Generate & Show Image", command=self.generate_and_show_image)
        self.generate_image_button.pack(pady=5)
        self.image_display_frame = ttk.Frame(generation_lf, width=130, height=130, relief="sunken", borderwidth=1)
        self.image_display_frame.pack(pady=5)
        # Prevent the frame from shrinking to the label's initial size (text only)
        self.image_display_frame.pack_propagate(False) 

        self.generated_image_label = ttk.Label(self.image_display_frame, text="No image generated yet.", compound="image", anchor="center")
        # Pack the label to expand within the frame
        self.generated_image_label.pack(fill="both", expand=True) 

    def update_summary_text_from_config(self):
        if not self.loaded_config_data:
            summary = "No configuration loaded. Please load a JSON config file from the 'Network Preview' tab."
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
            self.plot_losses_button.config(state="disabled")
            thread = threading.Thread(target=self._plot_losses_thread_task, daemon=True)
            thread.start()
        else:
            messagebox.showinfo("Info", "No training data to plot losses. Start training first.")

    def _plot_losses_thread_task(self):
        try:
            trainer_instance = self.gan_controller.get_trainer_instance()
            if trainer_instance:
                # Ensure matplotlib is imported in this thread's context if not globally visible sometimes
                # import matplotlib.pyplot as plt # Usually not needed if imported at top of file
                trainer_instance.plot_losses() 
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Plotting Error", f"Failed to plot: {e}"))
            print(f"Plotting error:", exc_info=True)
        finally:
            if hasattr(self, 'plot_losses_button') and self.plot_losses_button.winfo_exists():
                 self.root.after(0, lambda: self.plot_losses_button.config(state="normal"))
    
    def generate_and_show_image(self): 
        if not self.gan_controller or not self.gan_controller.generator:
            messagebox.showerror("Error", "Generator not initialized. Load config and start training first.")
            return
        
        self.generate_image_button.config(state="disabled")
        thread = threading.Thread(target=self._generate_image_thread_task, daemon=True)
        thread.start()

    def _generate_image_thread_task(self): 
        try:
            self.gan_controller.generator.eval() 
            noise = torch.randn(1, self.gan_controller.latent_dim, device=self.gan_controller.device)
            with torch.no_grad():
                generated_output = self.gan_controller.generator(noise)
            
            img_tensor_raw = generated_output[0].cpu() # Shape [C, H, W]
            
            # Denormalize from [-1,1] (Tanh output) to [0,1]
            img_tensor_norm = (img_tensor_raw + 1) / 2.0 
            img_tensor_norm = img_tensor_norm.clamp(0, 1) 
            
            # Convert to PIL Image
            # ToPILImage expects (C,H,W) or (H,W). If C=1, it handles it.
            pil_img = transforms.ToPILImage()(img_tensor_norm)
            
            pil_img_resized = pil_img.resize((128, 128), Image.Resampling.NEAREST) 

            self.root.after(0, self._update_generated_image_ui, pil_img_resized)
        except Exception as e:
            error_message = f"Failed to generate image: {e}"
            self.root.after(0, lambda: messagebox.showerror("Image Generation Error", error_message))
            print(f"Full traceback for image generation error:", exc_info=True) 
        finally:
            if hasattr(self, 'generate_image_button') and self.generate_image_button.winfo_exists():
                self.root.after(0, lambda: self.generate_image_button.config(state="normal"))


    def _update_generated_image_ui(self, pil_img):
        if hasattr(self, 'generated_image_label') and self.generated_image_label.winfo_exists(): 
            self.generated_image_pil_ref = ImageTk.PhotoImage(pil_img) 
            self.generated_image_label.config(image=self.generated_image_pil_ref, text="")


if __name__ == "__main__":
    import time # For timestamp in logger
    main_root = tk.Tk()
    app = GanConfigurator(main_root)
    app.update_button_states() # Initialize button states
    main_root.mainloop()