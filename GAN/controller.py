# GAN/controller.py
import threading
import torch
from .model_builder import NetworkBuilder 
from .train_manager import Trainer       
from .data_loader import DataLoader        

class GANController:
    def __init__(self, gen_config_arch, disc_config_arch, training_config_from_ui):
        self.gen_config_arch = gen_config_arch 
        self.disc_config_arch = disc_config_arch 
        self.training_config_ui = training_config_from_ui # Keep ref to latest full UI config

        self.gen_train_params_runtime = self.training_config_ui.get("generator", {}).copy()
        self.disc_train_params_runtime = self.training_config_ui.get("discriminator", {}).copy()
        
        self.auto_switch_config = self.training_config_ui.get("automatic_switching", {"enabled": False, "g_epochs": 1, "d_epochs": 1})

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Controller] Using device: {self.device}")
        
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        self.latent_dim = self.gen_config_arch.get("input_dim")
        if self.latent_dim is None: raise ValueError("Generator config must include 'input_dim'.")

        # Use G's batch size for DataLoader default, or D's, or make it a separate UI config
        # For consistency, let's assume the UI provides a batch_size for the DataLoader if needed,
        # or we can default to one of the network's batch sizes.
        dl_batch_size = self.gen_train_params_runtime.get("batch_size", 64) 
        self.data_loader_manager = DataLoader( 
            data_folder=self.training_config_ui.get("data_folder", "./data"), 
            batch_size=dl_batch_size, 
            image_size=28 # Assuming MNIST, make configurable if needed for other datasets
        )

        self.trainer = Trainer(
            self.generator, self.discriminator,
            gen_train_params=self.gen_train_params_runtime, 
            disc_train_params=self.disc_train_params_runtime, 
            device=self.device, data_loader=self.data_loader_manager, 
            latent_dim=self.latent_dim,
            controller_ref=self # Pass self to Trainer
        )
        self.trainer.update_auto_switch_config(self.auto_switch_config) # Initial set

        initial_net = self.training_config_ui.get("initial_network", "generator").lower()
        self.trainer.current_network = initial_net
        print(f"[Controller] Initial active network: {self.trainer.current_network}")

        self.generator_thread = None
        self.discriminator_thread = None
        self.threads_started_once = False 
        self.ui_callback_ref = None 

    def build_generator(self):
        input_dim=self.gen_config_arch.get("input_dim");layers=self.gen_config_arch.get("layers",[]);output_size=self.gen_config_arch.get("output_size");global_activation=self.gen_config_arch.get("global_activation","relu")
        builder=NetworkBuilder(input_dim,layers,output_size,global_activation); model=builder.build_network().to(self.device); print("[Controller] Generator built."); return model

    def build_discriminator(self):
        input_shape_list=self.disc_config_arch.get("input_shape"); assert input_shape_list, "Discriminator config needs 'input_shape'."
        input_shape_tuple=tuple(input_shape_list);layers=self.disc_config_arch.get("layers",[]);output_size=self.disc_config_arch.get("output_size");global_activation=self.disc_config_arch.get("global_activation","relu")
        builder=NetworkBuilder(input_shape_tuple,layers,output_size,global_activation,input_channels=input_shape_tuple[0]); model=builder.build_network().to(self.device); print("[Controller] Discriminator built."); return model

    def update_runtime_train_params_from_ui(self, new_ui_training_config):
        self.training_config_ui = new_ui_training_config # Store the latest
        
        self.gen_train_params_runtime.clear(); self.gen_train_params_runtime.update(self.training_config_ui.get("generator", {}))
        self.disc_train_params_runtime.clear(); self.disc_train_params_runtime.update(self.training_config_ui.get("discriminator", {}))
        
        new_auto_switch_config = self.training_config_ui.get("automatic_switching", {"enabled": False, "g_epochs": 1, "d_epochs": 1})
        # Update only if it's actually different to avoid unnecessary log/reset
        if self.auto_switch_config != new_auto_switch_config: 
            self.auto_switch_config = new_auto_switch_config
            if self.trainer: self.trainer.update_auto_switch_config(self.auto_switch_config) # This will reset cycles in trainer
            log_msg = f"Auto-switch config updated: G for {self.auto_switch_config.get('g_epochs',1)}, D for {self.auto_switch_config.get('d_epochs',1)}" if self.auto_switch_config.get("enabled") else "Auto-switch DISABLED."
            if self.ui_callback_ref: self.ui_callback_ref(f"[Controller Update] {log_msg}")
            print(f"[Controller Update] {log_msg}")

        new_data_folder = self.training_config_ui.get("data_folder")
        if new_data_folder and self.data_loader_manager.data_folder != new_data_folder:
            print(f"[Controller] Data folder updated to: {new_data_folder}")
            self.data_loader_manager.data_folder = new_data_folder
            
        # Update DataLoader's batch_size if UI changes it (e.g., via G's batch size field)
        # This assumes a single batch_size config for data loading, or you can tie it to D's batch size.
        new_dataloader_batch_size = self.gen_train_params_runtime.get("batch_size") 
        if new_dataloader_batch_size and self.data_loader_manager.batch_size != new_dataloader_batch_size:
            print(f"[Controller] DataLoader batch size updated to: {new_dataloader_batch_size}")
            self.data_loader_manager.batch_size = new_dataloader_batch_size

        if self.trainer: self.trainer.update_learning_rates_from_params() 

    def start_persistent_training(self, ui_callback):
        self.ui_callback_ref = ui_callback 
        if self.threads_started_once and \
           ((self.generator_thread and self.generator_thread.is_alive()) or \
            (self.discriminator_thread and self.discriminator_thread.is_alive())):
            if ui_callback: ui_callback("[Controller] INFO: Training threads seem to be already active (or were not properly joined). Using Resume/Switch is advised, or Stop then Start for a full reset.")
            # Optionally, try to resume the currently set active network
            # self.trainer.resume(self.trainer.current_network) # This might be too aggressive
            return

        if self.trainer: # Reset trainer state for a completely fresh start
            self.trainer.running_gen = True; self.trainer.running_disc = True
            self.trainer.current_epoch_gen = 1; self.trainer.current_epoch_disc = 1
            self.trainer.gen_losses = []; self.trainer.disc_losses = []
            self.trainer.g_epochs_done_in_cycle = 0; self.trainer.d_epochs_done_in_cycle = 0 
            self.trainer.pause_event_gen.clear(); self.trainer.pause_event_disc.clear() # Ensure both start paused until one is set
            self.trainer.update_auto_switch_config(self.auto_switch_config) # Pass current auto-switch settings

        if not self.generator_thread or not self.generator_thread.is_alive():
            self.generator_thread = threading.Thread(target=self.trainer.train_generator_loop, args=(ui_callback,), daemon=True)
            self.generator_thread.start()
            print("[Controller] Generator thread (re)started.")
        if not self.discriminator_thread or not self.discriminator_thread.is_alive():
            self.discriminator_thread = threading.Thread(target=self.trainer.train_discriminator_loop, args=(ui_callback,), daemon=True)
            self.discriminator_thread.start()
            print("[Controller] Discriminator thread (re)started.")
        
        self.threads_started_once = True # Mark that threads have been created
        
        # Set the initial network to train based on UI config
        # This ensures that if auto-switch is off, the user's choice is respected.
        # If auto-switch is on, Trainer will handle the first turn based on this current_network.
        initial_net_from_ui_config = self.training_config_ui.get("initial_network", "generator").lower()
        self.trainer.current_network = initial_net_from_ui_config 
        
        print(f"[Controller] Setting initial active network to: {self.trainer.current_network}")
        if self.trainer.current_network == "generator": 
            self.trainer.pause_event_gen.set() # Unpause G
            self.trainer.pause_event_disc.clear() # Ensure D is paused
        elif self.trainer.current_network == "discriminator": 
            self.trainer.pause_event_disc.set() # Unpause D
            self.trainer.pause_event_gen.clear() # Ensure G is paused
        
        log_msg = f"Training initialized. Active: {self.trainer.current_network}."
        if self.auto_switch_config.get("enabled"):
            log_msg += f" Auto-switch: G for {self.auto_switch_config.get('g_epochs',1)}, D for {self.auto_switch_config.get('d_epochs',1)}."
        if ui_callback: ui_callback(log_msg)
        print(log_msg)

    def pause_training(self): # Pauses the currently active network
        if self.trainer and self.threads_started_once: 
            self.trainer.pause(self.trainer.current_network)
            # UI callback for pause confirmation is handled by the caller (main.py)
        else: 
            msg = "[Controller] Cannot pause: No active training or threads not started."
            if self.ui_callback_ref: self.ui_callback_ref(msg)
            print(msg)

    def resume_training(self): # Resumes the currently active network
        if self.trainer and self.threads_started_once: 
            self.trainer.resume(self.trainer.current_network)
        else: 
            msg = "[Controller] Cannot resume: No active training or threads not started."
            if self.ui_callback_ref: self.ui_callback_ref(msg)
            print(msg)

    def switch_network(self, manual_switch=False): 
        if self.trainer and self.threads_started_once:
            if manual_switch and self.auto_switch_config.get("enabled"):
                msg = "[Controller] INFO: Manual switch ignored, auto-switching is enabled."
                if self.ui_callback_ref: self.ui_callback_ref(msg)
                print(msg)
                return # Don't proceed with manual switch if auto is on
            
            # Trainer's switch method handles pause/resume events and current_network update
            # It also now handles resetting auto-switch cycle counts if manual_override is true
            self.trainer.switch(manual_override=manual_switch) 
            
            # Log source of switch
            source = "MANUAL" if manual_switch else "AUTO (via Trainer)"
            msg = f"[Controller] {source} Switch. Now active: {self.trainer.current_network}"
            if self.ui_callback_ref: self.ui_callback_ref(msg) # UI will get this
            print(msg) # Console log

        else: 
            msg = "[Controller] Cannot switch: No active training or threads not started."
            if self.ui_callback_ref: self.ui_callback_ref(msg)
            print(msg)

    def signal_stop_training_threads(self): 
        if self.trainer: 
            print("[Controller] Signaling trainer to stop all loops.")
            self.trainer.stop() 
        else:
            print("[Controller] No trainer instance to stop.")


    def join_training_threads(self, timeout=5): 
        stopped_cleanly = True
        print("[Controller] Attempting to join training threads...");
        for thread_name, thread_obj_attr in [("generator", "generator_thread"), ("discriminator", "discriminator_thread")]:
            thread_obj = getattr(self, thread_obj_attr, None)
            if thread_obj and thread_obj.is_alive():
                print(f"[Controller] Waiting for {thread_name} thread to join (timeout {timeout}s)..."); 
                thread_obj.join(timeout=timeout)
                if thread_obj.is_alive(): 
                    print(f"[Controller] WARNING: {thread_name} thread did not join in time."); 
                    stopped_cleanly = False
                else:
                    print(f"[Controller] {thread_name} thread joined.")
            elif thread_obj: # Exists but not alive
                 print(f"[Controller] {thread_name} thread was already not alive.")
            else: # No thread object
                 print(f"[Controller] No {thread_name} thread object to join.")
            setattr(self, thread_obj_attr, None) # Clear thread reference

        self.threads_started_once = False # Critical: allow threads to be recreated by start_persistent_training
        print(f"[Controller] Training threads joined (cleanly: {stopped_cleanly}). Ready for fresh start if needed.")
        return stopped_cleanly
            
    def get_trainer_instance(self): 
        return self.trainer