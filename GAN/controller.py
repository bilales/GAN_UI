import threading
import torch
from .model_builder import NetworkBuilder # Relative import
from .train_manager import Trainer       # Relative import
from .data_loader import DataLoader        # Relative import
# from tkinter import messagebox # Controller shouldn't know about UI directly

class GANController:
    def __init__(self, gen_config, disc_config, training_config_from_ui):
        self.gen_config_arch = gen_config # Architecture config
        self.disc_config_arch = disc_config # Architecture config
        self.training_config_ui = training_config_from_ui # Training params from UI

        # Extract training parameters for G and D (these can be updated from UI)
        self.gen_train_params_runtime = self.training_config_ui.get("generator", {})
        self.disc_train_params_runtime = self.training_config_ui.get("discriminator", {})

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Controller] Using device: {self.device}")
        
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        self.latent_dim = self.gen_config_arch.get("input_dim")
        if self.latent_dim is None:
            raise ValueError("Generator config must include 'input_dim'.")

        self.data_loader_manager = DataLoader( # Use the manager class
            data_folder=self.training_config_ui.get("data_folder", "./data"), # Get from UI config
            batch_size=self.training_config_ui.get("generator", {}).get("batch_size", 64), # Example: use G's batch_size
            image_size=28 # MNIST default, make configurable if needed
        )

        self.trainer = Trainer(
            self.generator,
            self.discriminator,
            gen_train_params=self.gen_train_params_runtime, # Pass runtime params
            disc_train_params=self.disc_train_params_runtime, # Pass runtime params
            device=self.device,
            data_loader=self.data_loader_manager, # Pass the DataLoader manager instance
            latent_dim=self.latent_dim
        )

        # Set initial network from UI config
        initial_net = self.training_config_ui.get("initial_network", "generator").lower()
        self.trainer.current_network = initial_net
        print(f"[Controller] Initial active network: {self.trainer.current_network}")

        self.generator_thread = None
        self.discriminator_thread = None

    def build_generator(self):
        input_dim = self.gen_config_arch.get("input_dim")
        layers = self.gen_config_arch.get("layers", [])
        output_size = self.gen_config_arch.get("output_size") # Informational
        global_activation = self.gen_config_arch.get("global_activation", "relu")
        
        builder = NetworkBuilder(input_dim, layers, output_size, global_activation)
        model = builder.build_network().to(self.device)
        print("[Controller] Generator built.")
        # print(model) # For debugging model structure
        return model

    def build_discriminator(self):
        input_shape_list = self.disc_config_arch.get("input_shape") # e.g., [1, 28, 28]
        if input_shape_list is None:
            raise ValueError("Discriminator config must include 'input_shape'.")
        input_shape_tuple = tuple(input_shape_list) # (C, H, W)
        
        layers = self.disc_config_arch.get("layers", [])
        output_size = self.disc_config_arch.get("output_size") # Informational
        global_activation = self.disc_config_arch.get("global_activation", "relu")
        
        # input_channels for NetworkBuilder if input_size is (H,W)
        # Here input_shape_tuple is (C,H,W), so NetworkBuilder will use it directly.
        builder = NetworkBuilder(input_shape_tuple, layers, output_size, global_activation)
        model = builder.build_network().to(self.device)
        print("[Controller] Discriminator built.")
        # print(model) # For debugging model structure
        return model

    def update_runtime_train_params_from_ui(self, new_ui_training_config):
        """Updates the runtime training parameters if changed in UI, e.g., learning rate."""
        self.training_config_ui = new_ui_training_config # Store the latest full UI config
        self.gen_train_params_runtime.update(self.training_config_ui.get("generator", {}))
        self.disc_train_params_runtime.update(self.training_config_ui.get("discriminator", {}))
        
        # Update data folder if changed
        new_data_folder = self.training_config_ui.get("data_folder")
        if new_data_folder and self.data_loader_manager.data_folder != new_data_folder:
            print(f"[Controller] Data folder changed to: {new_data_folder}")
            self.data_loader_manager.data_folder = new_data_folder
            # Potentially re-initialize or clear cached data if your DataLoader does that

        # Batch sizes might also change, update DataLoader if necessary
        new_batch_size_g = self.gen_train_params_runtime.get("batch_size")
        # Assuming batch size is primarily driven by generator for data loading config
        if new_batch_size_g and self.data_loader_manager.batch_size != new_batch_size_g:
            print(f"[Controller] Batch size changed to: {new_batch_size_g}")
            self.data_loader_manager.batch_size = new_batch_size_g


        if self.trainer:
            self.trainer.update_learning_rates_from_params() # Tell trainer to update optimizers


    def start_persistent_training(self, ui_callback):
        if self.generator_thread and self.generator_thread.is_alive():
            print("[Controller] Generator thread already running.")
        else:
            self.trainer.running_gen = True # Reset flag in case of previous stop
            self.trainer.current_epoch_gen = 1 # Reset epoch count
            self.trainer.gen_losses = [] # Clear old losses
            self.generator_thread = threading.Thread(
                target=self.trainer.train_generator_loop,
                args=(ui_callback,), daemon=True # Daemon threads exit when main program exits
            )
            self.generator_thread.start()
            print("[Controller] Generator training thread started.")

        if self.discriminator_thread and self.discriminator_thread.is_alive():
            print("[Controller] Discriminator thread already running.")
        else:
            self.trainer.running_disc = True # Reset flag
            self.trainer.current_epoch_disc = 1 # Reset epoch count
            self.trainer.disc_losses = [] # Clear old losses
            self.discriminator_thread = threading.Thread(
                target=self.trainer.train_discriminator_loop,
                args=(ui_callback,), daemon=True
            )
            self.discriminator_thread.start()
            print("[Controller] Discriminator training thread started.")

        # Unpause the initially selected network
        if self.trainer.current_network == "generator":
            self.trainer.pause_event_gen.set()
            self.trainer.pause_event_disc.clear() # Ensure other is paused
        elif self.trainer.current_network == "discriminator":
            self.trainer.pause_event_disc.set()
            self.trainer.pause_event_gen.clear() # Ensure other is paused
        print(f"[Controller] Initializing training. Active: {self.trainer.current_network}. G_paused: {not self.trainer.pause_event_gen.is_set()}, D_paused: {not self.trainer.pause_event_disc.is_set()}")


    def pause_training(self):
        if self.trainer:
            # Pause the currently active network
            self.trainer.pause(self.trainer.current_network)
            print(f"[Controller] Paused {self.trainer.current_network} training.")

    def resume_training(self):
        """Resume whichever network is active, after updating params from UI."""
        if self.trainer:
            # The UI should call a method on controller to update its training_config_ui first
            # Then this resume can use the updated params.
            # self.trainer.update_learning_rates_from_params() # Already called by update_runtime_train_params_from_ui
            
            self.trainer.resume(self.trainer.current_network) # Resume the one that's supposed to be active
            print(f"[Controller] Resumed {self.trainer.current_network} training.")


    def switch_network(self):
        if self.trainer:
            self.trainer.switch()
            print(f"[Controller] Switched. New active network: {self.trainer.current_network}")

    def stop_training(self):
        if self.trainer:
            print("[Controller] Stopping training...")
            self.trainer.stop() # Signals loops to stop and unblocks events

            if self.generator_thread and self.generator_thread.is_alive():
                print("[Controller] Waiting for generator thread to join...")
                self.generator_thread.join(timeout=5)
                if self.generator_thread.is_alive():
                    print("[Controller] Warning: Generator thread did not join in time.")
            self.generator_thread = None # Allow restarting later

            if self.discriminator_thread and self.discriminator_thread.is_alive():
                print("[Controller] Waiting for discriminator thread to join...")
                self.discriminator_thread.join(timeout=5)
                if self.discriminator_thread.is_alive():
                    print("[Controller] Warning: Discriminator thread did not join in time.")
            self.discriminator_thread = None # Allow restarting later
            print("[Controller] Training stopped.")
            
    def get_trainer_instance(self): # For UI to access plot_losses etc.
        return self.trainer