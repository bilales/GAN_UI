import threading
import torch
from .model_builder import NetworkBuilder 
from .train_manager import Trainer       
from .data_loader import DataLoader        

class GANController:
    def __init__(self, gen_config_arch, disc_config_arch, training_config_from_ui):
        self.gen_config_arch = gen_config_arch 
        self.disc_config_arch = disc_config_arch 
        self.training_config_ui = training_config_from_ui 

        self.gen_train_params_runtime = self.training_config_ui.get("generator", {}).copy() # Use .copy()
        self.disc_train_params_runtime = self.training_config_ui.get("discriminator", {}).copy() # Use .copy()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Controller] Using device: {self.device}")
        
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        self.latent_dim = self.gen_config_arch.get("input_dim")
        if self.latent_dim is None:
            raise ValueError("Generator config must include 'input_dim'.")

        self.data_loader_manager = DataLoader( 
            data_folder=self.training_config_ui.get("data_folder", "./data"), 
            batch_size=self.gen_train_params_runtime.get("batch_size", 64), 
            image_size=28 
        )

        self.trainer = Trainer(
            self.generator,
            self.discriminator,
            gen_train_params=self.gen_train_params_runtime, 
            disc_train_params=self.disc_train_params_runtime, 
            device=self.device,
            data_loader=self.data_loader_manager, 
            latent_dim=self.latent_dim
        )

        initial_net = self.training_config_ui.get("initial_network", "generator").lower()
        self.trainer.current_network = initial_net
        print(f"[Controller] Initial active network: {self.trainer.current_network}")

        self.generator_thread = None
        self.discriminator_thread = None
        self.threads_started_once = False # To prevent re-creating threads if start is called again

    def build_generator(self):
        input_dim = self.gen_config_arch.get("input_dim")
        layers = self.gen_config_arch.get("layers", [])
        output_size = self.gen_config_arch.get("output_size") 
        global_activation = self.gen_config_arch.get("global_activation", "relu")
        
        builder = NetworkBuilder(input_dim, layers, output_size, global_activation)
        model = builder.build_network().to(self.device)
        print("[Controller] Generator built.")
        return model

    def build_discriminator(self):
        input_shape_list = self.disc_config_arch.get("input_shape")
        if input_shape_list is None:
            raise ValueError("Discriminator config must include 'input_shape'.")
        input_shape_tuple = tuple(input_shape_list) 
        
        layers = self.disc_config_arch.get("layers", [])
        output_size = self.disc_config_arch.get("output_size") 
        global_activation = self.disc_config_arch.get("global_activation", "relu")
        
        builder = NetworkBuilder(input_shape_tuple, layers, output_size, global_activation, input_channels=input_shape_tuple[0])
        model = builder.build_network().to(self.device)
        print("[Controller] Discriminator built.")
        return model

    def update_runtime_train_params_from_ui(self, new_ui_training_config):
        self.training_config_ui = new_ui_training_config 
        
        # Update runtime params, ensuring to merge and not just overwrite
        # This allows Trainer to hold references to these dicts and see changes
        self.gen_train_params_runtime.clear()
        self.gen_train_params_runtime.update(self.training_config_ui.get("generator", {}))
        
        self.disc_train_params_runtime.clear()
        self.disc_train_params_runtime.update(self.training_config_ui.get("discriminator", {}))
        
        new_data_folder = self.training_config_ui.get("data_folder")
        if new_data_folder and self.data_loader_manager.data_folder != new_data_folder:
            print(f"[Controller] Data folder changed to: {new_data_folder}")
            self.data_loader_manager.data_folder = new_data_folder
            
        new_batch_size_g = self.gen_train_params_runtime.get("batch_size")
        # Assuming Dataloader batch size is primarily driven by G's for UI simplicity,
        # or you can have separate UI fields for Dataloader's batch size.
        if new_batch_size_g and self.data_loader_manager.batch_size != new_batch_size_g:
            print(f"[Controller] DataLoader batch size reference (from G config) changed to: {new_batch_size_g}")
            self.data_loader_manager.batch_size = new_batch_size_g

        if self.trainer:
            self.trainer.update_learning_rates_from_params() 


    def start_persistent_training(self, ui_callback):
        if self.threads_started_once and \
           ((self.generator_thread and self.generator_thread.is_alive()) or \
            (self.discriminator_thread and self.discriminator_thread.is_alive())):
            print("[Controller] Training threads already exist and may be running/paused. Use Resume/Switch.")
            # If they were stopped and joined, this logic needs refinement to re-create them.
            # For now, assume start is only called once, or after a full stop and join.
            # A more robust solution would be to re-initialize Trainer and threads if start is called after stop.
            # For this version, let's assume UI prevents calling start if already started.
            return


        # Reset trainer state if starting fresh (e.g., after a stop)
        if self.trainer:
            self.trainer.running_gen = True
            self.trainer.running_disc = True
            self.trainer.current_epoch_gen = 1
            self.trainer.current_epoch_disc = 1
            self.trainer.gen_losses = []
            self.trainer.disc_losses = []
            # Ensure pause events are cleared initially for the one that will start
            self.trainer.pause_event_gen.clear()
            self.trainer.pause_event_disc.clear()


        if not self.generator_thread or not self.generator_thread.is_alive():
            self.generator_thread = threading.Thread(
                target=self.trainer.train_generator_loop,
                args=(ui_callback,), daemon=True 
            )
            self.generator_thread.start()
            print("[Controller] Generator training thread started/restarted.")

        if not self.discriminator_thread or not self.discriminator_thread.is_alive():
            self.discriminator_thread = threading.Thread(
                target=self.trainer.train_discriminator_loop,
                args=(ui_callback,), daemon=True
            )
            self.discriminator_thread.start()
            print("[Controller] Discriminator training thread started/restarted.")
        
        self.threads_started_once = True

        # Unpause the initially selected network from config (or current if resuming)
        initial_net_from_ui = self.training_config_ui.get("initial_network", "generator").lower()
        self.trainer.current_network = initial_net_from_ui # Ensure trainer knows
        
        print(f"[Controller] Setting initial active network to: {self.trainer.current_network}")
        if self.trainer.current_network == "generator":
            self.trainer.pause_event_gen.set() # Unpause G
            self.trainer.pause_event_disc.clear() # Ensure D is paused
        elif self.trainer.current_network == "discriminator":
            self.trainer.pause_event_disc.set() # Unpause D
            self.trainer.pause_event_gen.clear() # Ensure G is paused
        
        print(f"[Controller] Training initialized. Active: {self.trainer.current_network}. G_unpaused: {self.trainer.pause_event_gen.is_set()}, D_unpaused: {self.trainer.pause_event_disc.is_set()}")


    def pause_training(self): # Pauses the currently active network
        if self.trainer and self.threads_started_once:
            active_net = self.trainer.current_network
            self.trainer.pause(active_net) # pause method clears the event for that network
            print(f"[Controller] Paused {active_net} training.")
        else:
            print("[Controller] Trainer not active or threads not started. Cannot pause.")


    def resume_training(self):
        if self.trainer and self.threads_started_once:
            # UI should call update_runtime_train_params_from_ui before this if params changed
            active_net = self.trainer.current_network
            self.trainer.resume(active_net) # resume method sets the event for that network
            print(f"[Controller] Resumed {active_net} training.")
        else:
            print("[Controller] Trainer not active or threads not started. Cannot resume.")


    def switch_network(self):
        if self.trainer and self.threads_started_once:
            old_active = self.trainer.current_network
            self.trainer.switch()
            print(f"[Controller] Switched. Was: {old_active}, New active: {self.trainer.current_network}.")
        else:
            print("[Controller] Trainer not active or threads not started. Cannot switch.")

    def signal_stop_training_threads(self): # For UI to call, non-blocking
        if self.trainer:
            print("[Controller] Signaling training threads to stop...")
            self.trainer.stop() 

    def join_training_threads(self, timeout=5): # Blocking, for app exit
        stopped_cleanly = True
        print("[Controller] Attempting to join training threads...")
        if self.generator_thread and self.generator_thread.is_alive():
            print("[Controller] Waiting for generator thread to join...")
            self.generator_thread.join(timeout=timeout)
            if self.generator_thread.is_alive():
                print("[Controller] Warning: Generator thread did not join in time.")
                stopped_cleanly = False
        self.generator_thread = None 

        if self.discriminator_thread and self.discriminator_thread.is_alive():
            print("[Controller] Waiting for discriminator thread to join...")
            self.discriminator_thread.join(timeout=timeout)
            if self.discriminator_thread.is_alive():
                print("[Controller] Warning: Discriminator thread did not join in time.")
                stopped_cleanly = False
        self.discriminator_thread = None
        
        self.threads_started_once = False # Allow restart
        if stopped_cleanly:
            print("[Controller] Training threads joined successfully.")
        return stopped_cleanly
            
    def get_trainer_instance(self): 
        return self.trainer