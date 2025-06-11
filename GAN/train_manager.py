# GAN/train_manager.py
import torch
import torch.optim as optim
import torch.nn as nn
import threading
import time 
import matplotlib.pyplot as plt # Keep for plotting logic
import traceback 

class Trainer:
    def __init__(self, generator, discriminator, gen_train_params, disc_train_params,
                 device=None, data_loader=None, latent_dim=100, controller_ref=None):
        self.generator = generator
        self.discriminator = discriminator
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.gen_train_params = gen_train_params # This is a reference to controller's dict
        self.disc_train_params = disc_train_params # This is a reference to controller's dict
        
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        
        # Optimizers are created once. LR/betas can be updated.
        self._initialize_optimizers()
        
        self.gen_loss_fn = self._get_loss_function(self.gen_train_params.get('loss_function', 'BCEWithLogitsLoss'))
        self.disc_loss_fn = self._get_loss_function(self.disc_train_params.get('loss_function', 'BCEWithLogitsLoss'))
        
        self.running_gen = True # Flag to control generator loop execution
        self.running_disc = True # Flag to control discriminator loop execution

        self.pause_event_gen = threading.Event()
        self.pause_event_disc = threading.Event()
        self.pause_event_gen.clear() # Start paused until explicitly set
        self.pause_event_disc.clear() # Start paused until explicitly set

        self.data_loader_manager = data_loader 
        self.latent_dim = latent_dim
        
        self.gen_losses = []
        self.disc_losses = []

        self.current_epoch_gen = 1 # Overall epoch count for G
        self.current_epoch_disc = 1 # Overall epoch count for D
        
        self.controller_ref = controller_ref # Reference to GANController
        self.auto_switch_config = {"enabled": False, "g_epochs": 1, "d_epochs": 1} # Default
        self.g_epochs_done_in_cycle = 0 # For auto-switching
        self.d_epochs_done_in_cycle = 0 # For auto-switching

        self.current_network = "generator" # Default, controller will set based on UI

    def _initialize_optimizers(self):
        lr_g = self.gen_train_params.get('learning_rate', 0.0002)
        lr_d = self.disc_train_params.get('learning_rate', 0.0002)
        beta1_g = self.gen_train_params.get('beta1', 0.5)
        beta2_g = self.gen_train_params.get('beta2', 0.999)
        beta1_d = self.disc_train_params.get('beta1', 0.5)
        beta2_d = self.disc_train_params.get('beta2', 0.999)

        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=lr_g, betas=(beta1_g, beta2_g))
        self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=(beta1_d, beta2_d))
        print("[Trainer] Optimizers initialized.")


    def _get_loss_function(self, loss_name_str):
        loss_name = loss_name_str.lower()
        loss_dict = {
            "mseloss": nn.MSELoss(),
            "bcewithlogitsloss": nn.BCEWithLogitsLoss(),
            "crossentropyloss": nn.CrossEntropyLoss(),
            "bceloss": nn.BCELoss() 
        }
        if loss_name not in loss_dict: 
            print(f"Warning: Loss function '{loss_name_str}' not recognized. Defaulting to BCEWithLogitsLoss.")
            return nn.BCEWithLogitsLoss()
        return loss_dict[loss_name]

    def _get_num_batches_per_dataset_pass(self):
        if self.data_loader_manager:
            current_dl_batch_size = self.disc_train_params.get('batch_size', 64) # Use D's batch size for this calc
            original_bs = self.data_loader_manager.batch_size
            try:
                self.data_loader_manager.batch_size = current_dl_batch_size
                temp_loader = self.data_loader_manager.get_data_loader(train=True)
                num_batches = len(temp_loader)
                del temp_loader
                return num_batches if num_batches > 0 else 1
            except Exception as e: 
                print(f"Warning: Could not get length from data_loader: {e}. Using fallback.")
            finally: 
                self.data_loader_manager.batch_size = original_bs 
        
        bs_fallback = self.disc_train_params.get('batch_size', 64)
        return (60000 // bs_fallback) if bs_fallback > 0 else 100 

    def update_auto_switch_config(self, config): # Called by controller
        self.auto_switch_config = config
        if not self.auto_switch_config.get("enabled", False):
            self.g_epochs_done_in_cycle = 0 # Reset if auto-switch is disabled
            self.d_epochs_done_in_cycle = 0
        print(f"[Trainer] Auto-switch config received: Enabled={self.auto_switch_config.get('enabled')}, G_epochs={self.auto_switch_config.get('g_epochs')}, D_epochs={self.auto_switch_config.get('d_epochs')}")


    def _maybe_perform_auto_switch(self, just_completed_network_type, callback):
        if not self.auto_switch_config.get("enabled", False) or not self.controller_ref:
            return False 

        g_target_epochs_in_cycle = self.auto_switch_config.get("g_epochs", 1)
        d_target_epochs_in_cycle = self.auto_switch_config.get("d_epochs", 1)
        
        switched_by_auto_logic = False

        if just_completed_network_type == "generator":
            self.g_epochs_done_in_cycle += 1
            if self.g_epochs_done_in_cycle >= g_target_epochs_in_cycle:
                if self.current_network == "generator": # Check if still G to prevent re-switch
                    self.controller_ref.switch_network(manual_switch=False) # Tell controller it's an auto switch
                    if callback: callback(f"[AUTO] G cycle ({self.g_epochs_done_in_cycle}/{g_target_epochs_in_cycle}) done. Switching to D.")
                    # Cycle counts will be reset by self.switch() if it's an auto-switch related call
                    switched_by_auto_logic = True
        
        elif just_completed_network_type == "discriminator":
            self.d_epochs_done_in_cycle += 1
            if self.d_epochs_done_in_cycle >= d_target_epochs_in_cycle:
                if self.current_network == "discriminator":
                    self.controller_ref.switch_network(manual_switch=False)
                    if callback: callback(f"[AUTO] D cycle ({self.d_epochs_done_in_cycle}/{d_target_epochs_in_cycle}) done. Switching to G.")
                    switched_by_auto_logic = True
        
        return switched_by_auto_logic


    def train_generator_loop(self, callback=None):
        try:
            while self.running_gen:
                max_manual_epochs_g = self.gen_train_params.get('epochs', 50) # For manual override/limit
                if self.current_epoch_gen > max_manual_epochs_g:
                    if callback: callback(f"[G] Max manual epochs ({max_manual_epochs_g}) reached. Stopping G thread.")
                    print(f"[G] Max manual epochs ({max_manual_epochs_g}) reached. Stopping G thread.")
                    break 

                self.pause_event_gen.wait() 
                if not self.running_gen: break 

                if self.current_network != "generator":
                    self.pause_event_gen.clear() 
                    time.sleep(0.1) # Small sleep to yield CPU if rapidly switching
                    continue 

                current_batch_size_g = self.gen_train_params.get('batch_size', 64) 
                num_steps_this_g_conceptual_epoch = self._get_num_batches_per_dataset_pass() 

                self.generator.train(); self._freeze(self.discriminator, True); self._freeze(self.generator, False)
                
                epoch_g_loss_sum = 0.0; actual_g_steps_this_epoch = 0
                for _ in range(num_steps_this_g_conceptual_epoch):
                    if not self.running_gen or not self.pause_event_gen.is_set(): break # Check before each step
                    
                    noise = torch.randn(current_batch_size_g, self.latent_dim, device=self.device)
                    loss = self._train_generator_step(noise)
                    epoch_g_loss_sum += loss; actual_g_steps_this_epoch += 1
                
                if not self.running_gen: break # Check after inner loop

                if actual_g_steps_this_epoch > 0:
                    avg_epoch_g_loss = epoch_g_loss_sum / actual_g_steps_this_epoch
                    self.gen_losses.append(avg_epoch_g_loss)
                    msg = f"[G] Overall Ep {self.current_epoch_gen}/{max_manual_epochs_g} | Cycle Ep {self.g_epochs_done_in_cycle+1 if self.auto_switch_config.get('enabled') else '-'} | Loss: {avg_epoch_g_loss:.4f} ({actual_g_steps_this_epoch} steps)"
                    if callback: callback(msg); print(msg)
                    
                    self.current_epoch_gen += 1 # Increment overall G epoch count
                    self._maybe_perform_auto_switch("generator", callback) # Check for auto-switch
                elif self.running_gen and self.pause_event_gen.is_set(): 
                    msg = f"[G] Overall Ep {self.current_epoch_gen}/{max_manual_epochs_g} | No training steps this cycle."
                    if callback: callback(msg); print(msg)
                    time.sleep(0.2) 
            
            if callback: callback("[G] Training loop finished.")
            print("[G] Training loop finished.")
        except Exception as e:
            tb_str = traceback.format_exc(); error_msg = f"[G Thread ERROR] {e}\n{tb_str}"
            if callback: callback(error_msg); print(error_msg)
        finally: 
            self.running_gen = False # Ensure flag is set on exit

    def _train_generator_step(self, noise_batch):
        self.gen_optimizer.zero_grad(); fake_images = self.generator(noise_batch)
        target_labels = torch.ones(fake_images.size(0), 1, device=self.device)
        # Discriminator should be in eval mode if G is training and D is frozen, 
        # but _freeze handles requires_grad. D.eval() would affect dropout/BN if D has them.
        # For simplicity, let's assume D just passes through. If D has BN/Dropout, consider D.eval() when G trains.
        # self.discriminator.eval() # Optional: if D has dropout/batchnorm
        disc_output_on_fake = self.discriminator(fake_images)
        # self.discriminator.train() # Optional: set back if you used D.eval()
        loss = self.gen_loss_fn(disc_output_on_fake, target_labels)
        loss.backward(); self.gen_optimizer.step(); return loss.item()

    def train_discriminator_loop(self, callback=None):
        try:
            while self.running_disc:
                max_manual_epochs_d = self.disc_train_params.get('epochs', 50) 
                if self.current_epoch_disc > max_manual_epochs_d:
                    if callback: callback(f"[D] Max manual epochs ({max_manual_epochs_d}) reached. Stopping D thread.")
                    break
                self.pause_event_disc.wait()
                if not self.running_disc: break
                if self.current_network != "discriminator":
                    self.pause_event_disc.clear(); time.sleep(0.1); continue
                
                self.discriminator.train(); self._freeze(self.generator, True); self._freeze(self.discriminator, False)
                
                epoch_d_loss_sum = 0.0; actual_d_steps_this_epoch = 0
                # DataLoader batch size is updated by controller via data_loader_manager.batch_size
                real_data_loader = self.data_loader_manager.get_data_loader(train=True)

                for real_images, _ in real_data_loader:
                    if not self.running_disc or not self.pause_event_disc.is_set(): break
                    current_batch_size = real_images.size(0); real_images = real_images.to(self.device)
                    noise = torch.randn(current_batch_size, self.latent_dim, device=self.device)
                    with torch.no_grad(): fake_images = self.generator(noise).detach() 
                    loss = self._train_discriminator_step(real_images, fake_images)
                    epoch_d_loss_sum += loss; actual_d_steps_this_epoch += 1
                if not self.running_disc: break

                if actual_d_steps_this_epoch > 0:
                    avg_epoch_d_loss = epoch_d_loss_sum / actual_d_steps_this_epoch
                    self.disc_losses.append(avg_epoch_d_loss)
                    msg = f"[D] Overall Ep {self.current_epoch_disc}/{max_manual_epochs_d} | Cycle Ep {self.d_epochs_done_in_cycle+1 if self.auto_switch_config.get('enabled') else '-'} | Loss: {avg_epoch_d_loss:.4f} ({actual_d_steps_this_epoch} batches)"
                    if callback: callback(msg); print(msg)
                    self.current_epoch_disc += 1
                    self._maybe_perform_auto_switch("discriminator", callback)
                elif self.running_disc and self.pause_event_disc.is_set():
                    msg = f"[D] Overall Ep {self.current_epoch_disc}/{max_manual_epochs_d} | No training steps this cycle."
                    if callback: callback(msg); print(msg)
                    time.sleep(0.2)
            if callback: callback("[D] Training loop finished.")
            print("[D] Training loop finished.")
        except Exception as e:
            tb_str = traceback.format_exc(); error_msg = f"[D Thread ERROR] {e}\n{tb_str}"
            if callback: callback(error_msg); print(error_msg)
        finally: self.running_disc = False

    def _train_discriminator_step(self, real_batch, fake_batch):
        self.disc_optimizer.zero_grad()
        real_labels = torch.ones(real_batch.size(0),1,device=self.device); output_real = self.discriminator(real_batch); loss_real = self.disc_loss_fn(output_real,real_labels)
        fake_labels = torch.zeros(fake_batch.size(0),1,device=self.device); output_fake = self.discriminator(fake_batch); loss_fake = self.disc_loss_fn(output_fake,fake_labels)
        total_loss = (loss_real + loss_fake) / 2 
        total_loss.backward(); self.disc_optimizer.step(); return total_loss.item()

    def _freeze(self, model, freeze=True):
        for param in model.parameters(): param.requires_grad = not freeze

    def pause(self, network_to_pause): # Called by Controller
        if network_to_pause == "generator": self.pause_event_gen.clear()
        elif network_to_pause == "discriminator": self.pause_event_disc.clear()
        print(f"[Trainer] Pausing {network_to_pause} training.")


    def resume(self, network_to_resume): # Called by Controller
        # This only sets the event. The loop's `current_network` check determines if it actually trains.
        if network_to_resume == "generator": self.pause_event_gen.set()
        elif network_to_resume == "discriminator": self.pause_event_disc.set()
        print(f"[Trainer] Resume signal sent for {network_to_resume}. Will run if it's the active network ({self.current_network}).")


    def stop(self): # Called by Controller
        print("[Trainer] Signaling stop to G & D training loops."); 
        self.running_gen = False; self.running_disc = False
        self.pause_event_gen.set(); self.pause_event_disc.set() # Unblock threads so they can see running_ flags and exit

    def switch(self, manual_override=False): # Called by Controller
        print(f"[Trainer] Switch received. Was: {self.current_network}. Manual: {manual_override}")
        if self.current_network == "generator":
            self.pause_event_gen.clear(); self.current_network = "discriminator"; self.pause_event_disc.set()  
        elif self.current_network == "discriminator":
            self.pause_event_disc.clear(); self.current_network = "generator"; self.pause_event_gen.set()   
        
        # If auto-switching is enabled OR it's a manual override, reset cycle counts.
        # This ensures that a manual switch also resets the auto-switch progression.
        if self.auto_switch_config.get("enabled", False) or manual_override:
            print("[Trainer] Resetting auto-switch cycle epoch counts due to switch action.")
            self.g_epochs_done_in_cycle = 0
            self.d_epochs_done_in_cycle = 0
        print(f"[Trainer] Switched. Now active: {self.current_network}.")
    
    def update_learning_rates_from_params(self): # Called by Controller
        gen_lr = self.gen_train_params.get("learning_rate",0.0002); disc_lr = self.disc_train_params.get("learning_rate",0.0002)
        # Assuming betas are fixed for now, but could be made dynamic similarly
        beta1_g=self.gen_train_params.get('beta1',0.5); beta2_g=self.gen_train_params.get('beta2',0.999)
        beta1_d=self.disc_train_params.get('beta1',0.5); beta2_d=self.disc_train_params.get('beta2',0.999)
        
        if hasattr(self,'gen_optimizer'): 
            for pg in self.gen_optimizer.param_groups: pg['lr']=gen_lr; pg['betas']=(beta1_g,beta2_g)
        if hasattr(self,'disc_optimizer'): 
            for pg in self.disc_optimizer.param_groups: pg['lr']=disc_lr; pg['betas']=(beta1_d,beta2_d)
        print(f"[Trainer] LRs/Betas updated: G_lr={gen_lr}, D_lr={disc_lr}")

    def plot_losses(self): # Called by main UI thread (via controller.get_trainer_instance().plot_losses())
        if not self.gen_losses and not self.disc_losses: 
            print("[Trainer Plot] No losses to plot.")
            # If called from a non-main thread and plt.show() is problematic,
            # this method should ideally just prepare data or save to file.
            # The actual plt.show() should be main-thread if possible, or save to file.
            # For this version, we assume it's called from a context where plt.show() might work (like _plot_losses_thread_task in main.py that handles it)
            # but it's better if this just returns the figure or saves it.
            # For the fix, main.py's _plot_losses_thread_task handles plt.savefig and no plt.show here.
            # This method can just prepare the plot object if we were to return it.
            # Since main.py _plot_losses_thread_task directly calls matplotlib, this method here is mostly for conceptual separation.
            # The actual plotting for UI is handled by main.py's thread to avoid main thread GUI issues.
            return None # Or return the figure object if main.py was to display it directly

        fig = plt.figure(figsize=(10,5))
        if self.gen_losses: plt.plot(self.gen_losses, label="Generator Loss (Avg per G-Epoch)")
        if self.disc_losses: plt.plot(self.disc_losses, label="Discriminator Loss (Avg per D-Epoch)")
        plt.xlabel("Conceptual Epoch Count (respective to G or D)");plt.ylabel("Average Loss");
        plt.title("GAN Training Losses");plt.legend();plt.grid(True)
        return fig # Return the figure object for the caller (main.py's thread) to save/show