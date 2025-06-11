import torch
import torch.optim as optim
import torch.nn as nn
import threading
import time 
import matplotlib.pyplot as plt
import traceback # For more detailed error logging in threads

class Trainer:
    def __init__(self, generator, discriminator, gen_train_params, disc_train_params,
                 device=None, data_loader=None, latent_dim=100):
        self.generator = generator
        self.discriminator = discriminator
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.gen_train_params = gen_train_params # Reference to controller's dict
        self.disc_train_params = disc_train_params # Reference to controller's dict
        
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        
        lr_g = self.gen_train_params.get('learning_rate', 0.0002)
        lr_d = self.disc_train_params.get('learning_rate', 0.0002)
        beta1 = self.gen_train_params.get('beta1', 0.5) # Allow overriding betas
        beta2 = self.gen_train_params.get('beta2', 0.999)

        self.gen_optimizer = optim.Adam(
            self.generator.parameters(), lr=lr_g, betas=(beta1, beta2)
        )
        self.disc_optimizer = optim.Adam(
            self.discriminator.parameters(), lr=lr_d, betas=(beta1, beta2)
        )
        
        self.gen_loss_fn = self._get_loss_function(self.gen_train_params.get('loss_function', 'BCEWithLogitsLoss'))
        self.disc_loss_fn = self._get_loss_function(self.disc_train_params.get('loss_function', 'BCEWithLogitsLoss'))
        
        self.running_gen = True
        self.running_disc = True

        self.pause_event_gen = threading.Event()
        self.pause_event_disc = threading.Event()
        self.pause_event_gen.clear() 
        self.pause_event_disc.clear() 

        self.data_loader_manager = data_loader 
        self.latent_dim = latent_dim
        
        self.gen_losses = []
        self.disc_losses = []

        self.current_epoch_gen = 1
        self.current_epoch_disc = 1
        # Max epochs will be read dynamically from self.gen_train_params now

        self.current_network = "generator" 

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
            # Use a consistent batch size, e.g., discriminator's, for this calculation
            # as it determines how many batches are in a "full dataset pass"
            current_dl_batch_size = self.disc_train_params.get('batch_size', 64)
            # Create a temp loader with the specific batch size for calculation
            try:
                # Store original batch size of the manager
                original_bs = self.data_loader_manager.batch_size
                self.data_loader_manager.batch_size = current_dl_batch_size # Temporarily set
                
                temp_loader = self.data_loader_manager.get_data_loader(train=True)
                num_batches = len(temp_loader)
                del temp_loader
                
                self.data_loader_manager.batch_size = original_bs # Restore
                return num_batches if num_batches > 0 else 1 
            except Exception as e:
                print(f"Warning: Could not get length from data_loader: {e}. Using fallback.")
        
        # Fallback based on MNIST: 60000 images / batch_size
        bs_for_fallback = self.disc_train_params.get('batch_size', 64)
        return (60000 // bs_for_fallback) if bs_for_fallback > 0 else 100

    def train_generator_loop(self, callback=None):
        try:
            while self.running_gen:
                max_epochs_g = self.gen_train_params.get('epochs', 50) # Dynamic read
                if self.current_epoch_gen > max_epochs_g:
                    if callback: callback(f"[G] Max epochs ({max_epochs_g}) reached. Stopping G thread.")
                    print(f"[G] Max epochs ({max_epochs_g}) reached. Stopping G thread.")
                    break 

                self.pause_event_gen.wait() 
                if not self.running_gen: break 

                if self.current_network != "generator":
                    self.pause_event_gen.clear() 
                    time.sleep(0.1) # Small sleep to prevent busy-waiting if switched rapidly
                    continue 

                batch_size_g = self.gen_train_params.get('batch_size', 64) # Dynamic read
                num_steps_this_g_epoch = self._get_num_batches_per_dataset_pass() # G aims for similar num steps as D epoch

                self.generator.train()
                self._freeze(self.discriminator, True)
                self._freeze(self.generator, False)

                epoch_g_loss_sum = 0.0
                actual_g_steps_this_epoch = 0
                for _ in range(num_steps_this_g_epoch):
                    if not self.running_gen or not self.pause_event_gen.is_set(): break
                    
                    noise = torch.randn(batch_size_g, self.latent_dim, device=self.device)
                    loss = self._train_generator_step(noise)
                    epoch_g_loss_sum += loss
                    actual_g_steps_this_epoch += 1
                
                if not self.running_gen: break

                if actual_g_steps_this_epoch > 0:
                    avg_epoch_g_loss = epoch_g_loss_sum / actual_g_steps_this_epoch
                    self.gen_losses.append(avg_epoch_g_loss)
                    msg = f"[G] Ep {self.current_epoch_gen}/{max_epochs_g} | Loss: {avg_epoch_g_loss:.4f} ({actual_g_steps_this_epoch} steps)"
                else:
                    msg = f"[G] Ep {self.current_epoch_gen}/{max_epochs_g} | No steps (paused/stopped early)."
                
                if callback: callback(msg)
                print(msg)
                self.current_epoch_gen += 1
            
            if callback: callback("[G] Training loop finished.")
            print("[G] Training loop finished.")
        except Exception as e:
            tb_str = traceback.format_exc()
            error_msg = f"[G Thread ERROR] {e}\n{tb_str}"
            if callback: callback(error_msg)
            print(error_msg)
        finally:
            self.running_gen = False


    def _train_generator_step(self, noise_batch):
        self.gen_optimizer.zero_grad()
        fake_images = self.generator(noise_batch)
        target_labels = torch.ones(fake_images.size(0), 1, device=self.device)
        disc_output_on_fake = self.discriminator(fake_images)
        loss = self.gen_loss_fn(disc_output_on_fake, target_labels)
        loss.backward()
        self.gen_optimizer.step()
        return loss.item()


    def train_discriminator_loop(self, callback=None):
        try:
            while self.running_disc:
                max_epochs_d = self.disc_train_params.get('epochs', 50) # Dynamic read
                if self.current_epoch_disc > max_epochs_d:
                    if callback: callback(f"[D] Max epochs ({max_epochs_d}) reached. Stopping D thread.")
                    print(f"[D] Max epochs ({max_epochs_d}) reached. Stopping D thread.")
                    break

                self.pause_event_disc.wait()
                if not self.running_disc: break

                if self.current_network != "discriminator":
                    self.pause_event_disc.clear()
                    time.sleep(0.1) # Small sleep
                    continue
                
                # Batch size for Dataloader is now dynamic via controller updating data_loader_manager.batch_size
                # self.data_loader_manager.batch_size = self.disc_train_params.get('batch_size', 64) # Ensure it's current

                self.discriminator.train() # Set model to training mode
                self._freeze(self.generator, True)   
                self._freeze(self.discriminator, False) # CRITICAL: Ensure D params require grad

                # --- Debug for RuntimeError ---
                # print("\n[D-Loop Debug] Discriminator Parameter Grads BEFORE D_epoch steps:")
                # for name, param in self.discriminator.named_parameters():
                #     if not param.requires_grad:
                #         print(f"  WARNING: {name} does NOT require grad!")
                #     # print(f"  {name}: requires_grad={param.requires_grad}")
                # --- End Debug ---

                epoch_d_loss_sum = 0.0
                actual_d_steps_this_epoch = 0
                
                real_data_loader = self.data_loader_manager.get_data_loader(train=True)

                for real_images, _ in real_data_loader:
                    if not self.running_disc or not self.pause_event_disc.is_set(): break

                    current_batch_size = real_images.size(0)
                    real_images = real_images.to(self.device)
                    
                    noise = torch.randn(current_batch_size, self.latent_dim, device=self.device)
                    with torch.no_grad(): 
                        fake_images = self.generator(noise).detach() 

                    loss = self._train_discriminator_step(real_images, fake_images)
                    epoch_d_loss_sum += loss
                    actual_d_steps_this_epoch += 1
                
                if not self.running_disc: break

                if actual_d_steps_this_epoch > 0:
                    avg_epoch_d_loss = epoch_d_loss_sum / actual_d_steps_this_epoch
                    self.disc_losses.append(avg_epoch_d_loss)
                    msg = f"[D] Ep {self.current_epoch_disc}/{max_epochs_d} | Loss: {avg_epoch_d_loss:.4f} ({actual_d_steps_this_epoch} batches)"
                else:
                    msg = f"[D] Ep {self.current_epoch_disc}/{max_epochs_d} | No steps (paused/stopped early)."
                
                if callback: callback(msg)
                print(msg)
                self.current_epoch_disc += 1
            
            if callback: callback("[D] Training loop finished.")
            print("[D] Training loop finished.")
        except Exception as e:
            tb_str = traceback.format_exc()
            error_msg = f"[D Thread ERROR] {e}\n{tb_str}"
            if callback: callback(error_msg)
            print(error_msg)
        finally:
            self.running_disc = False


    def _train_discriminator_step(self, real_batch, fake_batch):
        # Ensure D is in train mode and grads are enabled for its params *before this step*
        # self.discriminator.train() # Already done in the outer loop
        # self._freeze(self.discriminator, False) # Also done in outer loop

        self.disc_optimizer.zero_grad()

        real_labels = torch.ones(real_batch.size(0), 0.9, device=self.device)
        output_real = self.discriminator(real_batch)
        loss_real = self.disc_loss_fn(output_real, real_labels)
        
        fake_labels = torch.zeros(fake_batch.size(0), 1, device=self.device)
        output_fake = self.discriminator(fake_batch)
        loss_fake = self.disc_loss_fn(output_fake, fake_labels)

        total_loss = (loss_real + loss_fake) / 2 
        
        # Check if total_loss requires grad before backward()
        # if not total_loss.requires_grad:
        #     print("ERROR in _train_discriminator_step: total_loss does not require grad!")
        #     # Further debugging:
        #     # print(f"  loss_real.requires_grad: {loss_real.requires_grad}, loss_real.grad_fn: {loss_real.grad_fn}")
        #     # print(f"  loss_fake.requires_grad: {loss_fake.requires_grad}, loss_fake.grad_fn: {loss_fake.grad_fn}")
        #     # print(f"  output_real.requires_grad: {output_real.requires_grad}, output_real.grad_fn: {output_real.grad_fn}")
        #     # print(f"  output_fake.requires_grad: {output_fake.requires_grad}, output_fake.grad_fn: {output_fake.grad_fn}")
        #     # Check D params again right here if issues persist
        #     # for name, param in self.discriminator.named_parameters():
        #     #     print(f"    D Param {name}: requires_grad={param.requires_grad}")
        #     return total_loss.item() # Or raise error

        total_loss.backward()
        self.disc_optimizer.step()
        return total_loss.item()

    def _freeze(self, model, freeze=True):
        # This is critical. When unfreezing (freeze=False), params MUST get requires_grad=True
        for param in model.parameters():
            param.requires_grad = not freeze
        # # Optional: verify after setting
        # if not freeze: # If unfreezing
        #     for name, param in model.named_parameters():
        #         if not param.requires_grad:
        #             print(f"WARNING in _freeze: After unfreezing {model.__class__.__name__}, param {name} STILL has requires_grad=False")


    def pause(self, network_to_pause):
        if network_to_pause == "generator":
            self.pause_event_gen.clear()
            print("[Trainer] Pausing Generator training.")
        elif network_to_pause == "discriminator":
            self.pause_event_disc.clear()
            print("[Trainer] Pausing Discriminator training.")


    def resume(self, network_to_resume):
        if network_to_resume == "generator" and self.current_network == "generator":
            self.pause_event_gen.set()
            print(f"[Trainer] Resuming Generator training (active: {self.current_network}).")
        elif network_to_resume == "discriminator" and self.current_network == "discriminator":
            self.pause_event_disc.set()
            print(f"[Trainer] Resuming Discriminator training (active: {self.current_network}).")
        elif network_to_resume != self.current_network:
            # This case means UI tried to resume a non-active network.
            # The pause event for that network might be set, but its loop won't proceed
            # past the `if self.current_network != ...: continue` check.
            print(f"[Trainer] '{network_to_resume}' is not the current active network ('{self.current_network}'). It will remain paused in its loop if it was.")
            if network_to_resume == "generator": self.pause_event_gen.set() # Allow it to proceed to its check
            if network_to_resume == "discriminator": self.pause_event_disc.set()


    def stop(self):
        print("[Trainer] Signaling all training loops to stop...")
        self.running_gen = False
        self.running_disc = False
        self.pause_event_gen.set() 
        self.pause_event_disc.set()

    def switch(self):
        # print(f"[Trainer] Attempting to switch. Current: {self.current_network}")
        if self.current_network == "generator":
            self.pause_event_gen.clear() 
            self.current_network = "discriminator"
            self.pause_event_disc.set()  
        elif self.current_network == "discriminator":
            self.pause_event_disc.clear() 
            self.current_network = "generator"
            self.pause_event_gen.set()   
        # print(f"[Trainer] Switched. New active: {self.current_network}. G_paused: {not self.pause_event_gen.is_set()}, D_paused: {not self.pause_event_disc.is_set()}")
    
    def update_learning_rates_from_params(self): 
        gen_lr = self.gen_train_params.get("learning_rate", 0.0002)
        disc_lr = self.disc_train_params.get("learning_rate", 0.0002)
        # Also update betas if they are made configurable
        beta1_g = self.gen_train_params.get('beta1', 0.5)
        beta2_g = self.gen_train_params.get('beta2', 0.999)
        beta1_d = self.disc_train_params.get('beta1', 0.5)
        beta2_d = self.disc_train_params.get('beta2', 0.999)

        if hasattr(self, 'gen_optimizer'):
            for param_group in self.gen_optimizer.param_groups:
                param_group['lr'] = gen_lr
                param_group['betas'] = (beta1_g, beta2_g)
        if hasattr(self, 'disc_optimizer'):
            for param_group in self.disc_optimizer.param_groups:
                param_group['lr'] = disc_lr
                param_group['betas'] = (beta1_d, beta2_d)
        print(f"[Trainer] LRs/Betas updated: G_lr={gen_lr}, D_lr={disc_lr}")

    def plot_losses(self):
        if not self.gen_losses and not self.disc_losses:
            print("No losses recorded yet to plot.")
            if plt: 
                plt.figure(figsize=(10, 5))
                plt.text(0.5, 0.5, "No training data to plot losses.", ha='center', va='center', fontsize=12)
                plt.title("GAN Training Losses")
                try: plt.show()
                except Exception as e: print(f"Plot error (no data): {e}")
            return

        plt.figure(figsize=(12, 6))
        if self.gen_losses:
            plt.plot(self.gen_losses, label=f"Generator Loss (Avg per G-Epoch)")
        if self.disc_losses:
            plt.plot(self.disc_losses, label=f"Discriminator Loss (Avg per D-Epoch)")
        
        plt.xlabel("Epoch Count (respective to G or D)")
        plt.ylabel("Average Loss")
        plt.title("GAN Training Losses")
        plt.legend()
        plt.grid(True)
        try:
            plt.show()
        except Exception as e:
            print(f"Could not display plot (maybe no GUI backend): {e}")