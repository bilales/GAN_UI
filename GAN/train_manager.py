import torch
import torch.optim as optim
import torch.nn as nn
import threading
import time # For potential sleeps if needed, or timing
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, generator, discriminator, gen_train_params, disc_train_params,
                 device=None, data_loader=None, latent_dim=100):
        self.generator = generator
        self.discriminator = discriminator
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.gen_train_params = gen_train_params
        self.disc_train_params = disc_train_params
        
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        
        lr_g = self.gen_train_params.get('learning_rate', 0.0002)
        lr_d = self.disc_train_params.get('learning_rate', 0.0002)
        beta1 = 0.5 # Common beta1 for GANs
        beta2 = 0.999 # Common beta2 for GANs

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
        self.pause_event_gen.clear() # Start paused
        self.pause_event_disc.clear() # Start paused

        self.data_loader_manager = data_loader # Renamed to avoid confusion with torch.utils.data.DataLoader
        self.latent_dim = latent_dim
        
        self.gen_losses = []
        self.disc_losses = []

        self.current_epoch_gen = 1
        self.current_epoch_disc = 1
        self.max_epochs_gen = self.gen_train_params.get('epochs', 50)
        self.max_epochs_disc = self.disc_train_params.get('epochs', 50)

        self.current_network = "generator" # Will be set by controller based on config

    def _get_loss_function(self, loss_name_str):
        loss_name = loss_name_str.lower()
        loss_dict = {
            "mseloss": nn.MSELoss(),
            "bcewithlogitsloss": nn.BCEWithLogitsLoss(),
            "crossentropyloss": nn.CrossEntropyLoss(),
            "bceloss": nn.BCELoss() # Use if output already has sigmoid
        }
        if loss_name not in loss_dict:
            print(f"Warning: Loss function '{loss_name_str}' not recognized. Defaulting to BCEWithLogitsLoss.")
            return nn.BCEWithLogitsLoss()
        return loss_dict[loss_name]

    def _get_num_batches_per_epoch(self):
        if self.data_loader_manager:
            try:
                temp_loader = self.data_loader_manager.get_data_loader(train=True)
                num_batches = len(temp_loader)
                del temp_loader
                return num_batches if num_batches > 0 else 1 # Ensure at least 1
            except Exception as e:
                print(f"Warning: Could not get length from data_loader: {e}. Using fallback.")
        # Fallback based on MNIST: 60000 images / batch_size
        batch_size = self.gen_train_params.get('batch_size', 64) # Use G's batch_size for G epoch
        return (60000 // batch_size) if batch_size > 0 else 100


    def train_generator_loop(self, callback=None):
        batch_size = self.gen_train_params.get('batch_size', 64)
        num_steps_per_g_epoch = self._get_num_batches_per_epoch()

        while self.running_gen and self.current_epoch_gen <= self.max_epochs_gen:
            self.pause_event_gen.wait() # Blocks if event is clear (paused)
            if not self.running_gen: break # Exit if stop was called

            if self.current_network != "generator":
                self.pause_event_gen.clear() # Ensure it's paused if not current
                continue # Re-check conditions

            self.generator.train()
            self._freeze(self.discriminator, True)
            self._freeze(self.generator, False)

            epoch_g_loss_sum = 0.0
            actual_g_steps_this_epoch = 0
            for i in range(num_steps_per_g_epoch):
                if not self.running_gen or not self.pause_event_gen.is_set(): break
                
                noise = torch.randn(batch_size, self.latent_dim, device=self.device)
                loss = self._train_generator_step(noise)
                epoch_g_loss_sum += loss
                actual_g_steps_this_epoch += 1
            
            if not self.running_gen: break

            if actual_g_steps_this_epoch > 0:
                avg_epoch_g_loss = epoch_g_loss_sum / actual_g_steps_this_epoch
                self.gen_losses.append(avg_epoch_g_loss)
                msg = f"[G] Ep {self.current_epoch_gen}/{self.max_epochs_gen} | Loss: {avg_epoch_g_loss:.4f} ({actual_g_steps_this_epoch} steps)"
            else:
                msg = f"[G] Ep {self.current_epoch_gen}/{self.max_epochs_gen} | No steps taken this epoch."
            
            if callback: callback(msg)
            print(msg)
            self.current_epoch_gen += 1
        
        print("[Generator] Training loop finished.")
        self.running_gen = False


    def _train_generator_step(self, noise_batch):
        self.gen_optimizer.zero_grad()
        fake_images = self.generator(noise_batch)
        # Target labels for generator are "real" (1s)
        target_labels = torch.ones(fake_images.size(0), 1, device=self.device)
        # Discriminator classifies the fake images
        disc_output_on_fake = self.discriminator(fake_images)
        # Calculate loss based on how well G fooled D
        loss = self.gen_loss_fn(disc_output_on_fake, target_labels)
        loss.backward()
        self.gen_optimizer.step()
        return loss.item()


    def train_discriminator_loop(self, callback=None):
        while self.running_disc and self.current_epoch_disc <= self.max_epochs_disc:
            self.pause_event_disc.wait()
            if not self.running_disc: break

            if self.current_network != "discriminator":
                self.pause_event_disc.clear()
                continue
            
            self.discriminator.train()
            self._freeze(self.generator, True)
            self._freeze(self.discriminator, False)

            epoch_d_loss_sum = 0.0
            actual_d_steps_this_epoch = 0
            
            # Get a fresh data loader instance for the epoch for shuffling
            real_data_loader = self.data_loader_manager.get_data_loader(train=True)

            for real_images, _ in real_data_loader:
                if not self.running_disc or not self.pause_event_disc.is_set(): break

                current_batch_size = real_images.size(0)
                real_images = real_images.to(self.device)
                
                # Generate fake images for this batch
                noise = torch.randn(current_batch_size, self.latent_dim, device=self.device)
                with torch.no_grad(): # Don't track gradients for G here
                    fake_images = self.generator(noise).detach()

                loss = self._train_discriminator_step(real_images, fake_images)
                epoch_d_loss_sum += loss
                actual_d_steps_this_epoch += 1
            
            if not self.running_disc: break

            if actual_d_steps_this_epoch > 0:
                avg_epoch_d_loss = epoch_d_loss_sum / actual_d_steps_this_epoch
                self.disc_losses.append(avg_epoch_d_loss)
                msg = f"[D] Ep {self.current_epoch_disc}/{self.max_epochs_disc} | Loss: {avg_epoch_d_loss:.4f} ({actual_d_steps_this_epoch} batches)"
            else:
                 msg = f"[D] Ep {self.current_epoch_disc}/{self.max_epochs_disc} | No steps taken this epoch."
            
            if callback: callback(msg)
            print(msg)
            self.current_epoch_disc += 1

        print("[Discriminator] Training loop finished.")
        self.running_disc = False


    def _train_discriminator_step(self, real_batch, fake_batch):
        self.disc_optimizer.zero_grad()

        # Train on real images
        real_labels = torch.ones(real_batch.size(0), 1, device=self.device)
        output_real = self.discriminator(real_batch)
        loss_real = self.disc_loss_fn(output_real, real_labels)

        # Train on fake images
        fake_labels = torch.zeros(fake_batch.size(0), 1, device=self.device)
        output_fake = self.discriminator(fake_batch) # fake_batch is already detached
        loss_fake = self.disc_loss_fn(output_fake, fake_labels)

        total_loss = (loss_real + loss_fake) / 2
        total_loss.backward()
        self.disc_optimizer.step()
        return total_loss.item()

    def _freeze(self, model, freeze=True):
        for param in model.parameters():
            param.requires_grad = not freeze

    def pause(self, network_to_pause):
        if network_to_pause == "generator":
            self.pause_event_gen.clear()
            print("[Trainer] Pausing Generator training.")
        elif network_to_pause == "discriminator":
            self.pause_event_disc.clear()
            print("[Trainer] Pausing Discriminator training.")
        # If pausing the "current" network, the loop will naturally pause on wait()
        # If pausing the "other" network, it will remain paused.

    def resume(self, network_to_resume):
        # Resuming only makes sense if it's the current_network
        if network_to_resume == "generator" and self.current_network == "generator":
            self.pause_event_gen.set()
            print("[Trainer] Resuming Generator training.")
        elif network_to_resume == "discriminator" and self.current_network == "discriminator":
            self.pause_event_disc.set()
            print("[Trainer] Resuming Discriminator training.")
        elif network_to_resume != self.current_network:
            print(f"[Trainer] Cannot resume {network_to_resume}, current active network is {self.current_network}.")


    def stop(self):
        print("[Trainer] Stopping all training loops...")
        self.running_gen = False
        self.running_disc = False
        self.pause_event_gen.set() # Unblock threads so they can check running_gen/disc and exit
        self.pause_event_disc.set()

    def switch(self):
        print(f"[Trainer] Switching active network. Was: {self.current_network}")
        if self.current_network == "generator":
            self.pause_event_gen.clear() # Pause current (generator)
            self.current_network = "discriminator"
            self.pause_event_disc.set()  # Unpause new (discriminator)
        elif self.current_network == "discriminator":
            self.pause_event_disc.clear() # Pause current (discriminator)
            self.current_network = "generator"
            self.pause_event_gen.set()   # Unpause new (generator)
        print(f"[Trainer] Now active: {self.current_network}")
    
    def update_learning_rates_from_params(self): # Renamed for clarity
        """ Called by controller if UI changes LRs """
        gen_lr = self.gen_train_params.get("learning_rate", 0.0002)
        disc_lr = self.disc_train_params.get("learning_rate", 0.0002)
        for param_group in self.gen_optimizer.param_groups:
            param_group['lr'] = gen_lr
        for param_group in self.disc_optimizer.param_groups:
            param_group['lr'] = disc_lr
        print(f"[Trainer] LRs updated: G={gen_lr}, D={disc_lr}")


    def plot_losses(self):
        if not self.gen_losses and not self.disc_losses:
            print("No losses recorded yet to plot.")
            if plt: plt.text(0.5, 0.5, "No data to plot", ha='center', va='center')
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