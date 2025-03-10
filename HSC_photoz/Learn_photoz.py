import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.utils
from torch.utils.data import DataLoader
import torch.utils.data
import time
from datetime import timedelta
import os


# Manually set the seed for reproducibility
torch.manual_seed(0)


# Define loss function (Mean Squared Error)
def loss_fn(predictions, targets):
    # return torch.mean(torch.abs(predictions - targets) / targets)
    return torch.mean((predictions - targets) ** 2)
    # return torch.mean((predictions - targets) ** 2 / (targets ** 2))


VERBOSE = True
TIMINGS = True


class LearnNdimPoly:
    """
        Class that learns a N-dimensional polynomial
    """

    def __init__(self,
                 n_epochs: int,
                 learning_rate: float = 1e-4,
                 batch_size: int = 64
                 ):
        
        # The number of input values to obtain a photo-z from
        # Here, we use the five grizy filters
        self.n_input: int = 5

        # We have a single output, the photo-z
        self.n_output: int = 1

        # The number of epochs to train our system over
        self.n_epochs: int = n_epochs

        # A good batch size (apparently?)
        self.batch_size = batch_size

        # Try to use CUDA acceleration if it is available
        self.device: str = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        print(f"Using {self.device} device", end='\t')

        use_parallel = False

        if use_parallel:
            print(f'We are using {torch.cuda.device_count()} GPUs!')

            # Use as many GPUs as we have available to us
            model = self.NeuralNetwork(self.n_input, self.n_output)
            self.model = nn.DataParallel(model).to(self.device)
        else:
            print('We are using just one GPU')

            # Send the model to one GPU device
            self.model = self.NeuralNetwork(self.n_input, self.n_output).to(self.device)

        if VERBOSE:
            print(self.model)

        # The learning rate
        # TODO: see what this actually changes
        self.learning_rate: float = learning_rate

        # The given optimiser
        # TODO: Also see what this changes too
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Lists to store the test & train history in
        self.loss_train_hist = np.zeros(self.n_epochs)
        self.loss_test_hist = np.zeros(self.n_epochs)

        # The plots directory
        self.plots_dir: str = './HSC_mean_sq_no_norm_2'

        # If the plots dir does not exist, then make it
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)


    def generate_data_tensors(self):
        use_hsc = True
        if use_hsc:
            data_df = pd.read_csv('./data/HSC_photoz_data.csv')
            data_df = data_df.sample(frac=1).reset_index(drop=True)

            # Trim down the data to redshift less than two for quality reasons
            data_df = data_df[data_df['specz_redshift'] <= 2]

            data_x_df = data_df[['g_cmodel_mag', 'r_cmodel_mag', 'i_cmodel_mag', 'z_cmodel_mag', 'y_cmodel_mag']]
            data_y_df = data_df[['specz_redshift']] * 10

        else:
            data_df = pd.read_csv('./data/SDSS_DR18.csv')
            data_df = data_df.sample(frac=1).reset_index(drop=True)

            # Trim down the data to redshift less than two for quality reasons
            data_df = data_df[(data_df['redshift'] <= 0.25) & (data_df['redshift'] > 0)]
            data_df = data_df[data_df['class'] == 'GALAXY']

            data_x_df = data_df[['u', 'g', 'r', 'i', 'z']]
            data_y_df = data_df[['redshift']]


        data_x_tr = torch.Tensor(data_x_df.to_numpy())
        data_y_tr = torch.Tensor(data_y_df.to_numpy())

        data_tensor_set = torch.utils.data.TensorDataset(data_x_tr, data_y_tr)

        self.train_dataset, self.test_dataset = torch.utils.data.random_split(data_tensor_set, [0.8, 0.2])

        # Then convert to DataLoaders and store in class
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, persistent_workers=False)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=0, persistent_workers=False)

    
    # The neural network class. Does all the heavy lifting for us!
    class NeuralNetwork(nn.Module):
        def __init__(self, n_input, n_output, n_hidden_layers: int = 2048):
            super().__init__()
            
            self.flatten = nn.Flatten()

            self.linear_relu_stack = nn.Sequential(
                nn.Linear(n_input, n_hidden_layers),
                nn.ReLU(),
                nn.Linear(n_hidden_layers, n_hidden_layers),
                nn.ReLU(),
                nn.Linear(n_hidden_layers, n_hidden_layers),
                nn.ReLU(),
                nn.Linear(n_hidden_layers, n_hidden_layers),
                nn.ReLU(),
                nn.Linear(n_hidden_layers, n_hidden_layers),
                nn.ReLU(),
                nn.Linear(n_hidden_layers, n_hidden_layers),
                nn.ReLU(),
                nn.Linear(n_hidden_layers, n_output)
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits
        
    # The function to train our model with
    def train(self, dataloader, n_epoch):
        size = len(dataloader.dataset)
        self.model.train()

        cum_loss = 0

        for batch, (X, y) in enumerate(dataloader):
            X = X.to(self.device)
            y = y.to(self.device)

            # Compute prediction error
            pred = self.model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()
            self.optimizer.zero_grad()

            cum_loss += loss.item()
        
        self.loss_train_hist[n_epoch] = cum_loss / size
        
        if VERBOSE and (n_epoch % 100 == 0):
            print(f'Training loss {(cum_loss / size):>8e}')

    def test(self, dataloader, n_epoch):
        size = len(dataloader.dataset)
        self.model.eval()
        
        test_loss = 0
        with torch.no_grad():
            for X, y in dataloader:
                X = X.to(self.device)
                y = y.to(self.device)

                pred = self.model(X)
                test_loss += loss_fn(pred, y).item()

        test_loss /= size

        if VERBOSE and (n_epoch % 100 == 0):
            print(f"Avg loss: {test_loss:>8e} \n")

        self.loss_test_hist[n_epoch] = test_loss

    def do_training(self):
        if TIMINGS:
            start_time = time.time()

        # Train the model!
        for n_epoch in range(self.n_epochs):
            if VERBOSE and (n_epoch % 100 == 0):
                print(f"Epoch {n_epoch}\n-------------------------------")
            
            self.train(self.train_dataloader, n_epoch)
            self.test(self.test_dataloader, n_epoch)

        if VERBOSE:
            print("Done!")
        
        if TIMINGS:
            time_delta = timedelta(seconds=time.time() - start_time)
            print(f'Time taken is {int(time_delta.total_seconds())} sec for ')


def compute_testing_samples(learn_class: LearnNdimPoly):
    """
    Class that takes a trained LearnNdimPoly object and evaluates the testing
    samples to obtain their best-estimates for the photometric redshifts.
    Returns spec-z and photo-z as Numpy arrays
    """
    true_spec_z = []
    est_photo_z = []

    learn_class.model.eval()
    with torch.no_grad():
        for X, y in learn_class.test_dataloader:
            X = X.to(learn_class.device)
            y = y.to(learn_class.device)

            pred = learn_class.model(X)

            y = y.to('cpu')
            pred = pred.to('cpu')
            
            true_spec_z.append(y)
            est_photo_z.append(pred)
            
    true_spec_z = torch.cat(true_spec_z, dim=0).numpy().flatten() / 10
    est_photo_z = torch.cat(est_photo_z, dim=0).numpy().flatten() / 10

    return true_spec_z, est_photo_z


def compute_testing_statistics(learn_class: LearnNdimPoly):
    """
    Function that computes statistics on the testing samples
    """
    from scipy.stats import pearsonr

    # Get the true spec-z and estimates photo-z values
    true_spec_z, est_photo_z = compute_testing_samples(learn_class)

    # Compute the Pearson-r statistic
    r_stat = pearsonr(true_spec_z, est_photo_z)
    print(f'The Pearson-r statistics are: {r_stat.statistic}, p-value: {r_stat.pvalue}')

    print(f'The final testing and training losses were: Train = {learn_class.loss_train_hist[-1]:.4e}, Test = {learn_class.loss_test_hist[-1]:.4e}')

    # Compute number of outliers and print
    num_outliers = np.sum(np.abs(est_photo_z - true_spec_z) > 0.01)
    num_catastrophic_outliers = np.sum(np.abs(est_photo_z - true_spec_z) > 0.05)

    print(f'The number of outliers are {num_outliers} which is {(num_outliers / len(true_spec_z) * 100):.2f}% of the total samples')
    print(f'The number of *catastrophic* outliers are {num_catastrophic_outliers} which is {(num_catastrophic_outliers / len(true_spec_z) * 100):.2f}% of the total samples')



def plot_training_hist(learn_classes: list[LearnNdimPoly]):
    #* Plot loss history
    from matplotlib import pyplot as plt
    import seaborn as sns
    sns.set(font_scale=1.0, rc={'text.usetex' : True})

    cmap = sns.color_palette('inferno', len(learn_classes))

    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, learn_class in enumerate(learn_classes):
        n_epochs = np.arange(1, learn_class.n_epochs + 1)
        ax.semilogy(n_epochs, learn_class.loss_train_hist, lw=2, c=cmap[idx], 
                    label=fr'$N_{{\mathrm{{epoch}}}}$={learn_class.n_epochs}, LR={learn_class.learning_rate}, $N_{{\mathrm{{batch}}}}$={learn_class.batch_size}')
        ax.semilogy(n_epochs, learn_class.loss_test_hist, lw=2, c=cmap[idx], ls='--')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Normalised loss history')

    ax.legend()

    fig.tight_layout()

    plt.savefig(f'{learn_classes[0].plots_dir}/Loss_history.pdf')


def plot_redshift_est(learn_classes: list[LearnNdimPoly]):
    """
    Function that plots the scatter graph of estimated photo-z redshift
    against true spec-z redshift
    """
    from matplotlib import pyplot as plt
    import seaborn as sns
    sns.set(font_scale=1.0, rc={'text.usetex' : True})
    sns.set_theme(style="whitegrid")

    for idx, learn_class in enumerate(learn_classes):
        fig, ax = plt.subplots(figsize=(6, 6))

        true_spec_z, est_photo_z = compute_testing_samples(learn_class)
        
        sns.scatterplot(x=true_spec_z, y=est_photo_z, s=5, color=".15")
        sns.histplot(x=true_spec_z, y=est_photo_z, bins=50, pthresh=.1, cmap="viridis")
        sns.kdeplot(x=true_spec_z, y=est_photo_z, levels=5, color="w", linewidths=1)

        ax.plot(np.linspace(0, 2), np.linspace(0, 2), c='grey', lw=2, ls='--')
        ax.set_xlim(left=0, right=2)
        ax.set_ylim(bottom=0, top=2)

        ax.set_title(fr'$N_{{\mathrm{{epoch}}}}$={learn_class.n_epochs}, LR={learn_class.learning_rate}, $N_{{\mathrm{{batch}}}}$={learn_class.batch_size}')
        ax.set_xlabel('True spec-$z$')
        ax.set_ylabel('Estimated photo-$z$')

        fig.tight_layout()
        plt.savefig(f'{learn_classes[0].plots_dir}/photo_z_binned_idx{idx}.pdf')

        g = sns.JointGrid(x=true_spec_z, y=est_photo_z, space=0)

        g.plot_joint(sns.histplot, bins=50, pthresh=.1, cmap="viridis")

        sns.histplot(x=true_spec_z, fill=False, linewidth=2, ax=g.ax_marg_x)
        sns.histplot(y=true_spec_z, fill=False, linewidth=2, ax=g.ax_marg_y, color='purple', alpha=1)
        sns.histplot(y=est_photo_z, fill=False, linewidth=2, ax=g.ax_marg_y, color='orange', alpha=0.75)

        g.ax_joint.set_xlim(left=0, right=2)
        g.ax_joint.set_ylim(bottom=0, top=2)
        g.ax_joint.plot(np.linspace(0, 2), np.linspace(0, 2), c='grey', lw=2, ls='--')

        g.ax_joint.set_xlabel('True spec-$z$')
        g.ax_joint.set_ylabel('Estimated photo-$z$')

        g.savefig(f'{learn_classes[0].plots_dir}/2D_plot_idx{idx}.pdf')



        # import pdb; pdb.set_trace()




# Save the model, optionally
# torch.save(model.state_dict(), "model.pth")
# print("Saved PyTorch Model State to model.pth")

# Optionally, load the model again
# model = NeuralNetwork().to(device)
# model.load_state_dict(torch.load("model.pth", weights_only=True))

N_EPOCHS = 2500

learn_1 = LearnNdimPoly(n_epochs=N_EPOCHS, learning_rate=1e-3, batch_size=256)
learn_1.generate_data_tensors()
learn_1.do_training()
compute_testing_statistics(learn_1)


plot_training_hist([learn_1])
plot_redshift_est([learn_1])

assert False

learn_2 = LearnNdimPoly(n_epochs=N_EPOCHS, learning_rate=1e-3, batch_size=128)
learn_2.generate_data_tensors()
learn_2.do_training()
compute_testing_statistics(learn_2)

learn_3 = LearnNdimPoly(n_epochs=N_EPOCHS, learning_rate=1e-2, batch_size=128)
learn_3.generate_data_tensors()
learn_3.do_training()
compute_testing_statistics(learn_3)

learn_4 = LearnNdimPoly(n_epochs=N_EPOCHS, learning_rate=1e-3, batch_size=64)
learn_4.generate_data_tensors()
learn_4.do_training()
compute_testing_statistics(learn_4)

learn_5 = LearnNdimPoly(n_epochs=N_EPOCHS, learning_rate=1e-3, batch_size=256)
learn_5.generate_data_tensors()
learn_5.do_training()
compute_testing_statistics(learn_5)

plot_training_hist([learn_1, learn_2, learn_3, learn_4, learn_5])
plot_redshift_est([learn_1, learn_2, learn_3, learn_4, learn_5])
