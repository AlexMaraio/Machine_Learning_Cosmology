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
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(font_scale=1.0, rc={'text.usetex' : True})

# Manually set the seed for reproducibility
# torch.manual_seed(0)


# Define loss function (Mean Squared Error)
# def loss_fn(predictions, targets):
    # return torch.mean(torch.abs(predictions - targets) / targets)
    # return torch.mean((predictions - targets) ** 2)
    # return torch.mean((predictions - targets) ** 2 / (targets ** 2))

loss_fn = nn.CrossEntropyLoss()


def accuracy_fn(y_true, y_pred):
    y_pred = torch.max(y_pred, 1)[1]
    correct = (y_pred == y_true).sum().item()
    acc = (correct / len(y_pred)) * 100 
    return correct


VERBOSE = False
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
        self.n_input: int = 31

        # We have a single output, the photo-z
        self.n_output: int = 3

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
        self.acc_train_hist = np.zeros(self.n_epochs)
        self.acc_test_hist = np.zeros(self.n_epochs)

        # The plots directory
        self.plots_dir: str = './Plots_SDSS'

        # If the plots dir does not exist, then make it
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)


    def generate_data_tensors(self):
        # Load in our SDSS csv file
        data_df = pd.read_csv('./data/SDSS_DR18.csv')
        data_df = data_df.sample(frac=1).reset_index(drop=True)

        # Select our inputs and outputs
        data_x_df = data_df[['u', 'g', 'r', 'i', 'z', 'petroRad_u', 'petroRad_g', 'petroRad_i', 'petroRad_r', 'petroRad_z', 'petroFlux_u', 'petroFlux_g', 'petroFlux_i', 'petroFlux_r', 'petroFlux_z', 'petroR50_u', 'petroR50_g', 'petroR50_i', 'petroR50_r', 'petroR50_z', 'psfMag_u', 'psfMag_r', 'psfMag_g', 'psfMag_i', 'psfMag_z', 'expAB_u', 'expAB_g', 'expAB_r', 'expAB_i', 'expAB_z', 'redshift']]
        data_y_df = data_df[['class']]

        data_y_df = data_y_df['class'].map({'GALAXY': 0, 'STAR': 1, 'QSO': 2})

        # Convert our data-frames to tensors
        data_x_tr = torch.Tensor(data_x_df.to_numpy(dtype=np.long))
        data_y_tr = torch.Tensor(data_y_df.to_numpy(dtype=np.long))

        data_tensor_set = torch.utils.data.TensorDataset(data_x_tr, data_y_tr)

        self.train_dataset, self.test_dataset = torch.utils.data.random_split(data_tensor_set, [0.8, 0.2])

        # Then convert to DataLoaders and store in class
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, persistent_workers=False)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=0, persistent_workers=False)

    
    # The neural network class. Does all the heavy lifting for us!
    class NeuralNetwork(nn.Module):
        def __init__(self, n_input, n_output, n_hidden_layers: int = 512):
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
        accuracy = 0

        for batch, (X, y) in enumerate(dataloader):
            y = y.type(torch.LongTensor)
            X = X.to(self.device)
            y = y.to(self.device)

            # Compute prediction error
            pred = self.model(X)

            # import pdb; pdb.set_trace()
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()
            self.optimizer.zero_grad()

            # import pdb; pdb.set_trace()

            cum_loss += loss.item()
            accuracy += accuracy_fn(y, pred)
        
        self.loss_train_hist[n_epoch] = cum_loss / size
        self.acc_train_hist[n_epoch] = accuracy / size
        
        if VERBOSE and (n_epoch % 100 == 0):
            print(f'Training loss {(cum_loss / size):>8e}')
            print(f'Training accuracy {(accuracy / size)*100:.2f}%')

    def test(self, dataloader, n_epoch):
        size = len(dataloader.dataset)
        self.model.eval()
        
        test_loss = 0
        test_accuracy = 0
        with torch.no_grad():
            for X, y in dataloader:
                y = y.type(torch.LongTensor)
                X = X.to(self.device)
                y = y.to(self.device)

                pred = self.model(X)
                test_loss += loss_fn(pred, y).item()
                test_accuracy += accuracy_fn(y, pred)

        test_loss /= size
        test_accuracy /= size

        if VERBOSE and (n_epoch % 100 == 0):
            print(f"Test loss: {test_loss:>8e}")
            print(f"Test accuracy: {test_accuracy*100:.2f}% \n")

        self.loss_test_hist[n_epoch] = test_loss
        self.acc_test_hist[n_epoch] = test_accuracy

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
            print(f'Time taken is {int(time_delta.total_seconds())} sec')

    def get_testing_samples(self):
        """
        Class that takes a trained LearnNdimPoly object and evaluates the testing
        samples to obtain their best-estimates for the photometric redshifts.
        Returns spec-z and photo-z as Numpy arrays
        """
        train_x = []
        train_y = []

        self.model.eval()
        with torch.no_grad():
            for X, y in self.test_dataloader:
                X = X.to(self.device)
                y = y.to(self.device)

                pred = self.model(X)

                X = X.to('cpu')
                y = y.to('cpu')
                # pred = pred.to('cpu')
                
                train_x.append(X)
                train_y.append(y)
                # est_photo_z.append(pred)
                
        train_x = torch.cat(train_x, dim=0)#.numpy().flatten()
        train_y = torch.cat(train_y, dim=0)#.numpy().flatten()

        return train_x, train_y

    def compute_testing_statistics(self, n_iter: int = 0):
        """
        Function that computes statistics on the testing samples
        """
        classes = ('Galaxies', 'Stars', 'QSOs')
        
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        confusion_matrix = np.zeros([len(classes), len(classes)])

        # again no gradients needed
        with torch.no_grad():
            for data in self.test_dataloader:
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                predictions = torch.max(outputs, 1)[1]

                labels = labels.to(dtype=torch.int)
                predictions = predictions.to(dtype=torch.int)

                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1

                    # Store the (label, pred) in our confusion matrix
                    confusion_matrix[label.cpu(), prediction.cpu()] += 1

        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

        conf_mat_df = pd.DataFrame(confusion_matrix / np.sum(confusion_matrix, axis=1)[:, None], index = [i for i in classes], columns = [i for i in classes])
        conf_mat_df.loc[:] *= 100

        fig, ax = plt.subplots(figsize = (12,7))
        sns.heatmap(conf_mat_df, cmap='plasma', annot=True, fmt='.2f', ax=ax, cbar_kws={'label': 'Percentage of objects'})

        ax.set_xlabel('True input')
        ax.set_ylabel('Recovered output')

        plt.tight_layout()
        plt.savefig(f'{self.plots_dir}/Confusion_matrix_normd_{n_iter}.pdf')
        
        conf_mat_df = pd.DataFrame(confusion_matrix , index = [i for i in classes], columns = [i for i in classes])

        fig, ax = plt.subplots(figsize = (12,7))
        sns.heatmap(conf_mat_df, cmap='plasma', annot=True, fmt='.0f', ax=ax, cbar_kws={'label': 'Number of objects'})

        ax.set_xlabel('True input')
        ax.set_ylabel('Recovered output')

        plt.tight_layout()
        plt.savefig(f'{self.plots_dir}/Confusion_matrix_no_norm_{n_iter}.pdf')

        # Now try and use feature importance
        # test_x, test_y = self.get_testing_samples()
        # unpermuted_rmse_pct = self.loss_test_hist[-1]
        # rng = np.random.default_rng(0)

        # n_repeats = 50 #number of trials per feature
        # permutation_metrics = np.empty([31, n_repeats])

        # #Shuffle each feature in turn, and get model's score
        # for col_idx in range(31):
        #     X_val_perm = test_x.clone()
            
        #     for repeat in range(n_repeats):
        #         shuffled_ixs = rng.permutation(len(test_x))
        #         X_val_perm[:, col_idx] = test_x[shuffled_ixs, col_idx]
                
        #         val_loader = DataLoader(X_val_perm, batch_size=self.batch_size, shuffle=True)
        #         permutation_metrics[col_idx, repeat] = self.test(val_loader, 0)

        # #Convert to change in score compared to unpermuted data
        # permutation_df = pd.DataFrame(permutation_metrics.T, columns=range(31)) - unpermuted_rmse_pct

        # permutation_melt = permutation_df.melt(var_name='feature', value_name='permuted_rmse_pct')
        # sns.boxplot(permutation_melt, y='feature', x='permuted_rmse_pct')
        # ax = sns.stripplot(permutation_melt, y='feature', x='permuted_rmse_pct', marker='.', color='tab:red')

        # ax.set_xlabel('drop in performance')
        # ax.set_ylabel('permuted feature')
        # ax.figure.set_size_inches(8, 2.5)
        # plt.savefig(f'{self.plots_dir}/Permuted_drop1.pdf')

        # #Bar chart of feature importances
        # normalised_scores = permutation_df.mean(axis=0) / permutation_df.mean(axis=0).sum() #scores 0-1
        # ax = sns.barplot(normalised_scores, color='tab:purple')
        # ax.tick_params(axis='x', rotation=45)
        # ax.set_xlabel('feature')
        # ax.set_ylabel('feature importance')
        # ax.figure.set_size_inches(4, 3)
        # plt.savefig(f'{self.plots_dir}/Permuted_drop2.pdf')




        # from sklearn.inspection import permutation_importance

        # test_x, test_y = self.get_testing_samples()

        # feat_imp = permutation_importance(self.model, test_x, test_y,  n_repeats=50)

        # sns.barplot(feat_imp.importances_mean)
        # plt.tight_layout()
        # plt.savefig(f'{self.plots_dir}/Feature_importance.pdf')


        # import pdb; pdb.set_trace()


def plot_training_hist(learn_classes: list[LearnNdimPoly]):
    #* Plot loss history
    cmap = sns.color_palette('tab20', len(learn_classes) * 2)

    fig, ax = plt.subplots(figsize=(8, 5))

    idx = 0
    for learn_class in learn_classes:
        n_epochs = np.arange(1, learn_class.n_epochs + 1)
        ax.semilogy(n_epochs, learn_class.loss_train_hist, lw=2, c=cmap[idx], 
                    label=fr'$N_{{\mathrm{{epoch}}}}$={learn_class.n_epochs}, LR={learn_class.learning_rate}, $N_{{\mathrm{{batch}}}}$={learn_class.batch_size}')
        ax.semilogy(n_epochs, learn_class.loss_test_hist, lw=2, c=cmap[idx + 1], ls='--')

        idx += 2

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Normalised loss history')

    ax.legend()

    fig.tight_layout()

    plt.savefig(f'{learn_classes[0].plots_dir}/Loss_history.pdf')
    
    # Accuracy plot
    fig, ax = plt.subplots(figsize=(8, 5))

    idx = 0
    for learn_class in learn_classes:
        n_epochs = np.arange(1, learn_class.n_epochs + 1)
        ax.plot(n_epochs, learn_class.acc_train_hist * 100, lw=2, c=cmap[idx], 
                    label=fr'$N_{{\mathrm{{epoch}}}}$={learn_class.n_epochs}, LR={learn_class.learning_rate}, $N_{{\mathrm{{batch}}}}$={learn_class.batch_size}')
        ax.plot(n_epochs, learn_class.acc_test_hist * 100, lw=2, c=cmap[idx + 1], ls='--')

        idx += 2

    ax.set_xlabel('Epoch')
    ax.set_ylabel(r'Accuracy (\%)')
    ax.set_ylim(top=100)

    ax.legend()

    fig.tight_layout()

    plt.savefig(f'{learn_classes[0].plots_dir}/Accuracy_history.pdf')


N_EPOCHS = 500

learn_1 = LearnNdimPoly(n_epochs=N_EPOCHS, learning_rate=1e-3, batch_size=256)
learn_1.generate_data_tensors()
learn_1.do_training()
learn_1.compute_testing_statistics(n_iter=1)

learn_2 = LearnNdimPoly(n_epochs=N_EPOCHS, learning_rate=1e-3, batch_size=128)
learn_2.generate_data_tensors()
learn_2.do_training()
learn_2.compute_testing_statistics(n_iter=2)

learn_3 = LearnNdimPoly(n_epochs=N_EPOCHS, learning_rate=1e-2, batch_size=128)
learn_3.generate_data_tensors()
learn_3.do_training()
learn_3.compute_testing_statistics(n_iter=3)

learn_4 = LearnNdimPoly(n_epochs=N_EPOCHS, learning_rate=1e-3, batch_size=64)
learn_4.generate_data_tensors()
learn_4.do_training()
learn_4.compute_testing_statistics(n_iter=4)

learn_5 = LearnNdimPoly(n_epochs=N_EPOCHS, learning_rate=1e-3, batch_size=256)
learn_5.generate_data_tensors()
learn_5.do_training()
learn_5.compute_testing_statistics(n_iter=5)


plot_training_hist([learn_1, learn_2, learn_3, learn_4, learn_5])
