import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from train import tanhPlusOne, swish
from pathlib import Path

def get_feature_importance(test_X, test_y, model, Signal_Cut, n, recreate=False):
    # Create Lists for
    feature_importance = []
    feature_importance_unc = []
    
    outfile = Path('out/FI.npy')
    if outfile.is_file() and not recreate:
        feature_importance = np.load('out/FI.npy')
        feature_importance_unc = np.load('out/FI_unc.npy')
    else:
        # Generate default prediction
        print('Getting default prediction...')
        y_pred = model.predict(test_X, verbose=0)
        # Evaluate default performance for non permuated test data by using sklearn accuarcy_score
        print('Getting default score...')
        score_default = mean_squared_error(test_y, y_pred)
        # Looping over the 11 different features
        for j in range(test_X.shape[1]):
            print('Feature ',j,' running...')
            # Create list for
            score_with_perm = []
            # Iterate over the same feature several times for smaller uncertainties as permutations are random
            for i in range(n):
                print('\t Iternation: ',i)
                # Generate a random permutation
                perm = np.random.permutation(range(test_X.shape[0]))
                # Generate a copy for which the one feature is permutated
                X_test_ = test_X.copy()
                X_test_[:, j] = test_X[perm, j]
                # Predict the labels for the test data with the permutated feature
                y_pred_ = model.predict(X_test_, verbose=0)
                # Evaluate performance with sklearn accuracy_score and append to score_list
                s_ij = mean_squared_error(test_y, y_pred_)
                score_with_perm.append(s_ij)
            print('\n')
            # append the values to the lists before repeating for other features
            feature_importance.append(np.absolute(score_default - np.mean(score_with_perm))) # Calc score difference wrt to default
            feature_importance_unc.append(np.std(score_with_perm)) # Use std devation as uncertainity for the feature importance
        np.save('out/FI.npy', feature_importance)
        np.save('out/FI_unc.npy', feature_importance_unc)
    return feature_importance, feature_importance_unc

def plot_feature_importance(X, y, model, feature_names, iteration=10, recreate=False):
    feature_importance, uncertainty = get_feature_importance(X, y, model, 0.8, iteration, recreate)
    idx = np.argsort(feature_importance)
    uncertainty_sorted = np.array(uncertainty)[idx]

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.barh(range(X.shape[1]), np.sort(feature_importance), color="r", alpha=0.7)
    ax.set_yticks(range(X.shape[1]), np.array(feature_names)[idx])
    # Add error bars representing the variable uncertainty
    ax.errorbar(np.sort(feature_importance), range(X.shape[1]), xerr=uncertainty_sorted, fmt='o', color='black', alpha=0.7)
    ax.set_xlabel('Feature Importance')
    plt.title("Iterations, N="+str(iteration))
    plt.show()
    plt.savefig('out/feature_importance_'+str(iteration)+'.png')


def main():
    dir_path = 'out'
    model = tf.keras.models.load_model(dir_path+"/model.h5", compile=False, custom_objects={'swish': swish, 'tanhPlusOne': tanhPlusOne})
    X = np.load(dir_path + '/x_test.npy')
    y = np.load(dir_path + '/trueResponse.npy')
    feature_names = ['clusterE', 'clusterEta',
                        'cluster_CENTER_LAMBDA', 'cluster_CENTER_MAG', 'cluster_ENG_FRAC_EM', 'cluster_FIRST_ENG_DENS',
                        'cluster_LATERAL', 'cluster_LONGITUDINAL', 'cluster_PTD', 'cluster_time', 'cluster_ISOLATION',
                        'cluster_SECOND_TIME', 'cluster_SIGNIFICANCE', 'nPrimVtx', 'avgMu',]
    print('Checking data sizes...')
    print('\t X shape: ', X.shape)
    print('\t y shape: ', y.shape)

    plot_feature_importance(X, y, model, feature_names, 1, True)

if __name__ == '__main__':
    main()
    pass