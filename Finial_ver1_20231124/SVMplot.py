
import torch

import numpy as np
import torchvision.transforms as T
import matplotlib.pyplot as plt
from getcat import getactboxmapver3
from imageget import getimage
from SVM_ver3 import SVMget
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
torch.set_grad_enabled(False)


"""
h,w,pred= SVMget(imageid)
im= getimage(imageid)
transform_show= T.Resize((h,w))

"""




def plotoverview(imageid):
    h, w, pred = SVMget(imageid)
    im = getimage(imageid)
    transform_show = T.Resize((h, w))
    img_show = transform_show(im)
    plt.imshow(img_show)
    pred_f = pred.view(h, w)
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    # Flatten x, y, and buffer_f for scatter plot
    x, y, sizes = x.flatten(), y.flatten(), pred_f.flatten()
    plt.scatter(x, y, s=sizes*100, c=sizes, cmap='BrBG_r', alpha=0.6)  # Multiplying size for visibility
    plt.colorbar(label='Value')


    ### get the annotation bbox location and map
    Act_tensor = getactboxmapver3(h,w,imageid)
    x_act, y_act = np.meshgrid(np.arange(w), np.arange(h))
    x_act, y_act, sizes = x_act.flatten(), y_act.flatten(), Act_tensor.flatten()
    plt.scatter(x_act, y_act, s=sizes * 100, c=sizes, cmap='PRGn', alpha=0.6)
    plt.show()




def plotannolassifer(imageid):
    h, w, pred = SVMget(imageid)
    im = getimage(imageid)
    transform_show = T.Resize((h, w))
    img_show = transform_show(im)
    plt.imshow(img_show)

    ### get the annotation bbox location and map
    Act_tensor = getactboxmapver3(h, w, imageid)
    x_act, y_act = np.meshgrid(np.arange(w), np.arange(h))
    x_act, y_act, sizes = x_act.flatten(), y_act.flatten(), Act_tensor.flatten()
    plt.scatter(x_act, y_act, s=sizes * 100, c=sizes, cmap='Reds', alpha=0.6)
    plt.show()






def plotattnclassifer(imageid):
    h, w, pred = SVMget(imageid)
    im = getimage(imageid)
    transform_show = T.Resize((h, w))

    pred_f = pred.view(h, w)
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    # Flatten x, y, and buffer_f for scatter plot
    x_pred, y_pred, sizes_pred = x.flatten(), y.flatten(), pred_f.flatten()
    train = np.column_stack((x_pred, y_pred, sizes_pred))

    ### get the annotation bbox location and map
    Act_tensor = getactboxmapver3(h, w, imageid)
    x_act, y_act = np.meshgrid(np.arange(w), np.arange(h))
    x_act, y_act, sizes_act = x_act.flatten(), y_act.flatten(), Act_tensor.flatten()
    test = [x_act, y_act, sizes_act]

    plt.scatter(test[0], test[1], c=np.where(test[2], 'red', 'blue'))
    kernel = 1.0 * RBF(length_scale=1.0)  # You can adjust the length_scale parameter
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

    #plt.scatter(train[:, 0], train[:, 1], c=np.where(train[:, 2], 'red', 'blue'), label='Training Data')

    # Gaussian Process Regressor
    kernel = 1.0 * RBF(length_scale=1.0) #Radial Basis Function (RBF) kernel SVM.
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

    x_train = train[:, :2]  # Extract the input features from train
    y_train = train[:, 2]  # Extract the target values from train
    gp.fit(x_train, y_train)

    x1_values = np.linspace(min(x_train[:, 0]), max(x_train[:, 0]), 100)
    x2_values = np.linspace(min(x_train[:, 1]), max(x_train[:, 1]), 100)
    x1_grid, x2_grid = np.meshgrid(x1_values, x2_values)
    X_grid = np.column_stack((x1_grid.ravel(), x2_grid.ravel()))

    # Predict using the GPR model
    y_pred, sigma = gp.predict(X_grid, return_std=True)
    y_pred = y_pred.reshape(x1_grid.shape)

    # Plot the GPR predictions
    plt.contourf(x1_grid, x2_grid, y_pred, cmap='viridis', alpha=0.5, levels=20)
    plt.colorbar(label='Predicted Value')
    plt.xlabel('Attention width')
    plt.ylabel('Attention height')
    plt.title('Scatter Plot with Encoder Attention Gaussian Process Regressor Predictions')
    plt.legend()
    plt.show()






