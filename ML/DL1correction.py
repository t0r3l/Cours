import keras
import numpy as np

#GPU desktop pour dl, + de ram classique que de video, ce qui compte le plus c'est le nombre de cahes GPU pour éviter les allers retours sur la RAM
#Recommandation si intéressé par hardware for DL computerenhance.com ou youtube Casey Muratori pour comprendre les GPU de a à z.


#Lorsqu'on parle de gradient descent propre, on parle de momentum pour acccélérer les descentes si aucune pente ne se présente

#momentum classique avec petit learning rate donne de bons réqultats mais expérimenter avec tout car il 
#théorème du pas de repas gratuit (il existera toujours un modèle qui poura prouver )

#categorical crossentropy => pas de sigmoïde ni tangente hyperbolique

#Sign SGD sur gd batch size peut être efficace

#Important de comprendre les papiers de recherche pour être le plus rapide en milieu concurrenciel 
#en pouvant implémenter une solution  sans dépendre de bibliothèques

#Il n'est pas possible de comparer les losses entre elles, pour comparer l'efficience d'un modèle il faut douc calculer l'accuracy

#Adam devient intéressant avec petits batch size et ?

#When writing a file it is not being written in real time but in a buffer which will send data at the end of of the writting
#To force writing instreaming directly to a file use flush



#Théorème Lepnitz&Kitz déterminer overfitting il faut avoir 10 plus d'individus d'entraînement que de couches
#=> lorsque trop nombreux paramètres d'entrée => overfiting

#Deep learning 2010 !!
#Pour décoréler le nombre de paramètres d'entrée de la fiabilité des modèles  => matrices de convolutions 


def my_softmax(x: keras.KerastTensor):
    x_max = keras.ops.max(x, axis=1, keepdims=True)
    negative_x = x - x_max
    return keras.ops.exp(x) /keras.ops.sum(keras.ops.exp(x))

def my_categorical_cross_entropy(y_true: keras.KerasTensor, y_pred: keras.KerasTensor):
    return - keras.ops.sum(y_true * keras.ops.log(y_pred + 1e-7), axis=1, keepdims=True)

def exp_tanh():
    for seed in [42, 51, 89, 66, 16]:
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

        keras.utils.set_random_seed(seed)

        model = keras.models.Sequential([
            keras.layers.Flatten(),
            #Première couche cachée à 32 entrées
            keras.layers.Dense(32, activation=keras.activations.tanh),
            #Deuxième couche cachée à 16 entrées
            keras.layers.Dense(16, activation=keras.activations.tanh),
            #Fonction  de seuil?
            keras.layers.Dense(10, activation=keras.activations.tanh),
        ]
        )

        x_test = (x_test - np.mean(x_train)) / np.std(x_train)
        x_train = (x_train - np.mean(x_train)) / np.std(x_train)
        
        y_test = keras.utils.to_categorical(y_train, num_classes:10) * 2.0 - 1.0
        y_train = keras.utils.to_categorical(y_train, num_classes:10) * 2.0 - 1.0
        

        exp_name = f"tanh_and_mse_seed_{seed}"

        model.compile(
            loss=keras.losses.mean_squared_error,
            optimizer=keras.optimizers.SGD(),
            metrics=[keras.metrics.categorical_accuracy]
        )

        model.fit(x_train, y_train, 32, 100,
                  callbacks=[keras.callbacks.TensorBoard(log_dir=f'./logs/{exp_name}')])

def exp_softmax():
    for seed in [42, 51, 89, 66, 16]:
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

        keras.utils.set_random_seed(seed)

        model = keras.models.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(32, activation=keras.activations.tanh),
            keras.layers.Dense(16, activation=keras.activations.tanh),
            keras.layers.Dense(10, activation=keras.activations.softmax),
        ]
        )

        x_train = (x_train - np.mean(x_train)) / np.std(x_train)
        x_test = (x_test - np.mean(x_train)) / np.std(x_train)

        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_train, 10)

        exp_name = f"softmax_and_cat_xentropy_seed_{seed}"

        model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.SGD(),
            metrics=[keras.metrics.categorical_accuracy]
        )

        model.fit(x_train, y_train, 32, 100,
                  callbacks=[keras.callbacks.TensorBoard(log_dir=f'./logs/{exp_name}')])

def exp_softmax_with_mse():
    for seed in [42, 51, 89, 66, 16]:
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

        keras.utils.set_random_seed(seed)

        model = keras.models.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(32, activation=keras.activations.tanh),
            keras.layers.Dense(16, activation=keras.activations.tanh),
            keras.layers.Dense(10, activation=keras.activations.softmax),
        ]
        )

        x_train = (x_train - np.mean(x_train)) / np.std(x_train)
        x_test = (x_test - np.mean(x_train)) / np.std(x_train)

        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_train, 10)

        exp_name = f"softmax_and_mse_seed_{seed}"

        model.compile(
            loss=keras.losses.mean_squared_error,
            optimizer=keras.optimizers.SGD(),
            metrics=[keras.metrics.categorical_accuracy]
        )

        model.fit(x_train, y_train, 32, 100,
                  callbacks=[keras.callbacks.TensorBoard(log_dir=f'./logs/{exp_name}')])

def run():
    exp_tanh()
    exp_softmax()
    exp_softmax_with_mse()
    my_softmax(x: keras.KerastTensor)

if __name__ == "__main__":
    run()


#Faire attention lorsqu'on compare les courbes des loss fonctionnes notament à l'évchelle des fonctions d'activation
