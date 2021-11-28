from MLP_alg import MLP_alogorythm

if __name__ == '__main__':
    mlp = MLP_alogorythm()
    epochs_tab=[]
    skut=[]
    for _ in range(10):
        epochs,howmuch=mlp.learn_algorythm_batch(7, -0.2, 0.2, 0.01, 0.1, 10,100)
        print("Epochs:",epochs, "skutecznosc: ", howmuch*100,"%")
        epochs_tab.append(epochs)
        skut.append(howmuch)
    print(epochs_tab)
    print(skut)


           