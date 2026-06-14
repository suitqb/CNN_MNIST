def convert(imgs, labels, outfile, n):
    imgf = open(imgs, 'rb')
    labelf = open(labels, 'rb')
    csvf = open(outfile, "w")
    imgf.read(16)
    labelf.read(8)
    images = []
    for i in range(n):
        image = [ord(labelf.read(1))]
        for j in range(28*28):
            image.append(ord(imgf.read(1)))
        images.append(image)
    for image in images:
        csvf.write(",".join(str(pix) for pix in image)+"\n")
    imgf.close()
    labelf.close()
    csvf.close()


mnist_train_x = "/home/suito/Documents/Projet/CNN_MNIST/mnist/train-images.idx3-ubyte"
mnist_train_y = "/home/suito/Documents/Projet/CNN_MNIST/mnist/train-labels.idx1-ubyte"
mnist_test_x = "/home/suito/Documents/Projet/CNN_MNIST/mnist/t10k-images.idx3-ubyte"
mnist_test_y = "/home/suito/Documents/Projet/CNN_MNIST/mnist/t10k-labels.idx1-ubyte"
convert(mnist_train_x, mnist_train_y,
        "/home/suito/Documents/Projet/CNN_MNIST/train.csv", 60000)
convert(mnist_test_x, mnist_test_y,
        "/home/suito/Documents/Projet/CNN_MNIST/test.csv", 10000)
