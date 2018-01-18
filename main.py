from model import *
import tensorflow as tf
import numpy as np

LATENT_SIZE = 100
SEED = 111 #seed for shuffle data

TRAINING_EPOCHS = 100
BATCH_SIZE = 100
NAME = 'GAN_MNIST'


def main():
    images, _ = load_mnist_datasets(SEED)
    sess = tf.Session()
    model = GAN(sess, latent_size=100, name=NAME)
    total_batch = int(len(images) / BATCH_SIZE)
    for epoch in range(TRAINING_EPOCHS):
        average_d_loss = 0
        average_g_loss = 0

        for idx in range(total_batch):
            batch_images = images[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
            latents = np.random.normal(size=(BATCH_SIZE, LATENT_SIZE))
            d_loss, g_loss = model.train(latents, batch_images)
            average_d_loss += d_loss / total_batch
            average_g_loss += g_loss / total_batch
        print(average_g_loss, average_d_loss)
        if epoch % 10 == 0:
            print(epoch)
            # if you want to generate images using same latent variables,
            # please takes out below lines to outside of loop
            test_latents = np.random.normal(size=(BATCH_SIZE, LATENT_SIZE))
            generated_images = model.generating_images(test_latents)
            reshaped_and_save_images(generated_images, model.result_path, epoch)

    print('Learning Finished')


if __name__ == '__main__':
    main()