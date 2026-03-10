# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset


## DESIGN STEPS

### STEP 1:


Import the required libraries and load the image dataset. Preprocess the images and add artificial noise to create noisy input data. 

### STEP 2:

Design a Convolutional Autoencoder model consisting of an encoder (for feature extraction and dimensionality reduction) and a decoder (for reconstructing the clean image from encoded features).
### STEP 3:

Train the model using noisy images as input and original images as target output using Mean Squared Error (MSE) loss and the Adam optimizer.

## PROGRAM

### Name:SANTHABABU  G
### Register Number:212224040292

```
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # [1,28,28] -> [32,14,14]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # [32,14,14] -> [64,7,7]
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # [64,7,7] -> [32,14,14]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),   # [32,14,14] -> [1,28,28]
            nn.Sigmoid()  # Output between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()               # Mean Squared Error for reconstruction
optimizer = optim.Adam(model.parameters(), lr=0.001)
summary(model, input_size=(1, 28, 28))


def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)

            optimizer.zero_grad()
            outputs = model(noisy_images)
            loss = criterion(outputs, images)   # Compare denoised output vs original clean image
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")


def visualize_denoising(model, loader, num_images=4):
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            break

    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()
    outputs = outputs.cpu().numpy()

    print("Name: SANTHABABU  G                  ")
    print("Register Number: 212224040292                 ")
    plt.figure(figsize=(18, 6))
    for i in range(num_images):
        # Original
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title("Original")
        plt.axis("off")

        # Noisy
        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(noisy_images[i].squeeze(), cmap='gray')
        ax.set_title("Noisy")
        plt.axis("off")

        # Denoised
        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        ax.set_title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


```

## OUTPUT

### Model Summary

<img width="654" height="435" alt="image" src="https://github.com/user-attachments/assets/e273567c-386e-47cc-a4c2-8e3bfc7a132f" />


### Original vs Noisy Vs Reconstructed Image

<img width="1727" height="788" alt="image" src="https://github.com/user-attachments/assets/766cd907-6ba8-43e7-acb6-1b279133486e" />


## RESULT

The Convolutional Autoencoder model was successfully implemented for image denoising. The model removed noise from the input images and reconstructed images close to the original images.
