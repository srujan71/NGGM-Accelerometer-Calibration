from PIL import Image

# path_uncorrected = "C:/Users/sruja/Desktop/Astro_class_code/nggm/SimulationOutput/figures/Verification/Orbit/Plot3D/Uncorrected/"
path_corrected = "C:/Users/sruja/Desktop/Astro_class_code/nggm/SimulationOutput/figures/Verification/Orbit/Plot3D/Corrected/"

image_files_uncorrected = []
image_files_corrected = []
for i in range(145):
    # image_files_uncorrected.append(path_uncorrected + f"{i*100}.png")
    image_files_corrected.append(path_corrected + f"{i*100}.png")

# images_uncorrected = [Image.open(image) for image in image_files_uncorrected]
images_corrected = [Image.open(image) for image in image_files_corrected]

# Save the images as a GIF
# images_uncorrected[0].save('output_uncorrected.gif', save_all=True, append_images=images_uncorrected[1:], duration=100, loop=0)
images_corrected[0].save('output_corrected.gif', save_all=True, append_images=images_corrected[1:], duration=100, loop=0)
